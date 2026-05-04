import unittest
from unittest.mock import patch

from agent import agent_tools
from agent.plant_chat_agent import (
    _asks_for_run_sample_ablation,
    _asks_for_parameter_sweep,
    _asks_to_plan_parameter_sweep,
    _asks_to_run_parameter_sweep,
    _fallback_router,
    _format_parameter_sweep_plan,
    _format_parameter_sweep_result,
)

FAILED_PROMPT = (
    "run inference only parameter sweep using the following details. "
    "Base paramateres - top_k: 5, max_images: 100, seed: 50, sampling_mode: sorted, max_images_per_class: 5 "
    "sweeps - top_k: 1,3,5,8,10 sampleng_mode: balance,random,sorted "
    "max_images: 10,25,50,100,200 max_images_per_class: 1,2,5,7,10 seeds: 7,25,42,123,200 "
    "metrics to plot are: TFLite top-1 accuracy, HOG+SVM top-1 accuracy, TFLite top-k accuracy, "
    "HOG+SVM top-k accuracy, agrrement rate"
)


class _DummySweepRunner:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def run_parameter_sweep(self, plan, *, allow_unsafe=False, write_charts=True):
        self.calls.append(
            {
                "plan": plan,
                "allow_unsafe": allow_unsafe,
                "write_charts": write_charts,
            }
        )
        return self.response


class ParameterSweepAgentSmokeTests(unittest.TestCase):
    def test_parameter_sweep_plan_intent_matches_target_prompts(self):
        prompts = [
            "Plan a parameter sweep varying top_k 1,3,5.",
            "Create a one-factor sweep with max_images 50,100,200.",
            "Use OFAT to vary seed 7,42,123.",
        ]
        for prompt in prompts:
            self.assertTrue(_asks_for_parameter_sweep(prompt), msg=prompt)
            self.assertTrue(_asks_to_plan_parameter_sweep(prompt), msg=prompt)
            self.assertFalse(_asks_for_run_sample_ablation(prompt), msg=prompt)

    def test_parameter_sweep_run_intent_matches_target_prompts(self):
        prompts = [
            "Run the parameter sweep we planned.",
            "Run a parameter sweep for top_k 1,3,5.",
        ]
        for prompt in prompts:
            self.assertTrue(_asks_for_parameter_sweep(prompt), msg=prompt)
            self.assertTrue(_asks_to_run_parameter_sweep(prompt), msg=prompt)

    def test_exact_failed_prompt_routes_to_parameter_sweep_handling(self):
        self.assertTrue(_asks_for_parameter_sweep(FAILED_PROMPT))
        self.assertTrue(_asks_to_run_parameter_sweep(FAILED_PROMPT))
        self.assertFalse(_asks_for_run_sample_ablation(FAILED_PROMPT))

    def test_unsupported_parameter_rejection_is_clear(self):
        response = _fallback_router("Can I sweep SVM C?")
        self.assertIn("not currently supported by the inference-only one-factor parameter sweep", response)
        self.assertIn("top_k", response)

    def test_plan_formatter_summarizes_valid_plan(self):
        formatted = _format_parameter_sweep_plan(
            {
                "success": True,
                "request": "Plan a parameter sweep varying top_k 1,3,5.",
                "supported_parameters": ["top_k", "sampling_mode", "max_images", "max_images_per_class", "seed"],
                "parameter_ranges": {"top_k": [1, 3, 5]},
                "baseline_values": {
                    "top_k": 5,
                    "sampling_mode": "balanced",
                    "max_images": 200,
                    "max_images_per_class": 1,
                    "seed": 42,
                },
                "selected_metrics": ["tflite_top1_accuracy", "hog_top1_accuracy"],
                "total_planned_runs": 3,
                "generated_sweep_points": [
                    {
                        "varied_parameter": "top_k",
                        "varied_value": 1,
                        "parameter_values": {
                            "top_k": 1,
                            "sampling_mode": "balanced",
                            "max_images": 200,
                            "max_images_per_class": 1,
                            "seed": 42,
                        },
                    }
                ],
                "warnings": [],
            }
        )
        self.assertIn("Inference-only", formatted)
        self.assertIn("Baseline values", formatted)
        self.assertIn("Total planned runs: 3", formatted)
        self.assertIn("top_k", formatted)
        self.assertIn("Selected metrics", formatted)

    def test_run_formatter_summarizes_result(self):
        formatted = _format_parameter_sweep_result(
            {
                "success": True,
                "status": "partial_success",
                "output_dir": r"D:\tmp\parameter_sweep_20260503_142701",
                "artifacts": {
                    "plan_json": r"D:\tmp\parameter_sweep_20260503_142701\parameter_sweep_plan.json",
                    "results_csv": r"D:\tmp\parameter_sweep_20260503_142701\parameter_sweep_results.csv",
                },
                "summary": {
                    "total_planned_runs": 5,
                    "total_completed_runs": 4,
                    "total_failed_runs": 1,
                    "charts": [
                        {"filename": "sweep_top_k_metrics.png"},
                        {"filename": "sweep_max_images_metrics.png"},
                    ],
                },
                "warnings": [],
                "errors": [],
            }
        )
        self.assertIn("partial_success", formatted)
        self.assertIn("parameter_sweep_20260503_142701", formatted)
        self.assertIn("Completed: 4", formatted)
        self.assertIn("sweep_top_k_metrics.png", formatted)
        self.assertIn("parameter_sweep_results.csv", formatted)

    def test_tool_wrapper_planning_returns_plan_dictionary(self):
        result = agent_tools.plan_parameter_sweep(
            request="Plan a parameter sweep varying top_k 1,3,5 and max_images 50,100.",
        )
        self.assertTrue(result["success"])
        self.assertEqual(result["parameter_ranges"]["top_k"], [1, 3, 5])
        self.assertEqual(result["parameter_ranges"]["max_images"], [50, 100])
        self.assertIn("dataset_path was not provided", " ".join(result["warnings"]))

    def test_tool_wrapper_running_respects_plan_validity_and_safety(self):
        valid_plan = {
            "success": True,
            "study_type": "parameter_sweep",
            "dataset_path": "dataset_sample",
            "split": "test",
            "generated_sweep_points": [
                {
                    "varied_parameter": "top_k",
                    "varied_value": 1,
                    "parameter_values": {
                        "top_k": 1,
                        "sampling_mode": "balanced",
                        "max_images": 200,
                        "max_images_per_class": 1,
                        "seed": 42,
                    },
                }
            ],
            "selected_metrics": ["tflite_top1_accuracy"],
            "warnings": [],
            "errors": [],
            "require_confirmation": False,
        }
        dummy_runner = _DummySweepRunner(
            {
                "success": True,
                "status": "completed",
                "sweep_id": "parameter_sweep_fake",
                "output_dir": "result\\parameter_sweep_fake",
                "plan": valid_plan,
                "results": [],
                "summary": {},
                "artifacts": {},
                "errors": [],
                "warnings": [],
            }
        )
        invalid_result = agent_tools.run_parameter_sweep_tool(plan={"success": False, "errors": ["bad plan"]})
        self.assertFalse(invalid_result["success"])
        self.assertEqual(invalid_result["status"], "invalid_plan")

        unsafe_plan = dict(valid_plan)
        unsafe_plan["require_confirmation"] = True
        unsafe_result = agent_tools.run_parameter_sweep_tool(plan=unsafe_plan)
        self.assertFalse(unsafe_result["success"])
        self.assertEqual(unsafe_result["status"], "requires_confirmation")

        with patch("agent.agent_tools._load_parameter_sweep_runner_module", return_value=dummy_runner):
            success_result = agent_tools.run_parameter_sweep_tool(plan=valid_plan, write_charts=False)
        self.assertTrue(success_result["success"])
        self.assertEqual(len(dummy_runner.calls), 1)
        self.assertFalse(dummy_runner.calls[0]["write_charts"])

    def test_exact_failed_prompt_produces_useful_run_summary_without_real_execution(self):
        fake_result = {
            "success": True,
            "status": "completed",
            "sweep_id": "parameter_sweep_fake",
            "output_dir": r"D:\tmp\parameter_sweep_fake",
            "plan": {},
            "results": [],
            "summary": {
                "total_planned_runs": 25,
                "total_completed_runs": 25,
                "total_failed_runs": 0,
                "charts": [{"filename": "sweep_top_k_metrics.png"}],
            },
            "artifacts": {
                "results_csv": r"D:\tmp\parameter_sweep_fake\parameter_sweep_results.csv",
            },
            "errors": [],
            "warnings": [],
        }
        with patch(
            "agent.plant_chat_agent.agent_tools.run_parameter_sweep_tool",
            return_value=fake_result,
        ):
            response = _fallback_router(FAILED_PROMPT)
        self.assertIn("inference-only", response)
        self.assertIn("Completed: 25", response)
        self.assertIn("sweep_top_k_metrics.png", response)
        self.assertIn("parameter_sweep_results.csv", response)

    def test_exact_failed_prompt_plan_summary_is_useful(self):
        plan = agent_tools.plan_parameter_sweep(request=FAILED_PROMPT)
        formatted = _format_parameter_sweep_plan(plan)
        self.assertIn("top_k", formatted)
        self.assertIn("sampling_mode", formatted)
        self.assertIn("max_images", formatted)
        self.assertIn("max_images_per_class", formatted)
        self.assertIn("seed", formatted)
        self.assertIn("Total planned runs: 23", formatted)
        self.assertIn("Selected metrics", formatted)
        self.assertIn("Inference-only", formatted)

    def test_langchain_registration_includes_parameter_sweep_tools_when_available(self):
        if not agent_tools.LANGCHAIN_AVAILABLE:
            self.skipTest("LangChain tool registry is not available in this environment.")
        names = {getattr(tool, "name", "") for tool in agent_tools.AGENT_TOOLS}
        self.assertIn("get_parameter_sweep_support", names)
        self.assertIn("plan_parameter_sweep", names)
        self.assertIn("run_parameter_sweep_tool", names)


if __name__ == "__main__":
    unittest.main()
