import unittest
from unittest.mock import patch

from agent.plant_chat_agent import (
    _SYSTEM_PROMPT,
    _asks_for_artifacts,
    _asks_for_ablation_control_explanation,
    _asks_for_ablation_history,
    _asks_for_ablation_options,
    _asks_for_ablation_results,
    _asks_for_ablation_recommendations,
    _asks_for_metrics,
    _asks_for_run_sample_ablation,
    _fallback_router,
    _format_ablation_control_detail_response,
    _format_ablation_comparison,
    _format_ablation_options_response,
    _format_ablation_recommendations,
)


class FallbackRoutingSmokeTests(unittest.TestCase):
    def test_system_prompt_mentions_ablation_control_tool_guidance(self):
        prompt = _SYSTEM_PROMPT
        self.assertIn("explain_ablation_control", prompt)
        self.assertIn("top_k", prompt)
        self.assertIn("max_images_per_class", prompt)
        self.assertIn("sampling_mode", prompt)
        self.assertTrue("SVM C" in prompt or "svm_c" in prompt)
        self.assertTrue("HOG pixels per cell" in prompt or "hog_pixels_per_cell" in prompt)
        self.assertIn("inference-only", prompt)

    def test_metrics_helper_accepts_generic_model_metrics(self):
        self.assertTrue(_asks_for_metrics("show model metrics"))

    def test_metrics_helper_rejects_ablation_results_phrase(self):
        self.assertFalse(_asks_for_metrics("show ablation results"))

    def test_metrics_helper_rejects_ambiguous_results_prompts(self):
        self.assertFalse(_asks_for_metrics("what do the results mean?"))
        self.assertFalse(_asks_for_metrics("can you explain the results?"))
        self.assertFalse(_asks_for_metrics("what does the report mean?"))

    def test_ablation_results_helper_matches_result_report_phrases(self):
        self.assertTrue(_asks_for_ablation_results("show ablation results"))
        self.assertTrue(_asks_for_ablation_results("download ablation report"))
        self.assertTrue(_asks_for_ablation_results("compare ablation runs"))

    def test_ablation_results_helper_rejects_ambiguous_results_prompts(self):
        self.assertFalse(_asks_for_ablation_results("what do the results mean?"))
        self.assertFalse(_asks_for_ablation_results("can you explain the results?"))
        self.assertFalse(_asks_for_ablation_results("what does the report mean?"))

    def test_ablation_history_helper_matches_history_phrase(self):
        self.assertTrue(_asks_for_ablation_history("show ablation history"))

    def test_metrics_helper_matches_specific_metrics_prompts(self):
        self.assertTrue(_asks_for_metrics("show model metrics"))
        self.assertTrue(_asks_for_metrics("summarize classifier metrics"))
        self.assertTrue(_asks_for_metrics("show TFLite metrics"))
        self.assertTrue(_asks_for_metrics("what are the HOG metrics?"))

    def test_recommendation_helper_still_matches_specific_prompts(self):
        self.assertTrue(_asks_for_ablation_recommendations("recommend ablation settings"))
        self.assertTrue(_asks_for_ablation_recommendations("what ablation should i run next?"))

    def test_artifact_status_helper_still_matches_availability_prompts(self):
        self.assertTrue(_asks_for_artifacts("check model artifacts"))
        self.assertTrue(_asks_for_artifacts("are the model files available?"))
        self.assertTrue(_asks_for_artifacts("is the HOG model available?"))

    def test_run_sample_ablation_helper_still_matches_run_prompts(self):
        self.assertTrue(_asks_for_run_sample_ablation("plan a balanced 200 image ablation study"))
        self.assertTrue(_asks_for_run_sample_ablation("run ablation on the default dataset"))

    def test_classical_ablation_options_helper_matches_failed_prompt(self):
        message = (
            "I want to conduct an ablation study using the classical approach. "
            "What hyperparameters are available for me to experiment with?"
        )
        self.assertTrue(_asks_for_ablation_options(message))

    def test_classical_ablation_options_helper_matches_related_prompts(self):
        self.assertTrue(_asks_for_ablation_options("What HOG+SVM settings can I vary in an ablation?"))
        self.assertTrue(_asks_for_ablation_options("What parameters can I experiment with for the classical model?"))
        self.assertTrue(_asks_for_ablation_options("What ablation options are available for HOG?"))

    def test_ablation_control_explanation_helper_matches_direct_questions(self):
        self.assertTrue(_asks_for_ablation_control_explanation("What does max_images_per_class mean?"))
        self.assertTrue(_asks_for_ablation_control_explanation("Explain sampling_mode."))
        self.assertTrue(_asks_for_ablation_control_explanation("What is top_k?"))
        self.assertTrue(_asks_for_ablation_control_explanation("Why is balanced sampling preferred?"))
        self.assertTrue(_asks_for_ablation_control_explanation("What does color_correct do?"))
        self.assertTrue(_asks_for_ablation_control_explanation("What does SVM C mean for this ablation?"))

    def test_ablation_control_explanation_helper_does_not_overtrigger(self):
        self.assertFalse(_asks_for_ablation_control_explanation("What does this result mean?"))
        self.assertFalse(_asks_for_ablation_control_explanation("Explain the image."))
        self.assertFalse(_asks_for_ablation_control_explanation("What is a model?"))

    def test_fallback_router_answers_direct_control_questions_without_artifact_status(self):
        response = _fallback_router(message="What does max_images_per_class mean?")
        self.assertIn("Maximum images per class", response)
        self.assertIn("supported inference-time ablation control", response)
        self.assertNotIn("TFLite model available:", response)

        response = _fallback_router(message="What does SVM C mean for this ablation?")
        self.assertIn("SVM C", response)
        self.assertIn("not currently exposed", response)

    def test_fallback_router_answers_classical_ablation_options_without_artifact_status(self):
        message = (
            "I want to conduct an ablation study using the classical approach. "
            "What hyperparameters are available for me to experiment with?"
        )
        response = _fallback_router(message=message)
        self.assertIn("inference-only", response)
        self.assertIn("HOG+SVM", response)
        self.assertIn("top_k", response)
        self.assertIn("sampling_mode", response)
        self.assertIn("max_images", response)
        self.assertIn("seed", response)
        self.assertNotIn("TFLite model available:", response)
        self.assertNotIn("HOG+SVM model available:", response)

    def test_ablation_options_formatter_mentions_fixed_training_time_settings(self):
        response = _format_ablation_options_response(
            "What HOG+SVM settings can I vary in an ablation?"
        )
        self.assertIn("inference-only", response)
        self.assertIn("HOG+SVM", response)
        self.assertIn("top_k", response)
        self.assertIn("sampling_mode", response)
        self.assertIn("max_images", response)
        self.assertIn("seed", response)
        self.assertIn("not expose classical training-time hyperparameters", response)

    def test_ablation_options_formatter_uses_shared_helper_output(self):
        mocked_controls = {
            "success": True,
            "inference_only": True,
            "shared_sampling_controls": ["split", "sampling_mode", "max_images", "max_images_per_class", "seed"],
            "shared_sampling_control_details": [
                {"name": "split", "values": ["eval", "test"], "description": "Which dataset split to evaluate."},
                {"name": "sampling_mode", "values": ["balanced", "sorted"], "description": "How images are sampled into the manifest."},
                {"name": "max_images", "suggested_values": [12, 34], "description": "Global sample size for the ablation run."},
                {"name": "max_images_per_class", "suggested_value": 2, "description": "Limits how many images can be selected from each class."},
                {"name": "seed", "default": 99, "description": "Random seed used for reproducible manifest sampling."},
            ],
            "classical_controls": ["top_k"],
            "classical_control_details": [
                {"name": "top_k", "values": [2, 4], "description": "Controls how many ranked HOG+SVM class predictions are returned and evaluated."}
            ],
            "deep_learning_controls": ["color_correct"],
            "deep_learning_control_details": [
                {"name": "color_correct", "values": ["none", "max_rgb"], "description": "Preprocessing color correction mode applied before TFLite inference."}
            ],
            "not_currently_swept": ["svm_c", "svm_kernel"],
            "not_currently_swept_details": [
                {"name": "svm_c", "label": "SVM C"},
                {"name": "svm_kernel", "label": "SVM kernel"},
            ],
        }
        with patch("agent.plant_chat_agent.agent_tools.get_supported_ablation_controls", return_value=mocked_controls):
            response = _format_ablation_options_response("What HOG+SVM settings can I vary in an ablation?")
        self.assertIn("2, 4", response)
        self.assertIn("balanced, sorted", response)
        self.assertIn("12 (quick) and 34 (stronger)", response)
        self.assertIn("default of 99", response)
        self.assertIn("SVM C, SVM kernel", response)

    def test_ablation_control_detail_formatter_is_concise_and_metadata_backed(self):
        response = _format_ablation_control_detail_response(
            {
                "success": True,
                "matched_name": "top_k",
                "matched_label": "Top-k depth",
                "category": "classical_control",
                "detail": {
                    "name": "top_k",
                    "label": "Top-k depth",
                    "description": "Controls how many ranked predictions are returned and evaluated.",
                    "applies_to": ["HOG+SVM"],
                    "values": [1, 3, 5],
                    "default": 5,
                    "examples": ["1", "3", "5"],
                    "notes": ["This is an inference/evaluation setting."],
                },
                "is_supported_control": True,
                "is_not_currently_swept": False,
            }
        )
        self.assertIn("Top-k depth", response)
        self.assertIn("supported inference-time ablation control", response)
        self.assertIn("Supported values: 1, 3, 5.", response)
        self.assertIn("Default: 5.", response)

    def test_ablation_comparison_formatter_includes_comparison_warnings(self):
        response = _format_ablation_comparison(
            {
                "success": True,
                "explanation": "Compared the explicitly requested ablation runs.",
                "comparison_rows": [
                    {
                        "run_id": "run_a",
                        "split": "test",
                        "sampling_mode": "balanced",
                        "max_images": 200,
                        "tflite_topk_accuracy": 0.28,
                        "hog_topk_accuracy": 0.02,
                        "model_agreement_rate": 0.01,
                        "tflite_color_stability_rate": 0.23,
                    }
                ],
                "comparison_warnings": ["Requested run ID run_MISSING was not found and was omitted."],
            }
        )
        self.assertIn("Comparison warnings:", response)
        self.assertIn("run_MISSING", response)
        self.assertNotIn("{'comparison_warnings':", response)

    def test_ablation_comparison_formatter_includes_failure_details_for_missing_runs(self):
        response = _format_ablation_comparison(
            {
                "success": False,
                "error": "At least two valid requested runs are required.",
                "requested_run_ids": ["run_A", "run_MISSING"],
                "matched_run_ids": ["run_A"],
                "missing_run_ids": ["run_MISSING"],
                "comparison_warnings": [],
            }
        )
        self.assertIn("At least two valid requested runs are required.", response)
        self.assertIn("Requested run IDs:", response)
        self.assertIn("Matched run IDs:", response)
        self.assertIn("Missing run IDs:", response)
        self.assertIn("run_MISSING", response)

    def test_recommendation_formatter_avoids_raw_dictionary_dump(self):
        result = {
            "success": True,
            "enough_history": True,
            "recommendation_status": "evidence_supported",
            "evidence_summary": "Best comparable group has 2 runs on the same balanced 200-image test configuration.",
            "recommended_config": {
                "profile": "robust",
                "dataset_label": "plantnet_300K",
                "split": "test",
                "sampling_mode": "balanced",
                "max_images": 200,
                "max_images_per_class": 1,
                "seed": 42,
            },
            "supporting_metrics": {
                "num_runs": 2,
                "evaluated_image_range": "200",
                "class_count_range": "200",
                "best_tflite_top1": 0.15,
                "best_tflite_topk": 0.285,
                "best_hog_top1": 0.0,
                "best_hog_topk": 0.02,
                "best_agreement": 0.015,
                "best_stability": 0.235,
            },
            "caveats": [
                "Accuracy metrics are only meaningful when manifest labels are available.",
                "Ablations are inference-only; no retraining is performed.",
            ],
            "next_best_actions": [
                "Run a balanced 50-image quick check.",
                "Repeat the same configuration with a different seed.",
            ],
            "recommendations": {
                "recommended_quick_run": {"split": "test", "sampling_mode": "balanced"},
                "recommended_robust_run": {"split": "test", "sampling_mode": "balanced"},
            },
            "history_summary": {"count": 6},
        }
        formatted = _format_ablation_recommendations(result)
        self.assertIn("Recommendation status: evidence_supported", formatted)
        self.assertIn("Recommended configuration:", formatted)
        self.assertIn("Supporting metrics:", formatted)
        self.assertIn("Caveats:", formatted)
        self.assertIn("Next best actions:", formatted)
        self.assertNotIn("{'split': 'test'", formatted)
        self.assertNotIn("recommended_quick_run={'", formatted)


if __name__ == "__main__":
    unittest.main()
