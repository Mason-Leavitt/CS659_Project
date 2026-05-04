import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from agent.parameter_sweep_planner import build_parameter_sweep_plan
from agent import parameter_sweep_runner


class ParameterSweepRunnerTests(unittest.TestCase):
    def test_chart_files_are_generated_per_varied_parameter(self):
        plan = build_parameter_sweep_plan(
            dataset_path="dataset_sample",
            parameter_ranges={
                "top_k": [1, 3, 5],
                "max_images": [50, 100],
            },
        )
        rows = [
            {
                "varied_parameter": "top_k",
                "varied_value": 1,
                "tflite_top1_accuracy": 0.70,
                "hog_top1_accuracy": 0.65,
            },
            {
                "varied_parameter": "top_k",
                "varied_value": 3,
                "tflite_top1_accuracy": 0.75,
                "hog_top1_accuracy": 0.68,
            },
            {
                "varied_parameter": "top_k",
                "varied_value": 5,
                "tflite_top1_accuracy": 0.78,
                "hog_top1_accuracy": 0.70,
            },
            {
                "varied_parameter": "max_images",
                "varied_value": 50,
                "tflite_top1_accuracy": 0.60,
                "hog_top1_accuracy": 0.58,
            },
            {
                "varied_parameter": "max_images",
                "varied_value": 100,
                "tflite_top1_accuracy": 0.66,
                "hog_top1_accuracy": 0.61,
            },
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            charts = parameter_sweep_runner._write_parameter_sweep_charts(
                plan=plan,
                summary={"baseline_values": dict(plan["baseline_values"])},
                rows=rows,
                output_dir=Path(tmpdir),
            )

            self.assertEqual(len(charts), 2)
            self.assertTrue((Path(tmpdir) / "charts" / "sweep_top_k_metrics.png").is_file())
            self.assertTrue((Path(tmpdir) / "charts" / "sweep_max_images_metrics.png").is_file())

    def test_chart_metadata_is_returned(self):
        plan = build_parameter_sweep_plan(
            dataset_path="dataset_sample",
            parameter_ranges={"top_k": [1, 3]},
        )
        rows = [
            {
                "varied_parameter": "top_k",
                "varied_value": 1,
                "tflite_top1_accuracy": 0.5,
                "hog_top1_accuracy": 0.4,
            },
            {
                "varied_parameter": "top_k",
                "varied_value": 3,
                "tflite_top1_accuracy": 0.6,
                "hog_top1_accuracy": 0.45,
            },
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            charts = parameter_sweep_runner._write_parameter_sweep_charts(
                plan=plan,
                summary={"baseline_values": dict(plan["baseline_values"])},
                rows=rows,
                output_dir=Path(tmpdir),
            )

        self.assertEqual(charts[0]["varied_parameter"], "top_k")
        self.assertIn("tflite_top1_accuracy", charts[0]["metrics"])
        self.assertIn("hog_top1_accuracy", charts[0]["metrics"])
        self.assertIn("sampling_mode", charts[0]["baseline_values"])
        self.assertEqual(charts[0]["filename"], "sweep_top_k_metrics.png")

    def test_missing_metrics_are_not_treated_as_zero(self):
        plan = build_parameter_sweep_plan(
            dataset_path="dataset_sample",
            parameter_ranges={"top_k": [1, 3]},
        )
        rows = [
            {
                "varied_parameter": "top_k",
                "varied_value": 1,
                "tflite_top1_accuracy": 0.0,
                "hog_top1_accuracy": 0.4,
            },
            {
                "varied_parameter": "top_k",
                "varied_value": 3,
                "tflite_top1_accuracy": None,
                "hog_top1_accuracy": 0.45,
            },
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            charts = parameter_sweep_runner._write_parameter_sweep_charts(
                plan=plan,
                summary={"baseline_values": dict(plan["baseline_values"])},
                rows=rows,
                output_dir=Path(tmpdir),
            )

        self.assertTrue(any("missing values" in warning for warning in charts[0]["warnings"]))
        self.assertIn("tflite_top1_accuracy", charts[0]["metrics"])

    def test_invalid_plan_does_not_create_output_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch.object(
                parameter_sweep_runner,
                "_parameter_sweep_output_root",
                return_value=Path(tmpdir),
            ):
                result = parameter_sweep_runner.run_parameter_sweep(
                    {"success": False, "study_type": "parameter_sweep", "errors": ["bad plan"]},
                )

                self.assertEqual(list(Path(tmpdir).iterdir()), [])

        self.assertFalse(result["success"])
        self.assertEqual(result["status"], "invalid_plan")
        self.assertIsNone(result["output_dir"])

    def test_unsafe_plan_requires_confirmation_and_does_not_run_by_default(self):
        plan = {
            "success": True,
            "study_type": "parameter_sweep",
            "require_confirmation": True,
            "dataset_path": "dataset_sample",
            "generated_sweep_points": [{"parameter_values": {"top_k": 1}}],
            "selected_metrics": ["tflite_top1_accuracy"],
            "warnings": [],
        }
        result = parameter_sweep_runner.run_parameter_sweep(plan)
        self.assertFalse(result["success"])
        self.assertEqual(result["status"], "invalid_plan")
        self.assertIn("requires confirmation", " ".join(result["errors"]))

    def test_runner_executes_one_point_per_generated_sweep_point(self):
        plan = build_parameter_sweep_plan(
            dataset_path="dataset_sample",
            parameter_ranges={
                "top_k": [1, 3],
                "max_images": [50],
            },
            baseline_values={
                "top_k": 5,
                "sampling_mode": "balanced",
                "max_images": 200,
                "max_images_per_class": 1,
                "seed": 42,
            },
        )
        calls: list[dict] = []

        def fake_run_point(*, plan, sweep_dir, sweep_id, point_index, point):
            calls.append(
                {
                    "point_index": point_index,
                    "point": point,
                    "sweep_dir": sweep_dir,
                    "sweep_id": sweep_id,
                }
            )
            return {
                "sweep_point_id": f"point_{point_index:03d}",
                "point_index": point_index,
                "varied_parameter": point["varied_parameter"],
                "varied_value": point["varied_value"],
                "parameter_values": dict(point["parameter_values"]),
                "status": "completed",
                "manifest": {"success": True, "manifest_path": "manifest.csv"},
                "ablation": {
                    "success": True,
                    "summary": {
                        "num_images_evaluated": 10,
                        "num_classes": 2,
                        "metrics": {
                            "tflite_top1_accuracy": 0.9,
                            "tflite_topk_accuracy": 1.0,
                            "hog_top1_accuracy": 0.8,
                            "hog_topk_accuracy": 0.95,
                            "model_agreement_rate": 0.7,
                        },
                    },
                },
                "warning": None,
                "error": None,
            }

        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch.object(
                parameter_sweep_runner,
                "_parameter_sweep_output_root",
                return_value=Path(tmpdir),
            ), mock.patch.object(parameter_sweep_runner, "_run_sweep_point", side_effect=fake_run_point):
                result = parameter_sweep_runner.run_parameter_sweep(plan)

        self.assertTrue(result["success"])
        self.assertEqual(len(calls), 3)
        self.assertEqual(calls[0]["point"]["parameter_values"]["top_k"], 1)
        self.assertEqual(calls[0]["point"]["parameter_values"]["max_images"], 200)
        self.assertEqual(calls[1]["point"]["parameter_values"]["top_k"], 3)
        self.assertEqual(calls[1]["point"]["parameter_values"]["max_images"], 200)
        self.assertEqual(calls[2]["point"]["parameter_values"]["top_k"], 5)
        self.assertEqual(calls[2]["point"]["parameter_values"]["max_images"], 50)

    def test_results_csv_and_summary_files_are_written(self):
        plan = build_parameter_sweep_plan(
            dataset_path="dataset_sample",
            parameter_ranges={"top_k": [1]},
        )

        def fake_run_point(*, plan, sweep_dir, sweep_id, point_index, point):
            return {
                "sweep_point_id": "point_001_top_k_1",
                "point_index": point_index,
                "varied_parameter": point["varied_parameter"],
                "varied_value": point["varied_value"],
                "parameter_values": dict(point["parameter_values"]),
                "status": "completed",
                "manifest": {"success": True, "manifest_path": "manifest.csv"},
                "ablation": {
                    "success": True,
                    "summary": {
                        "num_images_evaluated": 5,
                        "num_classes": 3,
                        "metrics": {
                            "tflite_top1_accuracy": 0.8,
                            "tflite_topk_accuracy": 0.8,
                            "hog_top1_accuracy": 0.6,
                            "hog_topk_accuracy": 0.6,
                            "model_agreement_rate": 0.4,
                        },
                    },
                },
                "warning": None,
                "error": None,
            }

        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch.object(
                parameter_sweep_runner,
                "_parameter_sweep_output_root",
                return_value=Path(tmpdir),
            ), mock.patch.object(parameter_sweep_runner, "_run_sweep_point", side_effect=fake_run_point):
                result = parameter_sweep_runner.run_parameter_sweep(plan)
                sweep_dir = Path(result["output_dir"])

                self.assertTrue((sweep_dir / "parameter_sweep_plan.json").is_file())
                self.assertTrue((sweep_dir / "parameter_sweep_results.csv").is_file())
                self.assertTrue((sweep_dir / "parameter_sweep_summary.json").is_file())
                self.assertTrue((sweep_dir / "parameter_sweep_summary.md").is_file())
                self.assertTrue((sweep_dir / "charts" / "sweep_top_k_metrics.png").is_file())

                with (sweep_dir / "parameter_sweep_results.csv").open("r", encoding="utf-8", newline="") as fh:
                    rows = list(csv.DictReader(fh))
                self.assertEqual(len(rows), 1)
                self.assertEqual(rows[0]["varied_parameter"], "top_k")

                summary_payload = json.loads((sweep_dir / "parameter_sweep_summary.json").read_text(encoding="utf-8"))
                self.assertEqual(summary_payload["total_completed_runs"], 1)
                self.assertEqual(summary_payload["charts"][0]["varied_parameter"], "top_k")
                markdown_text = (sweep_dir / "parameter_sweep_summary.md").read_text(encoding="utf-8")
                self.assertIn("Generated Charts", markdown_text)
                self.assertIn("sweep_top_k_metrics.png", markdown_text)

    def test_failed_point_is_recorded_and_remaining_points_continue(self):
        plan = build_parameter_sweep_plan(
            dataset_path="dataset_sample",
            parameter_ranges={"top_k": [1, 3]},
        )
        statuses = ["completed", "failed"]

        def fake_run_point(*, plan, sweep_dir, sweep_id, point_index, point):
            status = statuses[point_index - 1]
            success = status == "completed"
            return {
                "sweep_point_id": f"point_{point_index:03d}",
                "point_index": point_index,
                "varied_parameter": point["varied_parameter"],
                "varied_value": point["varied_value"],
                "parameter_values": dict(point["parameter_values"]),
                "status": status,
                "manifest": {"success": True, "manifest_path": "manifest.csv"},
                "ablation": {
                    "success": success,
                    "summary": {
                        "num_images_evaluated": 5 if success else 0,
                        "num_classes": 2 if success else None,
                        "metrics": {},
                    },
                },
                "warning": None,
                "error": None if success else "mock failure",
            }

        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch.object(
                parameter_sweep_runner,
                "_parameter_sweep_output_root",
                return_value=Path(tmpdir),
            ), mock.patch.object(parameter_sweep_runner, "_run_sweep_point", side_effect=fake_run_point):
                result = parameter_sweep_runner.run_parameter_sweep(plan)

        self.assertFalse(result["success"])
        self.assertEqual(result["status"], "partial_success")
        self.assertEqual(result["summary"]["total_completed_runs"], 1)
        self.assertEqual(result["summary"]["total_failed_runs"], 1)
        self.assertEqual(result["results"][1]["status"], "failed")

    def test_run_sweep_point_passes_top_k_and_disables_history(self):
        plan = build_parameter_sweep_plan(
            dataset_path="dataset_sample",
            parameter_ranges={"top_k": [3]},
        )
        point = plan["generated_sweep_points"][0]

        with tempfile.TemporaryDirectory() as tmpdir:
            sweep_dir = Path(tmpdir)
            with mock.patch.object(
                parameter_sweep_runner,
                "build_manifest",
                return_value={"success": True, "manifest_path": str(sweep_dir / "manifest.csv")},
            ) as build_manifest_mock, mock.patch.object(
                parameter_sweep_runner,
                "run_sample_ablation",
                return_value={"success": True, "summary": {"metrics": {}}},
            ) as run_ablation_mock:
                result = parameter_sweep_runner._run_sweep_point(
                    plan=plan,
                    sweep_dir=sweep_dir,
                    sweep_id="parameter_sweep_test",
                    point_index=1,
                    point=point,
                )

        self.assertEqual(result["status"], "completed")
        build_manifest_mock.assert_called_once()
        run_ablation_mock.assert_called_once()
        kwargs = run_ablation_mock.call_args.kwargs
        self.assertEqual(kwargs["summary_top_k"], 3)
        self.assertEqual(kwargs["top_k_values"], (1, 3))
        self.assertFalse(kwargs["append_history"])
        self.assertFalse(kwargs["write_charts"])

    def test_no_chart_generation_when_disabled(self):
        plan = build_parameter_sweep_plan(
            dataset_path="dataset_sample",
            parameter_ranges={"top_k": [1]},
        )

        def fake_run_point(*, plan, sweep_dir, sweep_id, point_index, point):
            return {
                "sweep_point_id": "point_001_top_k_1",
                "point_index": point_index,
                "varied_parameter": point["varied_parameter"],
                "varied_value": point["varied_value"],
                "parameter_values": dict(point["parameter_values"]),
                "status": "completed",
                "manifest": {"success": True, "manifest_path": "manifest.csv"},
                "ablation": {
                    "success": True,
                    "summary": {
                        "num_images_evaluated": 5,
                        "num_classes": 3,
                        "metrics": {
                            "tflite_top1_accuracy": 0.8,
                            "hog_top1_accuracy": 0.6,
                        },
                    },
                },
                "warning": None,
                "error": None,
            }

        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch.object(
                parameter_sweep_runner,
                "_parameter_sweep_output_root",
                return_value=Path(tmpdir),
            ), mock.patch.object(parameter_sweep_runner, "_run_sweep_point", side_effect=fake_run_point):
                result = parameter_sweep_runner.run_parameter_sweep(plan, write_charts=False)
                sweep_dir = Path(result["output_dir"])

                self.assertEqual(result["summary"]["charts"], [])
                self.assertNotIn("charts_dir", result["artifacts"])
                self.assertFalse((sweep_dir / "charts").exists())


if __name__ == "__main__":
    unittest.main()
