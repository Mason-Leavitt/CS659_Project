import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agent import ablation_history


def _entry(
    *,
    dataset_path="E:/school/659/plantnet_300K/plantnet_300K",
    split="test",
    sampling_mode="balanced",
    max_images=200,
    max_images_per_class=1,
    seed=42,
    num_images_evaluated=200,
    num_classes=200,
    run_id="run_1",
    output_dir="D:/tmp/sample_ablation_1",
    tflite_top1=0.10,
    tflite_topk=0.25,
    hog_top1=0.0,
    hog_topk=0.01,
    agreement=0.01,
    stability=0.20,
):
    return {
        "run_id": run_id,
        "created_at": "2026-05-03T00:00:00Z",
        "dataset_path": dataset_path,
        "split": split,
        "sampling_mode": sampling_mode,
        "max_images": max_images,
        "max_images_per_class": max_images_per_class,
        "seed": seed,
        "num_images_evaluated": num_images_evaluated,
        "num_classes": num_classes,
        "output_dir": output_dir,
        "metrics": {
            "tflite_top1_accuracy": tflite_top1,
            "tflite_topk_accuracy": tflite_topk,
            "hog_top1_accuracy": hog_top1,
            "hog_topk_accuracy": hog_topk,
            "model_agreement_rate": agreement,
            "tflite_color_stability_rate": stability,
        },
    }


def _group_summaries(entries):
    grouped = {}
    for item in entries:
        grouped.setdefault(ablation_history._group_key(item), []).append(item)
    return [
        ablation_history._build_group_summary(group_entries)
        for group_entries in grouped.values()
    ]


class AblationReportingSmokeTests(unittest.TestCase):
    def test_iso_timestamp_short_label_uses_month_day_hour_minute(self):
        labels = ablation_history.make_unique_ablation_run_labels(
            [{"created_at": "2026-05-03T14:27:01Z", "run_id": "run_a"}]
        )
        self.assertEqual(len(labels), 1)
        self.assertIn("0503_1427", labels[0])

    def test_run_id_timestamp_short_label_fallback(self):
        labels = ablation_history.make_unique_ablation_run_labels(
            [{"created_at": None, "run_id": "sample_ablation_20260503_142701"}]
        )
        self.assertEqual(len(labels), 1)
        self.assertIn("0503_1427", labels[0])

    def test_output_dir_timestamp_short_label_fallback(self):
        labels = ablation_history.make_unique_ablation_run_labels(
            [{"created_at": None, "run_id": None, "output_dir": "D:/tmp/sample_ablation_20260503_142701"}]
        )
        self.assertEqual(len(labels), 1)
        self.assertIn("0503_1427", labels[0])

    def test_duplicate_short_labels_become_unique(self):
        labels = ablation_history.make_unique_ablation_run_labels(
            [
                {"created_at": "2026-05-03T14:27:01Z", "run_id": "run_a"},
                {"created_at": "2026-05-03T14:27:45Z", "run_id": "run_b"},
            ]
        )
        self.assertEqual(len(labels), 2)
        self.assertNotEqual(labels[0], labels[1])
        self.assertTrue(all(label.startswith("0503_1427") for label in labels))

    def test_unknown_format_short_label_fallback_is_non_empty_and_short(self):
        labels = ablation_history.make_unique_ablation_run_labels(
            [{"created_at": None, "run_id": "some_unusual_run_identifier_with_long_prefix"}]
        )
        self.assertEqual(len(labels), 1)
        self.assertTrue(labels[0])
        self.assertLessEqual(len(labels[0]), 16)

    def test_history_markdown_includes_report_sections_and_values(self):
        entries = [
            _entry(run_id="run_a", seed=42, tflite_top1=0.10, tflite_topk=0.26),
            _entry(run_id="run_b", seed=7, tflite_top1=0.15, tflite_topk=0.285, agreement=0.015, stability=0.235),
            _entry(run_id="run_c", sampling_mode="random", max_images_per_class=None, num_classes=114, tflite_top1=0.465, tflite_topk=0.735, hog_top1=0.02, hog_topk=0.045, agreement=0.02, stability=0.345),
        ]
        markdown = ablation_history._build_history_markdown(
            entries,
            _group_summaries(entries),
            [],
            {
                "history_jsonl": "D:/tmp/ablation_history.jsonl",
                "table_csv": "D:/tmp/ablation_history_table.csv",
                "summary_json": "D:/tmp/ablation_history_summary.json",
                "summary_markdown": "D:/tmp/ablation_history_summary.md",
                "metrics_png": "D:/tmp/ablation_history_metrics.png",
            },
            "2026-05-03T00:00:00Z",
        )
        for section in (
            "# Ablation History Summary Report",
            "## Scope / Data Source",
            "## Executive Summary",
            "## Best Observed Runs",
            "## Comparable Group Summary",
            "## Recommendation Readiness / Evidence",
            "## Caveats and Interpretation Notes",
            "## Generated Artifacts",
        ):
            self.assertIn(section, markdown)
        for expected in ("run_c", "plantnet_300K", "balanced", "200", "1", "46.5%", "ablation_history_metrics.png"):
            self.assertIn(expected, markdown)

    def test_history_markdown_handles_sparse_old_records_without_crashing(self):
        sparse_entry = {
            "run_id": "old_run",
            "created_at": "2026-05-03T00:00:00Z",
            "dataset_path": None,
            "split": "test",
            "sampling_mode": "balanced",
            "max_images": 50,
            "max_images_per_class": None,
            "num_images_evaluated": None,
            "num_classes": None,
            "metrics": {},
        }
        markdown = ablation_history._build_history_markdown(
            [sparse_entry],
            _group_summaries([sparse_entry]),
            [],
            {
                "history_jsonl": "D:/tmp/ablation_history.jsonl",
                "table_csv": None,
                "summary_json": "D:/tmp/ablation_history_summary.json",
                "summary_markdown": "D:/tmp/ablation_history_summary.md",
                "metrics_png": None,
            },
            "2026-05-03T00:00:00Z",
        )
        self.assertIn("<unknown_dataset>", markdown)
        self.assertIn("balanced", markdown)
        self.assertIn("50", markdown)
        self.assertIn("## Caveats and Interpretation Notes", markdown)
        self.assertIn("missing dataset_path", markdown)

    def test_write_history_summary_writes_only_to_temporary_directory(self):
        entries = [_entry(run_id="run_a"), _entry(run_id="run_b", seed=7)]
        comparable_groups = _group_summaries(entries)
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            history_dir = base / "ablation_history"
            history_dir.mkdir()
            table_path = history_dir / "ablation_history_table.csv"
            table_path.write_text("run_id\nrun_a\n", encoding="utf-8")
            metrics_path = history_dir / "ablation_history_metrics.png"
            metrics_path.write_bytes(b"png")
            with patch.object(ablation_history, "_history_summary_json_path", return_value=history_dir / "ablation_history_summary.json"), patch.object(
                ablation_history, "_history_summary_md_path", return_value=history_dir / "ablation_history_summary.md"
            ), patch.object(ablation_history, "_history_path", return_value=history_dir / "ablation_history.jsonl"), patch.object(
                ablation_history, "_history_table_path", return_value=table_path
            ), patch.object(
                ablation_history, "_history_metrics_png_path", return_value=metrics_path
            ):
                result = ablation_history._write_history_summary(entries, comparable_groups, [])
            summary_json = Path(result["summary_json"])
            summary_md = Path(result["summary_markdown"])
            self.assertTrue(summary_json.is_file())
            self.assertTrue(summary_md.is_file())
            payload = json.loads(summary_json.read_text(encoding="utf-8"))
            self.assertEqual(payload["count"], 2)
            self.assertIn("## Generated Artifacts", summary_md.read_text(encoding="utf-8"))

    def test_export_ablation_history_artifacts_writes_only_through_explicit_export(self):
        entries = [_entry(run_id="run_a"), _entry(run_id="run_b", seed=7)]
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            history_dir = base / "ablation_history"
            history_dir.mkdir()
            with patch.object(
                ablation_history,
                "load_ablation_history",
                return_value={"success": True, "entries": entries, "corrupted_lines": []},
            ), patch.object(ablation_history, "_history_path", return_value=history_dir / "ablation_history.jsonl"), patch.object(
                ablation_history, "_history_table_path", return_value=history_dir / "ablation_history_table.csv"
            ), patch.object(
                ablation_history, "_history_summary_json_path", return_value=history_dir / "ablation_history_summary.json"
            ), patch.object(
                ablation_history, "_history_summary_md_path", return_value=history_dir / "ablation_history_summary.md"
            ), patch.object(
                ablation_history, "_history_metrics_png_path", return_value=history_dir / "ablation_history_metrics.png"
            ):
                result = ablation_history.export_ablation_history_artifacts()
            self.assertTrue(result["success"])
            self.assertTrue((history_dir / "ablation_history_table.csv").is_file())
            self.assertTrue((history_dir / "ablation_history_summary.json").is_file())
            self.assertTrue((history_dir / "ablation_history_summary.md").is_file())
            self.assertTrue((history_dir / "ablation_history_metrics.png").is_file())

    def test_compare_ablation_runs_returns_validity_and_rows_for_incompatible_runs(self):
        entries = [
            _entry(run_id="run_a", dataset_path="E:/school/659/plantnet_300K/plantnet_300K"),
            _entry(run_id="run_b", dataset_path="D:/other/dataset"),
        ]
        with patch.object(ablation_history, "load_ablation_history", return_value={"success": True, "entries": entries, "corrupted_lines": []}):
            result = ablation_history.compare_ablation_runs(run_ids=["run_a", "run_b"])
        self.assertTrue(result["success"])
        self.assertEqual(len(result["comparison_rows"]), 2)
        self.assertEqual(result["comparison_validity"]["status"], "not_direct")
        self.assertIn("dataset_path", result["comparison_validity"]["differing_fields"])
        self.assertEqual(result["requested_run_ids"], ["run_a", "run_b"])
        self.assertEqual(result["matched_run_ids"], ["run_a", "run_b"])
        self.assertEqual(result["missing_run_ids"], [])

    def test_compare_ablation_runs_seed_only_difference_stays_partial(self):
        entries = [
            _entry(run_id="run_a", seed=42),
            _entry(run_id="run_b", seed=7, tflite_top1=0.15, tflite_topk=0.285),
        ]
        with patch.object(ablation_history, "load_ablation_history", return_value={"success": True, "entries": entries, "corrupted_lines": []}):
            result = ablation_history.compare_ablation_runs(run_ids=["run_a", "run_b"])
        self.assertTrue(result["success"])
        self.assertEqual(result["comparison_validity"]["status"], "partial")
        self.assertNotEqual(result["comparison_validity"]["status"], "not_direct")
        self.assertIn("seed", result["comparison_validity"]["differing_fields"])

    def test_compare_ablation_runs_reports_missing_requested_ids_but_still_compares_found_subset(self):
        entries = [
            _entry(run_id="run_a", seed=42),
            _entry(run_id="run_b", seed=7, tflite_top1=0.15, tflite_topk=0.285),
        ]
        with patch.object(ablation_history, "load_ablation_history", return_value={"success": True, "entries": entries, "corrupted_lines": []}):
            result = ablation_history.compare_ablation_runs(run_ids=["run_a", "run_b", "run_missing"])
        self.assertTrue(result["success"])
        self.assertEqual(result["requested_run_ids"], ["run_a", "run_b", "run_missing"])
        self.assertEqual(result["matched_run_ids"], ["run_a", "run_b"])
        self.assertEqual(result["missing_run_ids"], ["run_missing"])
        self.assertTrue(any("run_missing" in warning for warning in result["comparison_warnings"]))
        self.assertEqual([row["run_id"] for row in result["comparison_rows"]], ["run_a", "run_b"])

    def test_compare_ablation_runs_fails_safely_when_fewer_than_two_valid_requested_runs_remain(self):
        entries = [_entry(run_id="run_a", seed=42)]
        with patch.object(ablation_history, "load_ablation_history", return_value={"success": True, "entries": entries, "corrupted_lines": []}):
            result = ablation_history.compare_ablation_runs(run_ids=["run_a", "run_missing"])
        self.assertFalse(result["success"])
        self.assertEqual(result["requested_run_ids"], ["run_a", "run_missing"])
        self.assertEqual(result["matched_run_ids"], ["run_a"])
        self.assertEqual(result["missing_run_ids"], ["run_missing"])
        self.assertIn("At least two valid requested runs are required", result["error"])

    def test_compare_ablation_runs_auto_selection_does_not_emit_missing_id_warning(self):
        entries = [
            _entry(run_id="run_a", seed=42),
            _entry(run_id="run_b", seed=7, tflite_top1=0.15, tflite_topk=0.285),
            _entry(run_id="run_c", seed=99, tflite_top1=0.14, tflite_topk=0.280),
        ]
        with patch.object(ablation_history, "load_ablation_history", return_value={"success": True, "entries": entries, "corrupted_lines": []}):
            result = ablation_history.compare_ablation_runs()
        self.assertTrue(result["success"])
        self.assertEqual(result["requested_run_ids"], [])
        self.assertEqual(result["missing_run_ids"], [])
        self.assertEqual(result["comparison_warnings"], [])

    def test_history_markdown_unrelated_runs_stay_not_recommendation_ready(self):
        entries = [
            _entry(run_id="run_a", dataset_path="E:/school/659/plantnet_300K/plantnet_300K", sampling_mode="balanced", max_images=50),
            _entry(run_id="run_b", dataset_path="E:/school/659/plantnet_300K/plantnet_300K", sampling_mode="random", max_images=200, max_images_per_class=None),
            _entry(run_id="run_c", dataset_path="D:/other/dataset", sampling_mode="balanced", max_images=50),
        ]
        markdown = ablation_history._build_history_markdown(
            entries,
            _group_summaries(entries),
            [],
            {
                "history_jsonl": "D:/tmp/ablation_history.jsonl",
                "table_csv": "D:/tmp/ablation_history_table.csv",
                "summary_json": "D:/tmp/ablation_history_summary.json",
                "summary_markdown": "D:/tmp/ablation_history_summary.md",
                "metrics_png": "D:/tmp/ablation_history_metrics.png",
            },
            "2026-05-03T00:00:00Z",
        )
        self.assertIn("Recommendation-ready history: `no`", markdown)
        self.assertIn("none repeat within the same dataset, split, sampling mode, max_images, and max_images_per_class group", markdown)


if __name__ == "__main__":
    unittest.main()
