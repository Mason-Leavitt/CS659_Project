import unittest
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
        "metrics": {
            "tflite_top1_accuracy": tflite_top1,
            "tflite_topk_accuracy": tflite_topk,
            "hog_top1_accuracy": hog_top1,
            "hog_topk_accuracy": hog_topk,
            "model_agreement_rate": agreement,
            "tflite_color_stability_rate": stability,
        },
    }


class AblationHistorySmokeTests(unittest.TestCase):
    def test_group_key_includes_normalized_dataset_path(self):
        entry = _entry(dataset_path="E:/school/659/plantnet_300K/plantnet_300K")
        key = ablation_history._group_key(entry)
        self.assertEqual(key[0], "E:\\school\\659\\plantnet_300K\\plantnet_300K")

    def test_comparison_seed_only_difference_is_partial(self):
        rows = [
            _entry(run_id="run_a", seed=42),
            _entry(run_id="run_b", seed=7),
        ]
        result = ablation_history.describe_comparison_validity(rows)
        self.assertEqual(result["status"], "partial")
        self.assertIn("seed", result["differing_fields"])

    def test_comparison_dataset_difference_is_not_direct(self):
        rows = [
            _entry(run_id="run_a", dataset_path="E:/school/659/plantnet_300K/plantnet_300K"),
            _entry(run_id="run_b", dataset_path="D:/other/dataset"),
        ]
        result = ablation_history.describe_comparison_validity(rows)
        self.assertEqual(result["status"], "not_direct")
        self.assertIn("dataset_path", result["differing_fields"])

    def test_comparison_sampling_difference_is_not_direct(self):
        rows = [
            _entry(run_id="run_a", sampling_mode="balanced"),
            _entry(run_id="run_b", sampling_mode="random", num_classes=120),
        ]
        result = ablation_history.describe_comparison_validity(rows)
        self.assertEqual(result["status"], "not_direct")
        self.assertIn("sampling_mode", result["differing_fields"])

    def test_comparison_missing_dataset_path_is_partial_without_crash(self):
        rows = [
            _entry(run_id="run_a", dataset_path=None),
            _entry(run_id="run_b", dataset_path=None, num_images_evaluated=198, num_classes=198),
        ]
        result = ablation_history.describe_comparison_validity(rows)
        self.assertEqual(result["status"], "partial")
        self.assertIn("dataset_path", result["unknown_fields"])

    def test_single_run_comparison_is_direct(self):
        result = ablation_history.describe_comparison_validity([_entry()])
        self.assertEqual(result["status"], "direct")

    def test_recommendations_include_transparency_fields_with_mock_history(self):
        entries = [
            _entry(run_id="run_a", seed=42, tflite_top1=0.10, tflite_topk=0.26, stability=0.23),
            _entry(run_id="run_b", seed=7, tflite_top1=0.15, tflite_topk=0.285, agreement=0.015, stability=0.235),
            _entry(run_id="run_c", sampling_mode="random", max_images_per_class=None, num_classes=114, tflite_top1=0.465, tflite_topk=0.735, hog_top1=0.02, hog_topk=0.045, agreement=0.02, stability=0.345),
        ]
        grouped = {}
        for item in entries:
            grouped.setdefault(ablation_history._group_key(item), []).append(item)
        comparable_groups = [
            ablation_history._build_group_summary(group_entries)
            for group_entries in grouped.values()
        ]
        history_summary = {
            "success": True,
            "count": len(entries),
            "comparable_groups": comparable_groups,
            "history_files": {},
            "corrupted_lines": [],
            "latest_runs": entries[-3:],
        }
        with patch.object(ablation_history, "load_ablation_history", return_value={"entries": entries, "corrupted_lines": [], "success": True}), patch.object(
            ablation_history,
            "summarize_ablation_history",
            return_value=history_summary,
        ):
            result = ablation_history.get_recommendations_from_history()
        self.assertTrue(result["success"])
        self.assertIn("recommended_quick_run", result["recommendations"])
        self.assertIn("recommended_robust_run", result["recommendations"])
        self.assertIn("recommendation_status", result)
        self.assertIn("evidence_summary", result)
        self.assertIn("supporting_metrics", result)
        self.assertIn("caveats", result)
        self.assertIn("next_best_actions", result)

    def test_summarize_ablation_history_does_not_call_export(self):
        entries = [_entry(run_id="run_a"), _entry(run_id="run_b", seed=7)]
        with patch.object(
            ablation_history,
            "load_ablation_history",
            return_value={"entries": entries, "corrupted_lines": [], "success": True},
        ), patch.object(ablation_history, "export_ablation_history_artifacts") as export_mock:
            result = ablation_history.summarize_ablation_history()
        export_mock.assert_not_called()
        self.assertTrue(result["success"])
        self.assertEqual(result["count"], 2)
        self.assertIn("comparable_groups", result)
        self.assertIn("history_files", result)

    def test_recommendations_do_not_force_artifact_export(self):
        entries = [_entry(run_id="run_a"), _entry(run_id="run_b", seed=7)]
        grouped = {}
        for item in entries:
            grouped.setdefault(ablation_history._group_key(item), []).append(item)
        comparable_groups = [
            ablation_history._build_group_summary(group_entries)
            for group_entries in grouped.values()
        ]
        history_summary = {
            "success": True,
            "count": len(entries),
            "comparable_groups": comparable_groups,
            "history_files": {},
            "corrupted_lines": [],
            "latest_runs": entries[-2:],
        }
        with patch.object(
            ablation_history,
            "load_ablation_history",
            return_value={"entries": entries, "corrupted_lines": [], "success": True},
        ), patch.object(
            ablation_history,
            "summarize_ablation_history",
            return_value=history_summary,
        ), patch.object(ablation_history, "export_ablation_history_artifacts") as export_mock:
            result = ablation_history.get_recommendations_from_history()
        export_mock.assert_not_called()
        self.assertTrue(result["success"])

    def test_recommendations_three_unrelated_runs_stay_insufficient(self):
        entries = [
            _entry(run_id="run_a", dataset_path="E:/school/659/plantnet_300K/plantnet_300K", sampling_mode="balanced", max_images=50),
            _entry(run_id="run_b", dataset_path="E:/school/659/plantnet_300K/plantnet_300K", sampling_mode="random", max_images=200, max_images_per_class=None),
            _entry(run_id="run_c", dataset_path="D:/other/dataset", sampling_mode="balanced", max_images=50),
        ]
        grouped = {}
        for item in entries:
            grouped.setdefault(ablation_history._group_key(item), []).append(item)
        comparable_groups = [
            ablation_history._build_group_summary(group_entries)
            for group_entries in grouped.values()
        ]
        history_summary = {
            "success": True,
            "count": len(entries),
            "comparable_groups": comparable_groups,
            "history_files": {},
            "corrupted_lines": [],
            "latest_runs": entries[-3:],
        }
        with patch.object(ablation_history, "load_ablation_history", return_value={"entries": entries, "corrupted_lines": [], "success": True}), patch.object(
            ablation_history,
            "summarize_ablation_history",
            return_value=history_summary,
        ):
            result = ablation_history.get_recommendations_from_history()
        self.assertFalse(result["enough_history"])
        self.assertEqual(result["recommendation_status"], "insufficient_history")
        self.assertIn("none repeat within the same comparable configuration", result["evidence_summary"])

    def test_recommendations_two_same_group_runs_are_limited(self):
        entries = [
            _entry(run_id="run_a", seed=42),
            _entry(run_id="run_b", seed=7, tflite_top1=0.15, tflite_topk=0.285, stability=0.235),
        ]
        grouped = {}
        for item in entries:
            grouped.setdefault(ablation_history._group_key(item), []).append(item)
        comparable_groups = [
            ablation_history._build_group_summary(group_entries)
            for group_entries in grouped.values()
        ]
        history_summary = {
            "success": True,
            "count": len(entries),
            "comparable_groups": comparable_groups,
            "history_files": {},
            "corrupted_lines": [],
            "latest_runs": entries[-2:],
        }
        with patch.object(ablation_history, "load_ablation_history", return_value={"entries": entries, "corrupted_lines": [], "success": True}), patch.object(
            ablation_history,
            "summarize_ablation_history",
            return_value=history_summary,
        ):
            result = ablation_history.get_recommendations_from_history()
        self.assertTrue(result["enough_history"])
        self.assertEqual(result["recommendation_status"], "evidence_limited")
        self.assertIn("2 run(s)", result["evidence_summary"])

    def test_recommendations_three_same_group_runs_are_supported(self):
        entries = [
            _entry(run_id="run_a", seed=42),
            _entry(run_id="run_b", seed=7, tflite_top1=0.15, tflite_topk=0.285, stability=0.235),
            _entry(run_id="run_c", seed=99, tflite_top1=0.14, tflite_topk=0.280, stability=0.230),
        ]
        grouped = {}
        for item in entries:
            grouped.setdefault(ablation_history._group_key(item), []).append(item)
        comparable_groups = [
            ablation_history._build_group_summary(group_entries)
            for group_entries in grouped.values()
        ]
        history_summary = {
            "success": True,
            "count": len(entries),
            "comparable_groups": comparable_groups,
            "history_files": {},
            "corrupted_lines": [],
            "latest_runs": entries[-3:],
        }
        with patch.object(ablation_history, "load_ablation_history", return_value={"entries": entries, "corrupted_lines": [], "success": True}), patch.object(
            ablation_history,
            "summarize_ablation_history",
            return_value=history_summary,
        ):
            result = ablation_history.get_recommendations_from_history()
        self.assertTrue(result["enough_history"])
        self.assertEqual(result["recommendation_status"], "evidence_supported")
        self.assertIn("repeated comparable group", result["evidence_summary"].lower())

    def test_recommendations_missing_dataset_path_stay_conservative_without_crashing(self):
        entries = [
            _entry(run_id="run_a", dataset_path=None),
            _entry(run_id="run_b", dataset_path=None, seed=7, tflite_top1=0.15, tflite_topk=0.285, stability=0.235),
            _entry(run_id="run_c", dataset_path=None, seed=99, tflite_top1=0.14, tflite_topk=0.280, stability=0.230),
        ]
        grouped = {}
        for item in entries:
            grouped.setdefault(ablation_history._group_key(item), []).append(item)
        comparable_groups = [
            ablation_history._build_group_summary(group_entries)
            for group_entries in grouped.values()
        ]
        history_summary = {
            "success": True,
            "count": len(entries),
            "comparable_groups": comparable_groups,
            "history_files": {},
            "corrupted_lines": [],
            "latest_runs": entries[-3:],
        }
        with patch.object(ablation_history, "load_ablation_history", return_value={"entries": entries, "corrupted_lines": [], "success": True}), patch.object(
            ablation_history,
            "summarize_ablation_history",
            return_value=history_summary,
        ):
            result = ablation_history.get_recommendations_from_history()
        self.assertTrue(result["success"])
        self.assertEqual(result["recommendation_status"], "evidence_limited")
        self.assertTrue(any("missing dataset_path" in caveat for caveat in result["caveats"]))


if __name__ == "__main__":
    unittest.main()
