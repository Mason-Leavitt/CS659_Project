import unittest

from agent import parameter_sweep_planner

FAILED_PROMPT = (
    "run inference only parameter sweep using the following details. "
    "Base paramateres - top_k: 5, max_images: 100, seed: 50, sampling_mode: sorted, max_images_per_class: 5 "
    "sweeps - top_k: 1,3,5,8,10 sampleng_mode: balance,random,sorted "
    "max_images: 10,25,50,100,200 max_images_per_class: 1,2,5,7,10 seeds: 7,25,42,123,200 "
    "metrics to plot are: TFLite top-1 accuracy, HOG+SVM top-1 accuracy, TFLite top-k accuracy, "
    "HOG+SVM top-k accuracy, agrrement rate"
)


class ParameterSweepPlannerTests(unittest.TestCase):
    def test_basic_plan_generation_varies_only_one_parameter(self):
        result = parameter_sweep_planner.build_parameter_sweep_plan(
            dataset_path="dataset_sample",
            split="test",
            parameter_ranges={"top_k": [1, 3, 5]},
            baseline_values={
                "top_k": 5,
                "sampling_mode": "balanced",
                "max_images": 200,
                "max_images_per_class": 1,
                "seed": 42,
            },
        )
        self.assertTrue(result["success"])
        self.assertEqual(result["total_planned_runs"], 3)
        self.assertEqual(len(result["generated_sweep_points"]), 3)
        for expected_value, point in zip([1, 3, 5], result["generated_sweep_points"]):
            self.assertEqual(point["varied_parameter"], "top_k")
            self.assertEqual(point["varied_value"], expected_value)
            self.assertEqual(point["parameter_values"]["top_k"], expected_value)
            self.assertEqual(point["parameter_values"]["sampling_mode"], "balanced")
            self.assertEqual(point["parameter_values"]["max_images"], 200)
            self.assertEqual(point["parameter_values"]["max_images_per_class"], 1)
            self.assertEqual(point["parameter_values"]["seed"], 42)

    def test_multiple_parameters_use_sum_not_cartesian_product(self):
        result = parameter_sweep_planner.build_parameter_sweep_plan(
            dataset_path="dataset_sample",
            parameter_ranges={
                "top_k": [1, 3, 5],
                "max_images": [50, 100],
            },
        )
        self.assertTrue(result["success"])
        self.assertEqual(result["total_planned_runs"], 5)

    def test_defaults_are_applied_and_marked(self):
        result = parameter_sweep_planner.build_parameter_sweep_plan(
            parameter_ranges={"seed": [1, 2]},
        )
        self.assertTrue(result["success"])
        self.assertEqual(result["baseline_values"]["top_k"], 5)
        self.assertEqual(result["baseline_values"]["sampling_mode"], "balanced")
        self.assertEqual(result["baseline_values"]["max_images"], 200)
        self.assertEqual(result["baseline_values"]["max_images_per_class"], 1)
        self.assertEqual(result["baseline_values"]["seed"], 42)
        self.assertEqual(result["baseline_sources"]["top_k"], "default")
        self.assertEqual(result["baseline_sources"]["seed"], "default")
        self.assertIn("dataset_path was not provided", " ".join(result["warnings"]))

    def test_invalid_sampling_mode_is_rejected(self):
        result = parameter_sweep_planner.build_parameter_sweep_plan(
            dataset_path="dataset_sample",
            parameter_ranges={"sampling_mode": ["balanced", "bad_mode"]},
        )
        self.assertFalse(result["success"])
        self.assertIn("bad_mode", " ".join(result["errors"]))

    def test_too_many_values_require_confirmation_and_fail(self):
        result = parameter_sweep_planner.build_parameter_sweep_plan(
            dataset_path="dataset_sample",
            parameter_ranges={"seed": [1, 2, 3, 4, 5, 6]},
        )
        self.assertFalse(result["success"])
        self.assertTrue(result["require_confirmation"])
        self.assertIn("exceeds the safety limit", " ".join(result["errors"]))

    def test_too_many_total_points_require_confirmation_and_fail(self):
        result = parameter_sweep_planner.build_parameter_sweep_plan(
            dataset_path="dataset_sample",
            parameter_ranges={
                "top_k": [1, 2, 3, 4, 5],
            },
            max_total_runs=4,
        )
        self.assertFalse(result["success"])
        self.assertTrue(result["require_confirmation"])
        self.assertIn("Total planned sweep points", " ".join(result["errors"]))

    def test_default_total_limit_is_reported_in_safety_limits(self):
        result = parameter_sweep_planner.build_parameter_sweep_plan(
            dataset_path="dataset_sample",
            parameter_ranges={
                "top_k": [1, 2, 3, 4, 5],
                "seed": [10, 20, 30, 40, 50],
                "max_images": [100, 200, 300, 400, 500],
                "max_images_per_class": [1, 2, 3, 4, 5],
                "sampling_mode": ["balanced", "random", "sorted"],
            },
        )
        self.assertTrue(result["success"])
        self.assertEqual(result["total_planned_runs"], 23)
        self.assertEqual(result["safety_limits"]["max_total_runs"], 25)

    def test_invalid_metric_is_rejected(self):
        result = parameter_sweep_planner.build_parameter_sweep_plan(
            dataset_path="dataset_sample",
            parameter_ranges={"top_k": [1, 3, 5]},
            selected_metrics=["tflite_top1_accuracy", "made_up_metric"],
        )
        self.assertFalse(result["success"])
        self.assertIn("made_up_metric", " ".join(result["errors"]))

    def test_no_parameter_ranges_fails_safely(self):
        result = parameter_sweep_planner.build_parameter_sweep_plan(
            dataset_path="dataset_sample",
            parameter_ranges={},
        )
        self.assertFalse(result["success"])
        self.assertIn("At least one non-empty supported parameter range is required", " ".join(result["errors"]))

    def test_supported_parameters_only(self):
        result = parameter_sweep_planner.build_parameter_sweep_plan(
            dataset_path="dataset_sample",
            parameter_ranges={
                "svm_c": [1, 10],
                "color_correct": ["none"],
                "top_k": [1, 3, 5],
            },
        )
        self.assertFalse(result["success"])
        self.assertIn("svm_c", " ".join(result["errors"]))
        self.assertIn("color_correct", " ".join(result["errors"]))

    def test_parse_parameter_sweep_request_extracts_simple_ranges_and_baselines(self):
        parsed = parameter_sweep_planner.parse_parameter_sweep_request(
            "I want a parameter sweep with top_k 1,3,5 and max_images 50,100,200. "
            "Use baseline top_k 5, sampling_mode balanced, max_images 200, max_images_per_class 1, seed 42."
        )
        self.assertTrue(parsed["success"])
        self.assertEqual(parsed["parameter_ranges"]["top_k"], ["1", "3", "5"])
        self.assertEqual(parsed["parameter_ranges"]["max_images"], ["50", "100", "200"])
        self.assertEqual(parsed["baseline_values"]["top_k"], "5")
        self.assertEqual(parsed["baseline_values"]["sampling_mode"], "balanced")
        self.assertEqual(parsed["baseline_values"]["seed"], "42")

    def test_exact_failed_prompt_parses_into_valid_plan_inputs(self):
        parsed = parameter_sweep_planner.parse_parameter_sweep_request(FAILED_PROMPT)
        self.assertTrue(parsed["success"])
        self.assertEqual(
            parsed["baseline_values"],
            {
                "top_k": "5",
                "max_images": "100",
                "seed": "50",
                "sampling_mode": "sorted",
                "max_images_per_class": "5",
            },
        )
        self.assertEqual(parsed["parameter_ranges"]["top_k"], ["1", "3", "5", "8", "10"])
        self.assertEqual(parsed["parameter_ranges"]["sampling_mode"], ["balanced", "random", "sorted"])
        self.assertEqual(parsed["parameter_ranges"]["max_images"], ["10", "25", "50", "100", "200"])
        self.assertEqual(parsed["parameter_ranges"]["max_images_per_class"], ["1", "2", "5", "7", "10"])
        self.assertEqual(parsed["parameter_ranges"]["seed"], ["7", "25", "42", "123", "200"])
        self.assertEqual(
            parsed["selected_metrics"],
            [
                "tflite_top1_accuracy",
                "hog_top1_accuracy",
                "tflite_topk_accuracy",
                "hog_topk_accuracy",
                "model_agreement_rate",
            ],
        )

        built = parameter_sweep_planner.build_parameter_sweep_plan(
            parameter_ranges=parsed["parameter_ranges"],
            baseline_values=parsed["baseline_values"],
            selected_metrics=parsed["selected_metrics"],
        )
        self.assertTrue(built["success"])
        self.assertEqual(built["total_planned_runs"], 23)
        self.assertIn("dataset_path was not provided", " ".join(built["warnings"]))

    def test_field_boundary_parsing_keeps_adjacent_fields_separate(self):
        parsed = parameter_sweep_planner.parse_parameter_sweep_request(
            "parameter sweep sweeps - sampling_mode: balance,random,sorted max_images: 10,25,50"
        )
        self.assertEqual(parsed["parameter_ranges"]["sampling_mode"], ["balanced", "random", "sorted"])
        self.assertEqual(parsed["parameter_ranges"]["max_images"], ["10", "25", "50"])

    def test_misspelled_labels_and_metric_names_are_normalized(self):
        parsed = parameter_sweep_planner.parse_parameter_sweep_request(
            "Base paramateres - sampleng_mode: sorted. "
            "sweeps - seeds: 7,25,42. "
            "metrics to plot are: agrrement rate"
        )
        self.assertEqual(parsed["baseline_values"]["sampling_mode"], "sorted")
        self.assertEqual(parsed["parameter_ranges"]["seed"], ["7", "25", "42"])
        self.assertEqual(parsed["selected_metrics"], ["model_agreement_rate"])

    def test_metrics_phrase_is_not_treated_as_a_metric(self):
        parsed = parameter_sweep_planner.parse_parameter_sweep_request(
            "metrics to plot are: TFLite top-1 accuracy, agreement rate"
        )
        self.assertEqual(
            parsed["selected_metrics"],
            ["tflite_top1_accuracy", "model_agreement_rate"],
        )

    def test_plural_seeds_alias_maps_to_seed(self):
        parsed = parameter_sweep_planner.parse_parameter_sweep_request(
            "parameter sweep sweeps - seeds: 7,25,42"
        )
        self.assertEqual(parsed["parameter_ranges"]["seed"], ["7", "25", "42"])


if __name__ == "__main__":
    unittest.main()
