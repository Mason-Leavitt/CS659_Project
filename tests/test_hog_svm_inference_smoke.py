import sys
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = REPO_ROOT / "DeepLearning-tensorFlowLite"
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

import infer_hog_svm


class _FakeFinalEstimator:
    def __init__(self, classes):
        self.classes_ = np.asarray(classes)


class _FakePipeline:
    def __init__(self, classes):
        self.steps = [("svc", _FakeFinalEstimator(classes))]


class HogInferenceSmokeTests(unittest.TestCase):
    def test_predict_proba_ranking_uses_estimator_classes_order(self):
        predictions = infer_hog_svm._rank_scores_with_classes(
            np.asarray([0.90, 0.10], dtype=np.float64),
            labels=["class_a", "class_b"],
            estimator_classes=["class_b", "class_a"],
            top_k=2,
            id_name_map={},
            score_type="probability",
        )
        self.assertEqual(predictions[0]["label"], "class_b")
        self.assertEqual(predictions[1]["label"], "class_a")

    def test_decision_function_ranking_uses_pipeline_final_estimator_classes(self):
        estimator_classes = infer_hog_svm._get_estimator_classes(
            _FakePipeline(["class_c", "class_a", "class_b"])
        )
        predictions = infer_hog_svm._rank_scores_with_classes(
            np.asarray([0.20, 1.50, 0.90], dtype=np.float64),
            labels=["class_a", "class_b", "class_c"],
            estimator_classes=estimator_classes,
            top_k=3,
            id_name_map={},
            score_type="decision_function",
        )
        self.assertEqual(
            [item["label"] for item in predictions],
            ["class_a", "class_b", "class_c"],
        )

    def test_legacy_fallback_without_classes_keeps_index_based_mapping(self):
        predictions = infer_hog_svm._rank_scores_with_classes(
            np.asarray([0.10, 0.90], dtype=np.float64),
            labels=["legacy_a", "legacy_b"],
            estimator_classes=None,
            top_k=2,
            id_name_map={},
            score_type="probability",
        )
        self.assertEqual(predictions[0]["label"], "legacy_b")
        self.assertEqual(predictions[1]["label"], "legacy_a")

    def test_binary_decision_function_expands_against_two_classes(self):
        predictions = infer_hog_svm._rank_scores_with_classes(
            np.asarray([2.5], dtype=np.float64),
            labels=["class_a", "class_b"],
            estimator_classes=["class_a", "class_b"],
            top_k=2,
            id_name_map={},
            score_type="decision_function",
        )
        self.assertEqual(
            [item["label"] for item in predictions],
            ["class_b", "class_a"],
        )


if __name__ == "__main__":
    unittest.main()
