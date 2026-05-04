import unittest

from agent import agent_tools


class AblationOptionsSmokeTests(unittest.TestCase):
    def test_supported_ablation_controls_include_classical_and_shared_controls(self):
        result = agent_tools.get_supported_ablation_controls()
        self.assertTrue(result["success"])
        self.assertTrue(result["inference_only"])

        shared = result["shared_sampling_controls"]
        classical = result["classical_controls"]
        deep = result["deep_learning_controls"]
        unswept = result["not_currently_swept"]

        for key in (
            "dataset_path",
            "split",
            "sampling_mode",
            "max_images",
            "max_images_per_class",
            "seed",
        ):
            self.assertIn(key, shared)

        self.assertIn("top_k", classical)
        self.assertIn("top_k", deep)
        self.assertIn("color_correct", deep)
        self.assertIn("svm_c", unswept)
        self.assertIn("svm_kernel", unswept)
        self.assertIn("hog_pixels_per_cell", unswept)

        caveat_text = " ".join(result.get("caveats", []))
        self.assertIn("inference-only", caveat_text)
        self.assertIn("do not apply to HOG+SVM", caveat_text)

    def test_shared_control_details_exist(self):
        result = agent_tools.get_supported_ablation_controls()
        details = result["shared_sampling_control_details"]
        by_name = {item["name"]: item for item in details}

        for key in (
            "dataset_path",
            "split",
            "sampling_mode",
            "max_images",
            "max_images_per_class",
            "seed",
        ):
            self.assertIn(key, by_name)
            self.assertTrue(by_name[key]["label"])
            self.assertTrue(by_name[key]["description"])

    def test_sampling_mode_details_mention_supported_values(self):
        result = agent_tools.get_supported_ablation_controls()
        sampling_mode = next(
            item for item in result["shared_sampling_control_details"] if item["name"] == "sampling_mode"
        )
        self.assertIn("sorted", sampling_mode["values"])
        self.assertIn("random", sampling_mode["values"])
        self.assertIn("balanced", sampling_mode["values"])
        notes_text = " ".join(sampling_mode["notes"]).lower()
        self.assertIn("balanced", notes_text)
        self.assertIn("random", notes_text)
        self.assertIn("sorted", notes_text)

    def test_classical_control_details_exist(self):
        result = agent_tools.get_supported_ablation_controls()
        top_k = next(item for item in result["classical_control_details"] if item["name"] == "top_k")
        self.assertEqual(top_k["label"], "Top-k depth")
        self.assertIn("HOG+SVM", top_k["applies_to"])
        self.assertIn("5", top_k["examples"])
        description_text = top_k["description"].lower()
        self.assertTrue("ranked" in description_text or "top-k" in description_text or "evaluation" in description_text)

    def test_deep_control_details_exist(self):
        result = agent_tools.get_supported_ablation_controls()
        by_name = {item["name"]: item for item in result["deep_learning_control_details"]}
        self.assertIn("top_k", by_name)
        self.assertIn("color_correct", by_name)
        color_correct = by_name["color_correct"]
        self.assertIn("none", color_correct["values"])
        self.assertIn("gray_world", color_correct["values"])
        self.assertIn("max_rgb", color_correct["values"])

    def test_unswept_classical_settings_are_marked_unsupported(self):
        result = agent_tools.get_supported_ablation_controls()
        by_name = {item["name"]: item for item in result["not_currently_swept_details"]}
        for key in ("svm_c", "svm_kernel", "hog_pixels_per_cell"):
            self.assertIn(key, by_name)
            text = (by_name[key]["description"] + " " + " ".join(by_name[key]["notes"])).lower()
            self.assertTrue("not currently" in text or "would require" in text or "inference-only" in text)

    def test_exact_control_lookup_succeeds(self):
        for query in ("max_images_per_class", "sampling_mode", "top_k", "color_correct", "seed"):
            result = agent_tools.get_ablation_control_detail(query)
            self.assertTrue(result["success"])
            self.assertEqual(result["matched_name"], query)
            self.assertIsNotNone(result["detail"])

    def test_alias_lookup_succeeds(self):
        expected = {
            "maximum images per class": "max_images_per_class",
            "sampling mode": "sampling_mode",
            "balanced sampling": "sampling_mode",
            "top k": "top_k",
            "top-k": "top_k",
            "color correction": "color_correct",
            "svm c": "svm_c",
            "hog pixels per cell": "hog_pixels_per_cell",
        }
        for query, matched_name in expected.items():
            result = agent_tools.get_ablation_control_detail(query)
            self.assertTrue(result["success"], msg=query)
            self.assertEqual(result["matched_name"], matched_name)

    def test_supported_vs_unswept_distinction(self):
        top_k = agent_tools.get_ablation_control_detail("top_k")
        sampling_mode = agent_tools.get_ablation_control_detail("sampling_mode")
        svm_c = agent_tools.get_ablation_control_detail("svm c")
        hog_pixels = agent_tools.get_ablation_control_detail("hog pixels per cell")

        self.assertTrue(top_k["is_supported_control"])
        self.assertFalse(top_k["is_not_currently_swept"])
        self.assertTrue(sampling_mode["is_supported_control"])
        self.assertFalse(sampling_mode["is_not_currently_swept"])
        self.assertFalse(svm_c["is_supported_control"])
        self.assertTrue(svm_c["is_not_currently_swept"])
        self.assertFalse(hog_pixels["is_supported_control"])
        self.assertTrue(hog_pixels["is_not_currently_swept"])

    def test_unknown_lookup_does_not_hallucinate(self):
        result = agent_tools.get_ablation_control_detail("banana hyperdrive")
        self.assertFalse(result["success"])
        self.assertIsNone(result["detail"])
        self.assertIn("not recognized", result["message"])

    def test_tool_wrapper_calls_shared_helper_for_supported_control(self):
        result = agent_tools.explain_ablation_control("max_images_per_class")
        self.assertTrue(result["success"])
        self.assertEqual(result["matched_name"], "max_images_per_class")
        self.assertEqual(result["matched_label"], "Maximum images per class")
        self.assertTrue(result["detail"]["description"])

    def test_tool_wrapper_marks_unswept_settings_as_not_currently_exposed(self):
        for query in ("svm c", "hog pixels per cell"):
            result = agent_tools.explain_ablation_control(query)
            self.assertTrue(result["success"])
            self.assertFalse(result["is_supported_control"])
            self.assertTrue(result["is_not_currently_swept"])

    def test_tool_wrapper_unknown_control_fails_safely(self):
        result = agent_tools.explain_ablation_control("totally_unknown_control")
        self.assertFalse(result["success"])
        self.assertIn("not recognized", result["message"])

    def test_langchain_registration_includes_ablation_control_tool_when_available(self):
        if not agent_tools.LANGCHAIN_AVAILABLE:
            self.skipTest("LangChain tool registry is not available in this environment.")
        names = {getattr(tool, "name", "") for tool in agent_tools.AGENT_TOOLS}
        self.assertIn("explain_ablation_control", names)


if __name__ == "__main__":
    unittest.main()
