import math
import sys
import types
import tempfile
import unittest
from pathlib import Path


if "streamlit" not in sys.modules:
    class _DummySessionState(dict):
        def __getattr__(self, name):
            try:
                return self[name]
            except KeyError as exc:
                raise AttributeError(name) from exc

        def __setattr__(self, name, value):
            self[name] = value

    class _DummySlot:
        def markdown(self, *args, **kwargs):
            return None

        def info(self, *args, **kwargs):
            return None

        def warning(self, *args, **kwargs):
            return None

        def success(self, *args, **kwargs):
            return None

        def empty(self):
            return None

    streamlit_stub = types.ModuleType("streamlit")
    streamlit_stub.session_state = _DummySessionState()
    streamlit_stub.set_page_config = lambda *args, **kwargs: None
    streamlit_stub.markdown = lambda *args, **kwargs: None
    streamlit_stub.empty = lambda: _DummySlot()
    streamlit_stub.spinner = lambda *args, **kwargs: None
    streamlit_stub.progress = lambda *args, **kwargs: None
    streamlit_stub.pyplot = lambda *args, **kwargs: None
    streamlit_stub.caption = lambda *args, **kwargs: None
    streamlit_stub.write = lambda *args, **kwargs: None
    streamlit_stub.info = lambda *args, **kwargs: None
    streamlit_stub.warning = lambda *args, **kwargs: None
    streamlit_stub.success = lambda *args, **kwargs: None
    streamlit_stub.rerun = lambda *args, **kwargs: None
    sys.modules["streamlit"] = streamlit_stub

from agent.streamlit_app import (
    _CHAT_HISTORY_HEIGHT,
    _chat_input_widget_key,
    _extract_parameter_sweep_payload,
    _is_parameter_sweep_plan_dict,
    _is_parameter_sweep_result_dict,
    _parameter_sweep_sample_prompt,
    _chart_display_settings,
    _chart_figure_size,
    _comparison_export_markdown,
    _comparison_export_payload,
    _comparison_metric_value,
    _comparison_warning_messages,
    _file_exists,
    _has_displayable_artifact,
    _parse_sweep_values_text,
    _build_parameter_sweep_ranges_from_inputs,
    _existing_parameter_sweep_chart_entries,
    _zip_chart_artifacts,
    _clamp_chart_index,
)
from agent.parameter_sweep_planner import build_parameter_sweep_plan, parse_parameter_sweep_request


class StreamlitComparisonSmokeTests(unittest.TestCase):
    def test_missing_values_become_nan(self):
        for value in (None, "", "N/A", "nan", "bad-value"):
            result = _comparison_metric_value(value)
            self.assertTrue(math.isnan(result), msg=f"Expected NaN for {value!r}")

    def test_real_zero_values_stay_zero(self):
        for value in (0, 0.0, "0", "0.0"):
            result = _comparison_metric_value(value)
            self.assertEqual(result, 0.0)

    def test_valid_numbers_parse_correctly(self):
        self.assertEqual(_comparison_metric_value(0.75), 0.75)
        self.assertEqual(_comparison_metric_value("0.75"), 0.75)
        self.assertEqual(_comparison_metric_value(1), 1.0)

    def test_comparison_warning_messages_uses_explicit_warnings(self):
        messages = _comparison_warning_messages(
            {"comparison_warnings": ["Requested run ID run_MISSING was not found."]}
        )
        self.assertEqual(len(messages), 1)
        self.assertIn("run_MISSING", messages[0])

    def test_comparison_warning_messages_synthesizes_missing_run_warning(self):
        messages = _comparison_warning_messages(
            {"comparison_warnings": [], "missing_run_ids": ["run_X"]}
        )
        self.assertEqual(len(messages), 1)
        self.assertIn("run_X", messages[0])

    def test_comparison_warning_messages_is_empty_when_no_warnings_exist(self):
        messages = _comparison_warning_messages(
            {"comparison_warnings": [], "missing_run_ids": []}
        )
        self.assertEqual(messages, [])

    def test_comparison_export_payload_includes_warning_metadata(self):
        payload = _comparison_export_payload(
            [{"run_id": "run_a", "split": "test", "sampling_mode": "balanced", "max_images": 200}],
            validity={"summary": "Partial comparison.", "differing_fields": ["seed"], "caveats": ["Seed differs."]},
            result={
                "requested_run_ids": ["run_a", "run_missing"],
                "matched_run_ids": ["run_a"],
                "missing_run_ids": ["run_missing"],
                "comparison_warnings": ["Requested run ID run_missing was not found and was omitted."],
                "explanation": "Compared the explicitly requested ablation runs.",
            },
        )
        self.assertIn("comparison_warning_messages", payload)
        self.assertIn("run_missing", payload["comparison_warning_messages"][0])
        self.assertEqual(payload["missing_run_ids"], ["run_missing"])

    def test_comparison_export_markdown_includes_warning_section(self):
        markdown = _comparison_export_markdown(
            {
                "comparison_rows": [
                    {
                        "run_id": "run_a",
                        "split": "test",
                        "sampling_mode": "balanced",
                        "max_images": 200,
                        "tflite_topk_accuracy": 0.28,
                        "hog_topk_accuracy": 0.02,
                    }
                ],
                "comparison_validity": {"summary": "Partial comparison.", "differing_fields": ["seed"], "caveats": ["Seed differs."]},
                "requested_run_ids": ["run_a", "run_missing"],
                "matched_run_ids": ["run_a"],
                "missing_run_ids": ["run_missing"],
                "comparison_warning_messages": ["Requested run ID run_missing was not found and was omitted."],
                "explanation": "Compared the explicitly requested ablation runs.",
            }
        )
        self.assertIn("## Comparison Warnings", markdown)
        self.assertIn("run_missing", markdown)
        self.assertIn("Requested run IDs:", markdown)
        self.assertIn("Missing run IDs:", markdown)

    def test_chart_display_settings_returns_numeric_defaults(self):
        settings = _chart_display_settings()
        self.assertIsInstance(settings["width_px"], int)
        self.assertIsInstance(settings["height_px"], int)
        self.assertIsInstance(settings["font_size"], int)
        self.assertGreater(settings["width_px"], 0)
        self.assertGreater(settings["height_px"], 0)
        self.assertGreater(settings["font_size"], 0)

    def test_chart_figure_size_uses_settings_without_crashing(self):
        width, height = _chart_figure_size({"width_px": 800, "height_px": 400, "dpi": 100})
        self.assertEqual(width, 8.0)
        self.assertEqual(height, 4.0)

    def test_file_exists_helper_only_accepts_real_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            existing_file = Path(tmpdir) / "artifact.txt"
            existing_file.write_text("ok", encoding="utf-8")
            missing_file = Path(tmpdir) / "missing.txt"

            self.assertTrue(_file_exists(str(existing_file)))
            self.assertFalse(_file_exists(str(missing_file)))
            self.assertFalse(_file_exists(None))
            self.assertFalse(_file_exists(""))

    def test_has_displayable_artifact_handles_single_paths_and_mappings(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            existing_file = Path(tmpdir) / "artifact.txt"
            existing_file.write_text("ok", encoding="utf-8")
            missing_file = Path(tmpdir) / "missing.txt"

            self.assertTrue(_has_displayable_artifact(str(existing_file)))
            self.assertFalse(_has_displayable_artifact(str(missing_file)))
            self.assertTrue(
                _has_displayable_artifact(
                    {"png": str(existing_file), "json": str(missing_file)}
                )
            )
            self.assertFalse(_has_displayable_artifact({"png": str(missing_file)}))

    def test_parse_sweep_values_text_splits_comma_values(self):
        self.assertEqual(_parse_sweep_values_text("1,3,5"), ["1", "3", "5"])
        self.assertEqual(_parse_sweep_values_text("[balanced, random]"), ["balanced", "random"])
        self.assertEqual(_parse_sweep_values_text(""), [])

    def test_build_parameter_sweep_ranges_from_inputs_only_keeps_enabled_parameters(self):
        ranges = _build_parameter_sweep_ranges_from_inputs(
            {
                "top_k": {"enabled": True, "values": "1,3,5"},
                "max_images": {"enabled": False, "values": "50,100"},
            }
        )
        self.assertEqual(ranges, {"top_k": ["1", "3", "5"]})

    def test_existing_parameter_sweep_chart_entries_only_returns_existing_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            existing_chart = Path(tmpdir) / "chart_a.png"
            existing_chart.write_bytes(b"png")
            missing_chart = Path(tmpdir) / "chart_b.png"
            entries = _existing_parameter_sweep_chart_entries(
                {
                    "summary": {
                        "charts": [
                            {"path": str(existing_chart), "filename": existing_chart.name},
                            {"path": str(missing_chart), "filename": missing_chart.name},
                        ]
                    }
                }
            )
            self.assertEqual(len(entries), 1)
            self.assertEqual(entries[0]["filename"], existing_chart.name)

    def test_zip_chart_artifacts_contains_existing_pngs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            chart_a = Path(tmpdir) / "chart_a.png"
            chart_b = Path(tmpdir) / "chart_b.png"
            chart_a.write_bytes(b"a")
            chart_b.write_bytes(b"b")
            zip_bytes = _zip_chart_artifacts(
                [
                    {"path": str(chart_a), "filename": chart_a.name},
                    {"path": str(chart_b), "filename": chart_b.name},
                ]
            )
            self.assertGreater(len(zip_bytes), 0)
            import zipfile
            from io import BytesIO

            with zipfile.ZipFile(BytesIO(zip_bytes), "r") as zf:
                self.assertEqual(sorted(zf.namelist()), ["chart_a.png", "chart_b.png"])

    def test_clamp_chart_index_handles_out_of_range_values(self):
        self.assertEqual(_clamp_chart_index(-1, 3), 0)
        self.assertEqual(_clamp_chart_index(5, 3), 2)
        self.assertEqual(_clamp_chart_index(1, 3), 1)
        self.assertEqual(_clamp_chart_index(2, 0), 0)

    def test_chat_input_widget_key_changes_when_version_changes(self):
        self.assertEqual(_chat_input_widget_key(0), "chat_input_widget_0")
        self.assertEqual(_chat_input_widget_key(1), "chat_input_widget_1")
        self.assertNotEqual(_chat_input_widget_key(0), _chat_input_widget_key(1))

    def test_chat_history_height_constant_is_positive(self):
        self.assertIsInstance(_CHAT_HISTORY_HEIGHT, int)
        self.assertGreater(_CHAT_HISTORY_HEIGHT, 0)

    def test_parameter_sweep_plan_detection_helper_accepts_valid_plan_shape(self):
        self.assertTrue(
            _is_parameter_sweep_plan_dict(
                {
                    "study_type": "parameter_sweep",
                    "parameter_ranges": {"top_k": [1, 3, 5]},
                    "baseline_values": {"top_k": 5},
                    "generated_sweep_points": [],
                }
            )
        )
        self.assertFalse(_is_parameter_sweep_plan_dict("not-a-plan"))

    def test_parameter_sweep_result_detection_helper_accepts_valid_result_shape(self):
        self.assertTrue(
            _is_parameter_sweep_result_dict(
                {
                    "status": "completed",
                    "summary": {},
                    "artifacts": {},
                    "results": [],
                    "output_dir": "result\\parameter_sweep_fake",
                }
            )
        )
        self.assertFalse(_is_parameter_sweep_result_dict({"message": "only text"}))

    def test_extract_parameter_sweep_payload_handles_plain_and_structured_agent_results(self):
        self.assertEqual(
            _extract_parameter_sweep_payload("plain text"),
            {"parameter_sweep_plan": None, "parameter_sweep_result": None},
        )
        plan = {
            "study_type": "parameter_sweep",
            "parameter_ranges": {"top_k": [1, 3, 5]},
            "baseline_values": {"top_k": 5},
            "generated_sweep_points": [],
        }
        result = {
            "status": "completed",
            "summary": {"charts": []},
            "artifacts": {},
            "results": [],
            "plan": plan,
            "output_dir": "result\\parameter_sweep_fake",
        }
        extracted_plan = _extract_parameter_sweep_payload({"message": "ok", "parameter_sweep_plan": plan})
        self.assertEqual(extracted_plan["parameter_sweep_plan"], plan)
        self.assertIsNone(extracted_plan["parameter_sweep_result"])
        extracted_result = _extract_parameter_sweep_payload({"message": "done", "parameter_sweep_result": result})
        self.assertEqual(extracted_result["parameter_sweep_result"], result)
        self.assertEqual(extracted_result["parameter_sweep_plan"], plan)

    def test_parameter_sweep_sample_prompt_contains_required_sections(self):
        prompt = _parameter_sweep_sample_prompt()
        self.assertIn("Dataset path:", prompt)
        self.assertIn("Split:", prompt)
        self.assertIn("Baseline parameters:", prompt)
        self.assertIn("Sweep ranges:", prompt)
        self.assertIn("Metrics to plot:", prompt)

    def test_parameter_sweep_sample_prompt_contains_supported_parameters_only(self):
        prompt = _parameter_sweep_sample_prompt()
        for token in ("top_k", "sampling_mode", "max_images", "max_images_per_class", "seed"):
            self.assertIn(token, prompt)
        for unsupported in ("SVM C", "SVM kernel", "HOG pixels per cell"):
            self.assertNotIn(unsupported, prompt)

    def test_parameter_sweep_sample_prompt_contains_all_supported_metrics(self):
        prompt = _parameter_sweep_sample_prompt()
        for metric_label in (
            "TFLite top-1 accuracy",
            "HOG+SVM top-1 accuracy",
            "TFLite top-k accuracy",
            "HOG+SVM top-k accuracy",
            "model agreement rate",
        ):
            self.assertIn(metric_label, prompt)

    def test_parameter_sweep_sample_prompt_parses_into_valid_plan(self):
        prompt = _parameter_sweep_sample_prompt()
        parsed = parse_parameter_sweep_request(prompt)
        self.assertEqual(parsed["dataset_path"], str((Path.cwd() / "dataset_sample").resolve()))
        self.assertEqual(parsed["split"], "test")
        self.assertEqual(parsed["baseline_values"]["top_k"], "5")
        self.assertEqual(parsed["baseline_values"]["sampling_mode"], "balanced")
        self.assertEqual(parsed["parameter_ranges"]["top_k"], ["1", "3", "5", "8", "10"])
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
        plan = build_parameter_sweep_plan(
            dataset_path=parsed["dataset_path"],
            split=parsed["split"],
            parameter_ranges=parsed["parameter_ranges"],
            baseline_values=parsed["baseline_values"],
            selected_metrics=parsed["selected_metrics"],
        )
        self.assertTrue(plan["success"])
        self.assertEqual(plan["split"], "test")
        self.assertEqual(plan["total_planned_runs"], 23)


if __name__ == "__main__":
    unittest.main()
