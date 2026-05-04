#!/usr/bin/env python3
"""
Conversational plant-classification agent layer.

This module supports two modes:

- Full LangChain/OpenAI agent execution when dependencies and API key exist.
- A deterministic fallback router that can still answer artifact, metrics,
  classification, and comparison requests without LangChain.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from agent import agent_tools

_DOTENV_LOADED = False
_DEFAULT_OPENAI_MODEL = "gpt-5-nano"
_SYSTEM_PROMPT = (
    "You are a plant species classification assistant. "
    "The primary available classifier is a TensorFlow Lite MobileNetV2 model. "
    "A HOG+SVM baseline is optional and only available when its saved artifacts exist. "
    "Metrics are only available when real result files exist on disk. "
    "Never invent predictions, model outputs, artifact status, or metrics. "
    "Use tools for image classification, classifier comparison, artifact inspection, "
    "metrics lookup, dataset-manifest building, sample ablation execution, ablation history inspection, ablation-result export, recommendation lookup, and ablation-feasibility inspection. "
    "If the user asks for classical, traditional, baseline, HOG, or SVM classification, use classify_with_hog_svm. "
    "If the user asks for deep learning, CNN, MobileNet, TensorFlow Lite, TensorFlow, or TFLite classification, use classify_with_tflite. "
    "If the user asks what ablation settings, options, or hyperparameters are available, explain the currently implemented inference-time controls and distinguish them from unsupported retraining-time hyperparameters. "
    "If the user asks what a specific ablation control or hyperparameter means, or whether a HOG/SVM training-time setting is currently supported, use explain_ablation_control. "
    "Examples include top_k, sampling_mode, max_images, max_images_per_class, seed, color_correct, SVM C, SVM kernel, and HOG pixels per cell. "
    "Do not claim that retraining-time HOG/SVM hyperparameter sweeps are supported unless explain_ablation_control says they are currently exposed by the inference-only runner. "
    "If the user asks to plan, explain, or run an inference-only one-factor parameter sweep, use get_parameter_sweep_support, plan_parameter_sweep, and run_parameter_sweep_tool as appropriate. "
    "Supported sweep parameters are top_k, sampling_mode, max_images, max_images_per_class, and seed. "
    "Do not claim that unsupported HOG/SVM training-time hyperparameters or TFLite-only color_correct are part of this parameter sweep unless a sweep tool says they are supported. "
    "If the user asks to compare models, use compare_classifiers. If the user asks to build a dataset manifest or inspect a test split, use build_dataset_manifest. If the user asks to plan or run an ablation study from a dataset path, use plan_ablation_study and run_planned_ablation. If the user asks about previous ablations, latest ablation results, comparisons, or exportable ablation reports, use the ablation history and export tools. "
    "If HOG+SVM artifacts are missing, say so clearly. "
    "If metrics are missing, say so clearly. "
    "Never claim that ablation experiments were run unless a tool explicitly provides real results."
)


def _load_env_if_available() -> None:
    """Load .env files if python-dotenv is installed; otherwise do nothing."""
    global _DOTENV_LOADED
    if _DOTENV_LOADED:
        return

    try:
        from dotenv import load_dotenv
    except Exception:
        _DOTENV_LOADED = True
        return

    candidate_paths = [
        os.path.join(os.path.dirname(__file__), ".env"),
        os.path.join(os.path.dirname(__file__), "..", ".env"),
    ]
    for path in candidate_paths:
        if os.path.isfile(path):
            load_dotenv(path, override=False)
    _DOTENV_LOADED = True


def _get_openai_api_key() -> str | None:
    _load_env_if_available()
    value = os.environ.get("OPENAI_API_KEY")
    return value.strip() if value and value.strip() else None


def _looks_like_openai_api_key(value: str | None) -> bool:
    if not value:
        return False
    text = value.strip()
    return text.startswith("sk-") and len(text) >= 20


def _probe_agent_runtime() -> tuple[bool, str | None]:
    if not agent_tools.LANGCHAIN_AVAILABLE or not agent_tools.AGENT_TOOLS:
        return False, (
            "LangChain agent mode is not available in this environment. "
            "You can still ask for artifact status, metrics availability, TFLite classification, "
            "or classifier comparison through the built-in fallback router."
        )

    try:
        from langchain_openai import ChatOpenAI  # noqa: F401
    except Exception:
        return False, (
            "LangChain agent mode is unavailable because langchain-openai is not installed. "
            "You can still use artifact status, metrics availability, TFLite classification, "
            "and comparison requests through the built-in fallback router."
        )

    try:
        from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder  # noqa: F401
        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage  # noqa: F401
    except Exception:
        return False, (
            "LangChain agent mode is unavailable because the installed langchain-core components are incomplete. "
            "You can still use artifact status, metrics availability, TFLite classification, "
            "and comparison requests through the built-in fallback router."
        )

    return True, None


def _langchain_setup_error() -> str | None:
    api_key = _get_openai_api_key()
    if api_key is None:
        return (
            "OpenAI agent mode is not configured because OPENAI_API_KEY is missing. "
            "You can still use artifact status, metrics availability, TFLite classification, "
            "and comparison requests through the built-in fallback router."
        )
    if not _looks_like_openai_api_key(api_key):
        return (
            "OpenAI agent mode is not configured because the API key format does not look valid. "
            "You can still use artifact status, metrics availability, TFLite classification, "
            "and comparison requests through the built-in fallback router."
        )
    runtime_ok, runtime_error = _probe_agent_runtime()
    if not runtime_ok:
        return runtime_error
    return None


def get_agent_mode_status(model_name: str | None = None) -> dict[str, Any]:
    selected_model = model_name or _DEFAULT_OPENAI_MODEL
    api_key = _get_openai_api_key()
    key_valid = _looks_like_openai_api_key(api_key)
    runtime_ok, _runtime_error = _probe_agent_runtime()
    agent_available = runtime_ok
    setup_error = _langchain_setup_error()
    mode = "Agent" if setup_error is None else "Fallback"
    if mode == "Agent":
        message = f"OpenAI API key detected. OpenAI model: {selected_model}"
    else:
        message = "No valid OpenAI API key detected, fallback in use."
    return {
        "mode": mode,
        "agent_available": agent_available,
        "api_key_present": api_key is not None,
        "api_key_valid": key_valid,
        "model_name": selected_model,
        "message": message,
        "error": setup_error,
    }


class _ToolBindingAgentExecutor:
    """Compatibility wrapper for environments without langchain.agents helpers."""

    def __init__(self, llm: Any, tools: list[Any], system_prompt: str) -> None:
        from langchain_core.messages import HumanMessage, SystemMessage

        self._llm = llm.bind_tools(tools)
        self._tools = tools
        self._tool_map = {getattr(tool, "name", f"tool_{i}"): tool for i, tool in enumerate(tools)}
        self._system_prompt = system_prompt
        self._HumanMessage = HumanMessage
        self._SystemMessage = SystemMessage

    def _invoke_tool(self, tool: Any, args: Any) -> Any:
        if hasattr(tool, "invoke"):
            payload = args if isinstance(args, dict) else (args or {})
            return tool.invoke(payload)
        if hasattr(tool, "func") and callable(tool.func):
            if isinstance(args, dict):
                return tool.func(**args)
            if args is None:
                return tool.func()
            return tool.func(args)
        if callable(tool):
            if isinstance(args, dict):
                return tool(**args)
            if args is None:
                return tool()
            return tool(args)
        raise RuntimeError(f"Tool '{getattr(tool, 'name', 'unknown')}' is not invokable.")

    def invoke(self, payload: dict[str, Any]) -> dict[str, Any]:
        from langchain_core.messages import ToolMessage

        messages: list[Any] = [self._SystemMessage(content=self._system_prompt)]
        chat_history = payload.get("chat_history") or []
        messages.extend(chat_history)
        messages.append(self._HumanMessage(content=str(payload.get("input", ""))))

        for _ in range(4):
            ai_message = self._llm.invoke(messages)
            messages.append(ai_message)
            tool_calls = getattr(ai_message, "tool_calls", None) or []
            if not tool_calls:
                content = getattr(ai_message, "content", "")
                return {"output": str(content) if content is not None else ""}

            for tool_call in tool_calls:
                if isinstance(tool_call, dict):
                    tool_name = tool_call.get("name")
                    tool_args = tool_call.get("args", {})
                    tool_call_id = tool_call.get("id") or tool_name or "tool_call"
                else:
                    tool_name = getattr(tool_call, "name", None)
                    tool_args = getattr(tool_call, "args", {})
                    tool_call_id = getattr(tool_call, "id", None) or tool_name or "tool_call"

                tool = self._tool_map.get(str(tool_name))
                if tool is None:
                    tool_output = {"error": f"Tool '{tool_name}' is not available."}
                else:
                    try:
                        tool_output = self._invoke_tool(tool, tool_args)
                    except Exception as exc:
                        tool_output = {"error": f"Tool '{tool_name}' failed: {exc}"}

                messages.append(
                    ToolMessage(
                        content=json.dumps(tool_output, ensure_ascii=True),
                        tool_call_id=str(tool_call_id),
                    )
                )

        return {
            "output": "Full agent mode could not complete tool-calling cleanly, so no final conversational response was produced."
        }


def create_vision_agent(model_name: str | None = None) -> Any:
    """
    Create the LangChain/OpenAI conversational agent when dependencies exist.

    Returns an AgentExecutor-compatible object when available.
    Raises RuntimeError with a readable message when setup is incomplete.
    """
    setup_error = _langchain_setup_error()
    if setup_error is not None:
        raise RuntimeError(setup_error)

    construction_errors: list[str] = []

    try:
        from langchain_openai import ChatOpenAI
    except Exception as exc:
        raise RuntimeError(
            "LangChain agent mode is unavailable because langchain-openai is not installed."
        ) from exc

    llm = ChatOpenAI(model=model_name or _DEFAULT_OPENAI_MODEL, temperature=0)

    try:
        from langchain.agents import AgentExecutor, create_tool_calling_agent
        from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", _SYSTEM_PROMPT),
                MessagesPlaceholder(variable_name="chat_history", optional=True),
                ("human", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )
        agent = create_tool_calling_agent(llm, agent_tools.AGENT_TOOLS, prompt)
        return AgentExecutor(agent=agent, tools=agent_tools.AGENT_TOOLS, verbose=False)
    except Exception as exc:
        construction_errors.append(f"tool-calling agent path failed: {exc}")

    try:
        return _ToolBindingAgentExecutor(llm=llm, tools=agent_tools.AGENT_TOOLS, system_prompt=_SYSTEM_PROMPT)
    except Exception as exc:
        construction_errors.append(f"bind_tools runnable path failed: {exc}")

    raise RuntimeError(
        "Full LangChain agent mode is unavailable because no compatible agent-construction path succeeded. "
        + " | ".join(construction_errors)
    )


def _coerce_chat_history_for_langchain(chat_history: list | None) -> list[Any]:
    """
    Convert Streamlit-style history dictionaries into LangChain message objects.

    Expected incoming format from Streamlit is:
    {"role": "user" | "assistant" | "system", "content": "..."}
    """
    if not chat_history:
        return []

    try:
        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
    except Exception as exc:
        raise RuntimeError("LangChain message conversion is unavailable in this environment.") from exc

    converted: list[Any] = []
    for item in chat_history:
        if item is None:
            continue
        if hasattr(item, "content") and hasattr(item, "type"):
            converted.append(item)
            continue
        if not isinstance(item, dict):
            continue
        role = str(item.get("role", "user")).strip().lower()
        content = str(item.get("content", ""))
        if not content:
            continue
        if role == "assistant":
            converted.append(AIMessage(content=content))
        elif role == "system":
            converted.append(SystemMessage(content=content))
        else:
            converted.append(HumanMessage(content=content))
    return converted


def _message_lower(message: str) -> str:
    return (message or "").strip().lower()


def _asks_for_artifacts(message: str) -> bool:
    text = _message_lower(message)
    return any(token in text for token in ("artifact", "artifacts", "status", "available", "availability"))


def _asks_for_metrics(message: str) -> bool:
    text = _message_lower(message)
    if "ablation" in text:
        return False
    explicit_metric_tokens = (
        "metric",
        "metrics",
        "performance metric",
        "performance metrics",
    )
    explicit_model_metric_phrases = (
        "model performance",
        "model accuracy",
        "model metrics",
        "classifier metrics",
        "classifier performance",
        "hog metrics",
        "hog performance",
        "tflite metrics",
        "tflite performance",
    )
    return any(token in text for token in explicit_metric_tokens) or any(
        phrase in text for phrase in explicit_model_metric_phrases
    )


def _asks_for_compare(message: str) -> bool:
    text = _message_lower(message)
    return any(token in text for token in ("compare", "comparison", "both models", "agree", "agreement"))


def _asks_for_classification(message: str) -> bool:
    text = _message_lower(message)
    return any(
        token in text
        for token in (
            "classify",
            "classification",
            "predict",
            "prediction",
            "identify",
            "species",
            "what plant",
        )
    )


def _asks_for_hog_classification(message: str) -> bool:
    text = _message_lower(message)
    model_tokens = (
        "hog",
        "svm",
        "hog+svm",
        "classical",
        "classical model",
        "classical approach",
        "classical baseline",
        "traditional",
        "traditional model",
        "traditional approach",
        "baseline model",
        "baseline",
        "shallow",
        "hand-crafted",
        "handcrafted",
        "feature-based",
        "use the baseline model",
        "use hog+svm only",
        "use the classical model",
    )
    return _asks_for_classification(message) and any(token in text for token in model_tokens)


def _asks_for_tflite_classification(message: str) -> bool:
    text = _message_lower(message)
    model_tokens = (
        "tflite",
        "tensorflow lite",
        "tensorflow",
        "mobilenet",
        "mobilenetv2",
        "cnn",
        "deep",
        "deep learning",
        "neural network",
    )
    return _asks_for_classification(message) and any(token in text for token in model_tokens)


def _asks_for_ablation(message: str) -> bool:
    text = _message_lower(message)
    return any(token in text for token in ("ablation", "hyperparameter", "experiment", "experiments"))


def _asks_for_ablation_options(message: str) -> bool:
    text = _message_lower(message)
    option_query_phrases = (
        "what hyperparameter",
        "what hyperparameters",
        "available hyperparameter",
        "available hyperparameters",
        "what parameters",
        "what settings",
        "settings can i vary",
        "parameters can i vary",
        "what can i vary",
        "experiment with",
        "ablation settings",
        "ablation options",
    )
    if not any(phrase in text for phrase in option_query_phrases):
        return False
    if not _asks_for_ablation(message):
        return False

    classical_tokens = (
        "classical",
        "baseline",
        "traditional",
        "hog",
        "svm",
        "hog+svm",
    )
    general_option_context = (
        "ablation settings",
        "ablation options",
        "ablation study",
    )
    return any(token in text for token in classical_tokens) or any(
        phrase in text for phrase in general_option_context
    )


def _asks_for_ablation_control_explanation(message: str) -> bool:
    text = _message_lower(message)
    if _asks_for_ablation_options(message):
        return False

    question_starts = (
        "what does ",
        "what is ",
        "explain ",
        "define ",
        "why is ",
    )
    if not any(text.startswith(prefix) for prefix in question_starts):
        return False

    control_hints = (
        "max_images_per_class",
        "max images per class",
        "maximum images per class",
        "sampling_mode",
        "sampling mode",
        "balanced sampling",
        "random sampling",
        "sorted sampling",
        "top_k",
        "top k",
        "top-k",
        "seed",
        "dataset_path",
        "dataset path",
        "split",
        "color_correct",
        "color correction",
        "colour correction",
        "svm c",
        "svm kernel",
        "svm gamma",
        "hog feature extraction",
        "hog orientations",
        "hog pixels per cell",
        "hog cells per block",
        "class weighting",
    )
    return any(hint in text for hint in control_hints)


def _asks_for_parameter_sweep(message: str) -> bool:
    text = _message_lower(message)
    direct_phrases = (
        "parameter sweep",
        "one-factor sweep",
        "one factor sweep",
        "one-factor-at-a-time",
        "one factor at a time",
        "ofat",
        "inference-only sweep",
        "vary only one parameter at a time",
    )
    if any(phrase in text for phrase in direct_phrases):
        return True

    sweep_hints = (
        "top_k",
        "top k",
        "top-k",
        "sampling_mode",
        "sampling mode",
        "max_images",
        "max images",
        "max_images_per_class",
        "max images per class",
        "seed",
        "svm c",
        "svm kernel",
        "svm gamma",
        "hog pixels per cell",
        "hog cells per block",
        "hog orientations",
        "color_correct",
        "color correction",
    )
    return ("sweep" in text or "vary " in text or "varying " in text) and any(
        hint in text for hint in sweep_hints
    )


def _asks_to_run_parameter_sweep(message: str) -> bool:
    text = _message_lower(message)
    if not _asks_for_parameter_sweep(message):
        return False
    return any(token in text for token in ("run", "execute", "start", "launch"))


def _asks_about_parameter_sweep_options(message: str) -> bool:
    text = _message_lower(message)
    if not _asks_for_parameter_sweep(message):
        return False
    return any(
        phrase in text
        for phrase in (
            "what is",
            "how does",
            "how do",
            "what parameters",
            "which parameters",
            "what settings",
            "which settings",
            "supported parameter",
            "supported parameters",
            "supported control",
            "available parameter",
            "available parameters",
            "available controls",
            "can i sweep",
        )
    )


def _asks_to_plan_parameter_sweep(message: str) -> bool:
    if not _asks_for_parameter_sweep(message):
        return False
    if _asks_to_run_parameter_sweep(message) or _asks_about_parameter_sweep_options(message):
        return False
    return True


def _asks_for_manifest(message: str) -> bool:
    text = _message_lower(message)
    return any(token in text for token in ("manifest", "scan dataset", "scan the dataset", "test split", "dataset split"))


def _asks_for_run_sample_ablation(message: str) -> bool:
    text = _message_lower(message)
    return any(token in text for token in ("run ablation", "sample ablation", "ablation study", "run the ablation"))


def _asks_for_ablation_recommendations(message: str) -> bool:
    text = _message_lower(message)
    return any(
        token in text
        for token in (
            "recommend ablation settings",
            "what ablation should i run",
            "what settings do you recommend",
            "ablation settings do you recommend",
        )
    )


def _asks_for_ablation_history(message: str) -> bool:
    text = _message_lower(message)
    return any(
        token in text
        for token in (
            "show previous ablation results",
            "summarize ablation history",
            "previous ablation",
            "prior runs",
            "prior ablations",
            "observations from prior runs",
            "ablation history",
        )
    )


def _asks_for_ablation_results(message: str) -> bool:
    text = _message_lower(message)
    return any(
        token in text
        for token in (
            "show ablation results",
            "show latest ablation",
            "summarize ablation results",
            "compare ablation runs",
            "compare the last two ablations",
            "show charts from ablation",
            "download ablation report",
            "export ablation results",
            "latest ablation",
        )
    )


def _extract_path_like_token(message: str) -> str | None:
    quoted = re.findall(r'"([^"]+)"|\'([^\']+)\'', message)
    for pair in quoted:
        value = next((part for part in pair if part), "")
        if value:
            return value

    parts = re.split(r"\s+", message.strip())
    for part in parts:
        cleaned = part.strip(" ,.;:()[]{}")
        if "\\" in cleaned or "/" in cleaned or cleaned.endswith(".csv") or cleaned.lower() == "dataset_sample":
            return cleaned
    return None


def _extract_split_name(message: str) -> str:
    text = _message_lower(message)
    for split in ("validation", "val", "train", "test"):
        if re.search(rf"\b{re.escape(split)}\b", text):
            return split
    return "test"


def _default_dataset_if_available() -> str | None:
    candidate = r"E:\school\659\plantnet_300K\plantnet_300K"
    return candidate if os.path.exists(candidate) else None


def _default_parameter_sweep_dataset_if_available() -> str | None:
    local_sample = (Path(__file__).resolve().parent.parent / "dataset_sample").resolve()
    if local_sample.exists():
        return str(local_sample)
    return _default_dataset_if_available()


def _format_artifact_status(status: dict[str, Any]) -> str:
    tflite_exists = bool(status.get("tflite_model", {}).get("exists"))
    labels_exists = bool(status.get("default_labels", {}).get("exists"))
    hog_model_exists = bool(status.get("hog_svm_model", {}).get("exists"))
    hog_labels_exists = bool(status.get("hog_svm_labels", {}).get("exists"))
    result_exists = bool(status.get("result_dir", {}).get("exists"))

    lines = [
        f"TFLite model available: {tflite_exists}.",
        f"Default labels available: {labels_exists}.",
        f"HOG+SVM model available: {hog_model_exists}.",
        f"HOG+SVM labels available: {hog_labels_exists}.",
        f"Result directory available: {result_exists}.",
    ]
    if not hog_model_exists or not hog_labels_exists:
        lines.append("HOG+SVM comparison is not currently available because its saved artifacts are missing.")
    if not result_exists:
        lines.append("Metrics are not currently available because the result directory does not exist.")
    return " ".join(lines)


def _format_ablation_options_response(message: str) -> str:
    text = _message_lower(message)
    classical_focus = any(
        token in text for token in ("classical", "baseline", "traditional", "hog", "svm", "hog+svm")
    )
    controls = agent_tools.get_supported_ablation_controls()
    shared_details = controls.get("shared_sampling_control_details", []) if isinstance(controls, dict) else []
    classical_details = controls.get("classical_control_details", []) if isinstance(controls, dict) else []
    deep_details = controls.get("deep_learning_control_details", []) if isinstance(controls, dict) else []
    unswept = controls.get("not_currently_swept_details", []) if isinstance(controls, dict) else []

    def _detail_by_name(entries: list[Any], name: str) -> dict[str, Any]:
        for item in entries:
            if isinstance(item, dict) and item.get("name") == name:
                return item
        return {}

    unswept_labels = [
        (
            str(item.get("label") or item.get("name"))
            if isinstance(item, dict)
            else str(item)
        )
        for item in unswept
        if (isinstance(item, dict) and (item.get("label") or item.get("name"))) or item
    ]

    split_detail = _detail_by_name(shared_details, "split")
    sampling_detail = _detail_by_name(shared_details, "sampling_mode")
    max_images_detail = _detail_by_name(shared_details, "max_images")
    max_images_per_class_detail = _detail_by_name(shared_details, "max_images_per_class")
    seed_detail = _detail_by_name(shared_details, "seed")
    classical_top_k_detail = _detail_by_name(classical_details, "top_k")
    deep_color_detail = _detail_by_name(deep_details, "color_correct")

    split_values = split_detail.get("values", ["test", "train", "val", "validation"])
    split = split_values[0] if split_values else "test"
    sampling_modes = sampling_detail.get("values", ["balanced", "random", "sorted"])
    seed = seed_detail.get("default", 42)
    max_images_values = max_images_detail.get("suggested_values", [50, 200])
    quick_images = max_images_values[0] if max_images_values else 50
    robust_images = max_images_values[1] if len(max_images_values) > 1 else quick_images
    max_images_per_class = max_images_per_class_detail.get("suggested_value", 1)
    top_k_values = classical_top_k_detail.get("values", [1, 3, 5])
    tflite_color_modes = deep_color_detail.get("values", ["none", "gray_world", "max_rgb"])

    parts = [
        "The current ablation system is inference-only. It evaluates saved model artifacts and does not retrain the classical HOG+SVM model.",
    ]

    if classical_focus:
        parts.extend(
            [
                "",
                "For the classical HOG+SVM path, the ablation controls currently exposed by the code are:",
                f"- `top_k` for ranked HOG+SVM evaluation: {', '.join(str(v) for v in top_k_values)}",
                "- `dataset_path`",
                f"- `split`: {split}, train, val, or validation",
                f"- `sampling_mode`: {', '.join(sampling_modes)}",
                f"- `max_images` for run size, with common presets like {quick_images} (quick) and {robust_images} (stronger)",
                f"- `max_images_per_class`, commonly {max_images_per_class} for balanced small-sample studies",
                f"- `seed`, with a default of {seed} for reproducible sampling",
                "",
                "Those settings change which images are evaluated and how the ranked HOG results are summarized, but they do not change the saved HOG feature extractor or SVM weights.",
                "",
                "The current inference-only runner does not expose classical training-time hyperparameters like "
                + ", ".join(unswept_labels or ["saved HOG feature extraction and SVM training settings"])
                + " as ablation knobs. Those would require a separate training or artifact-generation pipeline.",
                "",
                "What the main controls mean:",
                f"- `sampling_mode`: {sampling_detail.get('description', 'Chooses how images are sampled into the manifest.')} Balanced is preferred for small representative samples.",
                f"- `max_images`: {max_images_detail.get('description', 'Sets the global sample size for the run.')}",
                f"- `max_images_per_class`: {max_images_per_class_detail.get('description', 'Limits per-class sampling before or during manifest generation.')}",
                f"- `seed`: {seed_detail.get('description', 'Keeps sampling reproducible when randomness is used.')}",
                f"- `top_k`: {classical_top_k_detail.get('description', 'Controls how many ranked predictions are returned and evaluated.')}",
                "",
                f"TFLite-only ablation settings such as color correction ({', '.join(tflite_color_modes)}) exist for the deep model, but they do not apply to the classical HOG+SVM path.",
            ]
        )
    else:
        parts.extend(
            [
                "",
                "The currently implemented ablation controls are dataset and sampling oriented: dataset_path, split, sampling_mode, max_images, max_images_per_class, seed, and model-specific ranked-evaluation settings like `top_k`.",
                f"For HOG+SVM specifically, the exposed model-side knob is `top_k` ({', '.join(str(v) for v in top_k_values)}).",
                "Training-time hyperparameters are not currently swept by this inference-only ablation system.",
            ]
        )

    return "\n".join(parts)


def _format_ablation_control_detail_response(result: dict[str, Any]) -> str:
    if not result.get("success"):
        return result.get("message") or (
            "I did not recognize that as a supported ablation control. "
            "Ask for available ablation controls or try a known term like sampling_mode, max_images_per_class, top_k, seed, color_correct, or svm_c."
        )

    detail = result.get("detail", {}) if isinstance(result.get("detail"), dict) else {}
    label = result.get("matched_label") or detail.get("label") or result.get("matched_name") or "This control"
    name = result.get("matched_name") or detail.get("name") or "unknown"
    description = detail.get("description") or "No description is available."
    applies_to = detail.get("applies_to", [])
    examples = detail.get("examples", [])
    values = detail.get("values", [])
    notes = detail.get("notes", [])
    default = detail.get("default")

    parts = [f"{label} (`{name}`): {description}"]
    if result.get("is_supported_control"):
        parts.append("This is currently a supported inference-time ablation control.")
    elif result.get("is_not_currently_swept"):
        parts.append(
            "This is not currently exposed by the inference-only ablation runner. Changing it would require a separate training, model-generation, or alternate-artifact workflow."
        )

    if applies_to:
        parts.append("Applies to: " + ", ".join(str(item) for item in applies_to) + ".")
    if values:
        parts.append("Supported values: " + ", ".join(str(item) for item in values) + ".")
    if default is not None:
        parts.append(f"Default: {default}.")
    if examples:
        parts.append("Examples: " + ", ".join(str(item) for item in examples[:4]) + ".")
    if notes:
        parts.append("Notes: " + " ".join(str(item) for item in notes[:2]))
    return " ".join(parts)


def _format_parameter_sweep_options_response(message: str) -> str:
    support = agent_tools.get_parameter_sweep_support()
    supported_parameters = list(support.get("supported_parameters", []))
    supported_metrics = list(support.get("supported_metrics", []))
    default_baseline_values = dict(support.get("default_baseline_values", {}))
    default_selected_metrics = list(support.get("default_selected_metrics", []))

    detail = agent_tools.get_ablation_control_detail(message)
    if not detail.get("success"):
        lowered = _message_lower(message)
        stripped_query = lowered
        for prefix in ("can i sweep ", "could i sweep ", "may i sweep ", "sweep ", "vary "):
            if stripped_query.startswith(prefix):
                stripped_query = stripped_query[len(prefix) :].strip(" .?")
                break
        if stripped_query and stripped_query != lowered:
            detail = agent_tools.get_ablation_control_detail(stripped_query)
    if detail.get("success"):
        matched_name = str(detail.get("matched_name") or "")
        matched_label = str(detail.get("matched_label") or matched_name)
        if matched_name not in supported_parameters:
            if detail.get("is_not_currently_swept"):
                return (
                    f"No. {matched_label} (`{matched_name}`) is not currently supported by the inference-only one-factor parameter sweep. "
                    "It is treated as a training-time or alternate-artifact setting rather than a sweepable inference-time control. "
                    "Supported sweep parameters are: "
                    + ", ".join(f"`{item}`" for item in supported_parameters)
                    + "."
                )
            return (
                f"No. {matched_label} (`{matched_name}`) is not part of the current inference-only parameter sweep surface. "
                "Supported sweep parameters are: "
                + ", ".join(f"`{item}`" for item in supported_parameters)
                + "."
            )

    baseline_text = ", ".join(f"`{key}={value}`" for key, value in default_baseline_values.items())
    metric_text = ", ".join(f"`{item}`" for item in default_selected_metrics or supported_metrics[:2])
    return (
        "A one-factor-at-a-time parameter sweep is inference-only: it varies one supported parameter while holding the others fixed at baseline values. "
        "Supported sweep parameters are "
        + ", ".join(f"`{item}`" for item in supported_parameters)
        + ". "
        + "Default baseline values are "
        + baseline_text
        + ". "
        + "Default plotted metrics are "
        + metric_text
        + ". "
        + "Training-time HOG/SVM hyperparameters and TFLite-only `color_correct` are not part of this sweep."
    )


def _format_parameter_sweep_plan(result: dict[str, Any]) -> str:
    supported_parameters = list(result.get("supported_parameters", []))
    if not result.get("success"):
        parts = [
            "Inference-only parameter sweep planning did not produce a runnable plan.",
        ]
        errors = [str(item) for item in result.get("errors", []) if str(item).strip()]
        warnings = [str(item) for item in result.get("warnings", []) if str(item).strip()]
        if errors:
            parts.append("Errors:")
            parts.extend(f"- {item}" for item in errors)
        if warnings:
            parts.append("Warnings:")
            parts.extend(f"- {item}" for item in warnings)
        if supported_parameters:
            parts.append(
                "Supported sweep parameters: " + ", ".join(f"`{item}`" for item in supported_parameters) + "."
            )
        request = str(result.get("request") or "").strip()
        parsed_request = result.get("parsed_request", {}) if isinstance(result.get("parsed_request"), dict) else {}
        if request and not parsed_request.get("parameter_ranges"):
            parts.append(
                "Try a request like: `Plan a parameter sweep with top_k 1,3,5 and max_images 50,100,200.`"
            )
        return "\n".join(parts).strip()

    parameter_ranges = dict(result.get("parameter_ranges", {}))
    baseline_values = dict(result.get("baseline_values", {}))
    selected_metrics = list(result.get("selected_metrics", []))
    sweep_points = list(result.get("generated_sweep_points", []))
    parts = [
        "Inference-only one-factor parameter sweep plan ready.",
        "Parameters swept: " + ", ".join(f"`{name}` ({len(values)} values)" for name, values in parameter_ranges.items()) + ".",
        "Baseline values: " + ", ".join(f"`{key}={value}`" for key, value in baseline_values.items()) + ".",
        "Selected metrics: " + ", ".join(f"`{item}`" for item in selected_metrics) + ".",
        f"Total planned runs: {result.get('total_planned_runs', 0)}.",
    ]
    if result.get("require_confirmation"):
        parts.append("This plan exceeds a safety limit and should not be run automatically.")
    if result.get("dataset_path"):
        parts.append(f"Dataset path: `{result.get('dataset_path')}`.")
    else:
        parts.append("Dataset path is not set yet, so this plan is preview-only until you provide one.")
    warnings = [str(item) for item in result.get("warnings", []) if str(item).strip()]
    if warnings:
        parts.append("Warnings: " + " ".join(warnings))
    preview_lines = []
    for point in sweep_points[:3]:
        values = dict(point.get("parameter_values", {}))
        preview_lines.append(
            f"- vary `{point.get('varied_parameter')}` to `{point.get('varied_value')}` with "
            + ", ".join(f"`{key}={value}`" for key, value in values.items())
        )
    if preview_lines:
        parts.extend(["First sweep points:"] + preview_lines)
    parts.append("This remains inference-only and does not retrain either model.")
    return "\n".join(parts).strip()


def _format_parameter_sweep_result(result: dict[str, Any]) -> str:
    if not result.get("success"):
        parts = [f"Parameter sweep did not run successfully. Status: `{result.get('status', 'failed')}`."]
        errors = [str(item) for item in result.get("errors", []) if str(item).strip()]
        warnings = [str(item) for item in result.get("warnings", []) if str(item).strip()]
        if result.get("output_dir"):
            parts.append(f"Output directory: `{result.get('output_dir')}`.")
        if errors:
            parts.append("Errors:")
            parts.extend(f"- {item}" for item in errors)
        if warnings:
            parts.append("Warnings:")
            parts.extend(f"- {item}" for item in warnings)
        return "\n".join(parts).strip()

    summary = result.get("summary", {}) if isinstance(result.get("summary"), dict) else {}
    artifacts = result.get("artifacts", {}) if isinstance(result.get("artifacts"), dict) else {}
    charts = [item for item in summary.get("charts", []) if isinstance(item, dict) and item.get("filename")]
    parts = [
        f"Parameter sweep finished with status `{result.get('status', 'completed')}`.",
        f"Output directory: `{result.get('output_dir')}`.",
        f"Planned runs: {summary.get('total_planned_runs', 0)}. Completed: {summary.get('total_completed_runs', 0)}. Failed: {summary.get('total_failed_runs', 0)}.",
    ]
    if charts:
        parts.append("Generated charts: " + ", ".join(f"`{chart.get('filename')}`" for chart in charts) + ".")
    if artifacts:
        artifact_names = [f"`{Path(path).name}`" for path in artifacts.values() if path]
        if artifact_names:
            parts.append("Artifacts: " + ", ".join(artifact_names) + ".")
    warnings = [str(item) for item in result.get("warnings", []) if str(item).strip()]
    errors = [str(item) for item in result.get("errors", []) if str(item).strip()]
    if warnings:
        parts.append("Warnings: " + " ".join(warnings))
    if errors:
        parts.append("Errors: " + " ".join(errors[:3]))
    parts.append("This sweep is inference-only and does not update normal ablation history.")
    return "\n".join(parts).strip()


def _format_metrics_summary(summary: dict[str, Any]) -> str:
    if not summary.get("available"):
        return summary.get("error") or "Metrics are not available."

    runs = summary.get("runs", [])
    if not runs:
        return "Metrics are not available because no summary files were found."

    def _metric_preview(metrics: dict[str, Any]) -> str:
        candidates = [
            ("validation_accuracy", "validation_accuracy"),
            ("accuracy", "accuracy"),
            ("f1_macro", "f1_macro"),
            ("f1_weighted", "f1_weighted"),
            ("precision_macro", "precision_macro"),
            ("recall_macro", "recall_macro"),
            ("roc_auc_ovr_macro", "roc_auc_ovr_macro"),
        ]
        parts: list[str] = []
        for key, label in candidates:
            value = metrics.get(key)
            if isinstance(value, (int, float)):
                parts.append(f"{label}={value:.4f}")
        return ", ".join(parts)

    parts = [f"Metrics are available in {len(runs)} run location(s)."]
    for run in runs[:5]:
        run_dir = run.get("run_dir", "result/")
        method = run.get("method", "unknown")
        metrics = run.get("metrics", {}) if isinstance(run.get("metrics"), dict) else {}
        files = run.get("files", [])
        preview = _metric_preview(metrics)
        line = f"{method} run at {run_dir}"
        if preview:
            line += f": {preview}."
        else:
            line += "."
        if files:
            line += f" Files: {', '.join(files)}."
        if run.get("error"):
            line += f" Parse issue: {run['error']}."
        parts.append(line)
    if summary.get("error"):
        parts.append(summary["error"])
    return " ".join(parts)


def _format_ablation_feasibility(data: dict[str, Any]) -> str:
    if not data.get("success"):
        return data.get("error") or "Ablation feasibility is not available."

    hardware = data.get("hardware", {})
    possible_now = data.get("possible_now", [])
    requires_dataset = data.get("requires_dataset", [])
    requires_training = data.get("requires_training", [])
    skipped = data.get("skipped", [])

    hardware_parts: list[str] = []
    if hardware.get("ram_total_gb") is not None:
        hardware_parts.append(f"RAM {hardware['ram_total_gb']} GB")
    if hardware.get("gpu_name"):
        gpu_line = str(hardware["gpu_name"])
        if hardware.get("gpu_vram_gb") is not None:
            gpu_line += f" ({hardware['gpu_vram_gb']} GB VRAM)"
        hardware_parts.append(f"GPU {gpu_line}")
    elif hardware.get("cuda_available") or hardware.get("tensorflow_gpu_visible"):
        hardware_parts.append("GPU visible")
    else:
        hardware_parts.append("No GPU currently visible")

    parts = [data.get("explanation") or "Ablation feasibility summary generated."]
    if hardware_parts:
        parts.append("Hardware summary: " + "; ".join(hardware_parts) + ".")
    if possible_now:
        parts.append("Possible now: " + "; ".join(possible_now) + ".")
    if requires_dataset:
        parts.append("Requires dataset access: " + "; ".join(requires_dataset) + ".")
    if requires_training:
        parts.append("Requires training: " + "; ".join(requires_training) + ".")
    if skipped:
        parts.append("Skipped for now: " + "; ".join(skipped) + ".")
    return " ".join(parts)


def _format_manifest_result(result: dict[str, Any]) -> str:
    if not result.get("success"):
        return result.get("error") or (
            "Dataset manifest generation is not available. Provide dataset_sample/ or an absolute path to plantnet_300K/."
        )

    parts = [
        f"Manifest built for split '{result.get('split', 'test')}'.",
        f"Resolved image root: {result.get('resolved_image_root')}.",
        f"Images: {result.get('num_images', 0)} across {result.get('num_classes', 0)} classes.",
    ]
    if result.get("manifest_path"):
        parts.append(f"Manifest CSV: {result.get('manifest_path')}.")
    preview = result.get("preview", [])
    if preview:
        sample = preview[0]
        parts.append(
            f"Example row: label {sample.get('label')} ({sample.get('display_label')}), image {sample.get('image_path')}."
        )
    return " ".join(parts)


def _format_sample_ablation_result(result: dict[str, Any]) -> str:
    if not result.get("success"):
        return result.get("error") or (
            "Sample ablation could not run. Build a manifest first, then pass its CSV path."
        )

    summary = result.get("summary", {})
    metrics = summary.get("metrics", {}) if isinstance(summary.get("metrics"), dict) else {}
    parts = [
        f"Sample ablation finished. Images attempted: {summary.get('num_images_attempted', 0)}. Images evaluated: {summary.get('num_images_evaluated', 0)}.",
        f"Outputs were saved to {result.get('output_dir')}.",
    ]
    notes = metrics.get("metrics_notes", [])
    if notes:
        parts.append("Key results: " + " ".join(str(note) for note in notes[:5]))
    skipped = result.get("skipped", {})
    if skipped:
        parts.append(
            "Skipped components: " + "; ".join(f"{key}: {value}" for key, value in skipped.items()) + "."
        )
    files = result.get("files", {})
    if files:
        named = [f"{name}={path}" for name, path in files.items() if path]
        if named:
            parts.append("Saved files: " + "; ".join(named) + ".")
    return " ".join(parts)


def _format_ablation_recommendations(result: dict[str, Any]) -> str:
    if not result.get("success"):
        return result.get("error") or "Ablation recommendations are unavailable."

    def _labelize(key: str) -> str:
        return str(key).replace("_", " ").strip().capitalize()

    def _format_rate(value: Any) -> str | None:
        if isinstance(value, (int, float)):
            return f"{100.0 * float(value):.1f}%"
        return None

    def _config_lines(config: dict[str, Any]) -> list[str]:
        if not isinstance(config, dict) or not config:
            return []
        dataset_value = config.get("dataset_label") or config.get("dataset_path")
        rows = []
        mapping = [
            ("profile", config.get("profile")),
            ("dataset", dataset_value),
            ("split", config.get("split")),
            ("sampling mode", config.get("sampling_mode")),
            ("max images", config.get("max_images")),
            ("max images per class", config.get("max_images_per_class")),
            ("seed", config.get("seed")),
        ]
        for label, value in mapping:
            if value is None or value == "":
                continue
            rows.append(f"- {label}: `{value}`")
        return rows

    def _supporting_metric_lines(metrics: dict[str, Any]) -> list[str]:
        if not isinstance(metrics, dict) or not metrics:
            return []
        rows: list[str] = []
        simple_fields = [
            ("run count", metrics.get("num_runs")),
            ("evaluated image range", metrics.get("evaluated_image_range")),
            ("class count range", metrics.get("class_count_range")),
        ]
        for label, value in simple_fields:
            if value is None or value == "":
                continue
            rows.append(f"- {label}: `{value}`")
        rate_fields = [
            ("best TFLite top-1", metrics.get("best_tflite_top1")),
            ("best TFLite top-k", metrics.get("best_tflite_topk")),
            ("best HOG top-1", metrics.get("best_hog_top1")),
            ("best HOG top-k", metrics.get("best_hog_topk")),
            ("best agreement", metrics.get("best_agreement")),
            ("best stability", metrics.get("best_stability")),
        ]
        for label, value in rate_fields:
            formatted = _format_rate(value)
            if formatted is not None:
                rows.append(f"- {label}: `{formatted}`")
        return rows

    recommendations = result.get("recommendations", {})
    constraints = result.get("constraints", [])
    evidence_summary = result.get("evidence_summary")
    recommendation_status = result.get("recommendation_status")
    supporting_metrics = result.get("supporting_metrics", {})
    caveats = result.get("caveats", [])
    next_best_actions = result.get("next_best_actions", [])
    history_summary = result.get("history_summary", {})

    recommended_config = result.get("recommended_config")
    if not isinstance(recommended_config, dict) or not recommended_config:
        recommended_config = (
            recommendations.get("recommended_robust_run")
            or recommendations.get("recommended_quick_run")
            or recommendations.get("exploratory_run_option")
            or {}
        )

    lines = [
        f"Recommendation status: {recommendation_status or ('evidence_supported' if result.get('enough_history') else 'insufficient_history')}",
        f"Enough history: {'yes' if result.get('enough_history', False) else 'no'}",
    ]
    if evidence_summary:
        lines.extend(["", "Evidence summary:", str(evidence_summary)])

    config_rows = _config_lines(recommended_config if isinstance(recommended_config, dict) else {})
    if config_rows:
        lines.extend(["", "Recommended configuration:"])
        lines.extend(config_rows)

    metric_rows = _supporting_metric_lines(supporting_metrics if isinstance(supporting_metrics, dict) else {})
    if metric_rows:
        lines.extend(["", "Supporting metrics:"])
        lines.extend(metric_rows)

    if caveats:
        lines.extend(["", "Caveats:"])
        lines.extend(f"- {item}" for item in caveats[:4] if item)
    elif constraints:
        lines.extend(["", "Caveats:"])
        lines.extend(f"- {item}" for item in constraints[:4] if item)

    if next_best_actions:
        lines.extend(["", "Next best actions:"])
        lines.extend(f"- {item}" for item in next_best_actions[:4] if item)

    if isinstance(history_summary, dict) and history_summary.get("count") is not None:
        lines.extend(["", f"History records: {history_summary.get('count', 0)}"])

    return "\n".join(lines).strip()


def _format_ablation_history(result: dict[str, Any]) -> str:
    if not result.get("success"):
        return result.get("error") or "Ablation history is unavailable."
    entries = result.get("entries", [])
    if not entries:
        return "No persistent ablation history exists yet."
    parts = [f"There are {result.get('count', len(entries))} recorded ablation run(s)."]
    for entry in entries[:5]:
        metrics = entry.get("metrics", {}) if isinstance(entry.get("metrics"), dict) else {}
        parts.append(
            f"Run {entry.get('run_id')} used split={entry.get('split')} sampling={entry.get('sampling_mode')} max_images={entry.get('max_images')}; "
            f"evaluated {entry.get('num_images_evaluated')} image(s); "
            f"TFLite top-k={metrics.get('tflite_topk_accuracy')}; HOG top-k={metrics.get('hog_topk_accuracy')}."
        )
    return " ".join(parts)


def _format_ablation_run(result: dict[str, Any]) -> str:
    if not result.get("success"):
        return result.get("error") or "Ablation result is unavailable."
    run = result.get("run", {})
    metrics = run.get("metrics", {}) if isinstance(run.get("metrics"), dict) else {}
    generated_files = run.get("generated_files", {}) if isinstance(run.get("generated_files"), dict) else {}
    parts = [
        f"Run {run.get('run_id')} used split={run.get('split')} sampling={run.get('sampling_mode')} max_images={run.get('max_images')}.",
        f"Images evaluated: {run.get('num_images_evaluated', 0)} across {run.get('num_classes', 0)} classes.",
        f"TFLite top-1={metrics.get('tflite_top1_accuracy')}; TFLite top-k={metrics.get('tflite_topk_accuracy')}; "
        f"HOG top-1={metrics.get('hog_top1_accuracy')}; HOG top-k={metrics.get('hog_topk_accuracy')}.",
        f"Agreement={metrics.get('model_agreement_rate')}; TFLite stability={metrics.get('tflite_color_stability_rate')}.",
    ]
    if generated_files:
        parts.append("Artifacts: " + "; ".join(f"{key}={value}" for key, value in generated_files.items() if value) + ".")
    caveats = run.get("caveats", [])
    if caveats:
        parts.append("Caveats: " + " ".join(str(item) for item in caveats))
    return " ".join(parts)


def _comparison_warning_lines(result: dict[str, Any]) -> list[str]:
    if not isinstance(result, dict):
        return []
    explicit_warnings = [
        str(item).strip()
        for item in result.get("comparison_warnings", [])
        if str(item).strip()
    ]
    if explicit_warnings:
        return explicit_warnings
    missing_run_ids = [
        str(item).strip()
        for item in result.get("missing_run_ids", [])
        if str(item).strip()
    ]
    if missing_run_ids:
        return [
            "Requested run IDs were not found and were omitted: "
            + ", ".join(f"`{run_id}`" for run_id in missing_run_ids)
            + "."
        ]
    return []


def _format_ablation_comparison(result: dict[str, Any]) -> str:
    warning_lines = _comparison_warning_lines(result)
    if not result.get("success"):
        parts = [result.get("error") or "Ablation comparison is unavailable."]
        requested_run_ids = [str(item) for item in result.get("requested_run_ids", []) if str(item).strip()]
        matched_run_ids = [str(item) for item in result.get("matched_run_ids", []) if str(item).strip()]
        missing_run_ids = [str(item) for item in result.get("missing_run_ids", []) if str(item).strip()]
        if requested_run_ids:
            parts.append("Requested run IDs: " + ", ".join(f"`{item}`" for item in requested_run_ids) + ".")
        if matched_run_ids:
            parts.append("Matched run IDs: " + ", ".join(f"`{item}`" for item in matched_run_ids) + ".")
        if missing_run_ids:
            parts.append("Missing run IDs: " + ", ".join(f"`{item}`" for item in missing_run_ids) + ".")
        if warning_lines:
            parts.extend(["", "Comparison warnings:"])
            parts.extend(f"- {item}" for item in warning_lines)
        return "\n".join(parts).strip()
    rows = result.get("comparison_rows", [])
    if not rows:
        parts = [result.get("explanation") or "No ablation runs were available to compare."]
        if warning_lines:
            parts.extend(["", "Comparison warnings:"])
            parts.extend(f"- {item}" for item in warning_lines)
        return "\n".join(parts).strip()
    parts = [result.get("explanation") or "Compared ablation runs."]
    if warning_lines:
        parts.extend(["", "Comparison warnings:"])
        parts.extend(f"- {item}" for item in warning_lines)
    for row in rows[:3]:
        parts.append(
            f"{row.get('run_id')}: split={row.get('split')} sampling={row.get('sampling_mode')} max_images={row.get('max_images')} "
            f"TFLite top-k={row.get('tflite_topk_accuracy')} HOG top-k={row.get('hog_topk_accuracy')} "
            f"agreement={row.get('model_agreement_rate')} stability={row.get('tflite_color_stability_rate')}."
        )
    return " ".join(parts)


def _format_ablation_export(result: dict[str, Any]) -> str:
    if not result.get("success"):
        return result.get("error") or "Ablation export is unavailable."
    artifacts = result.get("artifacts", {})
    parts = [f"Exported ablation history artifacts for {result.get('count', 0)} recorded run(s)."]
    if artifacts:
        parts.append("Files: " + "; ".join(f"{key}={value}" for key, value in artifacts.items() if value) + ".")
    if result.get("corrupted_lines"):
        parts.append("Some corrupted history lines were skipped during export.")
    return " ".join(parts)


def _format_planned_ablation(result: dict[str, Any]) -> str:
    if not result.get("success"):
        return result.get("error") or "Ablation planning failed."
    if result.get("needs_more_info"):
        return result.get("prompt") or "More information is needed before running the ablation."
    plan = result.get("plan", {})
    warnings = result.get("warnings", [])
    parts = [
        f"Planned ablation: dataset_path={plan.get('dataset_path')}, split={plan.get('split')}, max_images={plan.get('max_images')}, "
        f"sampling_mode={plan.get('sampling_mode')}, seed={plan.get('seed')}, max_images_per_class={plan.get('max_images_per_class')}."
    ]
    if warnings:
        parts.append("Planner notes: " + " ".join(str(item) for item in warnings))
    return " ".join(parts)


def _format_run_planned_ablation(result: dict[str, Any]) -> str:
    if not result.get("success"):
        return result.get("error") or "Planned ablation failed."
    manifest = result.get("manifest", {})
    ablation = result.get("ablation", {})
    summary = ablation.get("summary", {}) if isinstance(ablation, dict) else {}
    metrics = summary.get("metrics", {}) if isinstance(summary.get("metrics"), dict) else {}
    parts = [
        f"Planned ablation completed. Manifest: {manifest.get('manifest_path')}. Output directory: {ablation.get('output_dir')}.",
        f"Images evaluated: {summary.get('num_images_evaluated', 0)} across {summary.get('num_classes', 0)} classes.",
    ]
    if metrics:
        parts.append(
            f"TFLite top-k accuracy={metrics.get('tflite_topk_accuracy')}; HOG top-k accuracy={metrics.get('hog_topk_accuracy')}; "
            f"agreement rate={metrics.get('model_agreement_rate')}; stability rate={metrics.get('tflite_color_stability_rate')}."
        )
    return " ".join(parts)


def _format_tflite_result(result: dict[str, Any]) -> str:
    if not result.get("success"):
        return f"TFLite classification is not available: {result.get('error') or 'unknown error'}"

    preds = result.get("predictions", [])
    if not preds:
        return "TFLite classification completed, but no predictions were returned."

    top1 = preds[0]
    top1_prob = float(top1.get("probability", 0.0))
    if top1_prob >= 0.90:
        confidence_text = "very confident"
    elif top1_prob >= 0.70:
        confidence_text = "confident"
    elif top1_prob >= 0.50:
        confidence_text = "moderately confident"
    else:
        confidence_text = "uncertain"

    parts = [
        f"TFLite predicts '{top1.get('label')}' as the top result with {100.0 * top1_prob:.2f}% confidence, so the model appears {confidence_text}."
    ]
    if len(preds) > 1:
        top2 = preds[1]
        top2_prob = float(top2.get("probability", 0.0))
        preview = []
        for pred in preds[: min(3, len(preds))]:
            preview.append(
                f"{pred.get('rank')}. {pred.get('label')} ({100.0 * float(pred.get('probability', 0.0)):.2f}%)"
            )
        parts.append("Top predictions: " + "; ".join(preview) + ".")
        if top1_prob - top2_prob < 0.15:
            parts.append(
                f"The top two classes are fairly close, so the model may be uncertain between '{top1.get('label')}' and '{top2.get('label')}'."
            )
    warning = result.get("warning")
    if warning:
        parts.append(warning)
    return " ".join(parts)


def _format_hog_result(result: dict[str, Any]) -> str:
    if not result.get("success"):
        return f"HOG+SVM classification is not available: {result.get('error') or 'unknown error'}"

    prediction = result.get("prediction", {})
    label = prediction.get("label")
    display_label = prediction.get("display_label") or label
    confidence = prediction.get("confidence")
    score = prediction.get("score")
    if display_label and label and display_label != label:
        parts = [f"HOG+SVM predicts {display_label} (PlantNet ID: {label})."]
    else:
        parts = [f"HOG+SVM predicts '{display_label}'."]
    if confidence is not None:
        parts.append(f"Confidence: {100.0 * float(confidence):.2f}%.")
    if score is not None:
        parts.append(f"Score: {float(score):.6f}.")
    predictions = result.get("predictions", [])
    score_type = str(result.get("score_type") or "label_only")
    if len(predictions) > 1:
        if score_type == "probability":
            parts.append("Additional values are HOG+SVM probabilities.")
        elif score_type == "decision_function":
            parts.append("Additional values are HOG+SVM decision scores, not probabilities.")
        preview = []
        for item in predictions[: min(3, len(predictions))]:
            item_label = item.get("display_label") or item.get("label")
            if item_label and item.get("label") and item_label != item.get("label"):
                item_label = f"{item_label} (PlantNet ID: {item.get('label')})"
            if item.get("confidence") is not None:
                value_text = f"{100.0 * float(item.get('confidence', 0.0)):.2f}%"
            elif item.get("score") is not None:
                value_text = f"{float(item.get('score', 0.0)):.6f}"
            else:
                value_text = "n/a"
            preview.append(f"{item.get('rank')}. {item_label} ({value_text})")
        parts.append("Top HOG+SVM predictions: " + "; ".join(preview) + ".")
    warning = result.get("warning")
    if warning:
        parts.append(warning)
    return " ".join(parts)


def _format_comparison(result: dict[str, Any]) -> str:
    if result.get("error"):
        return result.get("explanation") or result["error"]

    parts = [result.get("explanation") or "Comparison completed."]
    agreement = result.get("agreement")
    if agreement is True:
        parts.append("Both models agree on the top label.")
    elif agreement is False:
        parts.append(
            "The models disagree on the top label. This may indicate ambiguity in the image, "
            "differences in preprocessing, or the limitations of HOG grayscale features versus CNN learned features."
        )
    else:
        parts.append("The comparison is incomplete, so a direct agreement judgment is not available.")
    hog_result = result.get("hog_svm_result", {})
    if hog_result.get("warning"):
        parts.append(hog_result["warning"])
    if not hog_result.get("success"):
        parts.append(
            f"HOG+SVM is unavailable: {hog_result.get('error') or 'missing artifacts'}."
        )
    return " ".join(parts)


def _fallback_router(
    message: str,
    image_path: str | None = None,
    chat_history: list | None = None,
) -> str:
    return str(_fallback_router_with_payload(message=message, image_path=image_path, chat_history=chat_history).get("message", ""))


def _fallback_router_with_payload(
    message: str,
    image_path: str | None = None,
    chat_history: list | None = None,
) -> dict[str, Any]:
    _ = chat_history

    if _asks_about_parameter_sweep_options(message):
        return {"message": _format_parameter_sweep_options_response(message)}

    if _asks_to_run_parameter_sweep(message):
        dataset_path = _extract_path_like_token(message)
        if not dataset_path:
            dataset_path = _default_parameter_sweep_dataset_if_available()
        split = _extract_split_name(message)
        parsed_request = agent_tools.parse_parameter_sweep_request_tool(message)
        if not parsed_request.get("parameter_ranges"):
            return {
                "message": (
                    "I do not keep a persisted conversational parameter sweep plan in fallback mode. "
                    "Please restate at least one sweep range with comma-separated values, for example: "
                    "`Run a parameter sweep with top_k 1,3,5 and max_images 50,100,200 on dataset_sample.`"
                )
            }
        plan = agent_tools.plan_parameter_sweep(
            request=message,
            dataset_path=dataset_path,
            split=split,
        )
        if not plan.get("success") or plan.get("require_confirmation") or not plan.get("dataset_path"):
            return {
                "message": _format_parameter_sweep_plan(plan),
                "parameter_sweep_plan": plan,
            }
        result = agent_tools.run_parameter_sweep_tool(
            plan=plan,
            write_charts=True,
        )
        return {
            "message": _format_parameter_sweep_result(result),
            "parameter_sweep_plan": plan,
            "parameter_sweep_result": result,
        }

    if _asks_to_plan_parameter_sweep(message):
        dataset_path = _extract_path_like_token(message)
        if not dataset_path and "default dataset" in _message_lower(message):
            dataset_path = _default_dataset_if_available()
        plan = agent_tools.plan_parameter_sweep(
            request=message,
            dataset_path=dataset_path,
            split=_extract_split_name(message),
        )
        return {
            "message": _format_parameter_sweep_plan(plan),
            "parameter_sweep_plan": plan,
        }

    if _asks_for_ablation_recommendations(message):
        return {"message": _format_ablation_recommendations(agent_tools.get_ablation_recommendations())}

    if _asks_for_ablation_results(message):
        history = agent_tools.get_ablation_results(limit=10)
        entries = history.get("entries", [])
        text = _message_lower(message)
        if not entries:
            return {"message": "No ablation history exists yet. Run an ablation study first, then I can summarize or export the results."}
        if "latest" in text:
            return {"message": _format_ablation_run(agent_tools.get_latest_ablation_result())}
        if "compare" in text:
            return {"message": _format_ablation_comparison(agent_tools.compare_ablation_results())}
        if "export" in text or "download" in text:
            return {"message": _format_ablation_export(agent_tools.export_ablation_results())}
        run_id = None
        for entry in entries:
            candidate = str(entry.get("run_id", ""))
            if candidate and candidate in message:
                run_id = candidate
                break
        if run_id:
            return {"message": _format_ablation_run(agent_tools.get_ablation_result_by_id(run_id))}
        if len(entries) == 1:
            return {"message": _format_ablation_run(agent_tools.get_latest_ablation_result())}
        return {
            "message": (
                "There are multiple ablation runs available. "
                "Tell me whether you want the latest run, an all-runs summary, a comparison across runs, or a specific run_id."
            )
        }

    if _asks_for_ablation_history(message):
        return {"message": _format_ablation_history(agent_tools.get_ablation_history(limit=10))}

    if _asks_for_ablation_control_explanation(message):
        return {"message": _format_ablation_control_detail_response(agent_tools.get_ablation_control_detail(message))}

    if _asks_for_ablation_options(message):
        return {"message": _format_ablation_options_response(message)}

    if _asks_for_metrics(message):
        return {"message": _format_metrics_summary(agent_tools.get_metrics_summary())}

    if _asks_for_artifacts(message):
        return {"message": _format_artifact_status(agent_tools.get_artifact_status())}

    if _asks_for_manifest(message):
        dataset_path = _extract_path_like_token(message)
        if not dataset_path:
            return {
                "message": (
                    "I can scan a dataset and build a manifest once you provide dataset_sample/ or an absolute path "
                    "to plantnet_300K/. For the held-out dataset split, use split='test'."
                )
            }
        split = _extract_split_name(message)
        return {
            "message": _format_manifest_result(
                agent_tools.build_dataset_manifest(
                    dataset_path=dataset_path,
                    split=split,
                    max_images=50,
                )
            )
        }

    if _asks_for_run_sample_ablation(message):
        plan = agent_tools.plan_ablation_study(message=message)
        planned = plan.get("plan", {}) if isinstance(plan, dict) else {}
        if not planned.get("dataset_path") and "default dataset" in _message_lower(message):
            default_dataset = _default_dataset_if_available()
            if default_dataset:
                plan = agent_tools.plan_ablation_study(message=message, context={"dataset_path": default_dataset})
                planned = plan.get("plan", {})
        if plan.get("needs_more_info"):
            return {"message": _format_planned_ablation(plan)}
        if plan.get("parsed", {}).get("asks_for_full_run"):
            return {
                "message": (
                    _format_planned_ablation(plan)
                    + " This is a full-run request, so expect a longer runtime. Ask again with the same settings if you want to proceed explicitly."
                )
            }
        return {
            "message": _format_run_planned_ablation(
                agent_tools.run_planned_ablation(
                    dataset_path=str(planned.get("dataset_path")),
                    split=str(planned.get("split", "test")),
                    max_images=planned.get("max_images"),
                    sampling_mode=str(planned.get("sampling_mode", "balanced")),
                    seed=int(planned.get("seed", 42)),
                    max_images_per_class=planned.get("max_images_per_class", 1),
                )
            )
        }

    if _asks_for_ablation(message):
        if "plan" in _message_lower(message):
            return {"message": _format_planned_ablation(agent_tools.plan_ablation_study(message=message))}
        return {"message": _format_ablation_feasibility(agent_tools.get_ablation_feasibility())}

    if _asks_for_compare(message):
        if not image_path:
            return {"message": "I can compare the classifiers once you provide an image path or upload an image."}
        return {
            "message": _format_comparison(
                agent_tools.compare_classifiers(
                    image_path=image_path,
                    top_k=5,
                    color_correct="none",
                )
            )
        }

    if _asks_for_hog_classification(message):
        if not image_path:
            return {"message": "I can classify the plant with HOG+SVM once you provide an image path or upload an image."}
        return {"message": _format_hog_result(agent_tools.classify_with_hog_svm(image_path=image_path, top_k=5))}

    if _asks_for_tflite_classification(message):
        if not image_path:
            return {"message": "I can classify the plant with the TFLite model once you provide an image path or upload an image."}
        return {
            "message": _format_tflite_result(
                agent_tools.classify_with_tflite(
                    image_path=image_path,
                    top_k=5,
                    color_correct="none",
                )
            )
        }

    if _asks_for_classification(message):
        if not image_path:
            return {"message": "I can classify the plant once you provide an image path or upload an image."}
        formatted = _format_tflite_result(
            agent_tools.classify_with_tflite(
                image_path=image_path,
                top_k=5,
                color_correct="none",
            )
        )
        return {
            "message": (
                formatted
                + " This defaults to the primary TFLite model. If you want the classical baseline instead, ask for HOG, SVM, the classical model, or the baseline approach."
            )
        }

    status = agent_tools.get_artifact_status()
    actions: list[str] = []
    if image_path:
        actions.append("classification")
    if image_path and status.get("hog_svm_model", {}).get("exists") and status.get("hog_svm_labels", {}).get("exists"):
        actions.append("comparison")
    if status.get("result_dir", {}).get("exists"):
        actions.append("metrics")
    actions.append("dataset manifest building")
    actions.append("sample ablation studies")
    actions.append("ablation feasibility")

    return {
        "message": (
            "Here’s what I can do right now: "
            + ", ".join(actions)
            + ". "
            + _format_artifact_status(status)
        )
    }


def run_agent_turn(
    message: str,
    image_path: str | None = None,
    chat_history: list | None = None,
    model_name: str | None = None,
) -> dict[str, Any]:
    """
    Run one user message through the conversational layer and return a message plus any structured UI payloads.
    """
    try:
        if not str(message or "").strip():
            return {"message": "Please send a message describing what you want to do."}

        use_fallback = False
        if _asks_for_metrics(message) or _asks_for_artifacts(message):
            use_fallback = True
        elif _asks_for_parameter_sweep(message):
            use_fallback = True
        elif _asks_for_manifest(message) or _asks_for_run_sample_ablation(message):
            use_fallback = True
        elif _asks_for_ablation_recommendations(message) or _asks_for_ablation_history(message) or _asks_for_ablation_results(message):
            use_fallback = True
        elif _asks_for_ablation(message):
            use_fallback = True
        elif _asks_for_compare(message):
            use_fallback = True
        elif _asks_for_hog_classification(message):
            use_fallback = True
        elif _asks_for_tflite_classification(message):
            use_fallback = True
        elif (_asks_for_compare(message) or _asks_for_classification(message)) and not image_path:
            use_fallback = True
        elif _langchain_setup_error() is not None:
            use_fallback = True

        if use_fallback:
            return _fallback_router_with_payload(message=message, image_path=image_path, chat_history=chat_history)

        agent_executor = create_vision_agent(model_name=model_name)
        payload: dict[str, Any] = {
            "input": message if image_path is None else f"{message}\n\nCurrent image path: {image_path}",
            "chat_history": _coerce_chat_history_for_langchain(chat_history),
        }
        result = agent_executor.invoke(payload)
        if isinstance(result, dict):
            output = result.get("output")
            if output:
                return {"message": str(output)}
        return {"message": str(result)}
    except Exception as exc:
        setup_error = _langchain_setup_error()
        if setup_error is not None:
            fallback = _fallback_router_with_payload(message=message, image_path=image_path, chat_history=chat_history)
            return {"message": f"{setup_error} {fallback.get('message', '')}".strip()}
        fallback = _fallback_router_with_payload(message=message, image_path=image_path, chat_history=chat_history)
        return {
            "message": (
                "Full agent mode is unavailable right now. "
                + str(exc)
                + " "
                + str(fallback.get("message", ""))
            ).strip()
        }


def run_agent_message(
    message: str,
    image_path: str | None = None,
    chat_history: list | None = None,
    model_name: str | None = None,
) -> str:
    """
    Run one user message through the conversational layer and return only text.
    """
    return str(
        run_agent_turn(
            message=message,
            image_path=image_path,
            chat_history=chat_history,
            model_name=model_name,
        ).get("message", "")
    )
