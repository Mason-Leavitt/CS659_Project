# Project Spec

## Purpose

This document defines the implementation contract for the next phase of the repository: a Streamlit conversational AI assistant for plant species image classification.

The spec is based on the current repository state and is intentionally constrained to what actually exists today.

## Current Repository State

- Core executable model and training code currently lives under `DeepLearning-tensorFlowLite/`.
- The existing deep learning classifier is a MobileNetV2 model exported to TensorFlow Lite.
- The existing TensorFlow Lite CLI inference entry point is `DeepLearning-tensorFlowLite/infer_plant_tflite.py`.
- The existing TensorFlow Lite model artifact is `DeepLearning-tensorFlowLite/plant_classifier.tflite`.
- The existing label files are:
  - `DeepLearning-tensorFlowLite/plant_labels_scientific.txt`
  - `DeepLearning-tensorFlowLite/plant_labels_export.txt`
- The existing classical baseline training entry point is `DeepLearning-tensorFlowLite/train_hog_svm.py`.
- There is currently no standalone HOG+SVM inference entry point.
- There is currently no implemented Streamlit app.
- There is currently no implemented LangChain or OpenAI integration code.
- A top-level `agent/` package will own future conversational tools and UI entry points.
- There is currently no checked-in `result/` directory.
- There is currently no checked-in saved HOG+SVM artifact.
- Existing artifact paths are effectively relative to the process working directory, so centralized absolute path resolution is required before app integration.

## 1. Project Goal

Build a Streamlit conversational AI assistant for plant species image classification.

The assistant must:

- Classify uploaded plant images with the existing TensorFlow Lite MobileNetV2 model.
- Compare the deep TFLite classifier against a classical HOG+SVM baseline when both baseline artifacts are available.
- Provide conversational access to predictions, artifact status, and available metrics.

## 2. Current Artifact Contracts

The implementation must treat the following artifact contracts as the starting point:

- TFLite model:
  - `DeepLearning-tensorFlowLite/plant_classifier.tflite`
- TFLite labels:
  - Preferred: `DeepLearning-tensorFlowLite/plant_labels_scientific.txt`
  - Fallback: `DeepLearning-tensorFlowLite/plant_labels_export.txt`
- HOG+SVM model:
  - Optional and currently missing until trained
  - Expected filename: `hog_svm_model.joblib`
- HOG+SVM labels:
  - Optional and currently missing until trained
  - Expected filename: `hog_svm_labels.txt`
- Metrics:
  - Optional and currently missing until generated
  - Expected under `result/`

Implementation must not assume HOG+SVM artifacts or metrics already exist.

## 3. Required New Components

The next implementation phase must add the following components:

- `DeepLearning-tensorFlowLite/app_config.py`
  - Centralized absolute path resolution for repository-relative and `DeepLearning-tensorFlowLite/`-relative assets.
- Reusable TFLite inference API
  - Must preserve the existing `infer_plant_tflite.py` CLI behavior while exposing importable inference functions.
- `DeepLearning-tensorFlowLite/hog_svm_utils.py`
  - Shared HOG preprocessing helpers aligned with `train_hog_svm.py`.
- `DeepLearning-tensorFlowLite/infer_hog_svm.py`
  - Standalone HOG+SVM inference entry point for saved `joblib` artifacts.
- `agent/__init__.py`
  - Marks the top-level app/agent package for future imports.
- `agent/agent_tools.py`
  - LangChain tools for inference, artifact inspection, and metrics lookup.
- `agent/plant_chat_agent.py`
  - LangChain/OpenAI agent assembly and routing logic.
- `agent/streamlit_app.py`
  - Main Streamlit UI entry point.
- `agent/requirements-app.txt`
  - App-level dependencies including Streamlit and LangChain/OpenAI packages.
- `agent/.env.example`
  - Example environment variables for API key configuration and optional overrides.

Directory ownership rules:

- `DeepLearning-tensorFlowLite/` owns model code, preprocessing, training, inference wrappers, metrics helpers, model artifacts, and labels.
- `agent/` owns conversational tools, LangChain/OpenAI routing, and Streamlit UI code.
- Agent/app code must call into the model-layer modules in `DeepLearning-tensorFlowLite/` instead of re-implementing model logic locally.

## 4. Streamlit UI Behavior

The Streamlit application must support the following behavior:

- Allow the user to upload a plant image.
- Save the uploaded image into an `uploads/` folder.
- Display the uploaded image in the UI.
- Allow the user to ask chat questions about the uploaded image.
- Show TensorFlow Lite top-k predictions.
- Show HOG+SVM prediction only if the required HOG artifacts exist.
- Show model comparison only if both models are available.
- Show friendly warnings when HOG artifacts are missing.
- Show friendly warnings when metrics are missing.

The UI must degrade gracefully when optional artifacts do not exist.

## 5. Agent Behavior

The conversational agent must:

- Answer conversationally.
- Use tools for model inference and metrics lookup.
- Never hallucinate predictions.
- Never hallucinate metrics.
- Explain missing artifacts clearly.

The agent must route user requests to the appropriate tool path for:

- TFLite classification
- HOG+SVM classification
- model comparison
- artifact status
- metrics summary lookup

## 6. TFLite Inference Behavior

The reusable TFLite inference layer must accept:

- image path
- model path
- labels path
- `top_k`
- `color_correct`

It must:

- Read the model input shape from the TFLite artifact.
- Resize the image to the expected model size.
- Rescale pixels to `[0, 1]`.
- Support `color_correct` values:
  - `none`
  - `gray_world`
  - `max_rgb`
- Return structured top-k predictions.
- Preserve the existing CLI behavior of `infer_plant_tflite.py`.

The reusable inference API should be suitable for both Streamlit and agent tool usage.

## 7. HOG+SVM Behavior

The HOG+SVM support layer must:

- Create reusable preprocessing that matches `train_hog_svm.py`.
- Use the following preprocessing contract:
  - grayscale conversion
  - normalization to `[0, 1]`
  - resize to `img_size`
  - HOG feature extraction
- Support inference from a saved `joblib` pipeline when available.
- Never retrain during inference.
- Gracefully report missing model artifacts.
- Gracefully report missing label artifacts.

The implementation must treat HOG+SVM inference as optional runtime functionality.

## 8. Evaluation And Metrics Behavior

The app and agent must inspect the filesystem for real existing metrics only.

For CNN/TFLite runs, look for:

- `metrics_summary.json`
- `metrics_classification_report.txt`
- `confusion_matrix_normalized.png`
- `confusion_row_correlation.png`
- `roc_curves.png`

For HOG+SVM runs, look for:

- `summary.json`
- `classification_report.txt`
- `confusion_matrix_normalized.png`

Rules:

- Summarize only files that actually exist.
- Do not fabricate metrics.
- Do not fabricate plots.
- If no metrics exist, explicitly say so.

Note:

- The current logging code may emit CNN files with names prefixed by `metrics_`, so implementation should be robust to the repository’s actual output naming while still honoring the intended contract above.

## 9. Ablation Behavior

Future ablations should be described and supported conceptually, but they are not required to be implemented in this spec step.

Potential future CNN/TFLite ablations:

- `color_correct`
- `img_size`
- `k_folds`, if retraining is feasible

Potential future HOG+SVM ablations:

- `img_size`
- `orientations`
- `pixels_per_cell`
- `cells_per_block`
- SVM `C`
- SVM `kernel`

Constraints:

- Expensive training ablations should be skipped unless dataset access and suitable hardware are available.
- This spec phase does not require retraining workflows.

## 10. Path Handling Requirement

All paths must resolve relative to the repository root or the `DeepLearning-tensorFlowLite/` directory, not the current process working directory.

This requirement is mandatory because Streamlit may be launched from the repository root or from `agent/`, and process-relative path assumptions would otherwise break artifact discovery.

Centralized path resolution must therefore be implemented before or alongside app integration.

Additional import and path rules:

- Agent code must not rely on the current working directory.
- Agent code must use `app_config.py` path helpers for artifact discovery.
- Agent code must import model-layer modules from `DeepLearning-tensorFlowLite/` using explicit `sys.path` setup or a small import helper if needed.
- Future Streamlit app startup must work when launched from the repository root or from `agent/`.

## 11. Non-Goals

This spec explicitly excludes the following:

- Redesigning the existing training scripts
- Implementing code during this spec step
- Assuming missing artifacts already exist
- Retraining models as part of app startup or inference
- Introducing YOLO
- Introducing object detection
- Introducing bounding boxes
- Introducing mAP or IoU evaluation
- Introducing new training pipelines
- Changing existing script arguments
- Removing any existing behavior

## 12. Implementation Constraints (MANDATORY)

These constraints are mandatory for any later implementation.

### 12.1 Existing Scripts MUST NOT Be Broken

- `infer_plant_tflite.py` CLI behavior must remain unchanged.
- Existing `infer_plant_tflite.py` arguments must remain unchanged.
- `train_hog_svm.py` must remain runnable as-is.
- Existing `train_hog_svm.py` arguments must remain unchanged.
- Later implementation may refactor internals, but it must not remove or break current script behavior.

### 12.2 No Duplication Of Logic

- Shared preprocessing must be extracted into reusable modules.
- `infer_plant_tflite.py` must call reusable inference functions instead of duplicating preprocessing and prediction logic in multiple places.
- HOG preprocessing must be centralized in `hog_svm_utils.py`.
- Streamlit, LangChain tools, and standalone inference entry points must reuse the same inference helpers rather than maintain parallel implementations.

### 12.3 No Retraining During Inference

- HOG+SVM inference must only load saved `joblib` artifacts.
- CNN inference must only use the exported TFLite model.
- Inference code must not trigger training, fine-tuning, export, or artifact regeneration.

### 12.4 Path Handling Rules

- No hardcoded absolute paths are allowed in implementation code.
- All application-facing paths must resolve through `app_config.py`.
- Code must work regardless of the current process working directory.
- Streamlit launch from repository root must be supported.
- Streamlit launch from `agent/` must be supported.
- Artifact discovery logic must resolve paths relative to the repository root or `DeepLearning-tensorFlowLite/`.
- Agent code must not assume its own directory is the working directory.
- Agent code must import model-layer modules using explicit `sys.path` setup or a small import helper if needed.

### 12.5 Graceful Degradation Rules

- Missing HOG artifacts must not crash the app.
- Missing metrics must not crash the app.
- Missing optional comparison functionality must not crash the app.
- The agent must explicitly report missing resources.
- The UI must present a human-readable warning when optional resources are unavailable.

### 12.6 No Hallucinated Outputs

- Predictions must come from real model calls.
- Metrics must come from real files.
- Artifact status must come from real filesystem checks.
- If data is unavailable, the app or agent must return an explicit `not available` style result instead of guessing.

## 13. End-to-End Data Flow

The intended runtime flow is:

1. User uploads an image in Streamlit.
2. Streamlit saves the image into `uploads/`.
3. Streamlit passes `image_path` and relevant user intent into the `agent/` layer.
4. The agent decides which tool to call based on the user request.
5. The selected tool calls the reusable inference or metrics layer in `DeepLearning-tensorFlowLite/`.
6. The reusable inference layer loads the required model artifact and preprocesses the image.
7. The model returns a prediction or score.
8. The tool formats the raw prediction into structured output.
9. The agent converts that structured output into a conversational response.
10. Streamlit renders the prediction result, warnings, and assistant response in the UI.

Additional data flow requirements:

- Image preprocessing must happen inside the reusable inference layer, not inside Streamlit UI code and not inside the agent prompt layer.
- Model loading must happen inside the reusable inference layer or dedicated helper functions used by the tool layer.
- Output formatting into dictionaries and typed result structures must happen in the tool or inference helper layer before the agent writes conversational text.
- Streamlit should render already-structured results rather than re-implement prediction logic.

## 14. Tool Contracts

All tool contracts below are mandatory for later implementation.

### 14.1 TFLite Tool

Input:

- `image_path`
- `top_k`
- `color_correct`

Output:

- `predictions`: list of objects with:
  - `label`
  - `probability`
  - `rank`

Behavior requirements:

- Must use the reusable TFLite inference layer.
- Must not fabricate labels or probabilities.
- Must return a structured error payload when required artifacts are missing or image loading fails.

### 14.2 HOG+SVM Tool

Input:

- `image_path`

Output:

- `label`
- `confidence` or `score`

Behavior requirements:

- Must load a saved `joblib` pipeline if present.
- Must not retrain.
- Must use centralized HOG preprocessing.
- Must return a structured error payload when required artifacts are missing or image loading fails.

### 14.3 Comparison Tool

Input:

- `image_path`

Output:

- both model outputs
- `agreement` boolean
- `explanation` string

Behavior requirements:

- Must call the TFLite tool and HOG+SVM tool or their shared inference backends.
- Must only provide comparison output when both model results are available.
- If one model is unavailable, it must return a structured partial result plus a clear explanation.

### 14.4 Metrics Tool

Input:

- none, or optional run directory

Output:

- parsed metrics summary

Behavior requirements:

- Must inspect only real filesystem artifacts.
- Must support missing metrics without crashing.
- Must return a structured error or `not available` result when no metrics are found.

## 15. Failure Modes and Expected Behavior

All failure paths must return both:

- a structured error
- a human-readable explanation

### 15.1 TFLite Model Missing

Expected behavior:

- Return a structured error indicating the TFLite model artifact is missing.
- Return a human-readable explanation that classification with the deep model is not available until `plant_classifier.tflite` is present.
- Do not crash the app.

### 15.2 Label File Missing

Expected behavior:

- Return a structured error indicating that no usable label file was found.
- Return a human-readable explanation that predictions cannot be mapped to labels until a valid label file exists.
- Do not guess labels.
- Do not crash the app.

### 15.3 HOG Model Missing

Expected behavior:

- Return a structured error indicating the HOG+SVM artifact is unavailable.
- Return a human-readable explanation that baseline comparison is not available until `hog_svm_model.joblib` and `hog_svm_labels.txt` are generated.
- Keep TFLite-only functionality working.
- Do not crash the app.

### 15.4 Metrics Missing

Expected behavior:

- Return a structured `not available` or error result for metrics lookup.
- Return a human-readable explanation that no metrics files were found under `result/`.
- Do not invent summaries.
- Do not crash the app.

### 15.5 Invalid Image Input

Expected behavior:

- Return a structured error indicating the image path is invalid, unreadable, or unsupported.
- Return a human-readable explanation telling the user the uploaded file could not be processed as an image.
- Do not crash the app.

## Suggested File Targets For Later Implementation

The following files are expected to be created or modified in a later implementation step:

- Create `PROJECT_SPEC.md`
- Create `DeepLearning-tensorFlowLite/app_config.py`
- Modify `DeepLearning-tensorFlowLite/infer_plant_tflite.py`
- Create `DeepLearning-tensorFlowLite/hog_svm_utils.py`
- Modify `DeepLearning-tensorFlowLite/train_hog_svm.py`
- Create `DeepLearning-tensorFlowLite/infer_hog_svm.py`
- Create `agent/__init__.py`
- Create `agent/agent_tools.py`
- Create `agent/plant_chat_agent.py`
- Create `agent/streamlit_app.py`
- Create `agent/requirements-app.txt`
- Create `agent/.env.example`
- Modify `DeepLearning-tensorFlowLite/README.md`

## Acceptance Criteria For A Later Implementation Prompt

A later implementation should be considered aligned with this spec if it:

- Preserves the existing TFLite CLI flow
- Adds reusable inference APIs
- Adds optional HOG+SVM inference without requiring retraining
- Adds centralized path handling
- Adds a Streamlit UI
- Adds a LangChain/OpenAI conversational layer
- Clearly reports missing artifacts and missing metrics
- Avoids inventing unavailable predictions, metrics, or artifacts
- Does not duplicate preprocessing logic across scripts, tools, and UI code
- Does not change existing script arguments
- Does not remove existing behavior
