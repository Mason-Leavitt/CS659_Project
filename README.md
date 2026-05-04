# CS659 Plant Classification Project

## Project Overview

This project compares two plant species classification approaches:

- Deep learning: MobileNetV2 exported to TensorFlow Lite
- Classical computer vision: HOG + SVM

The repository also includes a conversational AI interface built with Streamlit + LangChain/OpenAI.

Current supported capabilities:

- Image classification
- Model comparison
- Metrics inspection
- Ablation feasibility analysis
- Conversational ablation planning, execution, and history review

## Repository Structure

### `DeepLearning-tensorFlowLite/`

Owns the model layer:

- training scripts
- preprocessing utilities
- standalone inference wrappers
- metrics helpers
- model artifacts and label files

### `agent/`

Owns the conversational app layer:

- Streamlit UI
- conversational routing
- tool wrappers
- safe imports into the model layer
- ablation feasibility and hardware inspection

## Setup Instructions

Run these commands from the repository root.

### Windows

```bash
uv venv .venv --python 3.10
.\.venv\Scripts\activate
uv pip install -r requirements.txt
streamlit run agent\streamlit_app.py
```

### macOS / Linux

```bash
uv venv .venv --python 3.10
source .venv/bin/activate
uv pip install -r requirements.txt
streamlit run agent/streamlit_app.py
```

## Environment Setup

Copy:

```text
agent/.env.example -> agent/.env
```

Set:

```text
OPENAI_API_KEY=...
```

OpenAI is required for full conversational agent mode.

Without an OpenAI key, the fallback mode still works for:

- artifact status
- metrics availability
- classification
- ablation feasibility

## Run the Application

From the repository root:

```bash
streamlit run agent/streamlit_app.py
```

Alternative:

```bash
cd agent
streamlit run streamlit_app.py
```

## Artifact Expectations

- TFLite model:
  `DeepLearning-tensorFlowLite/plant_classifier.tflite`
- Labels:
  `DeepLearning-tensorFlowLite/plant_labels_scientific.txt` preferred
  `DeepLearning-tensorFlowLite/plant_labels_export.txt` fallback
- HOG+SVM:
  requires trained artifacts and is not included by default
- Metrics:
  expected under `DeepLearning-tensorFlowLite/result/` after experiments

If you use checked-in HOG+SVM `joblib` artifacts from `result/`, install the pinned
`scikit-learn` version from `DeepLearning-tensorFlowLite/requirements-tflite.txt`.
Loading those artifacts with a different sklearn version may still run, but inference
can be unreliable and retraining/export in the current environment may be required.

## Features

- TFLite classification with top-k output
- Optional HOG+SVM baseline comparison
- Metrics parsing from `result/`
- Ablation feasibility analysis without retraining
- Conversational ablation planning with safe defaults and hard constraints
- Persistent ablation history stored under `DeepLearning-tensorFlowLite/result/ablation_history/`
- Conservative recommendations that only become performance-based when enough prior runs exist
- Conversational interface for classification and inspection tasks

## Conversational Ablation Workflow

You can now ask the agent or Streamlit app to:

- run an ablation study
- recommend ablation settings
- run a balanced 200-image ablation on the test split
- compare this run to previous ablations
- show observations from prior runs

Every completed no-retraining ablation is appended to:

`DeepLearning-tensorFlowLite/result/ablation_history/ablation_history.jsonl`

The history directory may also contain:

- `ablation_history_table.csv`
- `ablation_history_metrics.png`

Recommendations may be unavailable initially. When there is not enough history,
the agent falls back to hard constraints and safe defaults instead of fabricating
performance-based advice.

## Limitations

- HOG+SVM comparison requires trained HOG artifacts
- Checked-in HOG+SVM `joblib` artifacts are sensitive to the `scikit-learn` version used to load them
- Metrics inspection requires a populated `result/` directory
- Ablation support is limited to no-retraining inference-time and post-training studies
- Recommendation quality depends on the amount of persisted ablation history
- No object detection or YOLO workflow is included
