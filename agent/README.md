# Agent Layer

## What the Agent Does

This directory contains the conversational plant classification assistant.

It uses:

- the TFLite classifier as the primary model
- the optional HOG+SVM baseline when its saved artifacts exist

It provides:

- classification
- model comparison
- metrics summary
- ablation feasibility
- conversational ablation planning and execution
- persistent ablation history and observations
- conservative ablation recommendations

## File Overview

- `streamlit_app.py` → Streamlit UI
- `plant_chat_agent.py` → conversational routing and LangChain/OpenAI agent setup
- `agent_tools.py` → tool wrappers around the model-layer functions
- `model_imports.py` → safe path handling for imports from `DeepLearning-tensorFlowLite/`
- `ablation_utils.py` → hardware inspection and ablation feasibility reporting
- `ablation_planner.py` → parses conversational ablation requests and identifies missing parameters
- `ablation_history.py` → persistent JSONL history, summaries, and recommendation support

## Running the App

From the repository root:

```bash
streamlit run agent/streamlit_app.py
```

Or from this directory:

```bash
streamlit run streamlit_app.py
```

## Agent Behavior

- Uses LangChain + OpenAI when dependencies and `OPENAI_API_KEY` are available
- Falls back to deterministic routing when LangChain/OpenAI setup is unavailable
- Never fabricates predictions or metrics
- Always reports missing artifacts clearly
- Surfaces HOG+SVM artifact-version warnings if saved `joblib` files were produced with a different `scikit-learn` version
- Persists completed no-retraining ablation runs under `DeepLearning-tensorFlowLite/result/ablation_history/`
- Only makes performance-based ablation recommendations when enough prior history exists

## Example Prompts

- `classify this image`
- `compare models`
- `what metrics are available`
- `what artifacts exist`
- `what ablation is possible`
- `run a quick balanced ablation on the test split`
- `what ablation settings do you recommend`
- `show previous ablation results`
