# DeepLearning–TensorFlow Lite (Plant classifier)

## 1. What deep learning is used for here

This folder trains a **plant species image classifier**: photos go in, and the model outputs scores over many classes (PlantNet-style species IDs or scientific names in the label file). The network is a **MobileNetV2** backbone with a small dense “head” on top. After training, the model is **exported to TensorFlow Lite (`.tflite`)** so it can run on-device on Android with the same preprocessing (RGB, float32 or uint8, resized to the training height/width).

The model is a **convolutional neural network (CNN)**: deep learning for images here means stacked convolutional layers that learn spatial patterns (edges, textures, parts of leaves) before the final layers map those features to species classes. MobileNetV2 is a CNN architecture designed to stay efficient on mobile and edge devices.

**Image size (preprocessing and modeling):** The default input is **224×224** pixels. In `train_export_tflite.py`, the **`--img_size`** flag (default `224`) controls both **preprocessing**—`image_dataset_from_directory` resizes images to that height and width—and **modeling**—`MobileNetV2` and the Keras `Input` use the same `(img_size, img_size, 3)` shape, so the data pipeline and network stay matched. In `infer_plant_tflite.py`, images are resized to the **height and width baked into the exported `.tflite` model** (read from the model’s input tensor), which should match what you used when training; the script’s `--img_size` should agree with that value if you train with a non-default size.

**Color correction (optional):** After rescaling pixels to **[0, 1]**, you can apply **`--color_correct`** in both `train_export_tflite.py` and `infer_plant_tflite.py`. Implemented options are **gray-world** and **max-RGB** classical color constancy (see `color_correction.py` for formulas). **Gray-world** assumes the average scene color is neutral and scales R/G/B channel means—simple and stable for mild casts. **Max-RGB** uses per-channel maxima (good when neutral highlights exist; weaker if a channel saturates). We avoid aggressive **histogram equalization** as a default because it often distorts color ratios compared with mild white-balance-style fixes, and we skip **learned** color networks to limit complexity and keep train/deploy parity. **Use the same `--color_correct` value at training and inference** (and in any Android preprocessing) so the model does not see a distribution shift.

## 2. Libraries used

| Library | Role |
|--------|------|
| **TensorFlow / Keras** | Training (`image_dataset_from_directory`, MobileNetV2), TFLite export, and inference via the TFLite interpreter |
| **NumPy** | Arrays, softmax for readable probabilities in the inference script |
| **scikit-learn** | Validation metrics (CNN), HOG+SVM training (`train_hog_svm.py`), CSV/plots from `metrics_logging.py` |
| **matplotlib** | ROC, confusion matrix, and correlation figures (non-interactive `Agg` backend) |
| **scikit-image** | HOG feature extraction in `train_hog_svm.py` |
| **joblib** | Save/load the HOG+SVM `Pipeline` |

Python **3.10+** is expected. Version pins live in `requirements-tflite.txt`.

### Classical vs deep learning comparison

Use **`train_hog_svm.py`** on the **same `--data_dir`** as **`train_export_tflite.py`**. Defaults match where it matters for a fair headline comparison: **`--img_size 224`**, **`--validation_split 0.15`**, **`--seed 42`**. Train/val **image assignments** are not guaranteed to match Keras’ internal split exactly, but stratified fractions and seed align the experimental setup. The HOG pipeline uses **grayscale** edges/gradients only (no color correction options), which is typical for this baseline.

Example:

```bash
python train_hog_svm.py --data_dir /path/to/your/data
```

Then compare **`result/hog_svm_*/summary.json`** with the CNN **`metrics_summary.json`** under **`result/<cnn_run>/…`**.

## 3. What each file does

| File | Purpose |
|------|---------|
| **`train_export_tflite.py`** | Builds train/validation datasets from a folder-per-class tree, trains MobileNetV2 + classifier, writes **`plant_classifier.tflite`** and **`plant_labels_export.txt`** (sorted folder names = class indices). Optional **`--color_correct`**. Optional **stratified k-fold** via **`--k_folds`**. Writes **metric logs** under **`result/…`** unless **`--no_metric_logs`**. |
| **`metrics_logging.py`** | Per-epoch **CSV** (accuracy, F1 macro/weighted, precision/recall macro, ROC AUC OvR + Keras loss/acc) and, after training, **`classification_report.txt`**, **`metrics_summary.json`**, **`confusion_matrix_normalized.png`**, **`confusion_row_correlation.png`** (correlation of normalized confusion rows), **`roc_curves.png`**, and **`confusion_matrix_raw.npz`**. |
| **`color_correction.py`** | Per-image **gray-world** or **max-RGB** correction on RGB in **[0, 1]** (batched NHWC). Shared by training and inference; module docstring explains method choices. |
| **`map_plantnet_ids_to_names.py`** | Maps numeric PlantNet species IDs in a label file to scientific names using **`plantnet300K_species_id_2_name.json`**; keeps **line order** so indices still match the trained model. |
| **`infer_plant_tflite.py`** | Loads a `.tflite` model and a label file, preprocesses one image like the app, runs inference, prints **top-k** species with percentages. Optional **`--color_correct`** (must match training). |
| **`train_hog_svm.py`** | **Classical baseline:** same folder-per-class data as the CNN, **grayscale resize → HOG → StandardScaler → SVM** (`linear` or `rbf`). Writes **`result/hog_svm_<UTC>/`** (`summary.json`, confusion plot, `hog_svm_model.joblib`, label file). Compare **`validation_accuracy` / F1** to CNN metrics. |
| **`requirements-tflite.txt`** | Pip dependencies for training/export/inference. |
| **`plant_labels_export.txt`** / **`plant_labels_scientific.txt`** | Example label lists (one label per line, order = class index). Replace or regenerate to match your training data. |

## 4. How to install

From this directory:

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements-tflite.txt
```

Optional: install **NumPy** explicitly if anything fails to resolve (it is usually pulled in with TensorFlow):

```bash
pip install numpy
```

## 5. Training and exporting TFLite on an NVIDIA GPU (NVIDIA Brev)

You do **not** need to install NVIDIA drivers or CUDA on your own computer for this project. Use **NVIDIA Brev** to open a cloud machine that already has a GPU and a suitable Linux stack.

1. **Create a GPU environment** in the NVIDIA Brev console (pick a GPU-backed template / public environment as offered on the page):  
   [https://brev.nvidia.com/environment/new/public](https://brev.nvidia.com/environment/new/public)

2. **On that machine**, open the provided terminal (or SSH), upload or clone this project, and follow **[How to install](#4-how-to-install)** (section 4 above) in the same folder. For GPU-accelerated training on typical **Linux** cloud images, install TensorFlow with CUDA support after the venv is active, for example:

   ```bash
   pip install "tensorflow[and-cuda]>=2.14.0,<2.19.0"
   ```

   If the image already ships a compatible TensorFlow GPU build, you can keep `requirements-tflite.txt` only—check the next step.

3. **Verify** the GPU is visible before a long run:

   ```python
   import tensorflow as tf
   print(tf.__version__)
   print(tf.config.list_physical_devices("GPU"))
   ```

   You should see at least one `GPU` device. `train_export_tflite.py` also logs devices and sets **GPU memory growth** so VRAM is allocated as needed.

4. **Put your dataset** on the instance (upload, `git`, or cloud storage), then train (example):

   ```bash
   python train_export_tflite.py --data_dir /path/to/your/data --out_tflite plant_classifier.tflite --out_labels plant_labels_export.txt
   ```

   **K-fold cross-validation (`--k_folds` > 1):** The script runs **stratified** k-fold (each class split across folds), trains a fresh model per fold, and logs **mean ± std** of validation accuracy plus per-fold scores. By default it then runs a **final** training pass on the usual **`--validation_split`** of the **full** dataset and exports that model (better use of all data for deployment). Use **`--no_final_retrain`** to export weights from the **best fold** instead (faster, no extra fit). Each class should have at least **`k_folds`** images so every fold’s training set still contains all classes.

5. **Download** `plant_classifier.tflite` and your label file back to your laptop for the app or for local `infer_plant_tflite.py` tests.

**Note:** Final **TFLite conversion** is usually quick; **training** is what benefits most from the GPU. **K-fold multiplies training time** roughly by **k folds + 1** (final), unless you pass **`--no_final_retrain`**.

**Metric logs (default on):** Each training run creates a timestamped folder under **`result/`** (override with **`--log_dir`**) containing subfolders **`single_split`**, **`fold_*`**, or **`final_retrain`** as applicable. CSV files, JSON summary, classification report, and plots (**confusion matrix**, **correlation**, **ROC**) are written there. Disable file metrics with **`--no_metric_logs`**. The **correlation** image summarizes **which true classes have similar error distributions** (row-wise correlation of the row-normalized confusion matrix), not pixel RGB correlation. **ROC** uses **one-vs-rest** multiclass AUC (macro average in CSV); plots show a **micro-averaged** ROC curve plus a few **per-class** curves for classes with enough support.

**Optional — local NVIDIA GPU:** If you train on your own hardware instead, install matching drivers and follow TensorFlow’s pip guide: [https://www.tensorflow.org/install/pip](https://www.tensorflow.org/install/pip).

## 6. Converting PlantNet species IDs to scientific names

Training with folders named after **numeric PlantNet species IDs** produces `plant_labels_export.txt` with one ID per line. For readable names in the app or in `infer_plant_tflite.py`, map those IDs to **scientific names** with `map_plantnet_ids_to_names.py`.

1. **Get the ID-to-name JSON** from the **Pl@ntNet-300K** dataset on Zenodo (same vocabulary as the paper/dataset):  
   [https://zenodo.org/records/5645731](https://zenodo.org/records/5645731)  
   Download the dataset archive (`plantnet_300K.zip` is large) and extract it, or obtain the file **`plantnet300K_species_id_2_name.json`** from that distribution. The official utilities repo also documents the layout: [https://github.com/plantnet/PlantNet-300K/](https://github.com/plantnet/PlantNet-300K/).  
   If you publish work using the data, cite the dataset as requested on the Zenodo page (DOI [10.5281/zenodo.5645731](https://doi.org/10.5281/zenodo.5645731)).

2. **Run the mapper** (paths adjusted to where your JSON and label file live):

   ```bash
   python map_plantnet_ids_to_names.py \
     --labels plant_labels_export.txt \
     --species_json /path/to/plantnet300K_species_id_2_name.json \
     --out plant_labels_scientific.txt
   ```

   The script **does not reorder lines**: line *i* stays class index *i*. Only the text on each line changes from ID to scientific name. IDs missing from the JSON are left as-is and a warning is printed.

3. Use **`plant_labels_scientific.txt`** as `--labels` for `infer_plant_tflite.py` or copy lines into the Android app’s `plant_labels.txt` **in the same order** as `plant_labels_export.txt`.

## 7. Example: test the model in Python

```bash
python infer_plant_tflite.py --model plant_classifier.tflite --labels plant_labels_scientific.txt --image ./Lactuca-virosa.jpg --top_k 10
```

Example output:

```text
2026-04-16 18:07:35.088384: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: AVX2 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
INFO: Created TensorFlow Lite XNNPACK delegate for CPU.
Image: Lactuca-virosa.jpg
Model: plant_classifier.tflite  |  labels: plant_labels_scientific.txt  |  top-10

   1.  58.53%  Lactuca serriola L.
   2.  36.34%  Cirsium oleraceum (L.) Scop.
   3.   1.68%  Lactuca virosa L.
   4.   1.35%  Helminthotheca echioides (L.) Holub
   5.   0.53%  Erechtites hieraciifolius (L.) Raf. ex DC.
   6.   0.33%  Lapsana communis L.
   7.   0.24%  Lactuca virosa Habl.
   8.   0.20%  Cirsium arvense (L.) Scop.
   9.   0.15%  Epipactis helleborine (L.) Crantz
  10.   0.08%  Papaver somniferum L.
```

The informational lines about CPU features and XNNPACK are normal when running TFLite on CPU; top predictions depend on your weights and labels.

## 8. End-to-end: classical vs deep learning (with optional color correction)

Use the same folder-per-class dataset for both lines of work. Replace **`/path/to/your/data`** with your **`--data_dir`**.

### 8.1 One-time environment

```bash
cd DeepLearning-tensorFlowLite   # or your clone path
python3 -m venv .venv
source .venv/bin/activate         # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements-tflite.txt
```

*(GPU training: follow [§5](#5-training-and-exporting-tflite-on-an-nvidia-gpu-nvidia-brev) on NVIDIA Brev and install `tensorflow[and-cuda]` on Linux if needed.)*

### 8.2 Classical baseline (HOG + SVM)

No color-correction flags — HOG runs on **grayscale** after resize (standard for this baseline).

```bash
python train_hog_svm.py --data_dir /path/to/your/data
```

Artifacts: **`result/hog_svm_<timestamp>_UTC/`** (`summary.json`, confusion plot, `hog_svm_model.joblib`, `hog_svm_labels.txt`).

### 8.3 Deep learning (CNN → TFLite)

**Without** extra color constancy (default):

```bash
python train_export_tflite.py \
  --data_dir /path/to/your/data \
  --color_correct none \
  --out_tflite plant_classifier.tflite \
  --out_labels plant_labels_export.txt
```

**With** color correction (pick **one**; use the **same** value when testing images):

```bash
python train_export_tflite.py \
  --data_dir /path/to/your/data \
  --color_correct gray_world \
  --out_tflite plant_classifier.tflite \
  --out_labels plant_labels_export.txt
```

```bash
python train_export_tflite.py \
  --data_dir /path/to/your/data \
  --color_correct max_rgb \
  --out_tflite plant_classifier.tflite \
  --out_labels plant_labels_export.txt
```

Artifacts: **`plant_classifier.tflite`**, **`plant_labels_export.txt`**, and under **`result/<timestamp>_UTC/`** (unless `--no_metric_logs`): metrics CSV, plots, `metrics_summary.json` in **`single_split/`** (or **`fold_*`** / **`final_retrain/`** if you used **`--k_folds`**).

### 8.4 Test the TFLite model (match `--color_correct` to training)

```bash
python infer_plant_tflite.py \
  --model plant_classifier.tflite \
  --labels plant_labels_scientific.txt \
  --image ./your_photo.jpg \
  --top_k 10 \
  --color_correct gray_world
```

Use **`--color_correct none`** if you trained with **`none`**; use **`gray_world`** or **`max_rgb`** if you trained with that setting.

### 8.5 Optional: PlantNet ID → scientific names

Only if your **folder names / labels** are numeric species IDs and you have **`plantnet300K_species_id_2_name.json`** — see [§6](#6-converting-plantnet-species-ids-to-scientific-names).

### 8.6 Compare classical vs deep learning

| Step | Classical | Deep learning |
|------|-----------|----------------|
| Train | `train_hog_svm.py` | `train_export_tflite.py` (+ optional `--color_correct`, `--k_folds`) |
| Main metrics | `result/hog_svm_*/summary.json` | `result/<run>/single_split/metrics_summary.json` (or `final_retrain/`) |
| Deployed model | `hog_svm_model.joblib` (Python/sklearn) | `plant_classifier.tflite` (mobile) |

**Fair comparison:** use the same **`--data_dir`**, similar **`--validation_split`** / **`--seed`**, and report accuracy / F1 from each **`summary.json`**. CNN adds color constancy as an optional experiment; HOG stays grayscale-only by design.

