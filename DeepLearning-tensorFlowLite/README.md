# DeepLearning–TensorFlow Lite (Plant classifier)

## 1. What deep learning is used for here

This folder trains a **plant species image classifier**: photos go in, and the model outputs scores over many classes (PlantNet-style species IDs or scientific names in the label file). The network is a **MobileNetV2** backbone with a small dense “head” on top. After training, the model is **exported to TensorFlow Lite (`.tflite`)** so it can run on-device on Android with the same preprocessing (RGB, float32 or uint8, resized to the training height/width).

The model is a **convolutional neural network (CNN)**: deep learning for images here means stacked convolutional layers that learn spatial patterns (edges, textures, parts of leaves) before the final layers map those features to species classes. MobileNetV2 is a CNN architecture designed to stay efficient on mobile and edge devices.

**Image size (preprocessing and modeling):** The default input is **224×224** pixels. In `train_export_tflite.py`, the **`--img_size`** flag (default `224`) controls both **preprocessing**—`image_dataset_from_directory` resizes images to that height and width—and **modeling**—`MobileNetV2` and the Keras `Input` use the same `(img_size, img_size, 3)` shape, so the data pipeline and network stay matched. In `infer_plant_tflite.py`, images are resized to the **height and width baked into the exported `.tflite` model** (read from the model’s input tensor), which should match what you used when training; the script’s `--img_size` should agree with that value if you train with a non-default size.

## 2. Libraries used

| Library | Role |
|--------|------|
| **TensorFlow / Keras** | Training (`image_dataset_from_directory`, MobileNetV2), TFLite export, and inference via the TFLite interpreter |
| **NumPy** | Arrays, softmax for readable probabilities in the inference script |

Python **3.10+** is expected. Version pins live in `requirements-tflite.txt`.

## 3. What each file does

| File | Purpose |
|------|---------|
| **`train_export_tflite.py`** | Builds train/validation datasets from a folder-per-class tree, trains MobileNetV2 + classifier, writes **`plant_classifier.tflite`** and **`plant_labels_export.txt`** (sorted folder names = class indices). |
| **`map_plantnet_ids_to_names.py`** | Maps numeric PlantNet species IDs in a label file to scientific names using **`plantnet300K_species_id_2_name.json`**; keeps **line order** so indices still match the trained model. |
| **`infer_plant_tflite.py`** | Loads a `.tflite` model and a label file, preprocesses one image like the app, runs inference, prints **top-k** species with percentages. |
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

5. **Download** `plant_classifier.tflite` and your label file back to your laptop for the app or for local `infer_plant_tflite.py` tests.

**Note:** Final **TFLite conversion** is usually quick; **training** is what benefits most from the GPU.

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

