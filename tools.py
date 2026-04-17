# tools.py

import os
from langchain.tools import tool
from ultralytics import YOLO
import threading

import torch
import random
import numpy as np

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

# Load once (global, so it isn't reloaded every call)
model = YOLO("yolov8n.pt")

# warm up (important)
_ = model("warmupImg.jpg")

yolo_lock = threading.Lock()  # To ensure thread-safe access to the model

@tool
def detect_objects(image_path: str) -> dict:
    """
    Detect objects in an image using YOLO.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Dictionary with 'counts' (object counts) and 'detections' (list of objects with confidence)
    """
    with yolo_lock:
        results = model(image_path)

    detections = []
    counts = {}

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            name = model.names[cls_id]

            detections.append({
                "object": name,
                "confidence": conf
            })

            counts[name] = counts.get(name, 0) + 1

    return {
        "image_path": image_path,
        "counts": counts,
        "detections": detections
    }

@tool
def list_images(folder_path: str = "images") -> list:
    """
    List all image files in a folder.
    
    Args:
        folder_path: Path to the folder containing images (default: "images")
        
    Returns:
        List of image file paths
    """
    valid_exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

    try:
        files = os.listdir(folder_path)
    except FileNotFoundError:
        return []

    image_paths = [
        os.path.join(folder_path, f)
        for f in files
        if f.lower().endswith(valid_exts)
    ]

    return image_paths

@tool
def analyze_all_images(folder_path: str = "images") -> str:
    """
    Analyze all images in a folder and return a summary.
    
    Args:
        folder_path: Path to the folder containing images (default: "images")
        
    Returns:
        Human-readable summary of all detected objects
    """
    # Get list of images
    valid_exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    
    try:
        files = os.listdir(folder_path)
    except FileNotFoundError:
        return f"Folder '{folder_path}' not found."
    
    image_paths = [
        os.path.join(folder_path, f)
        for f in files
        if f.lower().endswith(valid_exts)
    ]
    
    if not image_paths:
        return f"No images found in '{folder_path}'."
    
    # Process each image
    results = []
    for path in image_paths:
        with yolo_lock:
            detections = model(path)
        
        counts = {}
        for r in detections:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                name = model.names[cls_id]
                counts[name] = counts.get(name, 0) + 1
        
        # Format the result
        if counts:
            items = ", ".join([f"{count} {obj}{'s' if count > 1 else ''}" for obj, count in counts.items()])
            results.append(f"{os.path.basename(path)}: {items}")
        else:
            results.append(f"{os.path.basename(path)}: No objects detected")
    
    return "\n".join(results)

TOOLS = [
    detect_objects,
    list_images,
    analyze_all_images,
]

if __name__ == "__main__":
    # Test detect_objects
    result = detect_objects.invoke({"image_path": "test.jpg"})
    print(result)
