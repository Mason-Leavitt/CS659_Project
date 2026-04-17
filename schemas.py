# schemas.py

from pydantic import BaseModel
from typing import List, Dict

class Detection(BaseModel):
    """Single object detection result"""
    object: str
    confidence: float

class ImageResult(BaseModel):
    """Results for a single image"""
    image_path: str
    counts: Dict[str, int]
    detections: List[Detection]

class ResponseModel(BaseModel):
    """Complete response with all image results"""
    results: List[ImageResult]
