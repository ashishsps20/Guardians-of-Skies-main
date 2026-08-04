import os
import cv2
import torch
import numpy as np
from segment_anything import SamPredictor, sam_model_registry
from ultralytics import YOLO
from glob import glob

DATASET_DIR = "/kaggle/working/dataset"
SAM_CHECKPOINT = "/kaggle/working/sam_vit_h.pth" 
MODEL_TYPE = "vit_h" 

device = "cuda" if torch.cuda.is_available() else "cpu"
sam = sam_model_registry[MODEL_TYPE](checkpoint=SAM_CHECKPOINT).to(device)
predictor = SamPredictor(sam)

def get_yolo_bbox(mask):
    """Convert SAM mask to YOLO format bounding box"""
    y_indices, x_indices = np.where(mask > 0) 
    if len(y_indices) == 0 or len(x_indices) == 0:
        return None  
    
    x_min, x_max = x_indices.min(), x_indices.max()
    y_min, y_max = y_indices.min(), y_indices.max()

    cx = (x_min + x_max) / 2
    cy = (y_min + y_max) / 2
    w = x_max - x_min
    h = y_max - y_min
    
    return cx, cy, w, h

