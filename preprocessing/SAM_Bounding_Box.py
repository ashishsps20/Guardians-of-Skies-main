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

