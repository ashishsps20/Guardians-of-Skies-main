import os
import pandas as pd
import shutil
from sklearn.model_selection import train_test_split

DATASET_DIR = r"C:\arpan\project2\dataset" 
SPLIT_RATIOS = (0.7, 0.2, 0.1) 

CLASS_NAMES = [
    "A10", "A400M", "AG600", "AH64", "AV8B", "An124", "An22", "An225", "An72", "B1", "B2", "B21", "B52", "Be200",
    "C130", "C17", "C2", "C390", "C5", "CH47", "CL415", "E2", "E7", "EF2000", "F117", "F14", "F15", "F16", "F18",
    "F22", "F35", "F4", "H6", "J10", "J20", "J35", "JAS39", "JF17", "JH7", "KAAN", "KC135", "KF21", "KJ600", "Ka27",
    "Ka52", "MQ9", "Mi24", "Mi26", "Mi28", "Mi8", "Mig29", "Mig31", "Mirage2000", "P3", "RQ4", "Rafale", "SR71",
    "Su24", "Su25", "Su34", "Su57", "TB001", "TB2", "Tornado", "Tu160", "Tu22M", "Tu95", "U2", "UH60", "US2", "V22",
    "Vulcan", "WZ7", "XB70", "Y20", "YF23", "Z19"
]
CLASS_MAP = {name: idx for idx, name in enumerate(CLASS_NAMES)}

for split in ['train', 'val', 'test']:
    os.makedirs(os.path.join(DATASET_DIR, split, 'images'), exist_ok=True)
    os.makedirs(os.path.join(DATASET_DIR, split, 'labels'), exist_ok=True)

images = sorted([f for f in os.listdir(DATASET_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
csv_files = sorted([f for f in os.listdir(DATASET_DIR) if f.lower().endswith('.csv')])

data = [(img, os.path.splitext(img)[0] + ".csv") for img in images if os.path.splitext(img)[0] + ".csv" in csv_files]

# Split dataset
train, temp = train_test_split(data, test_size=(1 - SPLIT_RATIOS[0]), random_state=42)
val, test = train_test_split(temp, test_size=SPLIT_RATIOS[2] / (SPLIT_RATIOS[1] + SPLIT_RATIOS[2]), random_state=42)

