import cv2
import os
from pathlib import Path
import matplotlib

image_path_list = []
folder_path = Path("Quantization/calib_data_preprocessed")
folder = os.listdir(folder_path)
for idx, image_name in enumerate(folder):
    image_path = os.path.join(folder_path, image_name)
    cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (224, 224))
        