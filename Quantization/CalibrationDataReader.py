from onnxruntime.quantization import CalibrationDataReader
import numpy as np
import os
from pathlib import Path
import onnxruntime as ort
import cv2

class FaceCalibrationDataReader(CalibrationDataReader):
    def __init__(self, folder_path, input_name):
        self.input_name = input_name
        self.img_paths = list(Path(folder_path).glob("*.jpg"))
        self.batch_paths = os.listdir(folder_path)
        self.idx = 0
    
    def _load_and_preprocess(self, path, size=(224, 224)):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = cv2.resize(img, size)
        img = img.astype(np.float32) / 255.0
        return np.transpose(img, (2, 0, 1))
    
    #Gonna have to change this soon
    def get_next(self):
        if self.index >= len(self.image_paths):
            return None
        batch = []
        for _ in range(self.batch_size):
            if self.index >= len(self.image_paths):
                break
            batch.append(self._preprocess(self.image_paths[self.index]))
            self.index += 1
        return {self.input_name: np.stack(batch)}
    




