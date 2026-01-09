from onnxruntime.quantization import CalibrationDataReader
import numpy as np
import os
from pathlib import Path
import onnxruntime as ort
import cv2

class FaceCalibrationDataReader(CalibrationDataReader):
    def __init__(self, folder, input_name, batch_size):
        self.input_name = input_name
        # self.img_paths = list(Path(folder_path).glob("*.jpg"))
        self.folder_path = folder
        self.folder = os.listdir(folder)
        self.batch_size = batch_size
        self.idx = 0
    
    # def _load_and_preprocess(self, path, size=(224, 224)):
    #     img = cv2.imread(path)
    #     if img is None:
    #         print(path)
    #         raise FileNotFoundError(path)
    #     # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    #     img = cv2.resize(img, size)
    #     # img = img.astype(np.float32) / 255.0
    #     mean = np.array(
    #         [78.4263377603, 87.7689143744, 114.895847746],
    #         dtype=np.float32
    #     )
    #     img -= mean  # BGR mean subtraction
    #     return np.transpose(img, (2, 0, 1))
    def _load_and_preprocess(self, path):

        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"Failed to load image: {path}")

        img = cv2.resize(img, (227, 227))

        # IMPORTANT: convert to float FIRST
        img = img.astype(np.float32)

        # Caffe-style mean subtraction (BGR)
        img -= np.array(
            [78.4263377603, 87.7689143744, 114.895847746],
            dtype=np.float32
        )

        # HWC → CHW
        # img = np.transpose(img, (2, 0, 1))

        return img
    
    #Gonna have to change this soon
    def get_next(self):
        batch = os.path.join(self.folder_path, self.folder[self.idx])
        self.idx += 1
        return {self.input_name: np.stack(batch)}
    




