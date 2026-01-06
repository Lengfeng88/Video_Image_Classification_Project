from onnxruntime.quantization import CalibrationDataReader
import numpy as np
from pathlib import Path
import onnxruntime as ort
import cv2

class FaceCalibrationDataReader(CalibrationDataReader):
    def __init__(self, folder_path, input_name, batch_size=500):
        self.input_name = input_name
        self.img_paths = list(Path(folder_path).glob("*.jpg"))
        self.batch_size = 500
        self.idx = 0


    def _load_and_preprocess(self, path):

        

    def __iter__(self):
        self.index = 0
        return self 

    def __next__(self):
        self.index += 1
        return self


