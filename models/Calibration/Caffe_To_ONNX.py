import cv2
import numpy as np
from onnxruntime.quantization import CalibrationDataReader
from CalibrationDataReader import AgeCalibrationDataReader

calibration_images = []