import onnxruntime as ort
import numpy as np
from CalibrationDataReader import FaceCalibrationDataReader
import pathlib
from pathlib import Path
import os
import glob
import cv2


BASE_DIR = Path(__file__).resolve().parent.parent  # Project root
MODEL_DIR = BASE_DIR / "models"

face = cv2.dnn.readNet(
    str(MODEL_DIR / "opencv_face_detector.pbtxt"),
    str(MODEL_DIR / "opencv_face_detector_uint8.pb")
)

# face = cv2.dnn.readNetFromCaffe("Video_Image_Classification_Project2/models/opencv_face_detector.pbtxt", "models/opencv_face_detector_uint8.pb")

model = ort.InferenceSession("Quantization/age_net_new_v11.onnx")

input_name = model.get_inputs()[0].name

from onnxruntime.quantization import (
    quantize_static,
    QuantType,
    QuantFormat,
    CalibrationMethod
)

#Preprocessing the images
# folder = Path("Quantization/calib_data_raw")
# folders = os.listdir(folder)

# output_folder = Path("Quantization/calib_data_preprocessed")
# output_folder.mkdir(exist_ok=True)
# image_files = []

# for idx, folder_name in enumerate(folders):
#     full_path = os.path.join(folder, folder_name)
#     image_files.extend(list(glob.glob(full_path + "/*.jpg")))

# for idx, img_path in enumerate(image_files):
#     img = cv2.imread(str(img_path))
    
#     if img is None:
#         continue

#     blob = cv2.dnn.blobFromImage(img, 1.0, (300, 300), [104, 117, 123], True, False)
#     face.setInput(blob)
#     detections = face.forward()

#     img_h = img.shape[0]
#     img_w = img.shape[1]
#     for i in range(detections.shape[2]):
#         confidence = detections[0, 0, i, 2]

#         if confidence > 0.9:
#             x1 = int(detections[0, 0, i, 3] * img_w)
#             y1 = int(detections[0, 0, i, 4] * img_h)
#             x2 = int(detections[0, 0, i, 5] * img_w)
#             y2 = int(detections[0, 0, i, 6] * img_h)
#             bbox = [x1, y1, x2, y2]
#             # suppose bbox = [x1, y1, x2, y2]
#             x1, y1, x2, y2 = map(int, bbox)

#             # clamp coordinates inside image
#             x1 = max(0, x1)
#             y1 = max(0, y1)
#             x2 = min(img.shape[1], x2)
#             y2 = min(img.shape[0], y2)

#             if x2 > x1 and y2 > y1:
#                 cropped = img[y1:y2, x1:x2]
#                 output_path = output_folder / f"face_{idx}_{i}.jpg"
#                 cv2.imwrite(str(output_path), cropped)
#             else:
#                 print(f"Skipping invalid crop: {x1},{y1},{x2},{y2}")


            # cropped = img[y1:y2, x1:x2]
            # if cropped is None:
            #     continue
            # # print(str(x1) + " " + str(y1) + " " + str(x2) + " " + str(y2))
            # output_path = output_folder / f"face_{idx}_{i}.jpg"
            # cv2.imwrite(output_path, cropped)

calibration_reader = FaceCalibrationDataReader(
    folder="Quantization/calib_data_preprocessed/",
    input_name=input_name,
    batch_size=1
)

quantize_static(
    model_input="Quantization/age_net_new_v11.onnx",
    model_output="Quantization/age_net_quant.onnx",
    calibration_data_reader=calibration_reader,
    quant_format=QuantFormat.QDQ,       # Recommended for accuracy
    activation_type=QuantType.QUInt8,
    weight_type=QuantType.QUInt8,
    calibrate_method=CalibrationMethod.MinMax  # or Entropy for better accuracy
)
print("Quantized model saved as age_net_quant.onnx")

calibration_reader.idx = 0

quantize_static(
    model_input= "Quantization/gender_net_v11.onnx",
    model_output="Quantization/gender_net_quant.onnx",
    calibration_data_reader=calibration_reader,
    quant_format=QuantFormat.QDQ,
    activation_type=QuantType.QUInt8,
    weight_type=QuantType.QUInt8,
    calibrate_method=CalibrationMethod  .MinMax
)
print("Quantized model saved as gender_net_quant.onnx")

