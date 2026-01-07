import onnxruntime as ort
import numpy as np
from CalibrationDataReader import FaceCalibrationDataReader
import pathlib
from pathlib import Path
import cv2

face = cv2.dnn.readNetFromCaffe("models/opencv_face_detector.pbtxt", "models/opencv_face_detector_uint8.pb")

model = ort.InferenceSession("models/Calibration/age_net_new.onnx")

# input_name = model.get_inputs()[0].name

# from onnxruntime.quantization import (
#     quantize_static,
#     QuantType,
#     QuantFormat,
#     CalibrationMethod
# )

# #Preprocessing the images
# folder = Path("/calib_data_raw")
# image_files = list(folder.glob("*.jpg"))

# output_folder = Path("/calib_data_preprocessed")
# output_folder.mkdir(exist_ok=True)

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
#             cropped = img[y1:y2, x1:x2]
#             output_path = output_folder / f"face_{idx}_{i}.jpg"
#             cv2.imwrite(str(output_path), cropped)

# calibration_reader = FaceCalibrationDataReader(
#     folder="calib_data_preprocessed/",
#     input_name=input_name,
#     batch_size=1
# )

# quantize_static(
#     model_input="age_net.onnx",
#     model_output="age_net_quant.onnx",
#     calibration_data_reader=calibration_reader,
#     quant_format=QuantFormat.QDQ,       # Recommended for accuracy
#     activation_type=QuantType.QUInt8,
#     weight_type=QuantType.QUInt8,
#     calibrate_method=CalibrationMethod.MinMax  # or Entropy for better accuracy
# )
# print("Quantized model saved as age_net_quant.onnx")

