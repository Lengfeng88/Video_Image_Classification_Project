import openvino as ov
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import numpy as np
import torch
from openvino.preprocess import PrePostProcessor
from Calib_Dataset import GenderCalibrationDataset
from Calib_Dataset import AgeCalibrationDataset
from torch.utils.data import random_split
import cv2
import nncf
import os
import random
from PIL import Image

gender_path = "Quantization/gender_net_v11.onnx"
age_path = "Quantization/age_net_new_v11.onnx"
output_folder = "Quantization/calib_data_preprocessed_improved_data/"
# calib_data_path = "Quantization/calib_data_preprocessed/"
calib_data_path = "UTKFace/"

# Core = ov.Core()

# gender = Core.read_model(gender_path)
# age = Core.read_model(age_path)
gender = ov.convert_model(gender_path)
age = ov.convert_model(age_path)

# for inp in gender.inputs:
#     print("Input name:", inp.get_any_name())
#     print("Input shape:", inp.get_partial_shape())
#     print("Input type:", inp.get_element_type())

# age_fp32 = ov.compile_model(age)
# gender_fp32 = ov.compile_model(gender)

# img = cv2.imread("UTKFace/25_1_0_20170116003458183.jpg.chip.jpg")
# assert img is not None

# img = cv2.resize(img, (227, 227))
# img = img.astype(np.float32)
# img -= np.array([104.0, 117.0, 123.0], dtype=np.float32)
# img = np.expand_dims(img, axis=0)  # (1,227,227,3)

# input_layer = gender_fp32.input(0)

# result = gender_fp32({
#     input_layer.get_any_name(): img
# })
# output_tensor = list(result.values())[0]

# print("Output shape:", output_tensor.shape)
# print("Raw output:", output_tensor) 

# pred = np.argmax(output_tensor, axis=1)
# print("Predicted gender class:", pred[0])

img_names = os.listdir(calib_data_path)

random.shuffle(img_names)

img_names = img_names[:7676]
for img_name in (img_names):
    img_path = str(calib_data_path + img_name)
    img = Image.open(img_path)
    img.save(str(output_folder + img_name))

overall_gender_data = GenderCalibrationDataset(img_dir=calib_data_path)
overall_age_data = AgeCalibrationDataset(img_dir=output_folder)

calib_size_gender = int(0.8 * len(overall_gender_data))
val_size_gender = len(overall_gender_data) - calib_size_gender

calib_size_age = int(0.8 * len(overall_age_data))
val_size_age = len(overall_age_data) - calib_size_age

gender_calib_dataset, gender_val_dataset = random_split(overall_gender_data, [calib_size_gender, val_size_gender])
age_calib_dataset, age_val_dataset = random_split(overall_age_data, [calib_size_age, val_size_age])

gender_calibration_loader = DataLoader(gender_calib_dataset, batch_size=1, shuffle=False)
gender_validation_loader = DataLoader(gender_val_dataset, batch_size=1, shuffle=False)

age_calibration_loader = DataLoader(age_calib_dataset, batch_size=1, shuffle=False)
age_validation_loader = DataLoader(age_val_dataset, batch_size=1, shuffle=False)

def transform_fn(data_item):
    images, _ = data_item
    return images.numpy()

def validate(model: ov.CompiledModel, 
    validation_loader: torch.utils.data.DataLoader) -> float:
    predictions = []
    references = []

    output = model.outputs[0]

    for images, target in validation_loader:
        pred = model(images)[output]
        
        # input_layer = model.input(0)
        # result = model({input_layer.get_any_name(): images})
        # pred = list(result.values())[0]

        predictions.append(np.argmax(pred, axis=1))
        references.append(target)

    predictions = np.concatenate(predictions, axis=0)
    references = np.concatenate(references, axis=0)
    return accuracy_score(predictions, references)

gender_calibration_dataset = nncf.Dataset(gender_calibration_loader, transform_fn)
gender_validation_dataset = nncf.Dataset(gender_validation_loader, transform_fn)

age_calibration_dataset = nncf.Dataset(age_calibration_loader, transform_fn)
age_validation_dataset = nncf.Dataset(age_validation_loader, transform_fn)

quantized_age = nncf.quantize_with_accuracy_control(
    age,
    calibration_dataset=age_calibration_dataset,
    validation_dataset=age_validation_dataset,
    validation_fn=validate,
    max_drop=0.01,
    drop_type=nncf.DropType.ABSOLUTE,
)

quantized_gender = nncf.quantize_with_accuracy_control(
    gender,
    calibration_dataset=gender_calibration_dataset,
    validation_dataset=gender_validation_dataset,
    validation_fn=validate,
    max_drop=0.01,
    drop_type=nncf.DropType.ABSOLUTE,
)

age_int8 = ov.compile_model(quantized_age)
gender_int8 = ov.compile_model(quantized_gender)

ov.save_model(quantized_gender, "Quantization/ONNX_To_OpenVINO/gender_ov_quant.xml", compress_to_fp16=False)
ov.save_model(quantized_age, "Quantization/ONNX_To_OpenVINO/age_ov_quant.xml", compress_to_fp16=False)