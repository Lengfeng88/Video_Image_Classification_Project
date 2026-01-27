import cv2
import os
import openvino as ov
import numpy as np
import time
import random

# This basically will list all the details about the model.

# List all layer names
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

# Converted from a caffe2 model, so it will use the same preprocessing other than blobbing. Using an numpy array to store model format instead.
def ov_preprocess(img_path, input_size = (224, 224), mean_values = (104.0, 117.0, 123.0), scale=1.0):
    img =cv2.imread(img_path)
    if img is None:
        raise ValueError
    
    img = cv2.resize(img, (227, 227))
    img = img.astype(np.float32)
    img -= np.array([104.0, 117.0, 123.0], dtype=np.float32)
    img = np.expand_dims(img, axis=0)  # (1,227,227,3)
    # blob = cv2.dnn.blobFromImage(img, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
    return img

# Self-explanatory
def accuracy_score(predictions, validation_dataset, validation_data_dir):
    cnts = 0
    total = 0
    for img_path in validation_data_dir:
        if img_path in predictions and img_path in validation_dataset:
            total += 1
            # print(str(predictions[img_path])+ " " + str(validation_dataset[img_path]) + "\n")

            # for ch in validation_dataset[img_path]:
            #     if ch > '9' or ch < '0':
            #         print(img_path)
            #         break
            if predictions[img_path] == validation_dataset[img_path]:
                cnts += 1
            # else: 
            #     print(str(predictions[img_path]) + " " + str(validation_dataset[img_path]) + "\n")
    # print(str(cnts) + "/" + str(total))
    return cnts

def get_validation_dataset(data_names, data_paths, model_type):
    cnts = 0

    gender_labels = {}
    age_labels = {}

    for idx, img_name in enumerate(data_names):
        img_path = data_paths[idx]

        # The age is always the first number in the label
        age_idx = 0
        while img_name[age_idx] != '_': age_idx += 1
        age_labels[img_path] = img_name[:age_idx]

        # The gender is always after the _ followed by the age label
        gender_idx = age_idx + 1
        gender_labels[img_path] = img_name[gender_idx:gender_idx + 1]

    # print(age_labels)
    # print(gender_labels)

    if model_type == "age":
        return age_labels
    
    return gender_labels


def get_metrics_ov(model_path, weight_path, model_type):
    core = ov.Core()

    model = core.read_model(model_path)   # <-- only XML
    compiled_model = core.compile_model(model, "CPU")

    # Getting the model size
    bin_size = os.path.getsize(weight_path) / (1024 * 1024)
    xml_size = os.path.getsize(model_path) / (1024 * 1024)

    print(f"BIN size: {bin_size:.2f} MB")
    print(f"XML size: {xml_size:.2f} MB")
    print(f"Total: {bin_size + xml_size:.2f} MB")

    total_params = 0
    for tensor in model.get_parameters():
        shape = tensor.get_shape()
        num = 1
        for dim in shape:
            num *= dim
        total_params += num

    print("Total parameters:", total_params)

    batchsize = 3
    folder_dir = "UTKFace/"
    img_names = os.listdir(folder_dir)[:batchsize]
    img_paths = []

    # random.shuffle(img_names)
    for img_name in img_names:
        img_paths.append(folder_dir + img_name)

    # Warming up the inference.
    predictions = {}
    validation_data = {}
    for i in range(batchsize):
        # preprocessing the image
        processed_img = ov_preprocess(img_paths[i])

        #Setting the input layer and running a forward pass
        input_layer = compiled_model.input(0)
        result = compiled_model({
            input_layer.get_any_name(): processed_img
        })

        # Finding the most confident prediction.
        output = list(result.values())[0]
        pred = np.argmax(output, axis=1)
        
        # predictions.append(pred)

    # Testing 1000 images
    time0 = time.time()
    for i, img_path in enumerate(img_paths):
        # preprocessing the image
        processed_img = ov_preprocess(img_path)

        #Setting the input layer and running a forward pass
        input_layer = compiled_model.input(0)
        result = compiled_model({
            input_layer.get_any_name(): processed_img
        })

        # Finding the most confident prediction.   
        output = list(result.values())[0]
        pred = np.argmax(output, axis=1)
        predictions[img_path] = pred[0]
    time1 = time.time()
    
    # print(img_names)
    # print(img_paths)
    validation_data = get_validation_dataset(img_names, img_paths, model_type)
    # print(str(predictions[img_path]))
    for img_path in img_paths:
        # print(str(predictions[img_path])+ " " + str(validation_data[img_path]))
        if predictions[img_path] == validation_data[img_path]:
            print(str(predictions[img_path])+ " " + str(validation_data[img_path]))
        # print(str(validation_data[img_path]))

    # model_accuracy = accuracy_score(predictions, get_validation_dataset(img_names, img_paths, model_type), img_paths)
    # print(predictions)
    # print(get_validation_dataset(data_names=img_names, model_type=model_type))
    # print("Model average accuracy: " + str(model_accuracy))
    print("Model average latency: " + str((time1 - time0) / 1000) + " seconds")

# Model paths
age_xml = "Quantization/ONNX_To_OpenVINO/age_ov_quant.xml"
age_bin = "Quantization/ONNX_To_OpenVINO/age_ov_quant.bin"

gender_xml = "Quantization\ONNX_To_OpenVINO\gender_ov_quant.xml"
gender_bin = "Quantization\ONNX_To_OpenVINO\gender_ov_quant.bin"

age_caffe2 = cv2.dnn.readNetFromCaffe("models/age_deploy.prototxt", "models/age_net.caffemodel")

# get_metrics_ov(age_xml, age_bin, model_type="age")
get_metrics_ov(gender_xml, gender_bin, model_type = "gender")


