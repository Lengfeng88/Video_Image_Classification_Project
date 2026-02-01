import onnx
import onnxruntime as ort
import random
import cv2
import numpy as np
import os
import time

# OV keeps giving me warning messages for weights appearing in graph inputs and not being treated as weights. 
# ort.set_default_logger_severity(3)

def preprocess(img_path, input_size = (224, 224), mean_values = (104.0, 117.0, 123.0), scale=1.0):
    img =cv2.imread(img_path)
    if img is None:
        raise ValueError
    
    img = cv2.resize(img, (227, 227))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = img.astype(np.float32)
    img -= np.array([104.0, 117.0, 123.0], dtype=np.float32)
    img = np.expand_dims(img, axis=0)  # (1,227,227,3)
    # print(img.shape)
    # blob = cv2.dnn.blobFromImage(img, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
    return img

def accuracy_score(predictions, validation_dataset, validation_data_dir):
    cnts = 0
    total = 0
    for img_path in validation_data_dir:
        # if img_path in predictions and img_path in validation_dataset:
        #     total += 1
        #     if int(predictions[img_path]) == int(validation_dataset[img_path]):
        #         cnts += 1
        try:
            total += 1
            if int(predictions[img_path]) == int(validation_dataset[img_path]):
                cnts += 1
        except:
            raise ValueError
    return cnts / total

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

    la = ['(0-2)', '(4-6)', '(8-12)', '(15-20)',
                '(25-32)', '(38-43)', '(48-53)', '(60-100)']
    
    age_range = [(0, 2), (4, 6), (8, 12), (15, 20), (25, 32), (38, 43), (60, 100)]

    tmp_age_labels = []
    for idx, label_path in enumerate(age_labels):
        age_labels[label_path] = int(age_labels[label_path])
        tmp_age_labels.append((label_path, age_labels[label_path]))

    tmp_age_labels = sorted(tmp_age_labels, key=lambda x: x[1])

    # print(tmp_age_labels)

    final_age_labels = {}
    j = 0
    for label_path, label in tmp_age_labels:
        while j < len(age_range) - 1 and label > age_range[j][1]:
            j += 1
        
        if label < age_range[j][0]: continue
        
        final_age_labels[label_path] = j

    # print(final_age_labels)

    if model_type == "age":
        return final_age_labels
    
    return gender_labels

def get_metrics(model_path, model_type):
    sess = ort.InferenceSession(
    model_path, providers=ort.get_available_providers())

    batchsize = 1
    folder_dir = "UTKFace/"
    img_names = os.listdir(folder_dir)
    # random.shuffle(img_names)
    img_names = img_names[:batchsize + 1]
    img_paths = []

    for img_name in img_names:
        img_paths.append(str(folder_dir + img_name))
    
    for input_meta in sess.get_inputs():
        print(f"Input Name: {input_meta.name}")
        print(f"Input Shape: {input_meta.shape}") # Shape is a list/tuple
        print(f"Data Type: {input_meta.type}")

    for output_meta in sess.get_outputs():
        print(f"output Name: {output_meta.name}")
        print(f"output Shape: {output_meta.shape}") # Shape is a list/tuple
        print(f"Data Type: {output_meta.type}")

    time0 = time.time()
    predictions = {}
    for img_name in img_names:
        img_path = folder_dir + img_name
        X_test = preprocess(img_path)
        sess = ort.InferenceSession(model_path, providers=ort.get_availa  ble_providers())
        input_name = sess.get_inputs()[0].name
        pred_onx = sess.run(None, {input_name: X_test})[0]
        predictions[img_path] = np.argmax(pred_onx[0])
        # print(predictions[img_path])
    time1 = time.time()
    
    # print(img_paths[0])
    validation = get_validation_dataset(img_names, img_paths, model_type)

    score = accuracy_score(predictions, validation, img_paths)

    print("Model Accuracy: " + str(score))
    print("Model average latency: " + str((time1 - time0) / batchsize))

age_v11 = "Quantization/age_net_new_v11.onnx"
gender_v11 = "Quantization/gender_net_v11.onnx"
age_net = "Quantization/age_net_new.onnx"
gender_net = "Quantization/gender_net.onnx"
age_quant = "Quantization/age_net_quant.onnx"
gender_quant = "Quantization/gender_net_quant.onnx"

get_metrics(age_v11, "age")