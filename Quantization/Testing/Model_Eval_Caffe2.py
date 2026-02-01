import cv2
import os
import numpy as np
import time
import random

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

# Self-explanatory
def accuracy_score(predictions, validation_dataset, validation_data_dir):
    cnts = 0
    total = 0
    for img_path in validation_data_dir:
        if img_path in predictions and img_path in validation_dataset:
            total += 1
            if int(predictions[img_path]) == int(validation_dataset[img_path]):
                cnts += 1
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

def get_metrics_caffe2(model_path, weights_path, model_type):
    model = cv2.dnn.readNet(model_path, weights_path)

    total_params = 0
    for layer_name in model.getLayerNames():
        layer_id = model.getLayerId(layer_name)
        layer = model.getLayer(layer_id)
        if hasattr(layer, "blobs"):
            for blob in layer.blobs:
                total_params += np.prod(blob.shape)

    print("Total parameters:", total_params)

    batchsize = 1000
    folder_dir = "UTKFace/"
    img_names = os.listdir(folder_dir)
    random.shuffle(img_names)
    img_names = img_names[:batchsize + 1]
    img_paths = []

    for img_name in img_names:
        img_paths.append(folder_dir + img_name)

    # Warming up inference.
    for i in range(20):
        img = cv2.imread(img_paths[i])
        blob = cv2.dnn.blobFromImage(img, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
        model.setInput(blob)
        pred = model.forward()
    
    # Timing the inference for 1000 images
    pred = {}
    t0 = time.time()
    for img_path in img_paths:
        img = cv2.imread(img_path)
        blob = cv2.dnn.blobFromImage(img, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
        model.setInput(blob)
        output = model.forward()
        pred[img_path] = output[0].argmax()
    t1 = time.time()
    print("Average latency: ", (t1 - t0) / batchsize)

    validation_dataset = get_validation_dataset(data_names=img_names, data_paths = img_paths, model_type = model_type)
    score = accuracy_score(pred, validation_dataset, img_paths)

    print("Model accuracy: " + str(score))

    size_mb = (os.path.getsize(model_path) + os.path.getsize(weights_path)) / (1024**2)
    print("Model size: ", size_mb)


age1 = "models/age_deploy.prototxt"
age2 = "models/age_net.caffemodel"
gen1 = "models/gender_deploy.prototxt"
gen2 = "models/gender_net.caffemodel"
face1 = "models/opencv_face_detector.pbtxt"
face2 = "models/opencv_face_detector_uint8.pb"

# get_metrics_caffe2(face2, face1)
get_metrics_caffe2(age2, age1, "age")
get_metrics_caffe2(gen2, gen1, "gender")
# img_dir = "UTKFace/"
# img_names = os.listdir(img_dir)
# img_path = str(img_dir + img_names[0])
# model = cv2.dnn.readNet(age2, age1)

# img = cv2.imread(img_path)
# blob = cv2.dnn.blobFromImage(img, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
# model.setInput(blob)
# output = model.forward()

# print(output.shape)