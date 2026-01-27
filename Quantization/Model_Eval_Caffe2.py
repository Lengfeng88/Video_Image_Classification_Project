import cv2
import os

# def get_metrics_caffe2(model_path, weights_path):
#     model = cv2.dnn.readNetFromCaffe("models/age_deploy.prototxt", "models/age_net.caffemodel")

#     total_params = 0
#     for layer_name in model.getLayerNames():
#         layer_id = model.getLayerId(layer_name)
#         layer = model.getLayer(layer_id)
#         if hasattr(layer, "blobs"):
#             for blob in layer.blobs:
#                 total_params += np.prod(blob.shape)

#     print("Total parameters:", total_params)

#     test_data_folder_path = "UTKFace/"

#     img_names = os.listdir(test_data_folder_path)
#     img_paths = []

#     for idx, img_name in enumerate(img_names):
#         if idx > 1001: 
#             break
#         img_paths.append(str(test_data_folder_path + img_name))

#     # Warming up inference.
    # for i in range(20):
    #     img = cv2.imread(img_path[i])
    #     blob = cv2.dnn.blobFromImage(img, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
    #     model.setInput(img)
    #     pred = model.forward()
    
#     # Timing the inference for 1000 images
#     t0 = time.time()
#     for img_path in img_paths:
#         img = cv2.imread(img_path)
#         blob = cv2.dnn.blobFromImage(img, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
#         model.setInput(blob)
#         pred = model.forward()
#     t1 = time.time()
#     print("Average latency: ", (t1 - t0) / 1000)

#     size_mb = (os.path.getsize(model_path) + os.path.getsize(weights_path)) / (1024**2)
#     print("Model size: ", size_mb)