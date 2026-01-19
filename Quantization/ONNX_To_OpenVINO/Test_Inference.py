import cv2
import numpy as np
import openvino as ov

core = ov.Core()

age_fp32 = core.compile_model(age, "CPU")
gender_fp32 = core.compile_model(gender, "CPU")

img = cv2.imread("UTKFace/25_1_0_20170116003458183.jpg.chip.jpg")
assert img is not None

img = cv2.resize(img, (227, 227))
img = img.astype(np.float32)
img -= np.array([104.0, 117.0, 123.0], dtype=np.float32)
img = np.expand_dims(img, axis=0)  # (1,227,227,3)

input_layer = gender_fp32.input(0)

result = gender_fp32({
    input_layer.get_any_name(): img
})
output_tensor = list(result.values())[0]

print("Output shape:", output_tensor.shape)
print("Raw output:", output_tensor) 

pred = np.argmax(output_tensor, axis=1)
print("Predicted gender class:", pred[0])
