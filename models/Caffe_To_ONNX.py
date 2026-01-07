import cv2

# Load your Caffe model (same as before)
net = cv2.dnn.readNetFromCaffe(
    "models/age_deploy.prototxt",
    "models/age_net.caffemodel"
)

# Export to ONNX
net.save("models/age_net_new.onnx")

print("✅ ONNX model saved!")