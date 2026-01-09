import onnx
from onnx import version_converter

# Load the old opset 6 model
model_old = onnx.load("models/Quantization/gender_net.onnx")

# Convert to opset 11 (or 13)
model_new = version_converter.convert_version(model_old, 11)

# Save new model
onnx.save(model_new, "models/Quantization/gender_net_v11.onnx")
