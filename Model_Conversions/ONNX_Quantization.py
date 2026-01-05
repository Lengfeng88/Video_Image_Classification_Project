import numpy as np
import onnx
import onnxruntime
from onnxruntime.quantization import quantize_static, QuantType, CalibrationDataReader

# 1. Check your model's input name
model = onnx.load("age_net.onnx")
input_name = model.graph.input[0].name
print(f"Model input name: {input_name}")  # Probably "data"

# 2. Calibration data reader
class FaceDataReader(CalibrationDataReader):
    def __init__(self, image_paths):
        self.image_paths = image_paths
        self.iterator = iter(image_paths)
    
    def get_next(self):
        try:
            img_path = next(self.iterator)
            # Load and preprocess image to match your model's expected input
            # For your Caffe model: [1, 3, 227, 227] (NCHW format)
            img = np.random.random((1, 3, 227, 227)).astype(np.float32)  # Replace with real preprocessing
            return {input_name: img}
        except StopIteration:
            return None

# 3. Prepare calibration data (use your actual image paths)
calibration_images = ["image1.jpg", "image2.jpg"]  # Add your 100 images
dr = FaceDataReader(calibration_images)

# 4. Quantize!
# Correct usage
quantize_static(
    model_input="age_net.onnx",
    model_output="age_net_quant.onnx",
    calibration_data_reader=dr,
    quant_format=onnxruntime.quantization.QuantFormat.QDQ,  # ← onnxruntime is now defined
    activation_type=QuantType.QUInt8,
    weight_type=QuantType.QInt8,
    op_types_to_quantize=['Conv', 'Gemm']
)

print("✅ Quantized ONNX model created: age_net_quant.onnx")