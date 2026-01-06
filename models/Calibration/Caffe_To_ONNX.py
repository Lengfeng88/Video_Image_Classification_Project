import onnxruntime as ort
from CalibrationDataReader import FaceCalibrationDataReader

model = ort.InferenceSession("age_net.onnx")

input_name = model.get_inputs()[0].name

from onnxruntime.quantization import (
    quantize_static,
    QuantType,
    QuantFormat,
    CalibrationMethod
)

calibration_reader = FaceCalibrationDataReader(
    folder="calib_data/",
    input_name=input_name,
    batch_size=1
)

quantize_static(
    model_input="age_net.onnx",
    model_output="age_net_quant.onnx",
    calibration_data_reader=calibration_reader,
    quant_format=QuantFormat.QDQ,       # Recommended for accuracy
    activation_type=QuantType.QUInt8,
    weight_type=QuantType.QUInt8,
    calibrate_method=CalibrationMethod.MinMax  # or Entropy for better accuracy
)
print("Quantized model saved as age_net_quant.onnx")

