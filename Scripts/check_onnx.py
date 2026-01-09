import onnx

# Check if file exists
model_path = "Quantization/age_net_new_v11.onnx"
try:
    model = onnx.load(model_path)
    print("✅ File exists")
    print("Opset version:", model.opset_import[0].version)
    print("Model IR version:", model.ir_version)
except Exception as e:
    print("❌ Load error:", e)