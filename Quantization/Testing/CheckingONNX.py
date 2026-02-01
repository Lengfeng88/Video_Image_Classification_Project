import onnxruntime as ort
import onnx

def print_onnx_inputs(model_path):
    model = onnx.load(model_path)
    graph = model.graph

    print("=== ONNX Model Inputs ===")
    for inp in graph.input:
        shape = []
        for dim in inp.type.tensor_type.shape.dim:
            if dim.dim_param:
                shape.append(dim.dim_param)   # symbolic (e.g. batch_size)
            elif dim.dim_value:
                shape.append(dim.dim_value)   # concrete number
            else:
                shape.append("?")

        print(f"Name: {inp.name}")
        print(f"Shape: {shape}")
        print("-" * 40)

model_path = "Quantization/gender_net_v11.onnx"

print_onnx_inputs(model_path)