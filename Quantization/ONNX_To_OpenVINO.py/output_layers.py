import onnx

model = onnx.load("Quantization/age_net_new_v11.onnx")

for i, node in enumerate(model.graph.node):
    print(f"{i:3d} | {node.op_type:15s} | inputs={node.input} | outputs={node.output}")

for inp in model.graph.input:
    print(inp.name, inp.type.tensor_type.shape)

for out in model.graph.output:
    print(out.name, out.type.tensor_type.shape)

for init in model.graph.initializer:
    print(init.name, init.dims)
