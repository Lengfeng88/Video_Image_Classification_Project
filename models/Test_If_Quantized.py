import caffe
net = caffe.Net('age_deploy.prototxt', 'age_net.caffe.quant', caffe.TEST)

for layer_name, param in net.params.items():
    weights = param[0].data  # shape: (out, in, h, w)
    print(f"{layer_name}: {weights.dtype}, min={weights.min():.3f}, max={weights.max():.3f}")