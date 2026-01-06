import numpy as np
from onnx import helper, TensorProto
import onnx

_weights_dict = dict()

def load_weights(weight_file):
    if weight_file == None:
        return

    try:
        weights_dict = np.load(weight_file, allow_pickle=True).item()
    except:
        weights_dict = np.load(weight_file, allow_pickle=True, encoding='bytes').item()

    return weights_dict


def KitModel(weight_file = None):
    global _weights_dict
    _weights_dict = load_weights(weight_file)


    data_orig       = helper.make_tensor_value_info('data_orig', TensorProto.FLOAT, (1, 227, 227, 3,))
    data            = helper.make_node('Transpose', inputs=['data_orig'], outputs=['data'], perm=[0, 3, 1, 2], name='data')
    conv1_weight_array = _weights_dict['conv1']['weights']
    conv1_weight_array = conv1_weight_array.transpose([3,2,0,1])
    conv1_weight    = helper.make_tensor_value_info('conv1_weight', onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[conv1_weight_array.dtype], list(conv1_weight_array.shape))
    conv1_weight_init = helper.make_tensor(name='conv1_weight', data_type=onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[conv1_weight_array.dtype], dims=conv1_weight_array.shape, vals=conv1_weight_array.flatten().astype(float))
    conv1_bias_array = _weights_dict['conv1']['bias'].squeeze()
    conv1_bias      = helper.make_tensor_value_info('conv1_bias', onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[conv1_bias_array.dtype], list(conv1_bias_array.shape))
    conv1_bias_init = helper.make_tensor(name='conv1_bias', data_type=onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[conv1_bias_array.dtype], dims=conv1_bias_array.shape, vals=conv1_bias_array.flatten().astype(float))
    conv1           = helper.make_node('Conv', inputs=['data', 'conv1_weight', 'conv1_bias'],outputs=['conv1'], dilations=[1, 1], group=1, kernel_shape=[7, 7], pads=[0, 0, 1, 1], strides=[4, 4], name='conv1')
    relu1           = helper.make_node('Relu', inputs=['conv1'], outputs=['relu1'], name='relu1')
    pool1           = helper.make_node('MaxPool', inputs=['relu1'],outputs=['pool1'], kernel_shape=[3, 3], pads=[0, 0, 1, 1], strides=[2, 2], name='pool1')
    norm1           = helper.make_node('LRN', inputs=['pool1'], outputs=['norm1'], alpha=9.999999747378752e-05, beta=0.75, bias=1.0, size=5, name='norm1')
    conv2_weight_array = _weights_dict['conv2']['weights']
    conv2_weight_array = conv2_weight_array.transpose([3,2,0,1])
    conv2_weight    = helper.make_tensor_value_info('conv2_weight', onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[conv2_weight_array.dtype], list(conv2_weight_array.shape))
    conv2_weight_init = helper.make_tensor(name='conv2_weight', data_type=onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[conv2_weight_array.dtype], dims=conv2_weight_array.shape, vals=conv2_weight_array.flatten().astype(float))
    conv2_bias_array = _weights_dict['conv2']['bias'].squeeze()
    conv2_bias      = helper.make_tensor_value_info('conv2_bias', onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[conv2_bias_array.dtype], list(conv2_bias_array.shape))
    conv2_bias_init = helper.make_tensor(name='conv2_bias', data_type=onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[conv2_bias_array.dtype], dims=conv2_bias_array.shape, vals=conv2_bias_array.flatten().astype(float))
    conv2           = helper.make_node('Conv', inputs=['norm1', 'conv2_weight', 'conv2_bias'],outputs=['conv2'], dilations=[1, 1], group=1, kernel_shape=[5, 5], pads=[2, 2, 2, 2], strides=[1, 1], name='conv2')
    relu2           = helper.make_node('Relu', inputs=['conv2'], outputs=['relu2'], name='relu2')
    pool2           = helper.make_node('MaxPool', inputs=['relu2'],outputs=['pool2'], kernel_shape=[3, 3], pads=[0, 0, 1, 1], strides=[2, 2], name='pool2')
    norm2           = helper.make_node('LRN', inputs=['pool2'], outputs=['norm2'], alpha=9.999999747378752e-05, beta=0.75, bias=1.0, size=5, name='norm2')
    conv3_weight_array = _weights_dict['conv3']['weights']
    conv3_weight_array = conv3_weight_array.transpose([3,2,0,1])
    conv3_weight    = helper.make_tensor_value_info('conv3_weight', onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[conv3_weight_array.dtype], list(conv3_weight_array.shape))
    conv3_weight_init = helper.make_tensor(name='conv3_weight', data_type=onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[conv3_weight_array.dtype], dims=conv3_weight_array.shape, vals=conv3_weight_array.flatten().astype(float))
    conv3_bias_array = _weights_dict['conv3']['bias'].squeeze()
    conv3_bias      = helper.make_tensor_value_info('conv3_bias', onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[conv3_bias_array.dtype], list(conv3_bias_array.shape))
    conv3_bias_init = helper.make_tensor(name='conv3_bias', data_type=onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[conv3_bias_array.dtype], dims=conv3_bias_array.shape, vals=conv3_bias_array.flatten().astype(float))
    conv3           = helper.make_node('Conv', inputs=['norm2', 'conv3_weight', 'conv3_bias'],outputs=['conv3'], dilations=[1, 1], group=1, kernel_shape=[3, 3], pads=[1, 1, 1, 1], strides=[1, 1], name='conv3')
    relu3           = helper.make_node('Relu', inputs=['conv3'], outputs=['relu3'], name='relu3')
    pool5           = helper.make_node('MaxPool', inputs=['relu3'],outputs=['pool5'], kernel_shape=[3, 3], pads=[0, 0, 1, 1], strides=[2, 2], name='pool5')
    fc6_0           = helper.make_node('Flatten', inputs=['pool5'], outputs=['fc6_0'], name='fc6_0')
    fc6_1_weight_array = _weights_dict['fc6_1']['weights']
    fc6_1_weight    = helper.make_tensor_value_info('fc6_1_weight', onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[fc6_1_weight_array.dtype], list(fc6_1_weight_array.shape))
    fc6_1_weight_init = helper.make_tensor(name='fc6_1_weight', data_type=onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[fc6_1_weight_array.dtype], dims=fc6_1_weight_array.shape, vals=fc6_1_weight_array.flatten().astype(float))
    fc6_1_bias_array = _weights_dict['fc6_1']['bias'].squeeze()
    fc6_1_bias      = helper.make_tensor_value_info('fc6_1_bias', onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[fc6_1_bias_array.dtype], list(fc6_1_bias_array.shape))
    fc6_1_bias_init = helper.make_tensor(name='fc6_1_bias', data_type=onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[fc6_1_bias_array.dtype], dims=fc6_1_bias_array.shape, vals=fc6_1_bias_array.flatten().astype(float))
    fc6_1           = helper.make_node('Gemm', inputs=['fc6_0', 'fc6_1_weight', 'fc6_1_bias'], outputs=['fc6_1'], name='fc6_1')
    relu6           = helper.make_node('Relu', inputs=['fc6_1'], outputs=['relu6'], name='relu6')
    drop6           = helper.make_node('Dropout', inputs=['relu6'], outputs=['drop6'], is_test=1, ratio=0.5, name='drop6')
    fc7_0           = helper.make_node('Flatten', inputs=['drop6'], outputs=['fc7_0'], name='fc7_0')
    fc7_1_weight_array = _weights_dict['fc7_1']['weights']
    fc7_1_weight    = helper.make_tensor_value_info('fc7_1_weight', onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[fc7_1_weight_array.dtype], list(fc7_1_weight_array.shape))
    fc7_1_weight_init = helper.make_tensor(name='fc7_1_weight', data_type=onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[fc7_1_weight_array.dtype], dims=fc7_1_weight_array.shape, vals=fc7_1_weight_array.flatten().astype(float))
    fc7_1_bias_array = _weights_dict['fc7_1']['bias'].squeeze()
    fc7_1_bias      = helper.make_tensor_value_info('fc7_1_bias', onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[fc7_1_bias_array.dtype], list(fc7_1_bias_array.shape))
    fc7_1_bias_init = helper.make_tensor(name='fc7_1_bias', data_type=onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[fc7_1_bias_array.dtype], dims=fc7_1_bias_array.shape, vals=fc7_1_bias_array.flatten().astype(float))
    fc7_1           = helper.make_node('Gemm', inputs=['fc7_0', 'fc7_1_weight', 'fc7_1_bias'], outputs=['fc7_1'], name='fc7_1')
    relu7           = helper.make_node('Relu', inputs=['fc7_1'], outputs=['relu7'], name='relu7')
    drop7           = helper.make_node('Dropout', inputs=['relu7'], outputs=['drop7'], is_test=1, ratio=0.5, name='drop7')
    fc8_0           = helper.make_node('Flatten', inputs=['drop7'], outputs=['fc8_0'], name='fc8_0')
    fc8_1_weight_array = _weights_dict['fc8_1']['weights']
    fc8_1_weight    = helper.make_tensor_value_info('fc8_1_weight', onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[fc8_1_weight_array.dtype], list(fc8_1_weight_array.shape))
    fc8_1_weight_init = helper.make_tensor(name='fc8_1_weight', data_type=onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[fc8_1_weight_array.dtype], dims=fc8_1_weight_array.shape, vals=fc8_1_weight_array.flatten().astype(float))
    fc8_1_bias_array = _weights_dict['fc8_1']['bias'].squeeze()
    fc8_1_bias      = helper.make_tensor_value_info('fc8_1_bias', onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[fc8_1_bias_array.dtype], list(fc8_1_bias_array.shape))
    fc8_1_bias_init = helper.make_tensor(name='fc8_1_bias', data_type=onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[fc8_1_bias_array.dtype], dims=fc8_1_bias_array.shape, vals=fc8_1_bias_array.flatten().astype(float))
    fc8_1           = helper.make_node('Gemm', inputs=['fc8_0', 'fc8_1_weight', 'fc8_1_bias'], outputs=['fc8_1'], name='fc8_1')
    prob            = helper.make_node('Softmax', inputs=['fc8_1'], outputs=['prob'], name='prob')
    prob_out        = helper.make_tensor_value_info('prob', TensorProto.FLOAT, (1, 1, 2,))
    graph = helper.make_graph([data, conv1, relu1, pool1, norm1, conv2, relu2, pool2, norm2, conv3, relu3, pool5, fc6_0, fc6_1, relu6, drop6, fc7_0, fc7_1, relu7, drop7, fc8_0, fc8_1, prob], 'mmdnn', [data_orig, conv1_bias, conv1_weight, conv2_bias, conv2_weight, conv3_bias, conv3_weight, fc6_1_weight, fc6_1_bias, fc7_1_weight, fc7_1_bias, fc8_1_weight, fc8_1_bias], [prob_out], [conv1_bias_init, conv1_weight_init, conv2_bias_init, conv2_weight_init, conv3_bias_init, conv3_weight_init, fc6_1_weight_init, fc6_1_bias_init, fc7_1_weight_init, fc7_1_bias_init, fc8_1_weight_init, fc8_1_bias_init])
    return helper.make_model(graph, opset_imports=[helper.make_opsetid('', 6)])
