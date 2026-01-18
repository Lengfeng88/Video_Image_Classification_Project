from openvino.preprocess import PrePostProcessor, ColorFormat, ResizeAlgorithm
from openvino.runtime import Core, Layout, Type
import openvino as ov

gender_path = "Quantization/ONNX_To_OpenVINO/gender_ov_quant.xml"
age_path = "Quantization/ONNX_To_OpenVINO/age_ov_quant.xml"
cvt_gender = ov.convert_model(gender_path)
cvt_age = ov.convert_model(age_path)

ppp_g = PrePostProcessor(cvt_gender)
ppp_a = PrePostProcessor(cvt_age)

# Input tensor: raw image
ppp_g.input().tensor() \
    .set_element_type(Type.u8) \
    .set_layout(Layout("NHWC"))

ppp_a.input().tensor() \
    .set_element_type(Type.u8) \
    .set_layout(Layout("NHWC"))

# Preprocessing (Caffe-style)
ppp_g.input().preprocess() \
    .convert_element_type(Type.f32) \
    .mean([104, 117, 123]) \
    .scale(1.0)

ppp_a.input().preprocess() \
    .convert_element_type(Type.f32) \
    .mean([104, 117, 123]) \
    .scale(1.0)

# Model expects NCHW
ppp_g.input().model().set_layout(Layout("NCHW"))
ppp_a.input().model().set_layout(Layout("NCHW"))

gender = ppp_g.build()
age = ppp_a.build()

output_gender = "Quantization/ONNX_To_OpenVINO/gender_ov.xml"
output_age = "Quantization/ONNX_To_OpenVINO/age_ov.xml"
ov.save_model(gender, output_gender)
ov.save_model(age, output_age)