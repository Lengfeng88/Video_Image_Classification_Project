import openvino as ov
import nncf
import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import torch
from sklearn.metrics import accuracy_score
import onnx
import onnxruntime
import cv2

#Fix preprocessing sometime later.
def convert(model_path, model_name):
    cvt_model = ov.convert_model(model_path)

    calib_dataset = ImageFolder("Quantization/calib_data_preprocessed/", 
                                transform=transforms.compose([
                                    transforms.Resize(224),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor()
                                ]))
    calibration_loader = torch.utils.data.DataLoader(calib_dataset, batchsize=16, shuffle=True)

    def transform_fn(data_item):
        images, _ = data_item
        return {images: images.numpy()}

    calibration_dataset = nncf.Dataset(calibration_loader, transform_fn)
    validation_dataset = nncf.Dataset(calibration_loader, transform_fn)


    def validate(model: ov.CompiledModel, 
                 validation_loader: torch.utils.data.DataLoader) -> float:
            predictions = []
            references = []

            output = model.outputs[0]

            for images, target in validation_loader:
                pred = model(images)[output]
                predictions.append(np.argmax(pred, axis=1))
                references.append(target)

            predictions = np.concatenate(predictions, axis=0)
            references = np.concatenate(references, axis=0)
            return accuracy_score(predictions, references)

    quantized_model = nncf.quantize_with_accuracy_control(
        cvt_model,
        calibration_dataset=calibration_dataset,
        validation_dataset=validation_dataset,
        validation_fn=validate,
        max_drop=0.01,
        drop_type=nncf.DropType.ABSOLUTE,
    )

    model_int8 = ov.compile_model(quantized_model)
    input_fp32 = cvt_model # FP32 model input
    res = model_int8(input_fp32)

    ov.save_model(quantized_model, str(model_name + "_quant.xml"), compress_to_fp16=False)

convert("Quantization/age_net_new_v11.onnx", "age_net")
convert("Quantization/gender_net_v11.onnx", "gender_net")


