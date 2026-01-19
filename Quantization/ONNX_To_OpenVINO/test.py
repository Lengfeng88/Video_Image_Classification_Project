import cv2
import os
from Calib_Dataset import GenderCalibrationDataset
from Calib_Dataset import AgeCalibrationDataset

la = ['(0-2)', '(4-6)', '(8-12)', '(15-20)',
                '(25-32)', '(38-43)', '(48-53)', '(60-100)']
lg = ['Male', 'Female']

age = AgeCalibrationDataset("UTKFace")

images = age.get_paths("UTKFace")

for idx, img_path in enumerate(images):
    # if age.img_labels[img_path] is None:
    #     print(img_path + " " + str())
    print(age.img_labels[img_path])

    # print(la[age.img_labels[img_path]] + " " + img_path)

# for img_path in images[:20]:
#     cached = age.img_labels[img_path]
#     recomputed = age.get_age_label(img_path)
#     print(img_path, cached, recomputed)