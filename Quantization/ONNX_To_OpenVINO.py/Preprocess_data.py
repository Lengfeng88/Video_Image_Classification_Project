import cv2
import os
from pathlib import Path
import skimage
from PIL import Image

def crop_center(img,cropx,cropy):
    y,x,c = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)    
    return img[starty:starty+cropy,startx:startx+cropx]

folder = Path("Quantization/calib_data_raw")
save_to = Path("Quantization/calib_data_preprocessed_improved_data")
images = os.listdir(folder)
image_paths = []

for img_name in enumerate(images):
    img_path = str(folder + "/" + img_name)

    img = cv2.imread(img_path)

    #Scaling the image
    input_height = 224
    input_width = 224
    aspect = img.shape[1]/float(img.shape[0])
    imgScaled
    if(aspect>1):
        # landscape orientation - wide image
        res = int(aspect * input_height)
        imgScaled = skimage.transform.resize(img, (input_width, res))
    if(aspect<1):
        # portrait orientation - tall image
        res = int(input_width/aspect)
        imgScaled = skimage.transform.resize(img, (res, input_height))
    if(aspect == 1):
        imgScaled = skimage.transform.resize(img, (input_width, input_height))
    
    finalImg = crop_center(imgScaled, 224, 224)
    Image.SAVE(str(save_to + "/" + img_name))
    
    



