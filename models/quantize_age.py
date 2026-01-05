import cv2
import numpy as np

# 1. Load model
net = cv2.dnn.readNetFromCaffe('models/age_deploy.prototxt', 'models/age_net.caffemodel')
face = cv2.dnn.readNet('models\opencv_face_detector_uint8.pb', 'models\opencv_face_detector.pbtxt')
cap = cv2.VideoCapture("Videos\Video.mp4")
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

def enhance_lighting(face_img):
    # Convert to LAB color space
    lab = cv2.cvtColor(face_img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    
    # Merge and convert back
    enhanced = cv2.merge((l, a, b))
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

def is_frontal_face(face_box, frame_shape):
    x1, y1, x2, y2 = face_box
    h, w = frame_shape[:2]
    
    # 1. Size check
    face_area = (x2 - x1) * (y2 - y1)
    if face_area < 0.02 * h * w:  # too small
        return False
        
    # 2. Aspect ratio (profile faces are narrow)
    aspect = (x2 - x1) / (y2 - y1)
    if aspect < 0.6 or aspect > 1.4:  # too wide or narrow
        return False
        
    # 3. Centered? (optional)
    center_x = (x1 + x2) / 2
    if abs(center_x - w/2) > w * 0.35:
        return False
        
    return True

# 2. Calibrate with representative data
cnts = 0
while True:
    # Simulate real input: [1, 3, H, W] BGR float32
    ret, frame = cap.read()

    if not ret:
        break

    if cnts >= 100: 
        break

    fr_cv = frame.copy()

    f_h = fr_cv.shape[0]
    f_w = fr_cv.shape[1]

    blob = cv2.dnn.blobFromImage(fr_cv, 1.0, (300, 300), [104, 117, 123], True, False)

    face.setInput(blob)
    detections = face.forward()

    faceBoxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.9:
            x1 = int(detections[0, 0, i, 3] * f_w)
            y1 = int(detections[0, 0, i, 4] * f_h)
            x2 = int(detections[0, 0, i, 5] * f_w)
            y2 = int(detections[0, 0, i, 6] * f_h)

            faceBoxes.append([x1, y1, x2, y2])

    for faceBox in faceBoxes:
        x1, y1, x2, y2 = faceBox
        img = fr_cv[y1:y2, x1:x2]
        if img.size == 0:
            continue

        if cnts >= 100:
            break

        if is_frontal_face(faceBox, fr_cv.shape):
            img = enhance_lighting(img)
            blob = cv2.dnn.blobFromImage(img, 1.0, (227, 227), (0, 0, 0), swapRB=False, crop=False)
            blob = blob.astype(np.float32)
            net.setInput(blob)
            net.forward()  
            cnts += 1

# 3. Save quantization info
quant_data = net.dump()
if isinstance(quant_data, str):
    # OpenCV 4.12.0 returns string - encode to bytes
    quant_data = quant_data.encode('latin1')
    
with open('models/age_net.caffemodel.quant', 'wb') as f:
    f.write(quant_data)
print("✅ Quantization file saved successfully!")