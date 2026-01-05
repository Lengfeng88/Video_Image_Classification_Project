import numpy as np
import cv2
import time

cap = cv2.VideoCapture("Videos/Video.mp4")

face1 = "models/opencv_face_detector.pbtxt"
face2 = "models/opencv_face_detector_uint8.pb"
age1 = "models/age_deploy.prototxt"
age2 = "models/age_net.caffemodel"
gen1 = "models/gender_deploy.prototxt"
gen2 = "models/gender_net.caffemodel"

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

cv2.setLogLevel(0)  # Shows OpenCV DNN debug info

# Using models
# Face
face = cv2.dnn.readNet(face2, face1)

# age
age = cv2.dnn.readNet(age2, age1)

# gender
gen = cv2.dnn.readNet(gen2, gen1)

# Categories of distribution
la = ['(0-2)', '(4-6)', '(8-12)', '(15-20)',
      '(25-32)', '(38-43)', '(48-53)', '(60-100)']
lg = ['Male', 'Female']

def box_iou(box1, box2):
    # box = (x1, y1, x2, y2)
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    if x2 <= x1 or y2 <= y1:
        return 0.0
        
    intersection = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return intersection / (area1 + area2 - intersection)

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

# Global: store (box, label, history) for each tracked face
tracked_faces = []  # list of { 'box': (x1,y1,x2,y2), 'label': str, 'history': deque }

curr_time = 0
prev_time = 0
while True:
    ret, frame = cap.read()

    if not ret:
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

    final_detections = []
    for faceBox in faceBoxes:
        x1, y1, x2, y2 = faceBox
        img = fr_cv[y1:y2, x1:x2]

        if img.size == 0: continue

        if is_frontal_face(faceBox, fr_cv.shape):
            img = enhance_lighting(img)
            #Extract the portion of the image with the face.

            blob = cv2.dnn.blobFromImage(img, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

            gen.setInput(blob)
            genderPreds = gen.forward()
            gender = lg[genderPreds[0].argmax()]

            age.setInput(blob)
            agePreds = age.forward()
            final_age = la[agePreds[0].argmax()]

            prev_frame_gender = gender
            prev_frame_age = final_age

            final_detections.append({
                'box': faceBox,
                'label': f"{gender}, {final_age}",

                # 'confidence': {
                #     'age': agePreds.max(),
                #     'gender': genderPreds.max()
                # },
            })

    # Calculate FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    # Display FPS on frame
    cv2.putText(fr_cv, f"FPS: {fps:.1f}", (10, 30), 
    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    # if faceBoxes:
    #     cv2.putText(fr_cv, f'{gender}, {final_age}', (x1, y1 - 10),
    #     cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
    # frame_cnts += 1
    # cv2.imshow("Age/Gender Prediction", fr_cv)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

    # Match current     faces to tracked faces using IoU
    new_tracked = []
    tracked_face = []
    used_current = [False] * len(final_detections)

    for tracked in tracked_faces:
        best_match = None
        best_iou = 0.3  # threshold

        for i, (curr_box, curr_label) in enumerate(final_detections):
            if used_current[i]:
                continue
            iou = box_iou(tracked['box'], curr_box)
            if iou > best_iou:
                best_iou = iou
                best_match = (i, curr_box, curr_label)

        if best_match:
            i, curr_box, curr_label = best_match
            used_current[i] = True
            # Update history
            tracked['history'].append(curr_label)
            if len(tracked['history']) > 3:
                tracked['history'].pop(0)
            # Smooth label
            smoothed_label = max(set(tracked['history']), key=tracked['history'].count)
            new_tracked.append({
                'box': curr_box,
                'label': smoothed_label,
                'history': tracked['history']
            })
        # Else: face disappeared → drop it

    # Add new faces (not matched)
    for i, (curr_box, curr_label) in enumerate(final_detections):
        if not used_current[i]:
            new_tracked.append({
                'box': curr_box,
                'label': curr_label,
                'history': [curr_label]
            })

    tracked_faces = new_tracked

    # Draw all tracked_faces
    for face in tracked_faces:
        x1, y1, x2, y2 = face['box']
        cv2.rectangle(fr_cv, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(fr_cv, face['label'], (x1, y1 - 10))

cap.release()
cv2.destroyAllWindows()