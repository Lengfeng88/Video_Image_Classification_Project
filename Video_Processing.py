import cv2
import matplotlib.pyplot as plt
import time
import heapq

cap = cv2.VideoCapture(0)

face1 = "opencv_face_detector.pbtxt"
face2 = "opencv_face_detector_uint8.pb"
age1 = "age_deploy.prototxt"
age2 = "age_net.caffemodel"
gen1 = "gender_deploy.prototxt"
gen2 = "gender_net.caffemodel"

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

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

prev_time = 0
frame_cnts = 0

most_confident_gender = 0
most_confident_age = 0
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

        if confidence > 0.7:
            x1 = int(detections[0, 0, i, 3] * f_w)
            y1 = int(detections[0, 0, i, 4] * f_h)
            x2 = int(detections[0, 0, i, 5] * f_w)
            y2 = int(detections[0, 0, i, 6] * f_h)

            faceBoxes.append([x1, y1, x2, y2])
            cv2.rectangle(fr_cv, (x1, y1), (x2, y2), (0, 255, 0), int(round(f_h/150)), 8)

    for faceBox in faceBoxes:
        x1, y1, x2, y2 = faceBox
        img = fr_cv[y1:y2, x1:x2]

        if img.size == 0: continue

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

    # Calculate FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    # Display FPS on frame
    cv2.putText(fr_cv, f"FPS: {fps:.1f}", (10, 30), 
    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    if faceBoxes:
        cv2.putText(fr_cv, f'{gender}, {final_age}', (x1, y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
    frame_cnts += 1
    cv2.imshow("Age/Gender Prediction", fr_cv)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
