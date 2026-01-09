import cv2
import matplotlib.pyplot as plt

image = cv2.imread('Videos/IMG_0122.jpg')
image = cv2.resize(image, (720, 340))

face1 = "models/opencv_face_detector.pbtxt"
face2 = "models/opencv_face_detector_uint8.pb"
age1 = "models/age_deploy.prototxt"
age2 = "models/age_net.caffemodel"
gen1 = "models/gender_deploy.prototxt"
gen2 = "models/gender_net.caffemodel"

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

# Using models
# Face
face = cv2.dnn.readNet(face2, face1)

# age
age = cv2.dnn.readNet(age2, age1)

# gender
gen = cv2.dnn.readNet(gen2, gen1)

# Categories of distribution
la = ['(0-3)', '(4-7)', '(8-14)', '(15-24)',
      '(25-37)', '(38-43)', '(48-53)', '(60-100)']
lg = ['Male', 'Female']

# Copy image
fr_cv = image.copy()

f_h = fr_cv.shape[0]
f_w = fr_cv.shape[1]

#Parameters: Image, scale factor, size, mean, swapRB (convert BGR to RGB), crop
blob = cv2.dnn.blobFromImage(fr_cv, 1.0, (300, 300), [104, 117, 123], True, False)

#Inputs Blob into the face detector model.
face.setInput(blob)

#Calls one forward pass through the DNN face model
detections = face.forward()

# Face bounding box creation
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

if not faceBoxes:
    print("No face detected, you buffoon")

#Classifying the gender and age by running the blob through the gender and age DNN.
for faceBox in faceBoxes:
    
    #Extracting the face from the photo based on the facebox.
    #Btw, I have no idea what is going on in this one.
    face = fr_cv[max(0, faceBox[1]-15):
                 min(faceBox[3]+15, fr_cv.shape[0]-1),
                 max(0, faceBox[0]-15):min(faceBox[2]+15,
                               fr_cv.shape[1]-1)]
    
    #Converting it to a blob
    blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
    
    #Predict the gender
    gen.setInput(blob)
    genderPreds = gen.forward()
    #No clue what is going on behind lg[]
    gender = lg[genderPreds[0].argmax()]

    #Predict the age group
    age.setInput(blob)
    agePreds = age.forward()
    final_age = la[agePreds[0].argmax()]

    

    cv2.putText(fr_cv,
                f'{gender}, {final_age}',
                (faceBox[0]-150, faceBox[1]+10),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.3,
                (217, 0, 0),
                4,
                cv2.LINE_AA)
    
plt.figure(figsize=(7, 7))
plt.imshow(fr_cv)
plt.show()










