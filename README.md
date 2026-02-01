# Video_Image_Classification_Project

The primary objective of the project is to fairly accurately and in a stable manner classify a person's gender and age based on their face using OpenCV facial detection, age, and gender classification DNN models. This assumes the face is aligned in a fair position to the camera. First creating a python demo script for a desktop version.

An addition to the primary objective is to first deploy using OpenCV mobile SDK and then deploy. 

Taking the original mode, and converting the models to TensorFlow Lite and apply quantization and then deploy the project to be usable on android mobile devices. (Expects the android device to have relatively up to date processing power and software, so around android 9)

And then comparing the two deployments.

Stage 1 - Pipeline

Pipeline for image_processing.

1. Reading inputs, setting global variables and constants such as MEAN model values and classification groups.
2. OpenCV facial detection, run one forward pass on the OpenCV DNN.
3. Gather detections, take x, y coordinates and save for later processing.
4. Use x, y coordinates of individual faces-box crops and run gender, age classifications.
5. Return the frame with appropriate face box and labels.

Pipeline for Video_processing
1. Input frame.
2. Pipeline for image_processing.
3. Return frame with proper edits.
4. Repeat steps 1 - 3 with every single frame.
5. Return the video.

Stage 2 - Video Processing

Image_Processing.py - A template where a basic image detection and attribute classification system is built for images.

Video_Processing - A basic version where a video detection and classification system is built inspired by the approach used in Image_Processing. Eseentially implemented image processing frame by frame and optimized by dealing with misaligned faces, bad lighting, and only processing faces that face forwards.

Video_Processing_Single_Face - A more optimized and stable version of Video_Processing but only meant for a single face. Improvements include: More stable processing and accurate predictions as well as less flickering and more stable predictions.

Video_Processing _Multi_Face - Optimized with IoU to be able to track multiple faces, and take the best prediction of n frames. Currently a work in progress.

*These are just demos/desktop versions, will have to re-implement in C++ in deployment to android.

Stage 3 - Quantization (Post Training Static)

Calibration (min-max): Used 100 face crops to determine the range [a, b] that should be chosen for the INT8 conversion.

Stage 4 - Android Deployment Preparation

Moving to Android Studio.

*Note: Originally my plan here is to quantize the gender and age models, then move to deployment as is. As you can imagine, I was not pleasantly surprised when I found out OpenCV DNN still runs it as FP32. So, now, falling to a backup plan which is to convert and quantize the original model to TFLite and then deploy on android studio using C++. Regardless, the comparison will still happen and I will implement the deployment in those 2 ways, however is it evident which one is better.

Update: Following model Post training quantization to INT8 using the Openvino framework. Model evaluation results indicate a near 10% drop in accuracy for the age model. Currently looking into it, but it may be a blog or post worthy problem if it is a problem with the MDNN conversion tools.

