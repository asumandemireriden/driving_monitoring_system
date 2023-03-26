# driving_monitoring_system
 The goal of Driving Monitoring Systems is to prevent driver-related traffic accidents by detecting the driver's drowsiness, and attention state. 

## Equipments of Project
- Raspberry Pi 4
- Camera Module
- Mediapipe Library
OpenCv has 68 facial landmarks.
Mediapipe has 468 facial landmarks.
In this project, Mediapipe library is chosen, since it gives more accurate result.

## Methodology
 The video sequence is divided into frames, each frame is converted to grayscale, and then face detection and tracking are performed. To analyze the face ratio, 468 feature points are extracted for the face using the Mediapipe library.

![Visualisation of the 68 Facial Landmarks](C:\Users\MONSTER\Downloads\Ek Açıklama 2023-03-26 115516.jpg)


