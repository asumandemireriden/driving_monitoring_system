# Driving Monitoring System
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

### Drowsiness Detection and Methodology
The 12 landmarks points selected for eyes:

- For left eye: [362, 385, 387, 263, 373, 380]
- For the right eye: [33,133,143,144,158, 160]
- For the mouth: [61, 291, 39, 181,0, 17, 269, 405]

EAR (Eye Aspect Ratio) is used to detect the state of being eye closed.

A person's yawning status can be determined using the MAR (Mouth Aspect Ratio).

![EAR_MAR_representation](https://user-images.githubusercontent.com/73910961/227766467-99623ec3-acbc-4e2c-b9e9-0b9244cd6341.png)

Figure 1 shows the representation of the EAR and MAR values.

Threshold Value
EAR: 0.25
MAR:0.75

## Distraction Detection and Methodology
 The distraction is detected according to the direction of the gaze. To detect the gaze, head pose estimation is made. The position of the head in different directions such as left, right, up, down, and forward is detected. In the methodology of application, the nose position is detected using the facial landmarks and according to the nose position the reference value is selected according to the x,y, and z-axis. After that, according to nose direction, the head position is obtained as left, right, up, down, and forward. 

## Outputs of the project

![final_outputs_on_edgedevice](https://user-images.githubusercontent.com/73910961/227766517-d8c859b1-16fb-4895-ab00-d6f273054153.png)

## Conclusion 
- The application is executed on an edge device which is Raspberry Pi 4 with a USB camera module.

- A driving monitoring system is a valuable tool for monitoring and improving driver behavior. It can help to reduce accidents and improve efficiency by providing real-time feedback to drivers and fleet managers.

- In future work, the project can be implemented using machine learning models.




## References
[1] B. -T. Dong and H. -Y. Lin, "An On-board Monitoring System for Driving Fatigue and Distraction Detection," 2021



