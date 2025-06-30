**Overview**
This project utilizes Convolutional Neural Network(CNN) to detect whether a driver's eyes are open or closed using frames captured from the video/webcam. If the eyes remanin closed over a certain threshold, a sound alarm
is triggered to wake up and keep the driver alert to prevent accidents.

**Features**
1. Real-time eye state detection using OpenCV package.
2. Using Haar Cascade Files to detect face and eyes.
3. CNN based eye-state classification model.
4. Drowsiness scoring logic to trigger and cut off alarm automatically.
5. Made the logic such that both eyes must be kept open to keep the score below threshold.

**Necessary libraries**: cv2,load_model function from keras,numpy,mixer from pygame,tensorflow and its keras dependencies to build model

**Make sure to download Haar cascade files suitable for face and eye detection**

**Performance metrics**
1. **_Accuracy attained on training data:_**: 98.6%
2. **_F1 Score_**: 0.86

**_Limitations_**
1. Necessity of good lighting to ensure the eyes are captured by camera properly.
2. Takes only the predominant cause eye open/closure into account while other factors like yawning or head orientation are not taken into account.


_**Dataset utilized from:**_: _https://www.kaggle.com/datasets/serenaraju/yawn-eye-dataset-new_
