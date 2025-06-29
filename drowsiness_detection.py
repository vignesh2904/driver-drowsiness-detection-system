import cv2
import os
from keras.models import load_model
import numpy as np
from pygame import mixer
import time

# Initialize sound system
mixer.init()
sound = mixer.Sound('alarm.wav')

# Load Haar cascades
face_cascade = cv2.CascadeClassifier('haar cascade files/haarcascade_frontalface_alt.xml')
left_eye_cascade = cv2.CascadeClassifier('haar cascade files/haarcascade_lefteye_2splits.xml')
right_eye_cascade = cv2.CascadeClassifier('haar cascade files/haarcascade_righteye_2splits.xml')

# Load trained model
model = load_model('models/vig.h5')
labels = ['Closed', 'Open']

# Webcam and setup
cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
score = 0
thicc = 2
alarm_on = False
path = os.getcwd()

while True:
    ret, frame = cap.read()
    height, width = frame.shape[:2]

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(25, 25))
    left_eyes = left_eye_cascade.detectMultiScale(gray)
    right_eyes = right_eye_cascade.detectMultiScale(gray)

    cv2.rectangle(frame, (0, height - 50), (200, height), (0, 0, 0), thickness=cv2.FILLED)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (100, 100, 100), 1)

    rpred = [1]
    lpred = [1]

    for (x, y, w, h) in right_eyes:
        r_eye = gray[y:y + h, x:x + w]
        r_eye = cv2.resize(r_eye, (24, 24)) / 255.0
        r_eye = r_eye.reshape(1, 24, 24, 1)
        rpred = np.argmax(model.predict(r_eye), axis=1)
        break

    for (x, y, w, h) in left_eyes:
        l_eye = gray[y:y + h, x:x + w]
        l_eye = cv2.resize(l_eye, (24, 24)) / 255.0
        l_eye = l_eye.reshape(1, 24, 24, 1)
        lpred = np.argmax(model.predict(l_eye), axis=1)
        break

    if rpred[0] == 0 and lpred[0] == 0:
        score += 1
        cv2.putText(frame, "Closed", (10, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
    else:
        score = max(score - 1, 0)
        cv2.putText(frame, "Open", (10, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.putText(frame, 'Score:' + str(score), (100, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

    if score > 15:
        cv2.imwrite(os.path.join(path, 'drowsy_frame.jpg'), frame)

        if not alarm_on:
            sound.play(-1)  # Loop until stopped
            alarm_on = True

        if thicc < 16:
            thicc += 2
        else:
            thicc = max(thicc - 2, 2)

        cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 255), thicc)
        cv2.putText(frame, "DROWSINESS ALERT!", (width // 4, height // 2), font, 2, (0, 0, 255), 2)

    else:
        if alarm_on:
            sound.stop()
            alarm_on = False

    cv2.imshow('Driver Drowsiness Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
