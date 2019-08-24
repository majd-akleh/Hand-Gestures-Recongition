import cv2
import os

def extractFace(RGBimg):
    path = ''
    # path = 'C:/Users/MAJD_/PycharmProjects/Hand Gestures Recognition/venv/Lib/site-packages/cv2/data/'
    path = os.path.join(path , 'haarcascade_frontalface_default.xml')
    face_cascade = cv2.CascadeClassifier(path)

    gray = cv2.cvtColor(RGBimg, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # cv2.rectangle(RGBimg,(x,y),(x+w,y+h),(255,0,0),2)
    return faces


