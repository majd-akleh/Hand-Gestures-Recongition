import Image_Processing_Module.FaceExtractor as fe
import Image_Processing_Module.SkinDetector as sd
import Image_Processing_Module.HandLocalizer as hl
import cv2
import numpy as np

URL = ''
cap = cv2.VideoCapture(0)
yellow = (0, 255, 255)
green = (0, 255, 0)
black = (0, 0, 0)

Allfaces = []
rawImage = []
while len(Allfaces) == 0:
    ret, rawImage = cap.read()
    Allfaces = fe.extractFace(rawImage)
(x, y, w, h) = Allfaces[0]
cropped_face = rawImage[y:y + h, x:x + w]

n = 500  # number of snapshots
for snapshot in range(n):

    ret, rawImage = cap.read()

    skinMask = sd.detectSkinUsingHist(rawImage, cropped_face)

    skin = cv2.bitwise_and(rawImage, skinMask)
    #skin = rawImage
    #skin[skinMask == False] = black

    cv2.rectangle(rawImage, (x, y), (x + w, y + h), green, 2)
    cv2.imshow('h1', np.hstack([rawImage, skin]))
    cv2.waitKey(0)

    # remove face from mask
    skinMask[y:y + h, x:x + w] = 0

    hand1, hand2, bounds1, bounds2 = hl.localizeHands(skinMask)

    cv2.rectangle(rawImage, (x, y), (x + w, y + h), green, 2)
    cv2.rectangle(rawImage, (bounds1[1], bounds1[0]), (bounds1[3], bounds1[2]), yellow, 2)
    cv2.rectangle(rawImage, (bounds2[1], bounds2[0]), (bounds2[3], bounds2[2]), yellow, 2)

    cv2.imshow('h1', np.hstack([rawImage, skin]))
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
