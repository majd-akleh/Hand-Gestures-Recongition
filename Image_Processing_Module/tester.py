

import cv2
import Image_Processing_Module.FaceExtractor as fe
import Image_Processing_Module.SkinDetector as sd
import Image_Processing_Module.HandLocalizer as hl


img = cv2.imread('C:/Users/MAJD_/Desktop/sign.jpg');
face = fe.extractFace(img)

(x,y,w,h) = face[0]
crop_img = img[y:y+h, x:x+w]

bw = sd.detectSkin(img , crop_img)

hand1 , hand2 , bounds1 , bounds2 = hl.localizeHands(bw , face)
cv2.rectangle(img,(bounds1[1],bounds1[0]),(bounds1[3],bounds1[2]),(255,0,0),2)
cv2.rectangle(img,(bounds2[1],bounds2[0]),(bounds2[3],bounds2[2]),(255,0,0),2)

cv2.imshow('h1', img )
cv2.waitKey(0)

cv2.destroyAllWindows()







