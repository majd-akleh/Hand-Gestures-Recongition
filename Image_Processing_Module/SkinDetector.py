
import numpy as np
import cv2

thresh = 0.78;

def detectSkin(img, cropped_face):

    faceYCrCb = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2YCR_CB)
    imgYCrCb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)

    face_cb = faceYCrCb[:, :, 1]
    face_cr = faceYCrCb[:, :, 2]

    median_cb = np.median(face_cb)
    median_cr = np.median(face_cr)

    img_cb = imgYCrCb[:, :, 1]
    img_cr = imgYCrCb[:, :, 2]

    cb_dist = np.subtract(img_cb, median_cb)
    cr_dist = np.subtract(img_cr, median_cr)

    skin_dist = np.sqrt(np.add(np.power(cb_dist, 2), np.power(cr_dist, 2)))

    # convert the values range from float to 0 - 1
    norm_image = cv2.normalize(skin_dist, None, norm_type=cv2.NORM_MINMAX)

    # invert image bitwisly
    norm_image = np.subtract(1, norm_image)

    # apply threshold
    ret, skinMask = cv2.threshold(norm_image, thresh , 1, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    skinMask = cv2.erode(skinMask, kernel, iterations=1)
    skinMask = cv2.dilate(skinMask, kernel, iterations=1)
    return skinMask


def detectSkinUsingHist(img, cropped_face):

    skin_hist = cv2.calcHist([cropped_face], [0, 1], None, [180, 256], [0, 180, 0, 256])
    normed_hist = cv2.normalize(skin_hist, skin_hist, 0, 255, cv2.NORM_MINMAX)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    dst = cv2.calcBackProject([hsv], [0, 1], normed_hist, [0, 180, 0, 256], 1)

    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    cv2.filter2D(dst, -1, disc, dst)

    ret, thresh2 = cv2.threshold(dst, 150, 255, cv2.THRESH_BINARY)

    thresh2 = cv2.merge((thresh2, thresh2, thresh2))

    return thresh2


from matplotlib import pyplot as plt
def pltHist(histr):
    color = ('b', 'g', 'r')
    for i, col in enumerate(color):
        plt.plot(histr, color=col)
    plt.show()