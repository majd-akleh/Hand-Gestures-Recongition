import cv2
import numpy as np
from skimage import measure


def localizeHands(bwImage):

    labeled = measure.label(bwImage)

    regions = measure.regionprops(labeled, coordinates='rc')

    areas = [r.area for r in regions]
    areas.sort()

    filtered = filter(lambda region: region.area == areas[-1], regions)
    hand1_region = list(filtered)[0]
    filtered = filter(lambda region: region.area == areas[-2], regions)
    hand2_region = list(filtered)[0]

    (minc1, minr1, maxc1, maxr1) = hand1_region.bbox
    (minc2, minr2, maxc2, maxr2) = hand2_region.bbox

    hand1_cropped = bwImage[minc1 : maxc1, minr1 : maxr1]
    hand2_cropped = bwImage[minc2 : maxc2, minr2 : maxr2]

    hand1 = applyMorphology(hand1_cropped)
    hand2 = applyMorphology(hand2_cropped)

    return hand1, hand2 , hand1_region.bbox , hand2_region.bbox


def applyMorphology(hand):
    kernel = np.ones((5, 5), np.uint8)
    return  cv2.morphologyEx(hand, cv2.MORPH_CLOSE, kernel)
