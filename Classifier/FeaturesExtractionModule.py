from skimage import measure
import cv2


def extract_features(binaryImage):

    # no need for labeling images as they already have one label
    # label_image = measure.label(binaryImage)

    region = measure.regionprops(binaryImage,coordinates='rc')

    # only one region is provided
    region = region[0]
    features = [region.eccentricity, region.extent, region.solidity]
    features.append(region.eccentricity ** 2)
    features.append(region.extent * region.solidity)

    moments = cv2.HuMoments(cv2.moments(binaryImage)).flatten()
    features.extend(moments)
    return features


"""
% % == == == == == PCA
Coefficients
% Coeff = extract_pca(binaryImage);
% features(13: 19) = Coeff;

% % == == == == == fingers
feats
% feats = finger_features(binaryImage);
% features(20: 25) = feats;
%

end
"""
