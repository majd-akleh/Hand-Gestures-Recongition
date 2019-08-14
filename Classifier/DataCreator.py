import os
import cv2
import random
import Classifier.FeaturesExtractionModule as ext
import pickle

DATADIR = 'C:/Users/MAJD_/PycharmProjects/DataSet'
Categories = ['A Sign', 'B Sign', 'C Sign', 'Five Sign', 'L Sign', 'One Sign', 'V Sign', 'Y Sign']
training_data = []

def create_training_data(DATADIR):
    for cat in Categories:
        path = os.path.join(DATADIR, cat)
        class_num = Categories.index(cat);
        for file in os.listdir(path):

            # read image
            img_path = os.path.join(path, file);
            img_array = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

            # convert to binary images
            ret, bw = cv2.threshold(img_array, 127, 255, cv2.THRESH_BINARY)

            # assign each image to the appropriate output class
            training_data.append([bw, class_num])


create_training_data(DATADIR)

random.shuffle(training_data)

X = []
y = []

for img, label in training_data:
    X.append(ext.extract_features(img))
    temp = [0] * 8
    temp[label] = 1
    y.append(temp)

dataset = [X , y]
# save training data
with open("RawData" , "wb") as f:
    pickle.dump( dataset,f)


