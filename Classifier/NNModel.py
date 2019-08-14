import pickle
import os

DataDir = "C:/Users/MAJD_/PycharmProjects/Hand Gestures Recognition/Classifier/Saves"

file = open(os.path.join(DataDir, "X_train"),"rb")
X = pickle.load(file)

file = open(os.path.join(DataDir, "y_train"),"rb")
y = pickle.load(file)

print(len(X))
print(type(X))