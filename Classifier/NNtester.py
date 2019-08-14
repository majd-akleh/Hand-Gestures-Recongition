import pickle
import os

DATADIR = "C:/Users/MAJD_/PycharmProjects/Hand Gestures Recognition/Classifier/Saves"

from keras.models import load_model
def loadNN():
    return load_model(os.path.join(DATADIR, "Brain"))


with open(os.path.join(DATADIR, "DataSets.pickle"), 'rb') as f:
    [X_train, y_train, X_test, y_test] = pickle.load(f)


y_pred = loadNN().predict(X_test)
y_pred = (y_pred > 0.5)

y_test = y_test.as_matrix()

print("accuracy is: " , len(y_test[y_pred == 1] == 1) / len(y_test) , "%")

