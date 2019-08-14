import pickle
import os
import pandas as pd

DataDir = "C:/Users/MAJD_/PycharmProjects/Hand Gestures Recognition/Classifier/Saves"

file = open(os.path.join(DataDir, "RawData"), "rb")
dataset = pickle.load(file)

Xall , yall = dataset[0][:] , dataset[1][:]

Xall = pd.DataFrame(Xall)
yall = pd.DataFrame(yall)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(Xall, yall, test_size=0.3)

input_layer = 12
hidden_layer = 10 # good practice is to set num of hidden neurons between num of input and num of output
output_layer = 8

from keras import Sequential
from keras.layers import Dense

classifier = Sequential()
#First Hidden Layer
classifier.add(Dense(hidden_layer, activation='relu', kernel_initializer='random_normal', input_dim=input_layer))

#Output Layer
classifier.add(Dense(output_layer, activation='sigmoid', kernel_initializer='random_normal'))

#Compiling the neural network
classifier.compile(optimizer ='adam',loss='binary_crossentropy', metrics =['accuracy'])

#Fitting the data to the training dataset
classifier.fit(X_train,y_train, batch_size=10, epochs=100)

eval_model=classifier.evaluate(X_train, y_train)
print(eval_model)

classifier.save(os.path.join(DataDir, "Brain"))

with open('DataSets.pickle', 'wb') as f:
    pickle.dump([X_train, y_train , X_test , y_test], f)






"""


W = {
    'h1' : tf.Variable(tf.ones([input_layer , hidden_layer])),
    'h2' : tf.Variable(tf.ones([hidden_layer, output_layer]))
}

b = {
    'b1' : tf.Variable(tf.ones([hidden_layer])),
    'b2' : tf.Variable(tf.ones([output_layer]))
}

x = tf.placeholder('float' , [None , input_layer])
x = tf.placeholder('float' , [None , output_layer])

l1 = tf.add(tf.matmul(x , W['h1']) , b['b1'])
l1_sig = tf.sigmoid(l1)

l2 = tf.add(tf.matmul(l1_sig , W['h2']), b['b2'])
l2_sig = tf.sigmoid(l2)

"""