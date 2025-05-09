import os
import cv2 #computer vision to load images
import numpy as np #used for arrays
import matplotlib.pyplot as plt
import tensorflow as tf #this is for ml

#loading data directly nor converting it into csv
mnist=tf.keras.datasets.mnist

#now we are  going to split training and testing data
(x_train,y_train),(x_test,y_test)=mnist.load_data()
x_train=tf.keras.utils.normalize(x_train,axis=1)
x_test=tf.keras.utils.normalize(x_test,axis=1)

model=tf.keras.models.Sequential()
#adds layer this flattens certain input shapes it doesn't forms a grid
model.add(tf.keras.layers.Flatten(input_shape=(28,28)))
#dense and most basic layer where all the neural networks work
model.add(tf.keras.layers.Dense(128,activation='relu'))
model.add(tf.keras.layers.Dense(128,activation='relu'))
#gives probability
model.add(tf.keras.layers.Dense(10,activation='softmax'))

model.compile(optimizer="adam",loss='sparse_categorical_crossentropy',metrics=['accuracy'])
#compiled model is now needed to fit
model.fit(x_train,y_train,epochs=3)

model.save('handwritten.keras')

loss,accuracy=model.evaluate(x_test,y_test)
print(loss)
print(accuracy)