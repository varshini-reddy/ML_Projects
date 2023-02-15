from tensorflow.keras.models import load_model
import keras
import numpy as np
import matplotlib.pyplot as plt
import argparse
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras import layers

from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D,Dense,BatchNormalization,Dropout
from tensorflow.keras.layers import Flatten
from keras.regularizers import l2
from keras.datasets import cifar10
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator


model = Sequential()
model.add(Conv2D(32, kernel_size = (5, 5), activation='relu',kernel_regularizer=l2(0.001), input_shape=(224,224, 1)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Conv2D(64, kernel_size=(5,5), kernel_regularizer=l2(0.001),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Conv2D(64, kernel_size=(5,5), kernel_regularizer=l2(0.001),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Conv2D(96, kernel_size=(5,5), kernel_regularizer=l2(0.001),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Conv2D(32, kernel_size=(5,5), kernel_regularizer=l2(0.001),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(26, activation = 'softmax'))
model.compile(loss='sparse_categorical_crossentropy', optimizer="Adam", metrics=['accuracy'])

model.load_weights("78000weights.h5")

parser = argparse.ArgumentParser()

parser.add_argument("-ip", "--image_path", 
	help="The base path of the directory", 
	default="./dataset/asl_alphabet_train/asl_alphabet_train/")

args=parser.parse_args()

def resize_keep_aspect_ratio(img_path):
    img=cv2.imread(img_path,0)
    img1=np.pad(img,((0,0),((img.shape[0]-img.shape[1])/2,(img.shape[0]-img.shape[1])/2)),'constant', constant_values =0)
    img2=cv2.resize(img1,(224,224))
    return img2

def model_predict(img):
	# img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	arr=model.predict_classes(img.reshape(1,224,224,1))
    print("Prediction for the input image is: ")
	print(arr)
    # mx=np.max(arr)
	# for i in range(len(arr)):
		# if mx==arr[i]:
			# print("the sign is "+chr(i+65)))
            
img_path=args.image_path
resized=resize_keep_aspect_ratio(img_path)
model_predict(resized)

