from tensorflow.keras.models import load_model
import keras
import numpy as np
import matplotlib.pyplot as plt

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

#loading pre-trained model weights
model.load_weights("78000weights.h5")

def model_predict(img):
	img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	arr=model.predict(img.reshape(1,224,224,1)).reshape(26)
	mx=np.max(arr)
	for i in range(len(arr)):
		if mx==arr[i]:
			return i

	    	
import cv2
import sys

video_capture = cv2.VideoCapture(0)
x1 = 200
y1 = 200
x2 = 424
y2 = 424
count = 1

classes=np.zeros(26)
while(True):
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    frame = cv2.flip(frame,1)
    cv2.rectangle(frame,(x1, y1),(x2, y2),(255,0,0), 2)
    cv2.imshow("Video",frame)
    crop_img = frame[200:424,200:424]
    if(count%50==0):
    	count=1
    	mx=np.max(classes)
    	for i in range(classes.shape[0]):
    		if(classes[i]==mx):
    			print(chr(i+65))
    	classes=np.zeros(26)
    else:
    	classes[model_predict(crop_img)]+=1
    count+=1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()

