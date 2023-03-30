import pandas as pd
import numpy as np
import pickle
import re
import os
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_addons as tfa
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPool2D, Dense, Flatten, Dropout
from tensorflow.keras.layers import Input, Concatenate, InputLayer, UpSampling2D, Conv2DTranspose, Resizing
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow_graphics.math.math_helpers import cartesian_to_spherical_coordinates, spherical_to_cartesian_coordinates

# CREATING THE DATA GENERATORS
def get_dataset(batch_size=512, with_test=True, random_seed=10, target_size=(300,300), path="/Users/varshini/Desktop/HARVARD/SPRING'22/MIT9.60-Human Vision/Project/Data/Image_Data"):

    if with_test:
        split = 0.05
        subset = 'validation'
    else:
        split = 0.8
        subset = 'training'
        
    # DATA GENERATORS
    datagen = ImageDataGenerator(validation_split=split,
                                rescale=1./255,
                                rotation_range=np.random.choice([45, 90, 135]),
                                width_shift_range = np.random.choice(range(10,100)),
                                height_shift_range =np.random.choice(range(10,100)),
                                shear_range =np.random.choice(np.linspace(10,100,8))
                                )


    directory = path

    if with_test:
        test_data = datagen.flow_from_directory(
            directory,target_size=target_size,
            color_mode='rgb', class_mode='categorical',
            batch_size=batch_size, shuffle=True,
            seed=random_seed, subset=subset
        )
        return test_data
    
    else:
        train_data = datagen.flow_from_directory(
            directory,target_size=target_size,
            color_mode='rgb', class_mode='categorical',
            batch_size=batch_size, shuffle=True,
            seed=random_seed, subset=subset
        )

        return train_data

    
#--------------------------------------------------------------------------------------------------------------------

# POLAR TRANSFORM CUSTOM LAYER

# #squeeze -                            Remove axis 0 as it is 1 for batch size of 1
# #transpose -                          To get the batch size axis first then the images, required for the map_fn as it loops over axis 0
# #reshape -                            To the image of shape (w,h) as 1 in axis[2] to make it (w,h,1) for conversion to rgb
# #grayscale_to_rgb -                   Convert grayscale image to RGB, the output is of shape (w,h,3)
# #cartesian_to_spherical_coordinates - Convert image to polar coordinate system, output shape is (w,h,3)
# #rgb_to_grayscale -                   Convert back to grayscale, output shape is (w,h,1)
# #squeeze -                            Remove the last axis as it was not present initally, output shape is (w,h)
# #reshape -                            Add one to the begining to give the same shape as the batch, output shape is (1,w,h)


class LogPolarLayer(tf.keras.layers.Layer):
  def __init__(self, shape=(1,150,150,64)):
    super(LogPolarLayer, self).__init__()
    self.shape = shape

  def call(self, inputs):
    self.inputs = tf.transpose(tf.squeeze(inputs, axis=0))

    func = lambda output : tf.squeeze(tf.image.rgb_to_grayscale(cartesian_to_spherical_coordinates(tf.image.grayscale_to_rgb(
                                tf.reshape(output, shape=(self.shape[1], self.shape[2], 1))))), axis=-1)

    output_e = tf.map_fn(func, self.inputs)
    output_e = tf.reshape(output_e, shape=self.shape)

    return output_e

    
    
#--------------------------------------------------------------------------------------------------------------------

# INVERSE POLAR TRANSFORM CUSTOM LAYER
    
# #squeeze -                            Remove axis 0 as it is 1 for batch size of 1
# #transpose -                          To get the batch size axis first then the images, required for the map_fn as it loops over axis 0
# #reshape -                            To the image of shape (w,h) as 1 in axis[2] to make it (w,h,1) for conversion to rgb
# #grayscale_to_rgb -                   Convert grayscale image to RGB, the output is of shape (w,h,3)
# #spherical_to_cartesian_coordinates - Convert image to cartesian coordinate system, output shape is (w,h,3)
# #rgb_to_grayscale -                   Convert back to grayscale, output shape is (w,h,1)
# #squeeze -                            Remove the last axis as it was not present initally, output shape is (w,h)
# #reshape -                            Add one to the begining to give the same shape as the batch, output shape is (1,w,h)


class InverseLogPolarLayer(tf.keras.layers.Layer):
  def __init__(self, shape=(1, 38, 38, 512)):
    super(InverseLogPolarLayer, self).__init__()
    self.shape = shape

  def call(self, inputs):
    self.inputs = tf.transpose(tf.squeeze(inputs, axis=0))

    func = lambda output : tf.squeeze(tf.image.rgb_to_grayscale(spherical_to_cartesian_coordinates(tf.image.grayscale_to_rgb(
                                tf.reshape(output, shape=(self.shape[1], self.shape[2], 1))))), axis=-1)

    output_e = tf.map_fn(func, self.inputs)
    output_e = tf.reshape(output_e, shape=self.shape)

    return output_e


    
    
#--------------------------------------------------------------------------------------------------------------------

# FINAL LP MODEL ARCHITECTURE

def lp_architecture():
    kernel_size = (9,9)
    strides1 = (3,3)
    strides2 = (2,2)

    input_layer = Input(shape =(300,300,3))
        
    conv1 = Conv2D(filters =64, kernel_size =3, padding ='same', activation='relu',
                  name='conv1')(input_layer)

    conv2 = Conv2D(filters =64, kernel_size =3, padding ='same', activation='relu',
                  name='conv2')(conv1)

    maxpool1 = MaxPool2D(pool_size =2, strides =2, padding ='same',
                        name='maxpool1')(conv2)

    lp = LogPolarLayer(shape=(1,150,150,64))(maxpool1)

    conv3 = Conv2D(filters =128, kernel_size =3, padding ='same', activation='relu',
                  name='conv3')(lp)

    conv4 = Conv2D(filters =128, kernel_size =3, padding ='same', activation='relu',
                  name='conv4')(conv3)

    maxpool2 = MaxPool2D(pool_size =2, strides =2, padding ='same',
                        name='maxpool2')(conv4)

    conv5 = Conv2D(filters =256, kernel_size =3, padding ='same', activation='relu',
                  name='conv5')(maxpool2)

    conv6 = Conv2D(filters =256, kernel_size =3, padding ='same', activation='relu',
                  name='conv6')(conv5)

    conv7 = Conv2D(filters =256, kernel_size =3, padding ='same', activation='relu',
                  name='conv7')(conv6)

    maxpool3 = MaxPool2D(pool_size =2, strides =2, padding ='same',
                        name='maxpool3')(conv7)

    conv8 = Conv2D(filters =512, kernel_size =3, padding ='same', activation='relu',
                  name='conv8')(maxpool3)

    conv9 = Conv2D(filters =512, kernel_size =3, padding ='same', activation='relu',
                  name='conv9')(conv8)

    conv10 = Conv2D(filters =512, kernel_size =3, padding ='same', activation='relu',
                   name='conv10')(conv9)

    # HERE THE INVERSE LOG POLAR OF THE FEATURE MAP
    inp = InverseLogPolarLayer((1, 38, 38, 512))(conv10)

    # Upscaling to get image input
    conv_up = Conv2DTranspose(128, kernel_size,strides = strides1, padding='same')(inp)
    conv_up = Conv2DTranspose(3, kernel_size,strides = strides2, padding='same')(conv_up)
    conv_up = Resizing(224,224)(conv_up)
    

    # VGG ARCHITECTURE FROM HERE
    vgg_conv1 = Conv2D(filters =64, kernel_size =3, padding ='same', activation='relu',
                  name='vgg_conv1')(conv_up)

    vgg_conv2 = Conv2D(filters =64, kernel_size =3, padding ='same', activation='relu',
                  name='vgg_conv2')(vgg_conv1)

    vgg_maxpool1 = MaxPool2D(pool_size =2, strides =2, padding ='same',
                        name='vgg_maxpool1')(vgg_conv2)

    vgg_conv3 = Conv2D(filters =128, kernel_size =3, padding ='same', activation='relu',
                  name='vgg_conv3')(vgg_maxpool1)

    vgg_conv4 = Conv2D(filters =128, kernel_size =3, padding ='same', activation='relu',
                  name='vgg_conv4')(vgg_conv3)

    vgg_maxpool2 = MaxPool2D(pool_size =2, strides =2, padding ='same',
                        name='vgg_maxpool2')(vgg_conv4)

    vgg_conv5 = Conv2D(filters =256, kernel_size =3, padding ='same', activation='relu',
                  name='vgg_conv5')(vgg_maxpool2)

    vgg_conv6 = Conv2D(filters =256, kernel_size =3, padding ='same', activation='relu',
                  name='vgg_conv6')(vgg_conv5)

    vgg_conv7 = Conv2D(filters =256, kernel_size =3, padding ='same', activation='relu',
                  name='vgg_conv7')(vgg_conv6)

    vgg_maxpool3 = MaxPool2D(pool_size =2, strides =2, padding ='same',
                        name='vgg_maxpool3')(vgg_conv7)

    vgg_conv8 = Conv2D(filters =512, kernel_size =3, padding ='same', activation='relu',
                  name='vgg_conv8')(vgg_maxpool3)

    vgg_conv9 = Conv2D(filters =512, kernel_size =3, padding ='same', activation='relu',
                  name='vgg_conv9')(vgg_conv8)

    vgg_conv10 = Conv2D(filters =512, kernel_size =3, padding ='same', activation='relu',
                   name='vgg_conv10')(vgg_conv9)

    vgg_maxpool4 = MaxPool2D(pool_size =2, strides =2, padding ='same',
                        name='vgg_maxpool4')(vgg_conv10)

    vgg_conv11 = Conv2D(filters =512, kernel_size =3, padding ='same', activation='relu',
                   name='vgg_conv11')(vgg_maxpool4)

    vgg_conv12 = Conv2D(filters =512, kernel_size =3, padding ='same', activation='relu',
                   name='vgg_conv12')(vgg_conv11)

    vgg_conv13 = Conv2D(filters =512, kernel_size =3, padding ='same', activation='relu',
                   name='vgg_conv13')(vgg_conv12)

    vgg_maxpool5 = MaxPool2D(pool_size =2, strides =2, padding ='same',
                        name='vgg_maxpool5')(vgg_conv13)

    vgg_flatten = Flatten(name='vgg_flatten')(vgg_maxpool5)

    vgg_dense1 = Dense(4096, activation='relu',name='vgg_dense1')(vgg_flatten)

    vgg_dense2 = Dense(4096, activation='relu', name='vgg_dense2')(vgg_dense1)

    vgg_output_layer = Dense(1000, activation='softmax', name='vgg_output')(vgg_dense2)

    return input_layer, vgg_output_layer
        



#--------------------------------------------------------------------------------------------------------------------

# SSD ARCHITECTURE
def ssd_architecture(out_layer_num=-1, input_shape=None, model_type = None):

    if input_shape==None:
        input_layer = Input(shape =(300,300,3))
    else:
        input_layer = Input(shape =input_shape)
        

    conv1 = Conv2D(filters =64, kernel_size =3, padding ='same', activation='relu',
                  name='conv1')(input_layer)

    conv2 = Conv2D(filters =64, kernel_size =3, padding ='same', activation='relu',
                  name='conv2')(conv1)

    maxpool1 = MaxPool2D(pool_size =2, strides =2, padding ='same',
                        name='maxpool1')(conv2)

    conv3 = Conv2D(filters =128, kernel_size =3, padding ='same', activation='relu',
                  name='conv3')(maxpool1)

    conv4 = Conv2D(filters =128, kernel_size =3, padding ='same', activation='relu',
                  name='conv4')(conv3)

    maxpool2 = MaxPool2D(pool_size =2, strides =2, padding ='same',
                        name='maxpool2')(conv4)

    conv5 = Conv2D(filters =256, kernel_size =3, padding ='same', activation='relu',
                  name='conv5')(maxpool2)

    conv6 = Conv2D(filters =256, kernel_size =3, padding ='same', activation='relu',
                  name='conv6')(conv5)

    conv7 = Conv2D(filters =256, kernel_size =3, padding ='same', activation='relu',
                  name='conv7')(conv6)

    maxpool3 = MaxPool2D(pool_size =2, strides =2, padding ='same',
                        name='maxpool3')(conv7)

    conv8 = Conv2D(filters =512, kernel_size =3, padding ='same', activation='relu',
                  name='conv8')(maxpool3)

    conv9 = Conv2D(filters =512, kernel_size =3, padding ='same', activation='relu',
                  name='conv9')(conv8)

    conv10 = Conv2D(filters =512, kernel_size =3, padding ='same', activation='relu',
                   name='conv10')(conv9)

    maxpool4 = MaxPool2D(pool_size =2, strides =2, padding ='valid',
                        name='maxpool4')(conv10)

    conv11 = Conv2D(filters =1024, kernel_size =3, padding ='same', activation='relu',
                   name='conv11')(maxpool4)

    conv12 = Conv2D(filters =1024, kernel_size =3, padding ='same', activation='relu',
                   name='conv12')(conv11)
    
    maxpool5 = MaxPool2D(pool_size =2, strides =2, padding ='same',
                        name='maxpool5')(conv12)

    conv13 = Conv2D(filters =512, kernel_size =3, padding ='same', activation='relu',
                   name='conv13')(maxpool5)

    maxpool6 = MaxPool2D(pool_size =2, strides =2, padding ='same',
                        name='maxpool6')(conv13)

    conv14 = Conv2D(filters =256, kernel_size =3, padding ='same', activation='relu',
                   name='conv14')(maxpool6)

    maxpool7 = MaxPool2D(pool_size =2, strides =2, padding ='same',
                        name='maxpool7')(conv14)

    conv15 = Conv2D(filters =256, kernel_size =3, padding ='same', activation='relu',
                   name='conv15')(maxpool7)

    maxpool8 = MaxPool2D(pool_size =2, strides =2, padding ='valid',
                        name='maxpool8')(conv15)

    conv16 = Conv2D(filters =256, kernel_size =3, padding ='same', activation='relu',
                   name='conv16')(maxpool8)
    
    flatten = Flatten(name='flatten')(conv16)

    dense1 = Dense(4096, activation='relu',name='dense1')(flatten)

    output_layer = Dense(1000, activation='softmax', name='output')(dense1)

    kernel_size = (9,9)
    strides1 = (3,3)
    strides2 = (2,2)
    if out_layer_num==1:
        if model_type=="decoder":
            conv10 = Conv2DTranspose(128, kernel_size,strides = strides1, padding='same')(conv10)
            conv10 = Conv2DTranspose(3, kernel_size,strides = strides2, padding='same')(conv10)
            conv10 = Resizing(224,224)(conv10)
        return input_layer, conv10
        
    elif out_layer_num==2:
        if model_type=="decoder":
            # NEEDS TO CHANGE
            conv12 = Conv2DTranspose(128, kernel_size,strides = strides1, padding='same')(conv12)
            conv12 = Conv2DTranspose(3, kernel_size,strides = strides2, padding='same')(conv12)
            conv12 = Resizing(224,224)(conv12)
        return input_layer, conv12
        
    elif out_layer_num==3:
        if model_type=="decoder":
            # NEEDS TO CHANGE
            conv13 = Conv2DTranspose(128, kernel_size,strides = strides1, padding='same')(conv13)
            conv13 = Conv2DTranspose(3, kernel_size,strides = strides2, padding='same')(conv13)
            conv13 = Resizing(224,224)(conv13)
        return input_layer, conv13
        
    else:
        return input_layer, output_layer
        

#--------------------------------------------------------------------------------------------------------------------

        
# ORIGINAL VGG ARCHITECTURE
def vgg_architecture():
    input_layer = Input(shape =(224,224,3))

    conv1 = Conv2D(filters =64, kernel_size =3, padding ='same', activation='relu',
                  name='conv1')(input_layer)

    conv2 = Conv2D(filters =64, kernel_size =3, padding ='same', activation='relu',
                  name='conv2')(conv1)

    maxpool1 = MaxPool2D(pool_size =2, strides =2, padding ='same',
                        name='maxpool1')(conv2)

    conv3 = Conv2D(filters =128, kernel_size =3, padding ='same', activation='relu',
                  name='conv3')(maxpool1)

    conv4 = Conv2D(filters =128, kernel_size =3, padding ='same', activation='relu',
                  name='conv4')(conv3)

    maxpool2 = MaxPool2D(pool_size =2, strides =2, padding ='same',
                        name='maxpool2')(conv4)

    conv5 = Conv2D(filters =256, kernel_size =3, padding ='same', activation='relu',
                  name='conv5')(maxpool2)

    conv6 = Conv2D(filters =256, kernel_size =3, padding ='same', activation='relu',
                  name='conv6')(conv5)

    conv7 = Conv2D(filters =256, kernel_size =3, padding ='same', activation='relu',
                  name='conv7')(conv6)

    maxpool3 = MaxPool2D(pool_size =2, strides =2, padding ='same',
                        name='maxpool3')(conv7)

    conv8 = Conv2D(filters =512, kernel_size =3, padding ='same', activation='relu',
                  name='conv8')(maxpool3)

    conv9 = Conv2D(filters =512, kernel_size =3, padding ='same', activation='relu',
                  name='conv9')(conv8)

    conv10 = Conv2D(filters =512, kernel_size =3, padding ='same', activation='relu',
                   name='conv10')(conv9)

    maxpool4 = MaxPool2D(pool_size =2, strides =2, padding ='same',
                        name='maxpool4')(conv10)

    conv11 = Conv2D(filters =512, kernel_size =3, padding ='same', activation='relu',
                   name='conv11')(maxpool4)

    conv12 = Conv2D(filters =512, kernel_size =3, padding ='same', activation='relu',
                   name='conv12')(conv11)

    conv13 = Conv2D(filters =512, kernel_size =3, padding ='same', activation='relu',
                   name='conv13')(conv12)

    maxpool5 = MaxPool2D(pool_size =2, strides =2, padding ='same',
                        name='maxpool5')(conv13)

    flatten = Flatten(name='flatten')(maxpool5)

    dense1 = Dense(4096, activation='relu',name='dense1')(flatten)

    dense2 = Dense(4096, activation='relu', name='dense2')(dense1)

    output_layer = Dense(1000, activation='softmax', name='output')(dense2)

    return input_layer, output_layer



#--------------------------------------------------------------------------------------------------------------------


# SET MODEL WEIGHTS
# Set weights to the current model using tensorflow VGG weights
def set_model_weights(model, model_type="custom", path = "HARVARD/SPRING'22/MIT9.60-Human Vision/Project/Notebooks/Model Weights/model_weights.h5"):
    
    if model_type=='custom':
        if os.path.exists(path):
            model.load_weights(path)
        else:
            weights_model = tf.keras.applications.vgg16.VGG16()
            counter = 0
            for idx, layers in enumerate(model.layers):
                model.layers[idx].trainable = np.true_divide 
                if layers.name[:4]=='vgg_':
                    model.layers[idx].set_weights = weights_model.layers[counter].get_weights()
                    counter+=1
                else:
                    continue
                
    else:
        print("Setting VGG weights from tensorflow")
        weights_model = tf.keras.applications.vgg16.VGG16()
        model.set_weights(weights_model.get_weights())
    try:
        del weights_model
    except:
        pass
    
    return model