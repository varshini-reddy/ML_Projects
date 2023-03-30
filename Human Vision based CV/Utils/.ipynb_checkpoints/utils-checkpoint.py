
import pandas as pd
import numpy as np
import pickle
import re
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_addons as tfa
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPool2D, Dense, Flatten, Dropout
from tensorflow.keras.layers import Input, Concatenate, InputLayer
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# CREATING THE DATA GENERATORS
def get_dataset(batch_size=512, with_test=True, random_seed=10):
    blurred_images1 = lambda input_images : tfa.image.gaussian_filter2d(input_images, filter_shape=15,
                                            sigma=200)

    blurred_images2 = lambda input_images : tfa.image.gaussian_filter2d(input_images, filter_shape=10,
                                            sigma=100)

    blurred_images3 = lambda input_images : tfa.image.gaussian_filter2d(input_images, filter_shape=8,
                                            sigma=50)

    blurred_images4 = lambda input_images : tfa.image.gaussian_filter2d(input_images, filter_shape=5,
                                            sigma=10)

    if with_test:
        split = 0.05
        subset = 'validation'
    else:
        split = 0.95
        subset = 'training'
        
    # DATA GENERATORS
    datagen1 = ImageDataGenerator(validation_split=split,rescale=1./255,preprocessing_function=blurred_images1)
    datagen2 = ImageDataGenerator(validation_split=split,rescale=1./255,preprocessing_function=blurred_images2)
    datagen3 = ImageDataGenerator(validation_split=split,rescale=1./255,preprocessing_function=blurred_images3)
    datagen4 = ImageDataGenerator(validation_split=split,rescale=1./255,preprocessing_function=blurred_images4)
    datagen5 = ImageDataGenerator(validation_split=split,rescale=1./255)

    directory = "/Users/varshini/Desktop/HARVARD/SPRING'22/MIT6.819-Advances in Computer Vision/Project/Data/Image_Data"

    train_data1 = datagen1.flow_from_directory(
        directory,target_size=(224, 224),
        color_mode='rgb', class_mode='categorical',
        batch_size=batch_size, shuffle=True,
        seed=random_seed, subset=subset
    )

    train_data2 = datagen2.flow_from_directory(
        directory,target_size=(224, 224),
        color_mode='rgb', class_mode='categorical',
        batch_size=batch_size, shuffle=True,
        seed=random_seed, subset=subset
    )

    train_data3 = datagen3.flow_from_directory(
        directory,target_size=(224, 224),
        color_mode='rgb', class_mode='categorical',
        batch_size=batch_size, shuffle=True,
        seed=random_seed, subset=subset
    )

    train_data4 = datagen4.flow_from_directory(
        directory,target_size=(224, 224),
        color_mode='rgb', class_mode='categorical',
        batch_size=batch_size, shuffle=True,
        seed=random_seed, subset=subset
    )

    train_data5 = datagen5.flow_from_directory(
        directory,target_size=(224, 224),
        color_mode='rgb', class_mode='categorical',
        batch_size=batch_size, shuffle=True,
        seed=random_seed, subset=subset
    )

    return (train_data1, train_data2, train_data3, train_data4, train_data5)





# CUSTOM MODEL ARCHITECTURE
def get_alternate_architecture(name="blurred_model"):
    input_layer = Input(shape =(224,224,3), name='main_input')

    conv1 = Conv2D(filters =64, kernel_size =3, padding ='same', activation='relu',
                  name='conv1')(input_layer)

    conv2 = Conv2D(filters =64, kernel_size =3, padding ='same', activation='relu',
                  name='conv2')(conv1)

    maxpool1 = MaxPool2D(pool_size =2, strides =2, padding ='same',
                        name='maxpool1')(conv2)

    # -----BLUR BLOCK 1----
    input_pos1 = Input(shape =(224,224,3), name='pos1_input')
    pos1 = Conv2D(filters = 64, kernel_size = 5, activation='relu', padding='same', name='pos1_conv1')(input_pos1)
    pos1 = Conv2D(filters = 64, kernel_size = 3, activation='relu',padding='same', name='pos1_conv2')(pos1)
    pos1 = MaxPool2D(pool_size = 2, strides = 2, padding='valid', name='pos1_maxpool')(pos1)
    # -----END OF BLUR BLOCK 1----

    concat1 = Concatenate(axis=3)([pos1, maxpool1])

    conv3 = Conv2D(filters =128, kernel_size =3, padding ='same', activation='relu',
                  name='conv3')(concat1)

    conv4 = Conv2D(filters =128, kernel_size =3, padding ='same', activation='relu',
                  name='conv4')(conv3)

    maxpool2 = MaxPool2D(pool_size =2, strides =2, padding ='same',
                        name='maxpool2')(conv4)

    # -----BLUR BLOCK 2----
    input_pos2 = Input(shape =(224,224,3), name='pos2_input')
    pos2 = Conv2D(filters = 64, kernel_size = 7,strides = 2, activation='relu', padding='same', name='pos2_conv1')(input_pos2)
    pos2 = Conv2D(filters = 64, kernel_size = 5,strides = 2, activation='relu', padding='same', name='pos2_conv2')(pos2)
    pos2 = Conv2D(filters = 64, kernel_size = 3, activation='relu',padding='same', name='pos2_conv3')(pos2)
    pos2 = MaxPool2D(pool_size = 4, strides = 1, padding='same', name='pos2_maxpool')(pos2)
    # -----END OF BLUR BLOCK 2----

    concat2 = Concatenate(axis=3)([pos2, maxpool2])

    conv5 = Conv2D(filters =256, kernel_size =3, padding ='same', activation='relu',
                  name='conv5')(concat2)

    conv6 = Conv2D(filters =256, kernel_size =3, padding ='same', activation='relu',
                  name='conv6')(conv5)

    conv7 = Conv2D(filters =256, kernel_size =3, padding ='same', activation='relu',
                  name='conv7')(conv6)

    maxpool3 = MaxPool2D(pool_size =2, strides =2, padding ='same',
                        name='maxpool3')(conv7)

    # -----BLUR BLOCK 3----
    input_pos3 = Input(shape =(224,224,3), name='pos3_input')
    pos3 = Conv2D(filters = 128, kernel_size = 7,strides = 2, activation='relu', padding='same', name='pos3_conv1')(input_pos3)
    pos3 = Conv2D(filters = 128, kernel_size = 5,strides = 2, activation='relu', padding='same', name='pos3_conv2')(pos3)
    pos3 = Conv2D(filters = 128, kernel_size = 3,strides = 2, activation='relu',padding='same', name='pos3_conv3')(pos3)
    pos3 = MaxPool2D(pool_size = 4, strides = 1, padding='same', name='pos3_maxpool')(pos3)
    # -----END OF BLUR BLOCK 3----

    concat3 = Concatenate(axis=3)([pos3, maxpool3])


    conv8 = Conv2D(filters =512, kernel_size =3, padding ='same', activation='relu',
                  name='conv8')(concat3)

    conv9 = Conv2D(filters =512, kernel_size =3, padding ='same', activation='relu',
                  name='conv9')(conv8)

    conv10 = Conv2D(filters =512, kernel_size =3, padding ='same', activation='relu',
                   name='conv10')(conv9)

    maxpool4 = MaxPool2D(pool_size =2, strides =2, padding ='same',
                        name='maxpool4')(conv10)

    # -----BLUR BLOCK 4----
    input_pos4 = Input(shape =(224,224,3), name='pos4_input')
    pos4 = Conv2D(filters = 256, kernel_size = 7,strides = 2, activation='relu', padding='same', name='pos4_conv1')(input_pos4)
    pos4 = Conv2D(filters = 256, kernel_size = 5,strides = 2, activation='relu', padding='same', name='pos4_conv2')(pos4)
    pos4 = Conv2D(filters = 256, kernel_size = 3,strides = 2, activation='relu',padding='same', name='pos4_conv3')(pos4)
    pos4 = MaxPool2D(pool_size = 4, strides = 2, padding='same', name='pos4_maxpool')(pos4)
    # -----END OF BLUR BLOCK 4----

    concat4 = Concatenate(axis=3)([pos4, maxpool4])

    conv11 = Conv2D(filters =512, kernel_size =3, padding ='same', activation='relu',
                   name='conv11')(concat4)

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

    # Getting the model
    model = Model(inputs=(input_layer,input_pos1,input_pos2,input_pos3,input_pos4), outputs=output_layer, name=name)
    return model


# ORIGINAL ARCHITECTURE OF VGG

def get_original_architecture():
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

    model = Model(inputs=input_layer, outputs=output_layer)

    return model


# Set weights to the current model using tensorflow VGG weights
def set_model_weights(model, model_type="blur"):
    weights_model = tf.keras.applications.vgg16.VGG16()
    
    if model_type=="original":
        model.set_weights(weights_model.get_weights())
    
    else:
        counter = 0
        for idx, blur_layers in enumerate(model.layers):

            if blur_layers.name[0]=='p':
                continue
            elif blur_layers.name[:6]=="concat":
                continue
            else:
                model.layers[idx].set_weights = weights_model.layers[counter].get_weights()
                model.layers[idx].trainable = False 
                counter+= 1

    # Delete the tensorflow version of VGG16
    try:
        del weights_model
    except:
        pass
    
    return model

# GENERATE ADVERSARIAL SAMPLES
@tf.function
def onestep_pgd_linf(model, X, y, epsilon, alpha, delta):
    with tf.GradientTape() as tape:
        tape.watch(delta)
        loss = tf.keras.losses.CategoricalCrossentropy(
            # from_logits=True,
            reduction=tf.keras.losses.Reduction.NONE
        )(y, model(X + delta))

    delta = tf.clip_by_value(delta + alpha*tf.sign(tape.gradient(loss, delta)), X-epsilon, X+epsilon)

    return delta

# Full run – import this
def pgd_linf(model, X, y, epsilon, alpha, num_iter):
    """ Construct FGSM adversarial examples on the examples X"""
    delta = tf.zeros_like(X)
    for t in range(num_iter):
        delta = onestep_pgd_linf(model, X, y, epsilon, alpha, delta)
    return delta


###############
###   L2   ####
###############

# Helper
def norm(Z):
    """Compute norms over all but the first dimension"""
    return tf.norm(tf.reshape(Z, (Z.shape[0], -1)), axis=1)

########### ROBUSTIFICATION ##############
def single_pgd_step_robust(model, X, y, alpha, delta):
    with tf.GradientTape() as tape:
        tape.watch(delta)
        loss = tf.keras.losses.MeanSquaredError(
            reduction=tf.keras.losses.Reduction.NONE
        )(y, model(X + delta)) # comparing to robust model representation layer

    grad = tape.gradient(loss, delta)
    normgrad = tf.reshape(norm(grad), (-1, 1, 1, 1))
    delta -= alpha*grad / (normgrad + 1e-10) # normalized gradient step
    delta = tf.math.minimum(tf.math.maximum(delta, -X), 1-X) # clip X+delta to [0,1]
    return delta, loss

def pgd_l2_robust(model, X, y, alpha, num_iter, epsilon=0, example=False):
    delta = tf.zeros_like(X)
    loss = 0
    fn = tf.function(single_pgd_step_robust)
    for t in range(num_iter):
      delta, loss = fn(model, X, y, alpha, delta)
    # Prints out loss to evaluate if it's actually learning (currently broken)
    if example:
        print(f'{num_iter} iterations, final MSE {loss}')
    return delta

# PGD L2 for Non-Robustifying #
def single_pgd_step_nonrobust(model, X, y, alpha, epsilon, delta):
    with tf.GradientTape() as tape:
        tape.watch(delta)
        loss = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True,
            reduction=tf.keras.losses.Reduction.NONE # Use no aggregation - will give gradient separtely for each ex.
            )(y, model(X + delta)) # comparing to label for original data point
    grad = tape.gradient(loss, delta) #tape.gradient(loss, delta)

    # equivalent to delta += alpha*grad / norm(grad), just for batching
    normgrad = tf.reshape(norm(grad), (-1, 1, 1, 1))
    # changed from plus to minus b/c trying to minimize with non-robust
    z = delta - alpha * (grad / (normgrad + 1e-10))
    normz = tf.reshape(norm(z), (-1, 1, 1, 1))
    delta = epsilon * z / (tf.math.maximum(normz, epsilon) + 1e-10)
    return delta, loss

def pgd_l2_nonrobust(model, X, y, alpha, num_iter, epsilon=0, example=False):
    fn = tf.function(single_pgd_step_nonrobust)
    delta = tf.zeros_like(X)
    loss = 0
    for t in range(num_iter):
        delta, loss = fn(model, X, y, alpha, epsilon, delta)

    if example:
        print(f'{num_iter} iterations, final MSE {loss}')
    return delta

# PGD L2 for Adversarial Examples #
def single_pgd_step_adv(model, X, y, alpha, epsilon, delta):
    with tf.GradientTape() as tape:
        tape.watch(delta)
        loss = tf.keras.losses.CategoricalCrossentropy(
            from_logits=False,
            reduction=tf.keras.losses.Reduction.NONE # Use no aggregation - will give gradient separtely for each ex.
            )(y, model(X + delta)) # comparing to label for original data point
    grad = tape.gradient(loss, delta)

    normgrad = tf.reshape(norm(grad), (-1, 1, 1, 1))
    z = delta + alpha * (grad / (normgrad + 1e-10))

    normz = tf.reshape(norm(z), (-1, 1, 1, 1))
    delta = epsilon * z / (tf.math.maximum(normz, epsilon) + 1e-10)
    return delta, loss

def adversarial_examples(model, X, y, alpha, num_iter, epsilon=0, example=False):
    fn = tf.function(single_pgd_step_adv)
    delta = tf.zeros_like(X)
    loss = 0
    for t in range(num_iter):
        delta, loss = fn(model, X, y, alpha, epsilon, delta)

    if example:
        print(f'{num_iter} iterations, final MSE {loss}')
    return delta

