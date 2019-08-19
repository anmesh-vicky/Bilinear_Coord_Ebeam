#!/usr/bin/env python
# coding: utf-8

           '''
Coord Conv Bilinear to predict the 2D crystal structure from sem data

          '''

#### import keras libraries
import random
import numpy as np
import keras
import keras.layers
import keras.applications
import keras.backend
import keras.preprocessing.image
import keras.utils
import tensorflow as tf
from coord import CoordinateChannel2D
from keras.layers import Input, Conv2D, Flatten, Softmax,Dropout,AveragePooling2D,Dense,BatchNormalization,MaxPooling2D
from keras.applications.inception_v3 import InceptionV3
from keras.initializers import glorot_normal
from keras.optimizers import Adam

def outer_product(x):
    """
    calculate outer-products of 2 tensors

        args 
            x
                list of 2 tensors
                , assuming each of which has shape = (size_minibatch, total_pixels, size_filter)
    """
    return keras.backend.batch_dot(x[0], x[1], axes=[1,1]) / x[0].get_shape().as_list()[1] 

def signed_sqrt(x):
    """
    calculate element-wise signed square root

        args
            x
                a tensor
    """
    return keras.backend.sign(x) * keras.backend.sqrt(keras.backend.abs(x) + 1e-9)

def L2_norm(x, axis=-1):
    """
    calculate L2-norm

        args 
            x
                a tensor
    """
    return keras.backend.l2_normalize(x, axis=axis)


def build_model(
    size_heigth=128
    ,size_width=128
    ,no_class=6
    ,no_last_layer_backbone=-3
    
    ,name_optimizer="Adam"
    ,rate_learning=1e-4
    ,rate_decay_learning=0.0
    ,rate_decay_weight=0.01
    
    ,name_initializer="glorot_normal"
    ,name_activation_logits="softmax"
    ,name_loss="categorical_crossentropy"

    ,flg_debug=True
    ,**kwargs
):
    
    keras.backend.clear_session()
    
    print("-------------------------------")
    print("parameters:")
    for key, val in locals().items():
        if not val == None and not key == "kwargs":
            print("\t", key, "=",  val)
    print("-------------------------------")
    
    ### 
    ### load pre-trained model
    ###
#     tensor_input = keras.layers.Input(shape=[size_heigth,size_width,3])
#     initial_model = VGG16(weights='imagenet', include_top=False)
    tensor_input = Input(shape=(128, 128, 1))
    out = CoordinateChannel2D()(tensor_input)
    x = Conv2D(50,5,padding='same',activation="relu")(out)
    x = Conv2D(32,5,padding='same',activation="relu")(x)
    x = Conv2D(32,5,padding='same',activation="relu")(x)
    x = Conv2D(32,5,padding='same',activation="relu")(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D()(x)
	#	x = Conv2D(40,self.kernel_size,padding='valid',activation="relu")(x)
# 		x = Dropout(self.dropout)(x, training=True)
    x = Conv2D(32,5,padding='same',activation="relu")(x)
    x = Conv2D(32,5,padding='same',activation="relu")(x)
    x = Conv2D(32,5,padding='same',activation="relu")(x)
    x = Conv2D(32,5,padding='same',activation="relu")(x)
    x = Conv2D(32,5,padding='same',activation="relu")(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D()(x)
    x = Conv2D(32,5,padding='same',activation="relu")(x)
    x = Conv2D(32,5,padding='same',activation="relu")(x)
    x = Conv2D(32,5,padding='same',activation="relu")(x)
    x = Conv2D(32,5,padding='same',activation="relu")(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D()(x)


    ### 
    ### bi-linear pooling
    ###

    # extract features from detector
    x_detector = x
    shape_detector = x
    print(type(shape_detector[1]))
    if flg_debug:
        print("shape_detector : {}".format(shape_detector))

    # extract features from extractor , same with detector for symmetry DxD model
    shape_extractor = shape_detector
    x_extractor = x_detector
    if flg_debug:
        print("shape_extractor : {}".format(shape_extractor))
        
    
    # rehape to (minibatch_size, total_pixels, filter_size)
    x_detector = keras.layers.Reshape(
            [
                16*16,32
            ])(x_detector)
    if flg_debug:
        print("x_detector shape after rehsape ops : {}".format(x_detector.shape))
        
    x_extractor = keras.layers.Reshape(
            [
               16*16,32
            ]
        )(x_extractor)
    if flg_debug:
        print("x_extractor shape after rehsape ops : {}".format(x_extractor.shape))
        
        
    # outer products of features, output shape=(minibatch_size, filter_size_detector*filter_size_extractor)
    x = keras.layers.Lambda(outer_product)(
        [x_detector, x_extractor]
    )
    if flg_debug:
        print("x shape after outer products ops : {}".format(x.shape))
        
        
    # rehape to (minibatch_size, filter_size_detector*filter_size_extractor)
    x = keras.layers.Reshape([32*32])(x)
    if flg_debug:
        print("x shape after rehsape ops : {}".format(x.shape))
        
        
    # signed square-root 
    x = keras.layers.Lambda(signed_sqrt)(x)
    if flg_debug:
        print("x shape after signed-square-root ops : {}".format(x.shape))
        
    # L2 normalization
    x = keras.layers.Lambda(L2_norm)(x)
    if flg_debug:
        print("x shape after L2-Normalization ops : {}".format(x.shape))



    ### 
    ### attach FC-Layer
    ###
    x = Dense(128,activation='relu')(x)
#     x = Dense(128,activation='relu')(x)
    x = Dropout(0.2)(x, training=True)
    x = Dense(6)(x)
    x = Softmax(axis=-1)(x)
    
    if flg_debug:
        print("x shape after Dense ops : {}".format(x.shape))
    tensor_prediction = keras.layers.Activation(name_activation_logits)(x)
    if flg_debug:
        print("prediction shape : {}".format(tensor_prediction.shape))

        

    ### 
    ### compile model
    ###
    model_bilinear = keras.models.Model(
                        inputs=[tensor_input]
                        , outputs=[tensor_prediction]
                    )
    
    
    # fix pre-trained weights

        
        
    # define optimizers
#     opt_adam = keras.optimizers.adam(
#                     lr=rate_learning
#                     , decay=rate_decay_learning
#                 )
#     opt_rms = keras.optimizers.RMSprop(
#                     lr=rate_learning
#                     , decay=rate_decay_learning
#                 )
#     opt_sgd = keras.optimizers.SGD(
#                     lr=rate_learning
#                     , decay=rate_decay_learning
#                     , momentum=0.9
#                     , nesterov=False
#                 )
#     optimizers ={
#         "adam":opt_adam
#         ,"rmsprop":opt_rms
#         ,"sgd":opt_sgd
#     }
    
    model_bilinear.compile(
        loss=name_loss
        , optimizer=Adam(lr=rate_learning, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        , metrics=['accuracy']
    )
    
    
    
    if flg_debug:
        model_bilinear.summary()
    
    return model_bilinear


# Model Build


Model=build_model()

### keras libraries

from keras.callbacks import ModelCheckpoint,TensorBoard,ReduceLROnPlateau
from dataLoader import data_loader

### save weight files

filepath = "weights-imp-{epoch:02d}-{val_acc:.2f}.hdf5"

#load any previous weight files \
#Model.load_weights("weights-imp-20-0.99.hdf5")

### load data for training, test and validatin

dataLoader = data_loader('./data/train/', './data/val/', './data/test/')
train_generator, validation_generator, test_generator = dataLoader.load_images()
checkpoint_1 = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
checkpoint_2 =  TensorBoard(log_dir='logs/{}', histogram_freq=0, write_graph=True, write_images=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=5, cooldown=0)
callbacks_list = [checkpoint_2,checkpoint_1,reduce_lr]
#NO. of Steps per epoch 
train_steps = np.math.ceil(train_generator.samples / train_generator.batch_size)
val_steps = np.math.ceil(validation_generator.samples / validation_generator.batch_size)
test_steps = np.math.ceil(test_generator.samples / test_generator.batch_size)

# Fit model

history = Model.fit_generator(train_generator, steps_per_epoch =train_steps, epochs = 20,callbacks=callbacks_list,validation_data=validation_generator,validation_steps = val_steps)
print('--------Test data--------')

x = Model.evaluate_generator(test_generator, steps = test_steps,verbose = 1)
print(x)


#save log file after training
import pandas as pd
acc_log = history.history['acc']
val_acc_log = history.history['val_acc']
loss_log = history.history['loss']
val_loss_log = history.history['val_loss']
acc_log = np.array(acc_log)
val_acc_log = np.array(val_acc_log)
loss_log = np.array(loss_log)
val_loss_log = np.array(val_loss_log)
mat = np.vstack((loss_log, acc_log, val_loss_log, val_acc_log))
mat = np.transpose(mat)
dataframe1 = pd.DataFrame(data=mat)
dataframe1.to_csv('save_log.csv', sep=',', header=False, float_format='%.7f', index=False)





