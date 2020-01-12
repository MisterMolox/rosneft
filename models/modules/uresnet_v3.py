import tensorflow.compat.v1 as tf
from keras.models import Model
from keras import backend as K
from keras.layers import Input, Conv2D, Conv2DTranspose, MaxPooling2D, concatenate, UpSampling2D, BatchNormalization, Add, Reshape
from keras.layers.core import Activation
from dice_metric import dice, dice_border, dice_coef, dice_coef_loss, bce_dice_coef_loss
from keras.optimizers import Adam
import numpy as np

# U-Net + ResNet 34
def Conv2D_34skipblock(filters, strides=(3, 3), activation="relu", padding="same"):
    assert(isinstance(strides, tuple))
    assert(2 == len(strides))

    def f(x):
        conv1 = Conv2D(filters, strides, padding=padding)(x)
        batch_norm1 = BatchNormalization()(conv1)
        act1 = Activation(activation=activation)(batch_norm1)
        conv2 = Conv2D(filters, strides, padding=padding)(act1)
        batch_norm2 = BatchNormalization()(conv2)
        act2 = Activation(activation=activation)(batch_norm2)
        add = Add()([x, act2])
        return add
    return f

def uresnet(skip_block, image_rows, image_cols, img_channels=1, optimizer=Adam(lr=3e-4)):

    inputs = Input((image_rows, image_cols, img_channels))
    conv0 = Conv2D(32, (3, 3), padding='same')(inputs)
    batch_norm0 = BatchNormalization()(conv0)
    act0 = Activation(activation='relu')(batch_norm0)

    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(act0)
    conv1 = skip_block(32, (3, 3), activation='relu', padding='same')(conv1)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    conv1 = skip_block(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = skip_block(64, (3, 3), activation='relu', padding='same')(conv2)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    conv2 = skip_block(64, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = skip_block(128, (3, 3), activation='relu', padding='same')(conv3)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    conv3 = skip_block(128, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = skip_block(256, (3, 3), activation='relu', padding='same')(conv4)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
    conv4 = skip_block(256, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = skip_block(512, (3, 3), activation='relu', padding='same')(conv5)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)
    conv5 = skip_block(512, (3, 3), activation='relu', padding='same')(conv5)

    up6 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv4], axis=3)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
    conv6 = skip_block(256, (3, 3), activation='relu', padding='same')(conv6)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)
    conv6 = skip_block(256, (3, 3), activation='relu', padding='same')(conv6)
    
    up7 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv3], axis=3)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
    conv7 = skip_block(128, (3, 3), activation='relu', padding='same')(conv7)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)
    conv7 = skip_block(128, (3, 3), activation='relu', padding='same')(conv7)

    up8 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv2], axis=3)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
    conv8 = skip_block(64, (3, 3), activation='relu', padding='same')(conv8)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)
    conv8 = skip_block(64, (3, 3), activation='relu', padding='same')(conv8)

    up9 = concatenate([UpSampling2D(size=(2, 2))(conv8), conv1], axis=3)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
    conv9 = skip_block(32, (3, 3), activation='relu', padding='same')(conv9)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)
    conv9 = skip_block(32, (3, 3), activation='relu', padding='same')(conv9)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    model = Model(inputs=[inputs], outputs=[conv10])
        
    model.compile(optimizer=optimizer, loss=bce_dice_coef_loss, metrics=[dice_coef])
    model.summary()

    return model

def uresnet2(image_rows, image_cols, img_channels=1, optimizer=Adam(lr=3e-4)):
  def batch_block(activation=True):
    def f(x):
      batch_norm = BatchNormalization()(x)
      if activation: 
        batch_norm = Activation(activation='relu')(batch_norm)
      return batch_norm
    return f

  def conv_block(filters_num):
    def f(x):
      batch = batch_block()(x)
      conv = Conv2D(filters_num, (3, 3), padding='same')(batch)
      return conv
    return f

  def conv_block_transp(filters_num):
    def f(x):
      batch = batch_block()(x)
      conv = Conv2DTranspose(filters_num, (3, 3), padding='same')(batch)
      return conv
    return f

  def stem(filters_num):
    def f(x):
      conv = Conv2D(filters_num, (3, 3), padding='same')(x)
      conv = conv_block(filters_num)(conv)
      shortcut = Conv2D(filters_num, (3, 3), padding='same')(x)
      shortcut = batch_block(False)(shortcut)
      return Add()([conv, shortcut])
    return f

  def residual_block(filters_num):
    def f(x):
      conv = conv_block(filters_num)(x)
      conv = conv_block(filters_num)(conv)
      shortcut = Conv2D(filters_num, (3, 3), padding='same')(x)
      shortcut = batch_block(False)(shortcut)
      return Add()([conv, shortcut])
    return f

  def residual_block_transp(filters_num):
    def f(x):
      conv = conv_block_transp(filters_num)(x)
      conv = conv_block(filters_num)(conv)
      shortcut = Conv2D(filters_num, (3, 3), padding='same')(x)
      shortcut = batch_block(False)(shortcut)
      return Add()([conv, shortcut])
    return f

  filters = [32, 64, 128, 256, 512]
  #ROOT
  inputs = Input((image_rows, image_cols, img_channels))
  root = stem(filters[0])(inputs)
  
  #ENCODER
  res1 = residual_block(filters[0])(root)
  pool1 = MaxPooling2D(pool_size=(2, 2))(res1)
  
  res2 = residual_block(filters[1])(pool1)
  pool2 = MaxPooling2D(pool_size=(2, 2))(res2)
  
  res3 = residual_block(filters[2])(pool2)
  pool3 = MaxPooling2D(pool_size=(2, 2))(res3)
  
  res4 = residual_block(filters[3])(pool3)
  pool4 = MaxPooling2D(pool_size=(2, 2))(res4)
  
  #BRIDGE
  res5 = residual_block(filters[4])(pool4)
  res5 = residual_block(filters[4])(res5)
  
  #DECODER
  up1 = concatenate([UpSampling2D(size=(2, 2))(res5), res4], axis=3)
  resT1 = residual_block_transp(filters[3])(up1)
  
  up2 = concatenate([UpSampling2D(size=(2, 2))(resT1), res3], axis=3)
  resT2 = residual_block_transp(filters[2])(up2)

  up3 = concatenate([UpSampling2D(size=(2, 2))(resT2), res2], axis=3)
  resT3 = residual_block_transp(filters[1])(up3)

  up4 = concatenate([UpSampling2D(size=(2, 2))(resT3), res1], axis=3)
  resT4 = residual_block_transp(filters[0])(up4)
  
  #OUTPUT
  outputs = Conv2D(1, (1, 1), padding="same", activation="sigmoid")(resT4)
  outputs = Reshape((384, 384, 2, 1))(concatenate([outputs, outputs], axis=3))
  
  model = Model(inputs=[inputs], outputs=[outputs])  
  model.compile(optimizer=optimizer, loss=bce_dice_coef_loss, metrics=[dice_coef])
  return model

def uresnet3(image_rows, image_cols, img_channels=1, optimizer=Adam(lr=3e-4)):
  def batch_block(activation=True):
    def f(x):
      batch_norm = BatchNormalization()(x)
      if activation: 
        batch_norm = Activation(activation='relu')(batch_norm)
      return batch_norm
    return f

  def conv_block(striders, filters_num):
    def f(x):
      batch = batch_block()(x)
      conv = Conv2D(filters_num, striders, padding='same')(batch)
      return conv
    return f

  def conv_block_transp(striders, filters_num):
    def f(x):
      batch = batch_block()(x)
      conv = Conv2DTranspose(filters_num, striders, padding='same')(batch)
      return conv
    return f

  def stem(filters_num):
    def f(x):
      conv = Conv2D(filters_num, (7, 7), padding='same')(x)
      conv = conv_block((3, 3), filters_num)(conv)
      shortcut = Conv2D(filters_num, (3, 3), padding='same')(x)
      shortcut = batch_block(False)(shortcut)
      return Add()([conv, shortcut])
    
    return f

  def residual_block(filters_num):
    def f(x):
      conv = conv_block((1, 1), filters_num)(x)
      conv = conv_block((3, 3), filters_num)(conv)
      conv = conv_block((1, 1), filters_num * 2)(conv)
      shortcut = Conv2D(filters_num * 2, (3, 3), padding='same')(x)
      shortcut = batch_block(False)(shortcut)
      return Add()([conv, shortcut])
    return f

  def residual_block_transp(filters_num):
    def f(x):
      conv = conv_block((1, 1), filters_num)(x)  
      conv = conv_block_transp((3, 3), filters_num)(conv)
      conv = conv_block((1, 1), filters_num * 2)(conv)
      shortcut = Conv2D(filters_num * 2, (3, 3), padding='same')(x)
      shortcut = batch_block(False)(shortcut)
      return Add()([conv, shortcut])
    return f

  filters = [32, 64, 128, 256, 512]
  #ROOT
  inputs = Input((image_rows, image_cols, img_channels))
  root = stem(filters[0])(inputs)
  
  #ENCODER
  res1 = residual_block(filters[0])(root)
  pool1 = MaxPooling2D(pool_size=(2, 2))(res1)
  
  res2 = residual_block(filters[1])(pool1)
  pool2 = MaxPooling2D(pool_size=(2, 2))(res2)
  
  res3 = residual_block(filters[2])(pool2)
  pool3 = MaxPooling2D(pool_size=(2, 2))(res3)
  
  res4 = residual_block(filters[3])(pool3)
  pool4 = MaxPooling2D(pool_size=(2, 2))(res4)
  
  #BRIDGE
  res5 = residual_block(filters[4])(pool4)
  res5 = residual_block(filters[4])(res5)
  
  #DECODER
  up1 = concatenate([UpSampling2D(size=(2, 2))(res5), res4], axis=3)
  resT1 = residual_block_transp(filters[3])(up1)
  
  up2 = concatenate([UpSampling2D(size=(2, 2))(resT1), res3], axis=3)
  resT2 = residual_block_transp(filters[2])(up2)

  up3 = concatenate([UpSampling2D(size=(2, 2))(resT2), res2], axis=3)
  resT3 = residual_block_transp(filters[1])(up3)

  up4 = concatenate([UpSampling2D(size=(2, 2))(resT3), res1], axis=3)
  resT4 = residual_block_transp(filters[0])(up4)
  
  #OUTPUT
  outputs = Conv2D(1, (1, 1), padding="same", activation="sigmoid")(resT4)
  outputs = Reshape((384, 384, 2, 1))(concatenate([outputs, outputs], axis=3))
  
  model = Model(inputs=[inputs], outputs=[outputs])  
  model.compile(optimizer=optimizer, loss=bce_dice_coef_loss, metrics=[dice_coef])
  return model
