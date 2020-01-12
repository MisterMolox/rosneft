import tensorflow.compat.v1 as tf
import numpy as np
from tqdm import tqdm, tqdm_notebook
from keras import backend as K
from keras.losses import binary_crossentropy


def dice(y_true, y_pred):                                  # Метрика сходства двух изображений
    y_true_f = K.flatten(y_true[:,:,:,0])
    y_pred_f = K.flatten(y_pred[:,:,:,0])
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + 0.001) / (K.sum(y_true_f) + K.sum(y_pred_f) + 0.001)

def dice_border(y_true, y_pred):
    y_true_f = K.flatten(y_true[:,:,:,1])
    y_pred_f = K.flatten(y_pred[:,:,:,0]*y_true[:,:,:,1])
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + 0.001) / (K.sum(y_true_f) + K.sum(y_pred_f) + 0.001)

def dice_coef(y_true, y_pred):
    return (dice(y_true, y_pred) + dice_border(y_true, y_pred)) /2 

def dice_coef_loss(y_true, y_pred):                        # Функция потерь для минимизации
    return 1.0 - dice_coef(y_true, y_pred)
  
def bce_dice_coef_loss(y_true, y_pred): 
    score_1 = 0.5*dice_coef_loss(y_true[:,:,:,:,0], y_pred[:,:,:,:,0])
    score_2 = 0.5*binary_crossentropy(y_true[:, :, :, 0, 0], y_pred[:, :, :, 0, 0])
    return score_1 + score_2

# предполагается что на вход подаётся полный датасет (т.е. все y_true, и все предикты для x_train)
def find_threshold(y_true, y_pred_soft):

    def dice_local(y_true, y_pred):
        y_true_f = y_true[:,:,:,0].flatten()
        y_pred_f = y_pred.flatten()
        return (2.0 * np.sum(y_true_f * y_pred_f) + 0.001) / (np.sum(y_true_f) + np.sum(y_pred_f) + 0.001)

    def dice_border_local(y_true, y_pred):
        y_true_f = (y_true[:,:,:,0] * y_true[:,:,:,1]).flatten()
        y_pred_f = (y_pred * y_true[:,:,:,1]).flatten()
        return (2.0 * np.sum(y_true_f * y_pred_f) + 0.001) / (np.sum(y_true_f) + np.sum(y_pred_f) + 0.001)

    def dice_coef_local(y_true, y_pred):
        return (dice_local(y_true, y_pred) + dice_border_local(y_true, y_pred)) / 2.0

    thresholds = np.linspace(0.01, 0.99, 99)
    thrs_dice_coefs = []
    SAMPLES = y_true.shape[0]

    for threshold in tqdm_notebook(thresholds):
        y_pred_hard = np.where(y_pred_soft < threshold, 0., 1.).astype('float32')
        dice_coefs = np.array([
            dice_coef_local(y_true[i][np.newaxis, :], y_pred_hard[i][np.newaxis, :]) for i in range(SAMPLES)
        ])
        thrs_dice_coefs.append((threshold, dice_coefs.mean()))
    thrs_dice_coefs = sorted(thrs_dice_coefs, key=lambda x: x[1], reverse=True)
    return thrs_dice_coefs[0][0]
