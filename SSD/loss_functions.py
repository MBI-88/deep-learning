"""
@autor: MBI
Description: Script of loss functions
"""
#==== Packages ====#
from __future__ import unicode_literals,absolute_import
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.losses import Huber 
import numpy as np  

# ==== Functions ====#

def focal_loss_ce(y_true,y_pred) :
    weight = (1 - y_true)
    weight *= weight
    weight *= 0.25
    return K.categorical_crossentropy(weight * y_true,y_pred)

def focal_loss_binary(y_true,y_pred):
    gamma:float = 2.0
    alpha:float = 0.25

    pt_1 = tf.where(tf.equal(y_true,1),y_pred,tf.ones_like(y_pred))
    pt_0 = tf.where(tf.equal(y_true,0),y_pred,tf.ones_like(y_pred))

    epsilon = K.epsilon()
    # Prevent Nan and Inf
    pt_1 = K.clip(pt_1,epsilon,1.0 - epsilon)
    pt_0 = K.clip(pt_0,epsilon,1.0 - epsilon)

    weight = alpha * K.pow(1. - pt_1,gamma)
    fl1 = - K.sum(weight * K.log(pt_1))
    weight = (1 - alpha) * K.pow(pt_0,gamma)
    fl0 = - K.sum(weight * K.log(1. - pt_0))

    return fl1 + fl0

def focal_loss_categorical(y_true,y_pred):
    gamma:float = 2.0
    alpha:float = 0.25

    y_pred /= K.sum(y_pred,axis=-1,keepdims=True)

    epsilon = K.epsilon()
    y_pred = K.clip(y_pred,epsilon,1.0 - epsilon)

    cross_entropy = - y_true * K.log(y_pred)

    # focal loss
    weight = alpha * K.pow(1 - y_pred,gamma)
    cross_entropy *= weight

    return K.sum(cross_entropy,axis=-1)

def mask_offset(y_true,y_pred):
    offset = y_true[...,0:4]
    mask = y_true[...,4:8]
    pred = y_pred[...,0:4]
    offset *= mask
    pred *= mask
    return offset,pred

def l1_loss(y_true,y_pred):
    offset,pred = mask_offset(y_true,y_pred)
    return K.mean(K.abs(pred - offset),axis= -1)

def smooth_l1_loss(y_true,y_pred):
    offset,pred = mask_offset(y_true, y_pred)
    return Huber()(offset,pred)


