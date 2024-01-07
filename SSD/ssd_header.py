"""
@aurtor: MBI
Description: Script to developmnet ssd header
"""
#==== Packages ====#
from __future__ import unicode_literals,absolute_import
from tensorflow.keras.layers import (Activation,BatchNormalization,
                                    Conv2D,Input,Concatenate,ELU,MaxPooling2D,
                                    Reshape)
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K 
from resnet import build_resnet
import numpy as np
#==== Functions ====#

def conv2d(inputs,filters=32,kernel_size=3,strides=1,name=None):
    conv = Conv2D(filters=filters,kernel_size=kernel_size,strides=strides,
                  kernel_initializer='he_normal',name=name,padding='same')
    return conv(inputs)

def conv_layer(inputs,filters=32,kernel_size=3,strides=1,use_maxpool=True,postfix=None,activation=None):
    x = conv2d(inputs,filters,kernel_size,strides,name='conv_'+postfix)
    x = BatchNormalization(name='bn_'+postfix)(x)
    x = ELU(name='elu_'+postfix)(x)
    if use_maxpool:
        x = MaxPooling2D(name='pool_'+postfix)(x)
    
    return x

def build_ssd(inputs_shape,n_layers=4,n_classes=4,aspect_ratio=(1,2,0.5)) -> tuple:
    n_anchors = len(aspect_ratio)  + 1
    inputs = Input(shape=inputs_shape)
    """
    backbone = build_resnet(inputs_shape,n_layers=4)
    base_outputs = backbone(inputs)

        backbone = ResNet50(
        weights = 'imagenet',
        include_top = False,
        input_shape = inputs.shape[1:]
    )
    for layer in backbone.layers:
        layer.trainable = False

    base_outputs = [layer.output for layer in backbone.layers[:len(backbone.layers)]]
    backbone = Model(inputs=backbone.input,outputs=base_outputs,name='Resnet50/backbone')
    base_outputs = backbone(inputs)
    """
    backbone = build_resnet(inputs_shape,n_layers=4)
    base_outputs = backbone(inputs)

    outputs = []
    feature_shapes = []
    out_cls = []
    out_off = []
    
   

    for i in range(n_layers):
        conv = base_outputs if n_layers == 1 else base_outputs[i]
        name = 'cls_'+str(i+1)
        classes = conv2d(conv,n_anchors * n_classes,kernel_size=3,name=name)
        
        name = 'off_'+str(i+1)
        offsets = conv2d(conv,n_anchors * 4,kernel_size=3,name=name)
        
        shape = np.array(K.int_shape(offsets))[1:]
        feature_shapes.append(shape)

        name = 'cls_res_'+str(i+1)
        classes = Reshape((-1,n_classes),name=name)(classes)
        
        name = 'off_res_'+str(i+1)
        offsets = Reshape((-1,4),name=name)(offsets)
        
        offsets = [offsets,offsets]
        name = 'off_cat_' + str(i+1)
        offsets = Concatenate(axis=-1,name=name)(offsets)
        
        out_off.append(offsets)

        name = 'cls_out_'+str(i+1)
        classes = Activation('softmax',name=name)(classes)
        out_cls.append(classes)
        
    if n_layers > 1:
        name = 'offsets'
        offsets = Concatenate(axis=1,name=name)(out_off)

        name = 'classes'
        classes = Concatenate(axis=1,name=name)(out_cls)
        
        
    else: 
        offsets = out_off[0]
        classes = out_cls[0]
        
    outputs = [classes,offsets]
    model = Model(inputs=inputs,outputs=outputs,name='ssd_header')
    

    return n_anchors,feature_shapes,model
