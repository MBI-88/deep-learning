"""
@autor: MBI
Description: Script for generating images of the file systems

"""
#==== Packages ====#
from __future__ import unicode_literals,absolute_import
from tensorflow.python.keras.utils.data_utils import Sequence
import numpy as np
import layer_utils
import os,skimage
from layer_utils import get_gt_data,anchor_boxes
from skimage.io import imread
from skimage.util import random_noise
from skimage import exposure

#==== Classes ====#

class DataGenerator(Sequence):
    def __init__(self,input_shape:tuple,dictionary:dict,n_classes:int,batch_size:int,feature_shapes:list=[],n_anchors:int=4,shuffle:bool=True,data_path:str='') -> None:
        self.dictionary = dictionary
        self.data_path = data_path
        self.n_classes = n_classes
        self.keys = np.array(list(self.dictionary.keys()))
        self.input_shape = input_shape
        self.feature_shapes = feature_shapes
        self.shuffle = shuffle
        self.layers = 4
        self.n_anchors = n_anchors
        self.batch_size = batch_size
        self.on_epoch_end()
        self.get_n_boxes()
    
    def __len__(self) -> int:
        blen = np.floor(len(self.dictionary)/ self.batch_size)
        return int(blen)
    
    def __getitem__(self,index) -> tuple:
        start_index = index  * self.batch_size
        end_index = (index + 1) * self.batch_size
        keys = self.keys[start_index : end_index]
        x,y = self.__data_generation(keys)
        return x,y
    
    def on_epoch_end(self) -> None:
        if self.shuffle == True:
            np.random.shuffle(self.keys)
    
    def get_n_boxes(self) -> int:
        self.n_boxes = 0
        for shape in self.feature_shapes:
            self.n_boxes += np.prod(shape) // self.n_anchors
        return self.n_boxes
    
    def apply_random_noise(self,image,percent=30) -> np.ndarray:
        random = np.random.randint(0,100)
        if random < percent:
            image = random_noise(image)
        return image
    
    def apply_random_intesity_rescale(self,image,percent=30)  -> np.ndarray:
        random = np.random.randint(0,100)
        if random < percent:
            v_min,v_max = np.percentile(image,(0.2,99.8))
            image = exposure.rescale_intensity(image,in_range=(v_min,v_max))
        return image
    
    def __data_generation(self,keys) -> tuple:
        x = np.zeros((self.batch_size,*self.input_shape))
        dim = ( self.batch_size,self.n_boxes,self.n_classes)
        gt_class = np.zeros(dim)
        dim = (self.batch_size,self.n_boxes,4)
        gt_offset = np.zeros(dim)
        gt_mask = np.zeros(dim)

        for i,key in enumerate(keys):
            image_path = os.path.join(self.data_path,key)
            image = skimage.img_as_float(imread(image_path))
            image = self.apply_random_intesity_rescale(image)
            image = self.apply_random_noise(image)
            x[i] = image
            labels = self.dictionary[key]
            labels = np.array(labels)
            boxes = labels[:,0:-1]
            for index,feature_shape in enumerate(self.feature_shapes):
                anchors = anchor_boxes(feature_shape,image.shape,index=index,n_layers = self.layers)
                anchors = np.reshape(anchors,[-1,4])
                iou = layer_utils.iou(anchors,boxes)
                gt = get_gt_data(iou,n_classes=self.n_classes,anchors=anchors,labels=labels,normalize=True,threshold=0.6)
                gt_cls,gt_off,gt_msk = gt

                if index == 0:
                    cls = np.array(gt_cls)
                    off = np.array(gt_off)
                    msk = np.array(gt_msk)
                
                else:
                    cls = np.append(cls,gt_cls,axis=0)
                    off = np.append(off,gt_off,axis=0)
                    msk = np.append(msk,gt_msk,axis=0)
            
            gt_class[i] = cls
            gt_offset[i] = off
            gt_mask[i] = msk
        
        y = [gt_class,np.concatenate((gt_offset,gt_mask),axis=-1)]
        return x,y
