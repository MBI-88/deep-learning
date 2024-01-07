"""
@autor : MBI
Description : Script for labeling images
"""
#==== Packages ====#
from __future__ import unicode_literals,absolute_import
import numpy as np
import csv 
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from random import randint

#==== Functions ====#

params:dict = {
    'classes':['background','Water','Soda','Juice'],
    'prices': [0.0,10.0,40.0,35.0]
}

def get_box_color(index=None) -> str:
    colors:list = ['w', 'r', 'b', 'g', 'c', 'm', 'y', 'g', 'c', 'm', 'k']
    if index is None:
        return colors[randint(0,len(colors) - 1)]
    return colors[index  % len(colors)]

def get_box_rgbcolor(index=None) -> tuple:
    colors:list[tuple] = [(0,0,0),(255,0,0),(0,0,255),(0,255,0),(128,128,0)]
    if index is None:
        return colors[randint(0,len(colors) - 1)]
    return colors[index % len(colors)]

def index2class(index:int=0) -> str:
    classes:list[str] = params['classes']
    return classes[index]

def class2index(class_:str='background') -> int:
    classes:list[str] = params['classes']
    return classes.index(class_)

def load_csv(path:str) -> np.ndarray:
    data:list[str] = []
    with open(path) as csvfile:
        rows = csv.reader(csvfile,delimiter=',')
        for row  in rows:
            data.append(row)
    return np.array(data)

def get_label_dictionary(labels:list,keys:list) -> dict:
    dictionary:dict[str,list] = {}
    for key in keys:
        dictionary[key] = []
    
    for label in labels:
        if len(label) != 6:
            print("Incomplete label: ",label[0])
            continue

        value = label[1:]

        if value[0] == value[1]:
            continue

        elif value[2] == value[3]:
            continue
        
        elif label[-1] == 0:
            print('No  object labelled as bg:  ',label[0])
        
        value = value.astype(np.float32)
        key:str = label[0]
        boxes:list = dictionary[key]
        boxes.append(value)
        dictionary[key] = boxes

    for key in keys:
        if len(dictionary[key]) == 0:
            del dictionary[key]
    
    return dictionary

def build_label_dictionary(path:str) -> tuple:
    labels = load_csv(path)
    labels = labels[1:]
    keys = np.unique(labels[:,0])
    dictionary = get_label_dictionary(labels,keys)
    classes = np.unique(labels[:,-1]).astype(int).tolist()
    classes.insert(0,0)
    print('Num of unique classes:  ',classes)
    return dictionary,classes

def show_labels(image,labels,ax=None) -> None:
    if ax is None:
        fig,ax = plt.subplot(1)
        ax.imshow(image)
    for label in labels:
        w = label[1] - label[0]
        h = label[3] - label[2]
        x = label[0]
        y = label[2]
        category = int(label[4])
        color = get_box_color(category)
        rect = Rectangle((x,y),w,h,linewidth=2,edgecolor=color,facecolor='none')
        ax.add_patch(rect)
    
    plt.show()
