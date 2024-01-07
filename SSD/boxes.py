"""
@autor: MBI
Description: Script to show boxes
"""
#==== Package ====#
from __future__ import absolute_import,unicode_literals,division,print_function
import numpy as np
import matplotlib.pyplot as plt
import layer_utils,label_utils
import math
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from layer_utils import anchor_boxes,minmax2centroid,centroid2minmax
from label_utils import index2class,get_box_color
#==== Functions ====#

def nms(args,classes,offsets,anchors) -> tuple:
    objects = np.argmax(classes,axis=1)
    nonbg = np.nonzero(objects)[0]

    indexes:list[int] = []
    while True:
        scores = np.zeros((classes.shape[0],))
        scores[nonbg] = np.amax(classes[nonbg],axis=1)

        score_idx = np.argmax(scores,axis=0)
        score_max = scores[score_idx]

        nonbg = nonbg[nonbg != score_idx]

        if score_max < args[0]: # class threshold
            break

        indexes.append(score_idx)
        score_anc = anchors[score_idx]
        score_off = offsets[score_idx][0:4]
        score_box = score_anc + score_off
        score_box = np.expand_dims(score_box,axis=0)
        nonbg_copy = np.copy(nonbg)

        for idx in nonbg_copy:
            anchor = anchors[idx]
            offset = offsets[idx][0:4]
            box = anchor + offset
            box = np.expand_dims(box,axis=0)
            iou = layer_utils.iou(box,score_box)[0][0]

            if args[1]: # soft nms
                iou = -2 * iou * iou
                classes[idx] *= math.exp(iou)
            
            elif iou >= args[2]: # iou threshold
                nonbg = nonbg[nonbg != idx]
        
        if  nonbg.size == 0:
            break
    
    scores = np.zeros((classes.shape[0],))
    scores[indexes] = np.amax(classes[indexes],axis=1)

    return objects,indexes,scores 

def show_boxes(args,image,classes,offsets,feature_shapes,show=True) -> tuple:
    anchors:list = []
    class_name = None
    rect = None
    for index,feature_shape in enumerate(feature_shapes):
        anchor = anchor_boxes(feature_shape,image.shape,index=index)
        anchor = np.reshape(anchor,(-1,4))

        if index == 0:
            anchors = anchor
        else:
            anchors = np.concatenate((anchors,anchor),axis=0)
    
    if  args[4]: # True o False 
        print('Normalize')
        anchors_centroid = minmax2centroid(anchors)
        offsets[:,0:2] *= 0.1
        offsets[:,0:2] *= anchors_centroid[:,2:4]
        offsets[:,0:2] += anchors_centroid[:,0:2]
        offsets[:,2:4] *= 0.2
        offsets[:,2:4] = np.exp(offsets[:,2:4])
        offsets[:,2:4] *= anchors_centroid[:,2:4]
        offsets = centroid2minmax(offsets)
        offsets[:,0:4] = offsets[:,0:4] - anchors
    
    objects,indexes,scores = nms(args,classes,offsets,anchors)

    class_names:list[str] = []
    rects:list[int] = []
    class_ids:list[int] = []
    boxes:list[int] = []
  
    if show:
        fig,ax = plt.subplots(1)
        ax.imshow(image)
    yoff = 1
    for idx in indexes:
        anchor = anchors[idx]
        offset = offsets[idx]

        anchor += offset[0:4]
        boxes.append(anchor)
        w = anchor[1] - anchor[0]
        h = anchor[3] - anchor[2]
        x = anchor[0]
        y = anchor[2]
        category = int(objects[idx])
        class_ids.append(category)
        class_name = index2class(category)
        class_name = "%s: %0.2f" % (class_name,scores[idx])
        class_names.append(class_name)
        rect = (x,y,w,h)
        print(class_name,rect)
        rects.append(rect)
        if show:
            color = get_box_color(category)
            rect = Rectangle((x,y),w,h,linewidth=2,edgecolor=color,facecolor='none')
        
            ax.add_patch(rect)
            bbox = dict(color='white',alpha=1.0)
            ax.text(anchor[0] + 2,anchor[2] - 16 + np.random.randint(0,yoff),
            class_name,color=color,bbox=bbox,fontsize=10,verticalalignment='top')
            yoff += 50

    if show:
        plt.savefig('detection.png',dpi=600)
        plt.show()
    
    return class_name,rect,class_ids,boxes

def show_anchors(image,feature_shape,anchors,maxiou_indexes=None,maxiou_per_gt=None,labels=None,show_grids=False) -> tuple:
    image_height,image_width,_ = image.shape
    _,feature_height,feature_width,_ = feature_shape
    fig,ax = plt.subplots(1)
    ax.imshow(image)
    class_name:str = ''

    if show_grids:
        grid_height = image_height // feature_height
        for i in range(feature_height):
            y = i * grid_height
            line = Line2D([0,image_width],[y,y])
            ax.add_line(line)

        grid_width = image_width // feature_width
        for i in range(feature_width):
            x = i * grid_width
            line = Line2D([x,x],[0,image_height])
            ax.add_line(line)
    
    for index in range(maxiou_indexes.shape[1]):
        i = maxiou_indexes[1][index]
        j = maxiou_indexes[2][index]
        k = maxiou_indexes[3][index]
        box = anchors[0][i][j][k]
        w = box[1] - box[0]
        h = box[3] - box[2]
        x = box[0]
        y = box[2]
        rect = Rectangle((x,y),w,h,linewidth=2,edgecolor='y',facecolor='none')
        ax.add_patch(rect)

        if maxiou_per_gt is  not None and labels is not None:
            iou = np.amax(maxiou_per_gt[index])
            label = labels[index]
            category = int(label[4])
            class_name = index2class(category)
            color = label_utils.get_box_color(category)
            bbox = dict(facecolor=color,color=color,alpha=1.0)
            ax.text(
                label[0],
                label[2], 
                class_name,
                color='w',
                fontweight='bold',
                bbox=bbox,
                fontsize=16,
                verticalalignment='top'
            )
            dxmin = label[0] - box[0]
            dxmax = label[1] - box[1]
            dymin = label[2] - box[2]
            dymax = label[3] - box[3]
            print(index, ":","(",class_name,")",iou,dxmin,dxmax,dymin,dymax,label[0],label[2])
    
    if labels is None:
        plt.show()
    
    return fig,ax







