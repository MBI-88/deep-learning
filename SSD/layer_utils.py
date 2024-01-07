"""
@autor: MBI
Description: Toolkit for SSD Model (Computing IOU,anchor boxes,mask,and bounding box offsets)
"""

#==== Packages ====#
from __future__ import unicode_literals,absolute_import
import numpy as np
import math 

#==== Functions ====#

def anchor_sizes(n_layers:int=4) -> list:
    s = np.linspace(0.2,0.9,n_layers + 1)
    sizes:list = []
    for i in range(len(s) - 1):
        size:list = [s[i],math.sqrt(s[i] * s[i + 1])]
        sizes.append(size)
    return sizes

def anchor_boxes(feature_shape:list,image_shape:list,index:int=0,
n_layers:int=4,aspect_ratios:tuple=(1,2,0.5)) -> tuple:
    size:int = anchor_sizes(n_layers)[index]
    n_boxes:int = len(aspect_ratios) + 1
    image_height,image_width,_ =  image_shape
    feature_height,feature_width, _ = feature_shape

    norm_height = image_height * size[0]
    norm_width = image_width * size[0]
    width_height:list[tuple] = []

    # Equation 11.2.3
    for ar in aspect_ratios:
        box_width = norm_width * np.sqrt(ar)
        box_height = norm_height / np.sqrt(ar)
        width_height.append((box_width,box_height))
    
    # Equation 11.2.4
    box_width:float = image_width * size[1]
    box_height:float = image_height * size[1]
    width_height.append((box_width,box_height))

    width_height = np.array(width_height)

    grid_width = image_width / feature_width
    grid_height = image_height / feature_height

    start:float = grid_width * 0.5
    end:float = (feature_width - 0.5) * grid_width
    cx = np.linspace(start,end,feature_width)

    start = grid_height * 0.5
    end = (feature_height - 0.5) * grid_height
    cy = np.linspace(start,end,feature_height)

    cx_grid,cy_grid = np.meshgrid(cx,cy)

    cx_grid = np.expand_dims(cx_grid, -1)
    cy_grid = np.expand_dims(cy_grid, -1)

    boxes = np.zeros((feature_height,feature_width,n_boxes,4))

    boxes[...,0] = np.tile(cx_grid,(1,1,n_boxes))
    boxes[...,1] = np.tile(cy_grid,(1,1,n_boxes))
    boxes[...,2] = width_height[:,0]
    boxes[...,3] = width_height[:,1]

    boxes = centroid2minmax(boxes)
    boxes = np.expand_dims(boxes,axis=0)
    return boxes

def centroid2minmax(boxes) -> np.ndarray:
    minmax = np.copy(boxes).astype(np.float)
    minmax[...,0] = boxes[...,0] - (0.5 * boxes[...,2])
    minmax[...,1] = boxes[...,0] + (0.5 * boxes[...,2])
    minmax[...,2] = boxes[...,1] - (0.5 * boxes[...,3])
    minmax[...,3] = boxes[...,1] + (0.5 * boxes[...,3])
    return minmax

def minmax2centroid(boxes) -> np.ndarray:
    centroid = np.copy(boxes).astype(np.float)
    centroid[...,0] = 0.5 * (boxes[...,1] - boxes[...,0])
    centroid[...,0] += boxes[...,0]
    centroid[...,1] = 0.5 * (boxes[...,3] - boxes[...,2])
    centroid[...,1] += boxes[...,2] 
    centroid[...,2] = boxes[...,1] - boxes[...,0]
    centroid[...,3] = boxes[...,3] - boxes[...,2]
    return centroid

def intersection(boxes1:np.ndarray,boxes2:np.ndarray) -> np.ndarray:
    m:int = boxes1.shape[0]
    n:int = boxes2.shape[0]

    xmin:int = 0
    xmax:int = 1
    ymin:int = 2
    ymax:int = 3
    
    boxes1_min = np.expand_dims(boxes1[:,[xmin,ymin]],axis=1)
    boxes1_min = np.tile(boxes1_min,reps=(1,n,1))
    boxes2_min = np.expand_dims(boxes2[:,[xmin,ymin]],axis=0)
    boxes2_min = np.tile(boxes2_min,reps=(m,1,1))
    min_xy = np.maximum(boxes1_min,boxes2_min)

    boxes1_max = np.expand_dims(boxes1[:,[xmax,ymax]],axis=1)
    boxes1_max = np.tile(boxes1_max,reps=(1,n,1))
    boxes2_max = np.expand_dims(boxes2[:,[xmax,ymax]],axis=0)
    boxes2_max = np.tile(boxes2_max,reps=(m,1,1))
    max_xy = np.minimum(boxes1_max,boxes2_max)

    side_lenths = np.maximum(0,max_xy - min_xy)

    intersection_area = side_lenths[:,:,0] * side_lenths[:,:,1]
    return intersection_area

def union(boxes1:np.ndarray,boxes2:np.ndarray,intersection_areas:np.ndarray) -> np.ndarray:
    m:int = boxes1.shape[0]
    n:int = boxes2.shape[0]

    xmin:int = 0
    xmax:int = 1
    ymin:int = 2
    ymax:int = 3

    width = (boxes1[:, xmax] - boxes1[:, xmin])
    height = (boxes1[:, ymax] - boxes1[:, ymin])
    areas = width * height
    boxes1_areas = np.tile(np.expand_dims(areas,axis=1),reps=(1,n))

    width = (boxes2[:, xmax] - boxes2[:, xmin])
    height = (boxes2[:, ymax] - boxes2[:, ymin])
    areas = width * height
    boxes2_areas =  np.tile(np.expand_dims(areas,axis=0),reps=(m,1))

    union_areas = boxes1_areas + boxes2_areas - intersection_areas
    return union_areas

def iou(boxes1:np.ndarray,boxes2:np.ndarray) -> np.ndarray:
    intersection_areas = intersection(boxes1,boxes2)
    union_areas = union(boxes1,boxes2,intersection_areas)
    return intersection_areas / union_areas

def get_gt_data(iou:np.ndarray,n_classes:int=4,anchors=None,labels:list=None,
normalize:bool=False,threshold:float=0.6) -> tuple:
    maxiou_per_gt = np.argmax(iou,axis=0)

    if threshold < 1.0:
        iou_gt_thresh = np.argwhere(iou > threshold)
        if iou_gt_thresh.size > 0:
            extra_anchors = iou_gt_thresh[:,0]
            extra_classes = iou_gt_thresh[:,1]
            extra_labels:int = labels[extra_classes]
            indexes:list = [maxiou_per_gt,extra_anchors]
            maxiou_per_gt = np.concatenate(indexes,axis=0)
            labels = np.concatenate([labels,extra_labels],axis=0)
    
    gt_mask = np.zeros((iou.shape[0],4))
    gt_mask[maxiou_per_gt] = 1.0 

    gt_class = np.zeros((iou.shape[0],n_classes))
    gt_class[:,0] = 1
    gt_class[maxiou_per_gt,0] = 0

    maxiou_col = np.reshape(maxiou_per_gt,(maxiou_per_gt.shape[0],1))
    label_col = np.reshape(labels[:,4],(labels.shape[0],1)).astype(int)

    row_col = np.append(maxiou_col,label_col,axis=1)
    gt_class[row_col[:,0],row_col[:,1]] = 1.0

    gt_offset = np.zeros((iou.shape[0],4))

    if normalize:
        anchors = minmax2centroid(anchors)
        labels = minmax2centroid(labels)
        # Equation 11.4.8
        offset1 = labels[:,0:2] - anchors[maxiou_per_gt,0:2]
        offset1 /= anchors[maxiou_per_gt,2:4]
        offset1 /= 0.1

        offset2 = np.log(labels[:,2:4]/anchors[maxiou_per_gt,2:4])
        offset2 /= 0.2 
        
        offsets = np.concatenate([offset1,offset2],axis=-1)

    else:
        offsets = labels[:,0:4] - anchors[maxiou_per_gt]
    
    gt_offset[maxiou_per_gt] = offsets

    return gt_class,gt_offset,gt_mask


