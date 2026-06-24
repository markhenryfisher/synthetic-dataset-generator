# -*- coding: utf-8 -*-
"""
Created on Thu Aug 28 16:10:41 2025

@author: mhf
"""

import math
import cv2
import numpy as np
from ..torchvision_utils.utils import label_image_to_gt
from ..synth_utils import pca
from ..synth_utils import compositor_transforms as tf
from ..synth_utils.utils import inter_union, pad_img_array
from ..debug_utils import visualize as vis


def rotate_point(x, y, ox, oy, angle_radians):
    """
    rotate_point (x,y) around origin (ox,oy)
    """
    
    # Translate point to origin
    translated_x = x - ox
    translated_y = y - oy
    
    # Apply rotation
    rotated_x = translated_x * math.cos(angle_radians) - translated_y * math.sin(angle_radians)
    rotated_y = translated_x * math.sin(angle_radians) + translated_y * math.cos(angle_radians)
    
    # Translate point back
    final_x = rotated_x + ox
    final_y = rotated_y + oy
    
    return final_x, final_y


def obb_img(img, info=None, debug=False):
    """
    mhf 28.08.25 obb_img - generate a oriented bbox filled polygon
    c.f. hull_img
    """
    if debug:
        vis.display_img(img, 'img_canvas_input')
    
    in_shape = img.shape[:2] 
    pad_width = max(in_shape)
    p_img = pad_img_array(img, pad_width)
        
    contours, hierarchy = cv2.findContours(p_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
   
    temp=[]
    for c in contours:
        temp.append(c)
    c = np.vstack(temp)
            
    ctr, angle = pca.getOrientation(c)
    # mhf use info from compositor if we have it
    if info:
        ctr = info['position'][0] + 512, info['position'][1] + 512
        angle = math.radians(info['orientation'])
    
    ox, oy = ctr
    # rotation reqd. for horizontal 
    rot_theta =  0. - angle
    
    # # centre of mask
    # image_size = img.shape[:2]
    tmp_lbl_img = tf.rotate(p_img, ox, oy, rot_theta)
    
    if debug:
        vis.display_img(tmp_lbl_img, 'rotated lbl_img')

    # HBB (for aligned mask)        
    boxes, masks = label_image_to_gt(tmp_lbl_img)

    assert len(boxes) == 1
    bbox = boxes[0]
    
    # HBB coords wrt ctr (all 4 corners)
    # hbox_coords = np.array([[bbox[0], bbox[1]],
    #                        [bbox[2], bbox[1]],
    #                        [bbox[2], bbox[3]],
    #                        [bbox[0], bbox[3]]], dtype=np.int32)

    # mhf 10.11.25 reordered points (shouldn't matter)    
    hbox_coords = np.array([[bbox[0], bbox[3]],
                           [bbox[0], bbox[1]],
                           [bbox[2], bbox[1]],
                           [bbox[2], bbox[3]]], dtype=np.int32)

    # tmp_img = img.copy()    
    # points = hbox_coords.reshape((-1, 1, 2)) # reshape for fillPoly       
    # cv2.fillPoly(tmp_img, pts = [points], color=(255,255,255)) 

    # vis.display_img(tmp_img, 'img canvas')    
    # assert len(np.unique(tmp_img)) == 2


    rbox_coords = np.zeros_like(hbox_coords)
    for i, row in enumerate(hbox_coords):
        x, y = rotate_point(row[0], row[1], ctr[0], ctr[1], angle)
        rbox_coords[i, 0] = round(x)
        rbox_coords[i, 1] = round(y)
        
    if debug:
        vis.show_obbs(p_img, np.expand_dims(rbox_coords, 0))
        
    output_img = np.zeros_like(img)
    offset_x = in_shape[1]
    offset_y = in_shape[0]
    rbox_coords[:,0] -= offset_x
    rbox_coords[:,1] -= offset_y
    
    points = rbox_coords.reshape((-1, 1, 2)) # reshape for fillPoly       
    cv2.fillPoly(output_img, pts = [points], color=(255,255,255)) 
            
     

    return output_img, rbox_coords

    

def obbox_iou(lbl_img, debug=False):
    """
    c.f. mask_iou
    """ 
    if debug:
        vis.display_img(lbl_img, 'obbox_iou_input')
    n_objects = len(np.unique(lbl_img))
    iou = np.ones((n_objects-1, n_objects-1), dtype=np.float32)
    canvas = np.zeros_like(lbl_img, dtype=np.ubyte)
    for n in range(1,n_objects):
        n_img = canvas.copy()
        n_img[lbl_img==n] = 1
        n_img, _ = obb_img(n_img)
        for m in range(1,n_objects):
            if n == m:
                pass
            else:
                m_img = canvas.copy()
                m_img[lbl_img==m] = 1                
                m_img, _ = obb_img(m_img)
                iou[n-1,m-1] = inter_union(n_img, m_img)

    return iou


def get_obbs(lbl_img, c_fish):
    n_objects = len(np.unique(lbl_img))
    canvas = np.zeros_like(lbl_img, dtype=np.ubyte)

    obbs = np.zeros((n_objects-1, 4, 2), dtype=np.int32)
    for n in range(1,n_objects):
        n_img = canvas.copy()
        n_img[lbl_img==n] = 1
        _, rbox_coords = obb_img(n_img, info=c_fish['fish{}'.format(n-1)])
        obbs[n-1] = rbox_coords
        
    return obbs
        
        
