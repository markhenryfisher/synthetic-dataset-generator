# -*- coding: utf-8 -*-
"""
Created on Thu Sep  4 17:01:21 2025

@author: mhf
"""
import numpy as np
import cv2
from ..synth_utils import augmentation
from ..debug_utils import visualize as vis

def rotate(img, ox, oy, angle_radians, debug=False):
    """ 
    """
    if debug:
        vis.display_img(img, 'rot_tf input')

    
    y, x = img.shape[:2]
    ctr = (ox, oy)
    ctr_np = np.array(ctr)
    rot_theta_np = np.array([angle_radians])
    
    centre_xf = augmentation.translation_matrices(-ctr_np[None, :])

    uncentre_xf = augmentation.translation_matrices(ctr_np[None, :])


    # rotation
    rot_xf = augmentation.rotation_matrices(rot_theta_np)

    xf = augmentation.cat_nx2x3(uncentre_xf, rot_xf, centre_xf)
    

    tmp_img = cv2.warpAffine(img, xf[0], (x, y), flags=cv2.INTER_NEAREST)


    if debug:
        vis.display_img(tmp_img, 'rot_tf output')
    return tmp_img 