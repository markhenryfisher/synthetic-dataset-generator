# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 14:00:53 2024

@author: mhf
"""
import os
import json
import torch
# import torch.nn.functional as F
import numpy as np
import cv2
import random

from image_labelling_tool import labelling_tool
from torchvision.ops import boxes as box_ops
from ..debug_utils import visualize as vis


TORCH_DEVICE = 'cuda:0'
# WHOLE_FISH_CLASS = 'whole_fish'
IMAGE_SIZE = (512, 512)
CENTRE = [256, 256]


def uint8img(array):
    if array.dtype == np.float64 or array.dtype == np.float32:       
        array *= 255
    
    return array.astype(np.uint8)


def safe_mkdir(folder):
    print('deprecated use os.makedirs(folder, exist_ok=True)' )
    if not os.path.exists(folder):
        os.mkdir(folder)

class OutputWriter:
    def __init__(self, iou_mode):
        self.iou_mode = iou_mode
        
    def __write_img(self, image, folder, filename):
        os.makedirs(folder, exist_ok=True)
        rgb_img=image[:, :, ::-1] * 255
        cv2.imwrite(os.path.join(folder, '{}.png'.format(filename)), rgb_img.astype(np.uint8))
 
        return

           
    def write_occlusions(self, occ_info_list, occ_folder, filename, debug=True):
        os.makedirs(occ_folder, exist_ok=True) 
        for i, occ_info in enumerate(occ_info_list): 
            fullfile = os.path.join(occ_folder, filename + 'unoccluded_{}.png'.format(i) )
            cv2.imwrite(fullfile, occ_info[0])
            fullfile = os.path.join(occ_folder, filename + 'mask_{}.png'.format(i) )        
            cv2.imwrite(fullfile, occ_info[1])          
    
        return       

        
    def write_coco(self, image, # image
                     label_image,
                     filename,
                     frame_folder, 
                     labels_folder,
                     c_fish):

        self.__write_img(image, frame_folder, filename)
    
        # Very bodgy, but: use the IDs as classes so they get correctly assigned, and then
        # move them into the object ID property and add fish classification so that the
        # labels can be properly read later
        ids = [i for i in list(range(1,max(label_image.flatten())+1 )) ]
        tool_labels = labelling_tool.ImageLabels.from_label_image_maintain_instances(label_image, ids)
        for i, label in enumerate(tool_labels.labels):
            key = 'fish' + str(i)
            fish = c_fish[key]
            label.object_id = int(label.classification)
            label.classification = fish['species']
        jason_labels = tool_labels.to_json()
        os.makedirs(labels_folder, exist_ok=True)
        with open(os.path.join(labels_folder, '{}__labels.json'.format(filename)), 'w') as f:
            json.dump(jason_labels, f)
            
        # df = pd.DataFrame(data)
        # df.to_csv(os.path.join(areas_folder, '{}.csv'.format(filename)), index=False)
        # with open(os.path.join(areas_folder, '{}.csv'.format(filename)), 'w', newline='') as csvfile:
        #     writer = csv.writer(csvfile)
        #     for i, id in enumerate(fish_ids):
        #         if visible_percentages[i] > 0:
        #             writer.writerow([id, visible_percentages[i], ious[i]])
    
        return 
    
    def write_dota(self, 
                     image, 
                     obbs, 
                     filename, 
                     frame_folder, 
                     ann_folder,
                     c_fish):
        
        self.__write_img(image, frame_folder, filename)
        
        dota_ann_list = []
        for obb, fish in zip(obbs, c_fish):
            this_fish=c_fish[fish]
            difficulty = 0 if this_fish['tgt_iou'] and this_fish['tgt_iou'] < 0.1 else 1
            arr_flat_str = ' '.join(map(str, obb.flatten()))
            dota_ann_str = arr_flat_str + ' ' + this_fish['species'] + ' ' + str(difficulty) + '\n'
            dota_ann_list.append(dota_ann_str)

        os.makedirs(ann_folder, exist_ok=True)      
        with open(os.path.join(ann_folder, '{}.txt'.format(filename)), 'w') as f:
            f.writelines(dota_ann_list)

def assess_morph_proc(lbl_img, debug=False):

    def show_contour(img, name_str):
        # find the largest contour in the threshold image
        cnts, _ = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        c = max(cnts, key=cv2.contourArea)
        text = "{}, num_pts={}".format(name_str, len(c))
        print("[INFO] {}".format(text))
        
        height, width = lbl_img.shape
        output = np.zeros((height, width, 3), dtype=np.uint8)
        cv2.drawContours(output, [c], -1, (0, 255, 0), 1)
        vis.display_img(output, name_str)

    img = lbl_img.copy()
    show_contour(img, 'original')

    for sz in range(7, 27, 2):  
        strel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (sz, sz))
        opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, strel, iterations=1)
        show_contour(opening, 'opening_{}'.format(sz))
       
    return

def morph_opening(lbl_img, strel_sizes = [1,3,5,7], debug=False):
    """ 
    perform morphological opening with strel (size selected randomly)
    """    
    sz = random.choice(strel_sizes)
    strel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (sz, sz))
    opening = cv2.morphologyEx(lbl_img, cv2.MORPH_OPEN, strel, iterations=1)
    
    if debug:
        vis.display_img(opening)

    return opening.astype(np.int32)     

def assess_approx_contour(lbl_img):
    """
    https://pyimagesearch.com/2021/10/06/opencv-contour-approximation/
    """
    # find the largest contour in the threshold image
    cnts, _ = cv2.findContours(lbl_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    c = max(cnts, key=cv2.contourArea)
    
    # draw the shape of the contour on the output image, compute the
    # bounding box, and display the number of points in the contour
    # output = lbl_img.copy()
    height, width = lbl_img.shape
    output = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.drawContours(output, [c], -1, (0, 255, 0), 1)
    (x, y, w, h) = cv2.boundingRect(c)
    text = "original, num_pts={}".format(len(c))
    # show the original contour image
    print("[INFO] {}".format(text))
    vis.display_img(output, 'Original Contour')
    
    # to demonstrate the impact of contour approximation, let's loop
    # over a number of epsilon sizes
    for eps in np.linspace(0.0, 0.002, num=10):
    	# approximate the contour
    	peri = cv2.arcLength(c, True)
    	approx = cv2.approxPolyDP(c, eps * peri, True)
    	# draw the approximated contour on the image
    	output = np.zeros((height, width, 3), dtype=np.uint8)
    	cv2.drawContours(output, [approx], -1, (0, 255, 0), 1)
    	text = "eps={:.4f}, num_pts={}".format(eps, len(approx))
    	# show the approximated contour image
    	print("[INFO] {}".format(text))
    	vis.display_img(output, 'Approximated Contour') 
    return

def approx_contour(lbl_img, eps_max=0.0015, debug=False):
    eps = random.uniform(0.0, eps_max)
    img_vals = np.unique(lbl_img) 
    lbl_val = int(img_vals[-1])
    
    # find the largest contour in the threshold image
    cnts, _ = cv2.findContours(lbl_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)    
    c = max(cnts, key=cv2.contourArea)
    height, width = lbl_img.shape
    orig_pts = len(c)
    
    if debug:
        output = np.zeros((height, width, 3), dtype=np.uint8)    
        cv2.drawContours(output, [c], -1, (255, 255, 255), -1)
        text = "original, num_pts={}".format(len(c))
        # show the original contour image
        print("[INFO] {}".format(text))
        vis.display_img(output, 'Original Contour')

    
    peri = cv2.arcLength(c, True) 
    approx = cv2.approxPolyDP(c, eps * peri, True)
    output = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.drawContours(output, [approx], -1, (lbl_val, lbl_val, lbl_val), -1)

    if debug:
        text = "eps={:.4f}, num_pts={}".format(eps, len(approx))
        print("[INFO] {}".format(text))
        vis.display_img(output)

    # if len(approx) > orig_pts/10:        
    #     dst = output[:,:,0].astype(np.int32)
    # else:
    #     print('Discarding approx. contour; num_pts={}'.format(num_pts))
    #     dst = lbl_img.astype(np.int32)
        
    dst = output[:,:,0].astype(np.int32)
    
    if debug:
        vis.display_img(dst)
            
    return dst


def box_density(boxes, debug=False):
    """
    BOX_DENSITY - computes density d_i as defined by
    Liu et al. Adaptive NMS: Refining Pedestrian Detection in a Crowd. 2019
 
    """
    if not torch.is_tensor(boxes):
        raise ValueError('box_density: input boxes is not Tensor')
    
    if len(boxes) == 0:
        raise ValueError('box_density: empty input boxes')
    elif len(boxes) == 1:
        result = torch.Tensor([0.0])
    else:
            
        # More efficient to compute (self) iou matrix  
        iou_mat = box_ops.box_iou(boxes, boxes)
        # set main diagonal to 0.0 (iou of box with itself = 1.0) 
        iou_mat.fill_diagonal_(0.0)
        max_iou_per_gt_box, ind = iou_mat.max(dim=1)
        result = max_iou_per_gt_box
    
    # mhf 17.06.24
    if debug:
        result = torch.ones_like(result)
                      
    return result

def inter_union(ground_truth, prediction):
    """ 
    S:\ruth_fish_archive2\fish_code\cctv4fisheries\marine_py2\segmentation\eval_labels.py
    https://github.com/carpenterlab/2019_caicedo_dsb/blob/master/evaluation.py
    """    
    # Count objects
    true_objects = len(np.unique(ground_truth))
    pred_objects = len(np.unique(prediction))
    
    # Compute intersection
    h = np.histogram2d(ground_truth.flatten(), prediction.flatten(), bins=(true_objects,pred_objects))
    intersection = h[0]
    
    # Area of objects
    area_true = np.histogram(ground_truth, bins=true_objects)[0]
    area_pred = np.histogram(prediction, bins=pred_objects)[0]
    
    # Calculate union
    area_true = np.expand_dims(area_true, -1)
    area_pred = np.expand_dims(area_pred, 0)
    union = area_true + area_pred - intersection
    
    # Exclude background from the analysis
    intersection = intersection[1:,1:]
    union = union[1:,1:]
    
    # Compute Intersection over Union
    union[union == 0] = 1e-9
    IOU = intersection/union
    
    return IOU

def pad_img_array(img, pad_width):
    if len(img.shape) == 3:
        pad_spec = ((pad_width,pad_width),(pad_width,pad_width),(0,0))
    else:
        pad_spec = pad_width
    out_img = np.pad(img, pad_width=pad_spec, mode='constant', constant_values=0)

    return out_img

# -----------------------------------------------------------------------------
# visualization

# https://pytorch.org/vision/stable/auto_examples/plot_visualization_utils.html
from torchvision.utils import draw_bounding_boxes
from IPython import get_ipython
import matplotlib.pyplot as plt

def display_image_with_bboxes(img, target):
    img_id = target['image_id'].item()
    img = img.mul(255).to(torch.uint8)
    res_bb = draw_bounding_boxes(img, target['boxes'], width=2)

    # https://stackoverflow.com/questions/29356269/plot-inline-or-a-separate-window-using-matplotlib-in-spyder-ide
    get_ipython().run_line_magic('matplotlib', 'inline')
    temp = res_bb.permute(1, 2, 0).byte().numpy()
    plt.imshow(temp)
    plt.title('image id = {}'.format(img_id))
    # plt.savefig(os.path.join(exp_params['output_folder'], exp_params['box_nms_mode']+".png"), dpi=1000)
    plt.show()