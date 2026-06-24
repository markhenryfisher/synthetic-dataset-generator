# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 13:36:53 2024

@author: mhf
"""
# import os
import numpy as np
import torch
import torch.nn.functional as F
import cv2
# import copy
import math
import random
import pprint

from ..image_labelling_tool import labelling_tool
from ..torchvision_utils.utils import label_image_to_gt
from ..synth_utils.augmentation import cv_to_torch
from ..synth_utils.synthetic__augmentation import augment_labels_and_scale
from ..synth_utils.repo_mgt import FishRepo
from ..synth_utils.utils import uint8img
from ..debug_utils import visualize as vis
from ..torchvision_ops import boxes as box_ops
from ..synth_utils.utils import inter_union, pad_img_array,\
        morph_opening, approx_contour
from ..synth_utils.obb_utils import obbox_iou


def render_lbl_individual(label_img, tf_info, debug=False):
    """
    label_img: np.array labelled individual fish
    flip: flip horizontal
    """  
    if debug:
        vis.display_img(label_img, 'input')

    # mhf Note: this translates label image to self.position        
    xf, label_img, scale_factor = augment_labels_and_scale(
        label_img,
        tf_info['scalefactor'], 
        tf_info['image_size'],
        tf_info['position'], 
        tf_info['rot_theta'],
        tf_info['flip'])               
    
    if debug:
        vis.display_img(label_img, 'transformed')
        
        
    # Find contours to get the bounding box of the mask
    contours, _ = cv2.findContours(label_img.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # there may be more than one contour
    temp=[]
    for c in contours:
        temp.append(c)
        
    c = np.vstack(temp)

    if contours:
        x, y, w, h = cv2.boundingRect(c)  # Get bounding box of the largest contour
        c_label_img = label_img[y:y+h, x:x+w]  # Crop the region of interest
    else:
        raise ValueError('__render_lbl_individual: No contours!')

    crop_info = (x, y, w, h)
    
    if debug:
        vis.display_img(c_label_img, 'output')

    
    return c_label_img, xf, crop_info


def hull_img(img, debug=False):
    """
    mhf 11.08.25 generate a convex hull image (input may have disjoint segments)

    """
    if debug:
        vis.display_img(img, 'img_canvas_input')
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    length = len(contours)

    temp=[]
    for i in range(length):
        temp.append(contours[i])
    cont = np.vstack(temp)

    hull = cv2.convexHull(cont)
    uni_hull = []
    uni_hull.append(hull)
    # cv2.drawContours(temp_img, uni_hull, -1,255,2)       
    cv2.fillPoly(img, pts = uni_hull, color=(255,255,255))

    if debug:
        vis.display_img(img, 'img_canvas')
        
    return img

    
def mask_iou(lbl_img, debug=False):
    """
    https://stackoverflow.com/questions/52197918
    """    
    n_objects = len(np.unique(lbl_img))
    iou = np.ones((n_objects-1, n_objects-1), dtype=np.float32)
    canvas = np.zeros_like(lbl_img, dtype=np.ubyte)
    for n in range(1,n_objects):
        n_img = canvas.copy()
        n_img[lbl_img==n] = 1
        n_img = hull_img(n_img)
        for m in range(1,n_objects):
            if n == m:
                pass
            else:
                m_img = canvas.copy()
                m_img[lbl_img==m] = 1                
                m_img = hull_img(m_img)
                iou[n-1,m-1] = inter_union(n_img, m_img)
                
    return iou
            

def inter(bg, fg, pstn):
    """
    Note: pstn = (y, x)
    """
    # bg bbox [x1,y1,x2,y2]
    bbox1 = [0, 0, bg.shape[1], bg.shape[0]]
    # switch pstn tuple to (x,y) 
    x, y = pstn[1], pstn[0]
    # fg bbox [x1, y1, x2, y2]
    bbox2 = [x, y, x+fg.shape[1], y+fg.shape[0]]
    # offset = [x, y, x, y]
    bbox2_offset = [fg.shape[1]//2, fg.shape[0]//2, fg.shape[1]//2, fg.shape[0]//2]
    # subtract offset (elemeent-wise)
    bbox2 = [x - y for x, y in zip(bbox2, bbox2_offset)]
    
    boxes1 = torch.tensor([bbox1], dtype=torch.int)
    boxes2 = torch.tensor([bbox2], dtype=torch.int)
    
    inter, union = box_ops._box_inter_union(boxes1, boxes2)
    
    return inter > 0.

def add_fg(bg, fg, label_img, pstn, debug=False):
    """
    ADD_FG: Add foreground
    https://docs.opencv.org/4.x/d0/d86/tutorial_py_image_arithmetics.html
    """
    # check fg is visible
    if not inter(bg, fg, pstn):
        return bg
    img1 = bg # background
    img2 = fg # foreground
    
    # pad bg
    # Pad with zeros (1 row/column on each side)
    # padded_array = np.pad(array, pad_width=1, mode='constant', constant_values=0)
    pad_width = max(img2.shape)
    # if len(img1.shape) == 3:
    #     pad_spec = ((pad_width,pad_width),(pad_width,pad_width),(0,0))
    # else:
    #     pad_spec = pad_width
    # img1 = np.pad(img1, pad_width=pad_spec, mode='constant', constant_values=0)
    img1 = pad_img_array(img1, pad_width)
    pstn = (pstn[0]+pad_width, pstn[1]+pad_width)
    
    # convert to 8 bit 
    if img1.dtype == np.float64 or img1.dtype == np.float32:       
        img1 *= 255
    if img2.dtype == np.float64 or img2.dtype == np.float32:
        img2 *= 255 
        
    img1=img1.astype(np.uint8)
    img2=img2.astype(np.uint8)
    
    if debug:
        vis.display_img(img1, cmap=None)
        vis.display_img(img2, cmap=None)

    # create ROI        
    rows,cols = img2.shape[0], img2.shape[1]
    origin = [pstn[0]-rows//2, pstn[1]-cols//2]
    roi = img1[origin[0]:origin[0]+rows, 
               origin[1]:origin[1]+cols]
    
    # Now create a mask of logo and create its inverse mask also    
    mask = label_img.astype(np.uint8)
    mask_inv = (mask==0).astype(np.uint8)
    
    # Now black-out the area of logo in ROI
    img1_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)
       
    # Take only region of logo from logo image.
    img2_fg = cv2.bitwise_and(img2,img2,mask = mask)
    
    if debug:
        vis.display_img(roi, cmap=None)
        vis.display_img(img1_bg, cmap=None)
        vis.display_img(img2_fg, cmap=None)
 
    # Put logo in ROI and modify the main image
    dst = cv2.add(img1_bg,img2_fg)
    img1[origin[0]:origin[0]+rows, origin[1]:origin[1]+cols] = dst

    # remove padding
    rows, cols = bg.shape[:2]
    img1 = img1[pad_width:pad_width+rows, pad_width:pad_width+cols]
       
    if debug:
        vis.display_img(img1, cmap=None)
        
    return img1
     

class FishBuff():
    """ 
    """
    def __init__(self):
        self._buff = list()
        self._ptr = 0

    @property
    def len(self):
        return len(self._buff)
    
    @property 
    def ptr(self):
        return self._ptr
        
     
    def next(self):
        self._ptr += 1
        
    def put(self, fish):
        self._buff.append(fish)
        
    def replace(self, fish):
        self._buff[self._ptr] = fish
        
    def get(self):
        return self._buff[self._ptr]
        
    def sort(self):
        b = self._buff
        sorted_b = list(sorted(b, reverse=True, key=lambda x: int(x['Area'])))
        self._buff = sorted_b        
        
    def __str__(self):
        return pprint.pformat(self._buff)
        

class CompositeImage():
    """
    """
    MAX_REJECTS = 10
    def __init__(self, 
                 num_fish,
                 fish_iter,
                 belt_bg_img,
                 iou_mode,
                 rotation_mode,
                 rotation,
                 area_thresh=100,
                 tgt_iou=[0.0]):
        
        self.num_fish = num_fish
        self.fish_iter = fish_iter
        self.belt_bg_img = belt_bg_img
        self.area_thresh = area_thresh
        self.rotation_mode = rotation_mode
        self.rotation = rotation
        self.compositor=Compositor(belt_bg_img, 
                                   num_fish,
                                   iou_mode,
                                   tgt_iou,
                                   random.choice([-16, +16])) # divisible by 2
        self.fish_buff = self.__init_buff()
        self.alt_idx_max = self.alt_idx = 1
        self.fail_cnt = 0
    
        return
    
    def __state_reset(self):
        self.fish_buff = self.__init_buff()
        self.alt_idx_max = self.alt_idx = 1
        self.fail_cnt = 0
        self.compositor.state_reset()
    
    def __init_buff(self, debug=False):
        fish_buff = FishBuff()
        fish_i = 0
        reject_i = 0
        while fish_i < self.num_fish:
            dict_entry = self.fish_iter.next_item()
            if int(dict_entry['Area']) > self.area_thresh:
                fish_buff.put(dict_entry)
                reject_i = 0; fish_i += 1
            else:
                reject_i += 1
                assert reject_i < CompositeImage.MAX_REJECTS, 'MAX_REJECTS exceeded; check area_thresh'
                            
        fish_buff.sort()
        
        return fish_buff
    
    def _set_rotation(self):
        """
        rotation is determined by heuristics
        ideally we would like whole fish to be composited
        i.e. not clipped
        so we limit rotation under certain circumstances
        its a bit of a dogs dinner - sorry
        if you want specific composits then use rotation_mode 'explicit'
        and give a list of rotation angles
        """        
        if self.rotation_mode in ['explicit']:
            assert len(self.rotation) == self.num_fish, 'rotation format error'
            self.compositor.rotation = self.rotation[self.fish_buff.ptr]
        elif self.rotation_mode in ['random']:
            if self.fish_buff.ptr == 0:
                if self.compositor.iou_mode[0] in ['hbbox_iou']:
                    # mhf 18.11.25 fish are now oriented
                    self.compositor.rotation = random.choice([0, 90, -90, 180])                  
                else:
                    # mhf 11.11.25 change rotation from (0, 90)                    
                    self.compositor.rotation = random.uniform(-90, +90)

                
            else:
                if self.compositor.tgt_iou < 0.2:
                    # limit rotation to ensure whole fish is on canvas
                    self.compositor.rotation = self.compositor.rotation[0] + random.uniform(-30, +30)
                else:
                    # mhf 11.11.25 change rotation from (0, 90)                    
                    self.compositor.rotation = random.uniform(-90, +90)
        else:
            raise ValueError('Unknown rotation_mode')
            
        return


    def run(self, debug=False):
        composite_list = []        
        while self.fish_buff.ptr < self.fish_buff.len:
            this_entry = self.fish_buff.get()
            self._set_rotation()
            result = self.compositor.add_fish_to_image(this_entry)
            if result == 0: # success
                if debug and composite_list:
                    print('Successfully achieved tgt iou')
                composite_list.append(this_entry)
                self.alt_idx = self.alt_idx_max
                self.fish_buff.next()
            elif result == -1:
                if debug and composite_list:
                    print('Failed to achieve tgt iou, recovery mode -1')
                # failed to achieve tgt iou - need a 'similar' fish
                self.alt_idx -= 1
                self.fail_cnt += 1
                assert self.alt_idx > -1, 'Failed to achieve target iou' 
                dict_entry = self.fish_iter.next_item(area=composite_list[-1]['Area'], idx=self.alt_idx)
                self.fish_buff.replace(dict_entry)
            elif result == -2:
                if debug and composite_list:
                    print('Failed to achieve tgt iou, recovery mode -2')
                # total failure, start again
                self.__state_reset()
                composite_list = []

        # get the occlusions        
        occ_info = self.compositor.occ.get_occlusions()

        
        return self.compositor.c_img, self.compositor.c_lbl_img, self.compositor.c_fish, occ_info
    

class Occlusion:
    """ 
    """
    def __init__(self):
        self._fish_label_int = 0
        self.c_buff = list()
        
    @property 
    def fish_label_int(self):
        return self._fish_label_int
    
    @fish_label_int.setter 
    def fish_label_int(self, fish_label_int):
        self._fish_label_int = fish_label_int
            
    def update(self, fish_label_int, c_lbl_img, c_img):
        lbl_img = uint8img(c_lbl_img)
        img = uint8img(c_img)
        
        self.fish_label_int = fish_label_int
        
        occluder = np.zeros_like(lbl_img)
        n_obj = len(np.unique(lbl_img))-1 # omit background
        # occluder is uppermost fish
        occluder[lbl_img == n_obj] = self.fish_label_int

        self.c_buff.append((occluder, img))
            
    def get_occlusions(self, debug=False):
        """
        """
        if len(self.c_buff) < 2:
            return
                
        occ_lst = []
        for i in range(0, len(self.c_buff)):
            lbl, img = self.c_buff[i]
            occluded = lbl.copy()
            occluded[occluded>0] = 1
            
            tmp_img = img.copy()
            seg_img = cv2.bitwise_and(tmp_img, tmp_img, dst=None, mask=occluded)

            # gather all occluders above occluded                         
            occluders = np.zeros_like(occluded)
            for j in range(i+1, len(self.c_buff)):
                occ, _ = self.c_buff[j]
                occluder = occ.copy()
                occluder[occluder>0] = 1
                occluders = cv2.bitwise_or(occluders, occluder, dst=None, mask=None )
           
            hidden = cv2.bitwise_and(occluded,occluders,dst=None, mask=None)   
            vis_and_hidden = occluded+hidden
            if debug:
                # vis.display_img(hidden, 'occluded region')
                vis.display_img(seg_img, name='segmented')
                vis.display_img(vis_and_hidden, 'vis and occ')

            occ_lst.append((seg_img, vis_and_hidden))

        return  occ_lst 
                
            
class  Compositor:
    """ 
    """
    def __init__(self, bg_img, num_fish, iou_mode, tgt_iou, delta):
        self.torch_device = 'cuda:0'
        self.bg_img = bg_img
        self.c_img = bg_img.copy()

        # image_size: (x,y)
        self.image_size = (bg_img.shape[1],
                           bg_img.shape[0])
        self.crop_size = self.image_size
        self.c_lbl_img = np.zeros((bg_img.shape[0], bg_img.shape[1]), dtype=np.int32)
        self._rotation = []
        self.flip = False # Warning: do not flip obboxs!
        self.iou_mode = iou_mode
        self.tgt_iou = tgt_iou
        self.bboxes = torch.zeros(num_fish, 4, dtype=torch.int)
        self.fish_label_int = 0
        self._scalefactor = [1.0, 1.0]
        self.position = None
        # choose increments for y position
        self.delta = delta
        # crop_info (bbox relating to individual)
        self.crop_info = (0, 0, 0, 0) # x, y, w, h
        self.occ = Occlusion()
        # mhf 10.11.25 dict of composited fish.
        self.c_fish = {}
        
    def state_reset(self):
        bg_img = self.bg_img.copy()
        self.c_img = bg_img.copy()
        self.c_lbl_img = np.zeros((bg_img.shape[0], bg_img.shape[1]), dtype=np.int32)
        self._rotation = []
        self.fish_label_int = 0
        self.position = None
        self.c_fish = {}
        
        
    @property 
    def scalefactor(self):
        return self._scalefactor
    
    @property 
    def rotation(self):
        return self._rotation
    
    @rotation.setter 
    def rotation(self, value):
        self._rotation.append(value)
    
    @scalefactor.setter 
    def scalefactor(self, value):
        self._scalefactor = value
        
    def __update_c_fish(self, fish_id, position, angle, species, tgt_iou):
        """
        Dictionary of parameters for each individual fish composited (c_fish)
        """
        self.c_fish[fish_id] = {
        "position": position,
        "orientation": angle,
        "species": species,
        "tgt_iou": tgt_iou}
        
        
    def __compute_iou_matrix(self, lbl_img):
        # calculate iou matrix
        if self.iou_mode[0] in ['hbbox_iou']:
            boxes, masks = label_image_to_gt(lbl_img)
            boxes = torch.from_numpy(boxes.astype(np.int32))
            iou_mat = box_ops.box_iou(boxes, boxes)
        # elif self.iou_mode in ['mask_iou']:
        #     iou_mat = mask_iou(lbl_img)
        elif self.iou_mode[0] in ['obbox_iou']:
            iou_mat = obbox_iou(lbl_img)
        else:
            raise ValueError("Unknown iou_mode: {}".format(self.iou_mode))

        return iou_mat
    

    def __render_img_canvas(self, img, mask, position, debug=False):
        # update canvas
        position = (position[1], position[0])
        temp_img = add_fg(self.c_img, img.numpy(), mask, position)
        temp_img = temp_img.astype(np.float64)/255
        if debug:
            vis.display_img(temp_img, 'temp_img_canvas')
            
        return temp_img


    def __render_lbl_canvas(self, lbl_img, mask, position, debug=False):
        # update labels
        position=(position[1], position[0])
        lbl_img = lbl_img * (self.fish_label_int+1)        
        temp_img = add_fg(self.c_lbl_img, lbl_img, mask, position)
        temp_img = temp_img.astype(np.int32)
        if debug:
            vis.display_img(temp_img, 'temp_img_lbl_canvas')
            
        return temp_img
    
    def __refine_rotation(self, c_label_img, debug=False):
        """ 
        """
        x,y,w,h = self.crop_info
        position = self.position
        n = self.fish_label_int
        tgt = self.tgt_iou
        # mhf why are we adding 90 degrees?
        last_rot = (self.rotation[n-1] * math.pi / 180) + math.pi/2
        alpha = 10. # 10 degrees
        
        # adjust rot
        # check we can meet tgt iou @ rot
        n_iter = 0
        while True:
            # mhf we also add 90 degrees to rot why?
            rot = (self.rotation[n] * math.pi / 180) + math.pi/2
            rel_rot = (last_rot - rot) % (math.pi * 2)
            tmp_lbl_img = self.__render_lbl_canvas(c_label_img, c_label_img, position) 
            if debug:
                vis.display_img(c_label_img, 'c_label_img') # input
                vis.display_img(self.c_lbl_img, 'self.c_lbl_img') # current canvas
                vis.display_img(tmp_lbl_img, 'tmp_lbl') # poss. new canvas

            iou_mat = self.__compute_iou_matrix(tmp_lbl_img)

            if iou_mat.shape[0] == n+1:
                # fish visible
                iou = iou_mat[n-1,n] 
            else:
                # fish invisible
                iou = 0.

            if debug:    
                print('iter={}, rot={:.2f}, rel_rot={:.2f}, iou={:.2f}'.format(n_iter, self.rotation[n], math.degrees(rel_rot), iou))
                
            if iou > tgt+0.1: 
                if debug:
                    print('Target IoU achievable at rotation={:.2f} deg.'.format(self.rotation[n]))
                result = 0
                break
            # elif abs(rel_rot) < math.radians(alpha) or n_iter > 36:
            elif n_iter > (36+1):
                if debug:
                    print('Target IoU NOT achievabe at rotation={:.2f} deg.'.format(self.rotation[n]))
                result = -1
                break
            else:                
                self.rotation[n] = (self.rotation[n] - alpha) % 360
                # c_label_img, _ = self.__render_lbl_individual(c_label_img,
                #                                               -math.radians(alpha),
                #                                               False,
                #                                               debug=False)
                self.tf_info['rot_theta'] = -math.radians(alpha)
                self.tf_info['flip'] = False
                c_label_img, _, _ = render_lbl_individual(c_label_img,
                                                          self.tf_info)
                
                
            n_iter += 1
                
        return  result
        
          
    def __refine_translation(self, c_label_img, debug=False):
        """ 
        """
        position = self.position
        n = self.fish_label_int
        tgt = self.tgt_iou
        trans_dirn = (self.rotation[n] * math.pi / 180) + math.pi/2
        delta = self.delta
        # n_cycles = 0
        # cycle_thr = 10
        

        if tgt > 0:
            beta = 0.05
        else:
            beta = 0.0
        increase_iou = False
        decrease_iou = False
        iter_cnt = 0
        max_iter = 200
        
    
        while True:
            tmp_lbl_img = self.__render_lbl_canvas(c_label_img, c_label_img, position) 
            if debug:
                # vis.display_img(self.c_lbl_img, 'c_lbl')
                vis.display_img(tmp_lbl_img, 'tmp_lbl')
                
            iou_mat = self.__compute_iou_matrix(tmp_lbl_img)

            if iou_mat.shape[0] == n+1:
                # fish visible
                iou = iou_mat[n-1,n] 
                iou_error = abs(tgt - iou)
            else:
                # fish invisible
                iou_error = 0.
            
            
            if debug:
                print('itr_cnt={}, iou_error={:.2f}, delta={}'.format(iter_cnt, iou_error, delta))
            
            if iou_error <= beta:
                if debug:
                    print('Successful placement')
                self.position = position
                result = 0 
                break
            else:
                # no. of iters exceeded or impossible to meet tgt iou 
                if (iter_cnt > max_iter):
                    if debug:
                        print('aborted translation placement')
                    result = -2
                    break
            
            if (iou < tgt):
                # increase iou
                if decrease_iou:
                    # n_cycles += 1
                    delta = (abs(delta) // 2) * (delta//abs(delta))
                    decrease_iou = False
                increase_iou = True
                step = delta
            elif (iou > tgt):
                # decrease iou
                if increase_iou:
                    # n_cycles += 1
                    delta =  (abs(delta) // 2) * (delta//abs(delta))
                    increase_iou = False
                decrease_iou = True
                step = -delta

            if abs(delta) > 0:
                step_x = step
                step_y = step
            else:
                # reposition
                if debug:
                    print('delta: {}, Repositioning!!!!!'.format(delta))
                step_x = random.choice([-self.delta, +self.delta])
                step_y = random.choice([-self.delta, +self.delta])
                decrease_iou = False
                increase_iou = False
                delta = self.delta # // abs(self.delta)

            x, y = position 
            dx, dy = step_x * math.cos(trans_dirn), step_y * math.sin(trans_dirn)
            position = x + round(dx), y + round(dy)                     
            iter_cnt += 1
        
        return result       
        
    def add_fish_to_image(self, fish, debug=False):
        """ 
        """
        fish_label_filename = fish['Fname']
        fish_image_filename = FishRepo.fish_image_file(fish_label_filename)
        single_limg = labelling_tool.PersistentLabelledImage(
                          fish_image_filename,
                          fish_label_filename)
        pixels = np.transpose(single_limg.pixels, (2, 0, 1))
        fish_image = pixels
        if debug:
            vis.display_img(fish_image, name='ind. Fish')
        label_img, _ = single_limg.render_label_instances(
            [fish['Species']], 
            multichannel_mask=False)
        
        original_image_shape = label_img.shape[:2]
               
        # rotation
        angle = float(fish['Orientation'])
        # angle = angle + math.pi
        rot_theta =  math.radians(self.rotation[self.fish_label_int]) - angle

        if debug:
            print('Area = {}'.format(fish['Area']))
            print('Rotate Angle = {:.2f} deg.'.format(math.degrees(rot_theta)))

        # position of first placement
        if self.position is None:
            self.position = [self.image_size[1]//2, self.image_size[0]//2]
            result = 0

        #transform_info
        self.tf_info = {
            "scalefactor": self.scalefactor,
            "image_size": self.image_size,
            "position": self.position,
            "rot_theta": rot_theta,
            "flip": self.flip}        
            
        c_label_img, xf, self.crop_info = render_lbl_individual(label_img,
                                                                self.tf_info)

        if debug:
            vis.display_img(c_label_img, 'cropped')
            # assess_approx_contour(c_label_img.astype(np.uint8))
            # assess_morph_proc(c_label_img.astype(np.uint8))
            # morph_opening(c_label_img.astype(np.uint8))
            # approx_contour(c_label_img.astype(np.uint8))
       
        if self.fish_label_int > 0:
            result = self.__refine_rotation(c_label_img)
            if result == 0:
                self.tf_info['rot_theta'] =  (self.rotation[self.fish_label_int] * math.pi / 180) - angle
                self.tf_info['flip'] = False
                c_label_img, xf, self.crop_info = render_lbl_individual(label_img,
                                                        self.tf_info)                
                result = self.__refine_translation(c_label_img)

        if result == 0:
            c_label_img = morph_opening(c_label_img.astype(np.uint8))
            c_label_img = approx_contour(c_label_img.astype(np.uint8))
            
            
            x,y,w,h = self.crop_info
            # mhf not sure why we're using this method
            xf = cv_to_torch(xf, self.crop_size, original_image_shape)
            # build tensors
            fish_image = torch.tensor(fish_image, dtype=torch.float, device=self.torch_device)
            xf = torch.tensor(xf, dtype=torch.float, device=self.torch_device)
            grid = F.affine_grid(xf, (1, fish_image.shape[1], self.crop_size[0], self.crop_size[1]), align_corners=False)
            fish_image = F.grid_sample(fish_image[None, :], grid, align_corners=False)
            fish_image = np.transpose(fish_image[0].cpu(), (1, 2, 0))
            c_fish_img = fish_image[y:y+h, x:x+w]
            if debug:
                vis.display_img(c_fish_img, 'cropped')
    
            
    
            # update the canvas        
            self.c_img = self.__render_img_canvas(c_fish_img, c_label_img, self.position) 
            self.c_lbl_img = self.__render_lbl_canvas(c_label_img, c_label_img, self.position)

    
            if debug:
                vis.display_img(self.c_img, 'c_img_{} iou: {:.2f}'.format(self.fish_label_int, self.tgt_iou))
                vis.display_img(self.c_lbl_img, 'c_lbl_img_{} iou: {:.2f}'.format(self.fish_label_int, self.tgt_iou))


            self.__update_c_fish('fish{}'.format(self.fish_label_int), 
                                 self.position, 
                                 self.rotation[self.fish_label_int],
                                 fish['Species'],
                                 self.tgt_iou)
        
            self.fish_label_int += 1

            # self.occ.fish_label_int += 1
            # self.occ.c_lbl_img = self.c_lbl_img
            self.occ.update(self.fish_label_int, 
                            self.c_lbl_img.copy(), 
                            self.c_img.copy())

       
        return result
       
