# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 11:03:33 2026

@author: mhf
"""
import os
import cv2
import numpy as np
from skimage import transform
import glob
from tqdm import tqdm

from .compositor import CompositeImage
from ..torchvision_utils.utils import label_image_to_gt
from ..debug_utils import visualize as vis
from ..synth_utils.repo_mgt import FishRepo
from ..synth_utils.utils import OutputWriter # , box_density 
from ..synth_utils.obb_utils import get_obbs


class CompositeDataset():
    """
    """
    from ..constants import CLS_FISH_CLASSES, IMAGES_FOLDER_NAME, \
            LABELS_FOLDER_NAME, DOTA_ANNOTATION_FOLDER_NAME
    
    
    def __init__(self,
                 params):
        self.params = params
        self._belt_name = None
        self._belt_bg_img = None
        self.show_output = params.getboolean('show', False)
        self.output_folder = params.get('output_folder', None)
        self.dest_belts = params['dest_belts']
        self.images_per_tgt_iou = images_per_tgt_iou = int(params['images_per_tgt_iou'])
        self.fish_per_image = fish_per_image = eval(params['fish_per_image'])

        self.rotation_mode = params['rotation_mode']      
        self.rotation = eval(params['rotation'])

        self.op_mode = params['op_mode']
        self.area_thresh = int(params.get('area_thresh', 100))
        self.iou_mode = eval(params['iou_mode'])
        # check tgt_iou values are ok
        tgt_iou = eval(params.get('tgt_iou', '[0.0]'))
        ok_iou = [round(x * 0.1, 1) for x in range(6)]
        for iou in tgt_iou:
            assert iou in ok_iou, 'iou value error: {}'.format(iou)
        self.tgt_iou = tgt_iou        
        self.background_images = glob.glob(params['backgrounds']+'*.png')
        self.supp_output = eval(params['supp_output'])

        if self.op_mode == 'train':
            infile = 'train.csv'
        elif self.op_mode == 'val':
            infile = 'val.csv'
        elif self.op_mode == 'test':
            infile = 'test.csv'
        else:
            raise RuntimeError('Unknown op_mode')        

        splits_folder = params['single_whole_fish_root']
        source_belts = eval(params.get('source_belts', None))
        source_species = eval(params.get('source_species', self.CLS_FISH_CLASSES))
        self.fish_iter = FishRepo(splits_folder, 
                                fname=infile, 
                                fish_per_image = fish_per_image,
                                images_per_tgt_iou = images_per_tgt_iou,
                                belts=source_belts, 
                                species=source_species)

        # class to handle output        
        self.output_writer = OutputWriter(self.iou_mode)
       
        return

    @property
    def belt_name(self):
        """Getter for the belt_id property."""
        return self._belt_name
    
    @belt_name.setter
    def belt_name(self, value):
        """Setter for the belt_name property."""
        if not value:
            raise ValueError("Belt_name cannot be empty!")
        self._belt_name = value

    @property        
    def belt_bg_img(self):
        """Getter for the belt_bg property """
        return self._belt_bg_img
    
    @belt_bg_img.setter
    def belt_bg_img(self, value):
        """Setter for the belt_bg property """
        if value is None:
            raise ValueError("Belt_bg_img cannot be empty")
        self._belt_bg_img = value
        
    def bg_img(self, belt_name):
        bg = None
        for i, path in enumerate(self.background_images):
            if self.belt_name in path:
                bg = cv2.imread(self.background_images[i])[:, :, ::-1]
                bg = transform.resize(bg, eval(self.params['canvas_size']))
                break
        
        assert bg is not None, 'Failed to read background image' 

        return bg
    
    def write_output(self, img_i, tgt_iou, img, lbl_img, c_fish, occ_info, debug=False):
        iou_name = 'iou_' + str(int(tgt_iou*10))
        # filename = iou_name + 'image_{}'.format(img_i)
        filename = 'P{}{num:05d}'.format(int(tgt_iou*10), num=img_i)
        iou_output_folder = os.path.join(self.output_folder, self.belt_name, iou_name, self.op_mode)
        all_output_folder = os.path.join(self.output_folder, self.belt_name, 'all', self.op_mode)

        for output_folder in [iou_output_folder, all_output_folder]:
            
            frames_folder = os.path.join(output_folder, self.IMAGES_FOLDER_NAME)
            labels_folder = os.path.join(output_folder, self.LABELS_FOLDER_NAME)
            dota_annotation_folder = os.path.join(output_folder, self.DOTA_ANNOTATION_FOLDER_NAME)


            # write image with json labels and DOTA annotation txt
            self.output_writer.write_coco(img, 
                                          lbl_img, 
                                          filename, 
                                          frames_folder, 
                                          labels_folder)
            self.output_writer.write_dota(img,
                            get_obbs(lbl_img, c_fish),
                            filename,
                            frames_folder,
                            dota_annotation_folder, # annotation_folder
                            c_fish)

        output_folder = iou_output_folder
        # output some occlusion data (masks + indv. fish)
        if self.supp_output['occ_folder']:
            occlusion_folder = os.path.join(output_folder, self.supp_output['occ_folder'])
            self.output_writer.write_occlusions(occ_info, occlusion_folder, filename)
            
        # write some images that can be used in pubs
        if self.supp_output['fig_folder'] and self.op_mode in ['test']:
            fig_folder = os.path.join(output_folder, self.supp_output['fig_folder'] )
            os.makedirs(fig_folder, exist_ok=True)
            obb_tgt = 'iou: {}'.format(tgt_iou) if self.iou_mode[0] == 'obbox_iou' else ''
            hbb_tgt = 'iou: {}'.format(tgt_iou) if self.iou_mode[0] == 'hbbox_iou' else ''
            if 'obbox_iou' in self.iou_mode:
                obbs = get_obbs(lbl_img, c_fish)
                vis.show_obbs(img, obbs, '{} {}'.format(img_i, obb_tgt), fig_folder)
            if 'hbbox_iou' in self.iou_mode:
                boxes, masks = label_image_to_gt(lbl_img)
                vis.show_bboxes(img, boxes, '{} {}'.format(img_i, hbb_tgt), fig_folder)

            if debug:
                for j, occ in enumerate(occ_info):
                    vis.display_img(occ[0], name = 'img_{}_iou_{}_comp_{}'.format(img_i, tgt_iou, j), folder = fig_folder)
                    vis.display_img(occ[1], name = 'img_{}_iou_{}_mask_{}'.format(img_i, tgt_iou, j), folder = fig_folder)

        
        
        
    
    def composite_images(self, belt_name, tgt_iou=None, debug=False):
        num_images = self.images_per_tgt_iou
        
        self.belt_name = belt_name
        belt_bg = self.bg_img(belt_name)
        failed_composits = 0
        
        for img_i in tqdm(range(num_images), desc='Compositing IoU: {}'.format(tgt_iou)):
        # for img_i in range(num_images):

            if debug:
                print(f"Image: {img_i}")

            # set num                
            if len(self.fish_per_image) == 2:
                num_fish = np.random.randint(self.num_fish_per_image[0], self.num_fish_per_image[1])
            else:
                num_fish = self.fish_per_image[0]
                
            # instatiate CompositeImage object
            Cimg = CompositeImage(                
                num_fish,
                self.fish_iter,
                belt_bg,
                self.iou_mode,
                self.rotation_mode,
                self.rotation,
                area_thresh=self.area_thresh,
                tgt_iou = tgt_iou
                )
            
            img, lbl_img, c_fish, occ_info = Cimg.run()
            failed_composits += Cimg.fail_cnt
            
            if self.show_output:
                vis.display_img(img, name = 'c_lbl_img_{}_iou_{}'.format(img_i, tgt_iou))
            
            if self.output_folder:
                self.write_output(img_i, tgt_iou, img, lbl_img, c_fish, occ_info)                   


        print('\nNumber of Substituted Fish: {}'.format(failed_composits))
        return
        
    
    def run_placement(self, belt_name, debug=True):
        for iou in eval(self.params['tgt_iou']):
            self.composite_images(belt_name, tgt_iou=iou)
            
        return