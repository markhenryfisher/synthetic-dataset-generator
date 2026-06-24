# -*- coding: utf-8 -*-
"""
Created on Sun Jul 13 10:33:56 2025

@author: mhf
"""
import os
import glob
import csv
import math
import numpy as np
import random
import cv2
from tqdm import tqdm

from ..torchvision_utils.utils import label_image_to_gt
from image_labelling_tool import labelling_tool
from ..synth_utils import pca
from ..debug_utils import visualize as vis


class FishRepo:
    """
    mhf 12.07.25 Methods for managing Ruth's Whole Fish Repository
    """
    from syndatasetgen.constants import CLS_FISH_CLASSES, CLS_BELTS, IMAGE_SUFFIX, \
        CSV_SUFFIX, LABEL_SUFFIX, IMAGES_FOLDER_NAME, \
            LABELS_FOLDER_NAME, AREAS_FOLDER_NAME
            
    FIELDNAMES = ["Folder", "Fname", "Belt", "Species", "BBox", "Orientation", "Length"]

    
    def __init__(self, 
                 root,
                 fname=None,
                 fish_per_image=None,
                 images_per_tgt_iou=None,
                 belts=None,
                 species=None
                 ):
        # directory containing inventroy.csv, train.csv, test.csv
        self.root = root

        self.label_dir =  'seg_labels_json'
        self.image_dir = 'label_frames'
        self.belt_dir = 'all'

        self.belts = FishRepo.CLS_BELTS if belts is None else belts
        self.species = FishRepo.CLS_FISH_CLASSES if species is None else species

        self.num_fish_reqd = fish_per_image[-1] * images_per_tgt_iou if fish_per_image else None

        self.data = None if fname is None else self.__init_data(self.read_csv(fname))
    
        self.index = 0
        
    @property
    def num_fish(self):
        return len(self.data)

    def __init_data(self, data):
        data = self.__filter(data)
            
        return data
    
    def __filter(self, data):
        result = []
        for dict_entry in data:
            if dict_entry["Belt"] in self.belts and dict_entry["Species"] in self.species:
                result.append(dict_entry)
        print('{} fish loaded'.format(len(result)))        
        if self.num_fish_reqd and len(result) < self.num_fish_reqd:
            print('Insufficient Fish to meet compositing spec.')
            
        return result
                
    def write_inventory(self, debug=False):
        root = os.path.join(self.root, self.label_dir, self.belt_dir)
        labels = list(sorted(glob.glob(os.path.join(root, '*'+FishRepo.LABEL_SUFFIX))))
        # get corresponding images
        images = []
        for label in labels:
            images.append(FishRepo.label_to_image(label))

        data=[]            
        for label_fname, img_fname in zip(tqdm(labels), images):
            limg = labelling_tool.PersistentLabelledImage(img_fname, label_fname)
            label_img, _ = limg.render_label_instances(FishRepo.CLS_FISH_CLASSES, multichannel_mask=False)

            angle = FishRepo.alignment(limg)
            
            angle, length = FishRepo.componsate_pose(angle, label_img)
            # only expect one label per image
            boxes, masks = label_image_to_gt(label_img)
            assert len(boxes) == 1, 'Expeccting only one fish per limg'
                
            mask_area = np.sum(masks[0])
            

            species = FishRepo.species_substring(label_fname)
            belt = FishRepo.belt_substring(label_fname)
            fname = os.path.basename(label_fname)
            
            dict_entry = {"Folder": self.belt_dir, 
                         "Fname": fname, 
                         "Belt": belt, 
                         "Species": species, 
                         "BBox": boxes[0].tolist(),
                         "Orientation": round(angle,2), 
                         "Length": round(length,2),
                         "Area": mask_area
                         }
            
            if debug:
                print('Folder: {}, Fname: {}, Belt: {}, Species: {}, BBox: {}, Orientation: {:.2f}, Length: {:.2f}'.format(self.belt_dir, fname, belt, species, boxes[0].tolist(), math.degrees(angle), length))
        
            data.append(dict_entry)
            
        # Write CSV file
        self.write_csv('inventory.csv', data)
        
        return
        
    def read_csv(self, fname):
        data = []
        filename = os.path.join(self.root, fname)
        with open(filename, mode='r') as file:
            reader = csv.DictReader(file)
            data = [row for row in reader]  # List of dictionaries
                
        return data
    
    def write_csv(self, fname, data):
        filename = os.path.join(self.root, fname)
        if os.path.exists(filename):
            raise ValueError('{} exists! Please delete'.format(filename))
        with open(filename, mode="w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=data[0].keys())
            
            writer.writeheader()
            writer.writerows(data)
        print()
        print("{} written successfully!".format(fname))
        
        return
            
    def write_splits(self, split):
        """
        write train, validate, test splits
        """
        
        data = self.data
        if data is None:
            data = self.read_csv('inventory.csv')
               
        total_num_fish = len(data)
        num_train_fish = int(total_num_fish * split[0] / 100)
        num_val_fish = int(total_num_fish * split[1] / 100)
        num_test_fish = int(total_num_fish * split[2] / 100)
        random.shuffle(data)


        train_data = data[:num_train_fish]
        val_data = data[num_train_fish : num_train_fish+num_val_fish]
        test_data = data[num_train_fish+num_val_fish:]
        
        for fname, data in zip(['train.csv', 'val.csv', 'test.csv'], [train_data, val_data, test_data]):
            self.write_csv(fname, data)
                    
        return
        
    def read_data(self, infile):
        assert infile in ['train.csv', 'test.csv']
        
        return self.read_csv(infile)


    def next_item(self, area=None, idx=None, debug=False):

        if area:
            data = self.data
            list1 = [int(area)-int(d["Area"]) for d in data]
            list2 = data
            # Sort both lists based on the order of list1
            # positive_numbers = [num for num in numbers if num >= 0]
            temp = zip(list1, list2)
            pos_tuple = (data_tuple for data_tuple in temp if data_tuple[0] >=0)

            # https://stackoverflow.com/questions/9764298/           
            result1, result2 = zip(*sorted(pos_tuple, key=lambda x: x[0]))
            dict_entry = result2[idx]
            if debug:
                print('Substitute Fish: Area: {}'.format(dict_entry['Area']))
        else:
            dict_entry = self.data[self.index]
            self.index += 1
            if self.index >= self.num_fish:
                print('Warning: Insufficient Fish to meet compositing spec...reseting index!')
                self.index = 0


        # complete Fname 
        basename=os.path.basename(dict_entry['Fname'])
        dict_entry['Fname'] = os.path.join(self.root, self.label_dir, self.belt_dir, basename)
        # add index key
        # dict_entry['Index'] = self.index

        
        return dict_entry
        
    @staticmethod
    def species_substring(f):
        for s in FishRepo.CLS_FISH_CLASSES:
            if s in f:
                return s
            
        raise ValueError('Unknown species: {}'.format(f))
                
    @staticmethod
    def belt_substring(f):
        for b in FishRepo.CLS_BELTS:
            if b in f:
                return b
        
        raise ValueError('Unknown belt: {}'.format(f))
        
    @staticmethod 
    def fish_image_file(fish_label_filename):
        parts = os.path.split(fish_label_filename)
        single_whole_fish_image_dir = parts[0].replace('seg_labels_json', 'label_frames')
        fish_image_filename = os.path.join(single_whole_fish_image_dir,
                     parts[-1].replace(FishRepo.LABEL_SUFFIX,
                                       FishRepo.IMAGE_SUFFIX))
        return fish_image_filename      

    @staticmethod
    def alignment(single_labelled_fish_image):
        """ 
        # Find orientation using principal components
        #
        """
        
        labels = single_labelled_fish_image.labels
        assert len(labels) == 1, 'only expecting ONE polygon label'
        # mhf contour pts
        # if there are > 1 regions (there shouldn't be)
        # then get the longest contour
        npts = [len(region) for region in labels[0].regions]
        m = max(npts)
        n = npts.index(m)            
        region = labels[0].regions[n]
        
        pts = region.reshape(-1, 1, 2)    
        _, angle = pca.getOrientation(pts, single_labelled_fish_image.pixels)
        
        return angle
    
    @staticmethod 
    def componsate_pose(angle, label_img, debug=False):
        """ 
        angle found by PCA is only principal axis. It is invarient to fish pose.
        Beware! This function will only determine pose for long fish!
        """
        IMG_SZ = (512, 512)
        from ..core.compositor import render_lbl_individual
        
        if debug:
            vis.display_img(label_img)
        
        # transform info    
        tf_info = {
            "scalefactor": [1.0, 1.0],
            "image_size": IMG_SZ,
            "position": [IMG_SZ[1]//2, IMG_SZ[0]//2],
            "rot_theta": -angle,
            "flip": False}
            
        c_label_img, xf, crop_info = render_lbl_individual(label_img,
                                                           tf_info)
        sz_y, sz_x = c_label_img.shape
        
        if debug:
            vis.display_img(c_label_img)
            
        moments = cv2.moments(c_label_img.astype(np.uint8))
        
        # Calculate skewness
        # if moments['mu20'] > 0 and moments['mu02'] > 0: # Avoid division by zero
        #     skew_x = moments['mu30'] / (moments['mu20'] ** 1.5)
        #     skew_y = moments['mu03'] / (moments['mu02'] ** 1.5)
        #     print(f"Skewness in x-direction: {skew_x}")
        #     print(f"Skewness in y-direction: {skew_y}")
        # else:
        #     print("Skewness cannot be calculated due to zero central moments.")

        if moments['mu20'] > 0: # Avoid division by zero
            skew_x = moments['mu30'] / (moments['mu20'] ** 1.5)
            if debug:
                print(f"Skewness in x-direction: {skew_x}")
        else:
            raise RuntimeError("Skewness cannot be calculated due to zero central moments.")
                     
        if skew_x < 0:
                angle = angle + math.pi
            
        return angle, sz_x

    @staticmethod 
    def label_to_image(label):
        return label.replace(FishRepo.LABEL_SUFFIX, FishRepo.IMAGE_SUFFIX).replace(FishRepo.LABELS_FOLDER_NAME, FishRepo.IMAGES_FOLDER_NAME)
    
    @staticmethod      
    def image_to_label(image):
        return image.replace(FishRepo.IMAGE_SUFFIX, FishRepo.LABEL_SUFFIX).replace(FishRepo.IMAGES_FOLDER_NAME, FishRepo.LABELS_FOLDER_NAME)

    @staticmethod     
    def label_to_csv(label):
        return label.replace(FishRepo.LABEL_SUFFIX, FishRepo.CSV_SUFFIX).replace(FishRepo.LABELS_FOLDER_NAME, FishRepo.AREAS_FOLDER_NAME)

