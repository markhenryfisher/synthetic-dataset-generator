# -*- coding: utf-8 -*-
"""
Shpw belt data annotation (UEA json annotation format).
e.g. Usage: 
    python belt_data_browser.py "./data/ruth/datasets/belt_data_natural" "MRV SCOTIA" 

Created on Thu Feb  1 10:44:30 2024

@author: mhf@uea.ac.uk
@filename: belt2coco_from_json.py

last update: 20.06.26

Spyder command line...
runfile('S:/mhf/GitHub/synthetic-dataset-generator/tools/dataset_converters/belt2coco_from_json.py', \
                 wdir='S:/mhf/GitHub/synthetic-dataset-generator')

"""
# import argparse
import configparser
import os
import numpy as np
import cv2
import random
import time
import json
from PIL import Image
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from syndatasetgen.misc_utils import file_utils
from syndatasetgen.constants import CLS_FISH_CLASSES

# minimum area of a region (smaller areas raise an error)
MIN_AREA = 20

############## folder renaming utils ##########
# Constants for file / folder renaming
IMAGES_FOLDER_NAME = 'label_frames'
LABELS_FOLDER_NAME = 'seg_labels_json'
IMAGE_SUFFIX = '.png'
LABEL_SUFFIX = '__labels.json'

# label types
PRIMITIVE_LABELS = ['polygon']
COMPOUND_LABELS = ['group']

def label_to_image(label):
    return label.replace(LABEL_SUFFIX, IMAGE_SUFFIX).replace(LABELS_FOLDER_NAME, IMAGES_FOLDER_NAME)

def image_to_label(image):
    return image.replace(IMAGE_SUFFIX, LABEL_SUFFIX).replace(IMAGES_FOLDER_NAME, LABELS_FOLDER_NAME)

def today_yyyymmdd():
    # Get current date and format it as YYYYMMDD
    today_str = datetime.now().strftime("%Y%m%d")
    
    return(today_str)

############### image plotting ##############
    
def display(img, contours=None, show=True):
    """
    render / display an image (with optional contour plot)

    Parameters
    ----------
    img : TYPE
        DESCRIPTION.
    contours : TYPE, optional
        DESCRIPTION. The default is None.
    show : TYPE, optional
        DESCRIPTION. The default is True.

    Returns
    -------
    out_img : np array (RGB image)
        DESCRIPTION.

    """
    if img.dtype == np.bool_:
        # convert binay mask to image
        gray = img.astype(np.float32)
        img = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)       
        
    h, w = img.shape[:2]
    
    # Display the image and plot all contours found
    fig, ax = plt.subplots()

    # Display image so it nicely fills the figure
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)    
    # Get the size of the data
    dim = np.array(img.shape[:2][::-1]) + np.array([0, 0.5])
    # Set the figure sizes in inches
    fig.set_size_inches(dim / fig.dpi)
   
    ax.imshow(img)
    # mhf 22.12.25 bugfix
    x_min = w
    x_max = 0
    y_min = h
    y_max = 0  
    if contours is not None:
        # ax.imshow(img)
        for c in contours:
            px = c[:, 0]; py = c[:, 1]
            
            x_min = min( x_min, np.min(px) )
            x_max = max( x_max, np.max(px) )
            y_min = min( y_min, np.min(py) )
            y_max = max( y_max, np.max(py) )
                
                
            ax.plot(px, py, 'w', linewidth=2)

        # Create a Rectangle patch
        rect = patches.Rectangle((x_min, y_min), x_max-x_min, y_max-y_min, linewidth=2, edgecolor='r', facecolor='none')

        # Add the patch to the Axes
        ax.add_patch(rect)


    ax.axis('image')
    ax.set_xticks([])
    ax.set_yticks([])

    
    s, (width, height) = fig.canvas.print_to_buffer() 
    
    # Convert to a NumPy array.
    out_img = np.frombuffer(s, np.uint8).reshape((height, width, 4))    
    out_img = cv2.cvtColor(out_img, cv2.COLOR_RGBA2RGB)
    out_img = cv2.resize(out_img, (w, h))

    if show:    
        plt.show()
    
    plt.close(fig)
        
    return out_img

def generate_splits(img_list):
    # randomize dataset and split into train and test 
    n_train = 9 * len(img_list) // 10 
    n_test = len(img_list) - n_train  
    idxs = list(range(len(img_list)))
    # get the same random behaviour every time
    random.seed(1)
    random.shuffle(idxs)
    train_idxs = idxs[:n_train]
    test_idxs = idxs[-n_test:]
    
    train_list = [img_list[x] for x in train_idxs]
    val_list = [img_list[x] for x in test_idxs]
    
    return train_list, val_list


def fix_polygon_vertices(regions, img_size):
    
    def clip_polygon(region, img_size):
        w, h = img_size 
        region[:,0] = np.clip(region[:,0], a_min=0, a_max=w-1)
        region[:,1] = np.clip(region[:,1], a_min=0, a_max=h-1)
        
        return region
    
    def close_polygon(region):    
        region = np.rint(region)
        # close the contour
        closed = region[0,:] == region[-1,:]
        if not closed.all():
            region = np.vstack([region, region[0,:]])
            
        return region
                
    for i, region in enumerate(regions):
        region = close_polygon(region)
        region = clip_polygon(region, img_size)
        regions[i] = region
        
    return regions




def unpack_polygon(label_json):
    
    def verify_area(regions, debug=False):
        filtered_regions = []
        for c in regions:
            px = c[:, 0]; py = c[:, 1]
            
            x_min = np.min(px)
            x_max = np.max(px)
            y_min = np.min(py)
            y_max = np.max(py)
            
            area = (x_max - x_min) * (y_max - y_min)
            
            if area < MIN_AREA:
                if debug:
                    print('Unpacking Error: Region area too small')
            else:
                filtered_regions.append(c)

        return filtered_regions
        
    def verify_regions(regions_json):
        for region in regions_json:
            if len(region) > 3:
                pass
            else:
                raise RuntimeError('Unpacking Error: Insufficient Vertices')                              
        return
       
    if 'vertices' in label_json:
        regions_json = [label_json['vertices']]
    else:
        regions_json = label_json['regions']

    verify_regions(regions_json)        
    regions = [np.array([[v['x'], v['y']] for v in region_json]) for region_json in regions_json]
    regions = verify_area(regions)

    return regions
      

def convert_label(mask, label, debug=False):
    h, w = mask.shape[:2]
        
    # simple polygon
    if label['label_type'] in PRIMITIVE_LABELS:
        if debug:
            print('{} {}'.format(label['label_type'], label['object_id']))
        regions = unpack_polygon(label)
        regions = fix_polygon_vertices(regions, (w,h))
        if debug:
            mask = display(mask, regions, show=True)
    # group of polygon 'components'
    elif label['label_type'] in COMPOUND_LABELS:
        if debug:
            print(label['label_type'])
            print(label['object_id'])
        for component in label['component_models']:
            if debug:
                print(component['label_type'])
                print(component['object_id'])
            regions = convert_label(mask, component)
    else:
        raise RuntimeError('Unknown label type')
        
    return regions
        
    
def convert2coco(img_list, img_folder, out_file, debug=False):
    annotations = []
    images = []
    obj_count = 0 
    
    for idx in range(len(img_list)):
    # for idx in range(0,7):
        img_filename = img_list[idx]
        img_path = os.path.join(img_folder, img_filename)
        lbl_path = image_to_label(img_path) 
      
        img = Image.open(img_path).convert("RGB")
        width, height = img.size 

        with open(lbl_path, 'r', encoding='utf-8') as file:
                    labels = json.load(file)  # Parse JSON into Python dict/list
        
        if debug:
            if len(labels) != 2:
                print('idx: {}, File name: {}'.format(idx, img_filename))
                print('Num fish: {}'.format(len(labels)))
        else:
           print('idx: {}, File name: {}'.format(idx, img_filename)) 
        in_fish = len(labels)
        out_fish = 0
        for i, label in enumerate(labels):

            has_poly = False
            regions = convert_label(np.array(img), label, debug=not(out_file))
            if not(out_file):
                time.sleep(2)
                
            # if debug:
            #     print('Num Regions: {}'.format(len(regions)))

            # mhf 22.12.25 bugfix
            x_min = width
            x_max = 0
            y_min = height
            y_max = 0                
            seg = []
            for c in regions:
                has_poly = True
                px = c[:, 0]; py = c[:, 1]
                x_min = min(x_min, np.min(px))
                x_max = max(x_max, np.max(px))
                y_min = min(y_min, np.min(py))
                y_max = max(y_max, np.max(py))
                
                poly = [(x, y) for x, y in zip(px, py)]
                poly = [p for x in poly for p in x]
                seg.append(poly)

            data_anno = dict(
                image_id=idx,
                id=obj_count,
                category_id=CLS_FISH_CLASSES.index(label['label_class']),
                bbox=[x_min, y_min, x_max - x_min, y_max - y_min],
                area=(x_max - x_min) * (y_max - y_min),
                segmentation=seg,
                iscrowd=0)
        
            annotations.append(data_anno)
            obj_count += 1
            out_fish += 1
        
        assert in_fish == out_fish, 'Conversion Error!'
        # image contains at least one contour
        if out_file and has_poly:
            # add image to list
            basename, __ = os.path.splitext(img_filename)
            jpg_filename = basename + '.jpg'
            images.append(dict(id=idx, file_name=jpg_filename, height=height, width=width))
            # save jpg image 
            fullfile = os.path.join(os.path.dirname(out_file), jpg_filename)
            # assert not os.path.isfile(fullfile), '{} Exists!'.format(fullfile)
            img.save(fullfile)
        else:
            time.sleep(2)
            
    if debug:
        print('Obj cnt: {}'.format(obj_count))

    temp = []        
    for i, species in enumerate(CLS_FISH_CLASSES):
        temp.append(dict(name=species, id=i))

     
    coco_format_json = dict(
    info={"description": "CFISH 2025 Dataset","url": "http://uea.ac.uk","version": "1.0","year": 2025,"contributor": "School of Computing Sciences","date_created": today_yyyymmdd()},
    images=images,
    annotations=annotations,
    categories=temp)
    
    if out_file:
        # assert not os.path.isfile(out_file), '{} Exists!'.format(out_file)
        with open(out_file, "w") as json_file:
            json.dump(coco_format_json, json_file)

    
if __name__ == '__main__':
    # args = parse_args()
    # args.belt=args.belt.replace('_', ' ')
    config=configparser.ConfigParser()
    config.read('./configs/config.ini')    
    params = config['convert2coco.example']
    
    dataset = params.get('data_root', None)
    out_folder = params.get('out_folder', None)

    
    img_folder = os.path.join(dataset, 'label_frames')
    seg_labels_folder = os.path.join(dataset, 'seg_labels_json')
    img_list = file_utils.list_files_by_extension(img_folder, '.png')


    if out_folder:
        os.makedirs(out_folder, exist_ok=True)
        out_file = os.path.join(out_folder, 'annotation_coco.json')
    else:
        out_file = None
    
    convert2coco(img_list, 
                  img_folder, 
                  out_file = out_file)

     
