# -*- coding: utf-8 -*-
"""
catch monitor (cm) to everyfish_coco dataset converter
@date 25.06.26

@author: Hang.Zhou@uea.ac.uk, Mark.Fisher@uea.ac.uk

Spyder command line...
runfile('S:/mhf/GitLab/synthetic-dataset-generator/tools/dataset_converters/belt2everyfish_coco.py', \
                 wdir='S:/mhf/GitLab/synthetic-dataset-generator')
"""
import configparser
import os
from tqdm import tqdm
import json
from syndatasetgen.misc_utils import file_utils


def create_template():
    IMAGE_DICTIONARY_TEMPLATE = {'license': None, 
                    'file_name': None,
                    'width': None,
                    'height': None,
                    'coco_url': None,
                    'date_captured': None,
                    'flickr_url': None,
                    'id': None,
                    'annotation_types': []}

    ANNOTATION_DICTIONARY_TEMPLATE = {'segmentation': None,
                         'bbox': None,
                         'area': None,
                         'length': None,
                         'weight':  None,
                         'iscrowd': None,
                         'image_id': None,
                         'category_id': None,
                         'annotator_type': 'human',
                         'id': None}
       
    COCO_TEMPLATE = {'image': IMAGE_DICTIONARY_TEMPLATE, 
                    'annotation':[]}
    return ANNOTATION_DICTIONARY_TEMPLATE, COCO_TEMPLATE

def save_json(output_folder, value, file_name):
    save_file = open('{}/{}'.format(output_folder, file_name.replace('png','json')), "w")
    json.dump(value, save_file, indent=6)
    save_file.close()

def transfer_anno_cm2coco(cm, coco):
    # transfer instance-wise annotion e.g. object_id, bbox.
    for annotation_cm in cm['labels']:
        # if we want all objects have species_id and segmentation mask, uncomment it 
        #if annotation_cm['label_class'] not in FISH_SPECIES_FAO.keys():
        #    continue
        anno_instance, _ = create_template()
        'some instance have no regions but have vertices instead'
        #try:
        #    anno_instance['segmentation'] = annotation_cm['regions']
        #    anno_instance['id'] = annotation_cm['object_id']
        #    anno_instance['category_id'] = annotation_cm['label_class']
        #    coco['annotation'].append(anno_instance)
        #except:
        #    continue
        if 'regions' in annotation_cm.keys() or 'vertices' in annotation_cm.keys():
            try: 
                anno_instance['segmentation'] = annotation_cm['regions']
            except:
                anno_instance['segmentation'] = [annotation_cm['vertices']]
        else:
            continue
        anno_instance['id'] = annotation_cm['object_id']
        anno_instance['category_id'] = annotation_cm['label_class']
        # using FAO code as category_id
        # anno_instance['category_id'] = FISH_SPECIES_FAO[annotation_cm['label_class']]
        coco['annotation'].append(anno_instance)
    return coco

def transfer_cm2coco(cm, coco, species_dict):
    # transfer image-wise attributes e.g. file name

    # image level
    coco['image']['file_name'] = cm['image_name']

    # convert completed_tasks to ['classification', 'segmentation', 'detection', 'geometry measurement']
    # as CatchMonitor only has the below 2 tasks completed    
    if "species_id" in cm.get('completed_tasks', ''):
        coco['image']['annotation_types'].append('classification')
    
    if "segmentation" in cm.get('completed_tasks', ''):
        coco['image']['annotation_types'].append("segmentation")

    # # object level
    old_number_objects = len(cm['labels'])
    # # count number of species in CM
    # for annotation_cm in cm['labels']:
    #     if annotation_cm['label_class'] not in species_dict.keys():
    #         species_dict[annotation_cm['label_class']] = annotation_cm['label_class']
    coco = transfer_anno_cm2coco(cm, coco)
    new_number_objects = len(coco['annotation'])

    return coco, old_number_objects, new_number_objects
    
def convertor_patch(input_folder, output_folder, img_list, anno_list):
    species_dict = {}
    # f = open('error_anno.txt', 'w')
    # annotations = open(annotation_list, 'r')
    # annos = annotations.readlines()
    total_number_objects_cm = 0
    total_number_objects_coco = 0
    for anno, img in zip(tqdm(anno_list), img_list):
        json_path = input_folder + '/' + anno
        temp = json.load(open(json_path, 'r'))
        cm = {"labels": temp,
              "image_name": img}
        _, coco = create_template()
        coco_anno, old_number_objects, new_number_objects = transfer_cm2coco(cm, coco, species_dict)

        total_number_objects_cm += old_number_objects 
        total_number_objects_coco += new_number_objects
        if new_number_objects != 0:
            save_json(output_folder, coco_anno, img)
        #if old_number_objects != new_number_objects:
        #if new_number_objects == 0 and old_number_objects != 0:
        #    diff = old_number_objects - new_number_objects
        #    f.write(anno[:-1] + '\n')
        #    f.write('numbers of object from CM:{}'.format(old_number_objects)+ ' ' + 'diff:{}'.format(diff) + '\n')
    print('total_number_objects_coco', total_number_objects_coco)
    print('total_number_objects_cm', total_number_objects_cm)
    print('total_number_species', len(species_dict.keys()))

if __name__ == '__main__':   
    config=configparser.ConfigParser()
    config.read('./configs/config.ini')    
    params = config['convert2everyfish_coco.example']
    
    dataset = params.get('data_root', None)
    out_folder = params.get('out_folder', None)
   
    img_folder = os.path.join(dataset, 'label_frames')
    seg_labels_folder = os.path.join(dataset, 'seg_labels_json')
    img_list = file_utils.list_files_by_extension(img_folder, '.png')
    seg_labels_list = file_utils.list_files_by_extension(seg_labels_folder, '.json')

    if out_folder:
        os.makedirs(out_folder, exist_ok=True)
        
    convertor_patch(seg_labels_folder, out_folder, img_list, seg_labels_list)
    print('file saved!')
