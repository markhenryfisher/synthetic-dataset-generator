## demo.py

Demonstrates basic functionality of synthetic-dataset-generator. 
The program generates composited imagery and associated annotation. 
The output is written to subdirectories:  
```python
annfiles
label_frames
seg_labels_json
```
- annfiles: contains oriented bounding box (OBB) annotations in [DOTA v1 format](https://captain-whu.github.io/DOTA/index.html).  
- label_frames contains the output images (.png)
- seg_labels_json contans horizontal bounding box (HBB) annotations and masks (.json). These are stored in legacy UEA format (similar to COCO).  

Legacy UEA HBB annotations can be converted to 
[COCO format](https://labelformat.com/formats/object-detection/coco/) 
using the data_converter tool "belt2coco_from_json.py". This will produce a new subdirectory:
```python
coco
```