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
- [COCO format](https://labelformat.com/formats/object-detection/coco/) 
using the data_converter tool "belt2coco_from_json.py"
- [EVERYFISH_COCO v1.0 format](https://www.dropbox.com/scl/fi/vx3x4i13hx9ildc3gy4qz/EVERYFISH-Data-Structure.pdf?rlkey=w3mup6q0ql5emw9211ng7sn0b&st=ny5y4wlt&dl=0)


