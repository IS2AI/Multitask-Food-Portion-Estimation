1. Install

Mine CUDA Version: 11.0

Pip install the ultralytics package including all requirements in a Python>=3.8 environment with PyTorch>=1.8.

pip install ultralytics

2. Download model from the google drive:

   https://drive.google.com/file/d/12H1VLq9AvpgsA7lVS4RttCvuqNS7KkRH/view?usp=sharing

4. Run python code for inference:

   python3 webin.py

5. You can specify parameters in code:
   
   path_to_model="last.pt"
   path_to_image="ex/RAFdb_test_0003.jpg" #if you want use image from the Internet, replace path with None
   image_url=None #if you want use image from the Internet, replace None with URL
   threshold_bboxes=0.3 

#### References

* [https://github.com/ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)

* [https://github.com/deepcam-cn/yolov5-face](https://github.com/deepcam-cn/yolov5-face)

* [https://github.com/derronqi/yolov7-face](https://github.com/derronqi/yolov7-face)
