<div align="center">

<h1>YOLOv12-FoodWeight</h1>
<h3>Multi-Task Real-Time Food Detection and Weight Estimation</h3>

<p align="center">
  <img src="assets/model.png" width=90%><br>
  <em>Proposed YOLOv12-FoodWeight architecture with an additional regression head for food weight prediction</em>
</p>

</div>

---

## Overview

This project builds upon the YOLOv12 architecture to perform **multi-task learning**: 

- **Object Detection**: Detect food items.
- **Weight Prediction**: Predict the weight (in grams) of each detected food item.

We introduce an additional regression head to YOLOv12 to predict weights, enabling simultaneous localization and portion estimation from a single image.

Our model is trained and evaluated on a specialized food dataset with annotated bounding boxes and weight labels in grams, available on Hugging Face:

➡️ [Download FoodWeight Dataset on Hugging Face](https://huggingface.co/datasets/your-dataset-link)

## Main Features

- **Multi-task** Food object detection and weight (in grams) prediction.
- **Single unified model**: Jointly trained for classification, localization, and regression.
- **Flexible prediction outputs**: Save predictions in `.txt` or `.csv` formats.
- **Evaluation metrics**: Includes MAE (Mean Absolute Error) for weight estimation.


## Results

<p align="center">
  <img src="assets/training_results.png" width=80%><br>
  <em>Training results comparing the different versions of the YOLOv8 and YOLOv12 models</em>
</p>


## Pretrained Weights

You can download the best-performing pretrained YOLOv12-M model weights here:

- [YOLOv12-FoodWeight Medium (best checkpoint)](https://huggingface.co/your-model-link-small)


## Installation

```bash
conda create -n yolov12_foodweight python=3.11
conda activate yolov12_foodweight

# Install dependencies
pip install -r requirements.txt
pip install -e .

# (Optional) For FlashAttention support
wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/flash_attn-2.7.3+cu11torch2.2cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
pip install flash_attn-2.7.3+cu11torch2.2cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
```

## Training

Train the multi-task model from scratch or fine-tune a pretrained YOLOv12 model.

```python
from ultralytics import YOLO

model = YOLO('yolov12m.yaml')  

results = model.train(
    data='data.yaml',
    task='detect', 
    epochs=200, 
    imgsz=640, 
    batch=8, 
    device="cuda:0"
)
```



Outputs:
- Standard detection metrics (mAP, precision, recall)
- MAE for weight estimation
- Optionally, annotated images with predicted bounding boxes and weights

## Prediction

We provide two scripts to generate predictions:

- **`predict_txt.py`**: Runs inference and saves the predictions in a `.txt` format.
- **`predict_csv.py`**: Runs inference and saves the predictions in a `.csv` format.

Each prediction contains:
- `image_name`, `class_id`, `xmin`, `ymin`, `xmax`, `ymax`, `weight`, `confidence`

Choose the format depending on your post-processing or evaluation needs.




## Acknowledgment

This project is based on [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics) and [YOLOv12](https://github.com/sunsmarterjie/yolov12). We extend the original work with an additional regression head for food weight prediction.

## Citation

Please also cite our work if you use the FoodWeight model. (Citation will be added after publication.)

```BibTeX
@article{,
  title={},
  author={},
  journal={},
  year={2025}
}
```
