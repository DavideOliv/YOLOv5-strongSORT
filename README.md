
# YOLOv5 + strongSORT
The aim of this work is to study solutions for the **detection** and **tracking** of objects imaged by **drones** using Deep Learning approaches.
Particular attention was paid to studying the relationship between the accuracy and inference speed of the models, in order to run them on embedded boards and test their benefits in Real Time. 
The dataset chosen for this work is [**Vis-Drone 2019**](https://github.com/VisDrone/VisDrone-Dataset), it represents a challenge, still completely open due to its enormous
difficulties derived from the high unbalanced of classes and the size of the objects.

## Multi Object Detection using YOLOv5
In the first part of the project we will look at the results obtained for Multi Object Detection through YOLOv5 (You Only Look Once) and compare them with the state-of-the-art results for the VisDrone 2019 Challenge.
The official GitHub [YOLOv5](https://github.com/ultralytics/yolov5) repository was used for this part.
In the path you will find all models (N, S, M, L, X) trained on VisDrone.
>cd ./yolov5/results/

**Train model on VisDrone**
> python train.py --img 640 --batch 16 --epochs 100 --data VisDrone.yaml --weights yolov5s.pt --project results --name yolo5S_Vis --device 0 --patience 50 --conf-thres 0.5 --conf-iou 0.25

We recommend increasing the confidence threshold to 0.25 rather than 0.01 as in the official guide. Because we would have a lot of background in the VisDrone dataset.

**Validation model**
> python val.py --weights /usr/src/yolov5/results/yolo5S_Vis/weights/best.pt --data VisDrone.yaml --batch-size 16 --project testing/val_VisDrone --name yolov5n --device 0 --verbose --conf-thres 0.5 --conf-iou 0.25

**Test model**
**Validation model**
> python val.py --weights /usr/src/yolov5/results/yolo5S_Vis/weights/best.pt --data VisDrone.yaml --batch-size 1 --project testing/VisDrone --name yolov5n --device 0 --verbose --conf-thres 0.5 --conf-iou 0.25 --task test

**Export model**
> python export.py --device 0 --include onnx --weights /usr/src/yolov5/results/yolo5S_Vis/weights/best.pt --dynamic --simplify

> python export.py --device 0 --include engine --weights /usr/src/yolov5/results/yolo5S_Vis/weights/best.pt --half 

This way we will get the models in **engine** format with **tensorRT** that we can quantify them.

## Quantization
For Quatization the model with **TensorRT** use the folder *tensorrt-quantization*.
 > cd ./tensorrt-quantization
 
 The quantization is done on the type of dataset used in the train, in the folder is the calibration.cache file processed on VisDrone.
In case you use a custom dataset, make a calibration by setting in the file convert_trt_quant.py the variable: **CALIB_IMG_DIR=[path/dataset/images]**

 
 for FP16 quantization use:
 
 > python convert_trt_quant.py --input [path/model.onnx] --output [path/modelFP16.engine] --fp16
 
 for INT8 quantization use:
 
 > python convert_trt_quant.py --input [path/model.onnx] --output [path/modelINT8.engine] --int8
 

## Multi Object Tracking adding strongSORT
The second part of the project focuses on solving the problem of **Multi Object Tracking** of images taken by drones. 
For this purpose, a cascade approach was adopted between two algorithms, the first of which uses YOLOv5 for object
detection, while the second **(strongSORT)** takes the output of YOLOv5 as input and creates traces for each detected object by assigning it an ID. This approach was tested first on a single person tracking task taken from a drone, and then on video sequences from the VisDrone Tracking dataset, achieving good
results.

> cd ..

> python track.py --yolo-weights ./yolov5/[yolo_model.pt] --strong-sort-weights ./weights/osnet_x0_25_msmt17.pt --source [path/dataset/images] --conf-thres 0.5 --device 0 --save-txt --save-vid --project ./test_track/ 

The results saved in **'/test_track/'** must be processed before calculating the metrics with **/mot_evaluation**.
If you would like to test the algorithm with a video sequence from the **VisDrone Tracking** dataset in order to analyse the MOT results, run it first:

> python track_processing_visdrone.py --project_in /usr/src/test_track/[folder_with_gt_and_resultarck] --save /usr/src/mot_evaluation/data/[name] 

Then put in the 'mot_evaluation' folder for the automatic calculation of metrics:

> cd /usr/src/mot_evaluation/seqmaps/test.txt 

write list of folder to be analysed containing **gt.txt** (ground thruth) and **res.txt** (results of tarcking). Then:

> python evaluate_tracking.py --seqmap /usr/src/mot_evaluation/seqmaps/test.txt --track /usr/src/mot_evaluation/data/ --gt /usr/src/mot_evaluation/data/ --save
