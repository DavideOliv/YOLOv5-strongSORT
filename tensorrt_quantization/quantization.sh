#!bin/bash

cd /usr/src/tensorrt_quantization
#python convert_trt_quant.py --input /usr/src/yolov5/results/yolo5N_Vis/weights/best.onnx --output /usr/src/yolov5/results/yolo5N_Vis/weights/yolov5n_fp16.engine --fp16
#python convert_trt_quant.py --input /usr/src/yolov5/results/yolo5N_Vis/weights/best.onnx --output /usr/src/yolov5/results/yolo5N_Vis/weights/yolov5n_int8.engine --int8

python convert_trt_quant.py --input /usr/src/yolov5/results/yolo5S_Vis/weights/best.onnx --output /usr/src/yolov5/results/yolo5S_Vis/weights/yolov5s_fp16.engine --fp16 > loggert_fp16_s.txt
python convert_trt_quant.py --input /usr/src/yolov5/results/yolo5S_Vis/weights/best.onnx --output /usr/src/yolov5/results/yolo5S_Vis/weights/yolov5s_int8.engine --int8 > logger_int8_s.txt

python convert_trt_quant.py --input /usr/src/yolov5/results/yolo5M_Vis/weights/best.onnx --output /usr/src/yolov5/results/yolo5M_Vis/weights/yolov5m_fp16.engine --fp16 > logger_fp16_m.txt
python convert_trt_quant.py --input /usr/src/yolov5/results/yolo5M_Vis/weights/best.onnx --output /usr/src/yolov5/results/yolo5M_Vis/weights/yolov5m_int8.engine --int8 > logger_int8_m.txt

python convert_trt_quant.py --input /usr/src/yolov5/results/yolo5L_Vis/weights/best.onnx --output /usr/src/yolov5/results/yolo5L_Vis/weights/yolov5l_fp16.engine --fp16 > logger_fp16_l.txt
python convert_trt_quant.py --input /usr/src/yolov5/results/yolo5L_Vis/weights/best.onnx --output /usr/src/yolov5/results/yolo5L_Vis/weights/yolov5l_int8.engine --int8 > logger_int8_l.txt

python convert_trt_quant.py --input /usr/src/yolov5/results/yolo5X_Vis/weights/best.onnx --output /usr/src/yolov5/results/yolo5X_Vis/weights/yolov5x_fp16.engine --fp16
python convert_trt_quant.py --input /usr/src/yolov5/results/yolo5X_Vis/weights/best.onnx --output /usr/src/yolov5/results/yolo5X_Vis/weights/yolov5x_int8.engine --int8
