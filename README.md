# YOLOv5 + strongSORT
The aim of this work is to study solutions for the detection and tracking of objects imaged by drones using Deep Learning approaches.
Particular attention was paid to studying the relationship between the accuracy
and inference speed of the models, in order to run them on embedded boards
and test their benefits in Real Time. 
The dataset chosen for this work is *Vis-Drone 2019*, it represents a challenge, still completely open due to its enormous
difficulties derived from the high unbalanced of classes and the size of the objects.

## Multi Object Detection using YOLOv5
In the first part of the project we will look at the results obtained for Multi
Object Detection through YOLOv5 (You Only Look Once) and compare them
with the state-of-the-art results for the VisDrone 2019 Challenge.

## Multi Object Tracking adding strongSORT
The second part of the project focuses on solving the problem of Multi Object
Tracking of images taken by drones. For this purpose, a cascade approach was
adopted between two algorithms, the first of which uses YOLOv5 for object
detection, while the second (strongSORT) takes the output of YOLOv5 as
input and creates traces for each detected object by assigning it an ID. This
approach was tested first on a single person tracking task taken from a drone,
and then on video sequences from the VisDrone Tracking dataset, achieving good
results.
