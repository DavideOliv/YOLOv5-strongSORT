# L'approccio proposto per aumentare le classi in questo è script è il seguente:

# 1) vengono analizzate tutte le immagini del train
# 2) per ognuna si cercano i box contenenti la classe da aumentare (ad esempio van)
# 3) vengono ritagliati con un box più ampio, trasformati e aggiunto un padding per tornare alle dimensioni originali
# 4) le nuove immagine (con un singolo oggetto) vengono salvate con le relative label

import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import os, zipfile
import albumentations as A
from datetime import datetime
from PIL import Image
import argparse




"""
Function for validate boxes trasformed to albumentations format.

check_bbox():
    * seq: array of boxes
    * W: weight image
    * H: Height image
"""
def check_bbox(seq, W, H):
    bbox_albu = A.core.bbox_utils.convert_bboxes_to_albumentations(seq, source_format='yolo', rows=H, cols=W)
    bbox_yolo = A.core.bbox_utils.convert_bboxes_from_albumentations(bbox_albu, target_format='yolo', rows=H, cols=W)
    return np.array(bbox_yolo)


"""
Function for detele all box near edge of image.

delete_noise():
    * transformed_bboxes: array of boxes
    * transformed_class_labels: array
"""
def delete_noise(transformed_bboxes, transformed_class_labels):
    idx = []
    for i in range(transformed_bboxes.shape[0]):
        x = transformed_bboxes[i][0]
        y = transformed_bboxes[i][1]
        w = transformed_bboxes[i][2]
        h = transformed_bboxes[i][3]
        th1 = x-w/2
        th2 = y-h/2
        
        if th1 < 0.03 or th1 > 0.95 or th2 < 0.03 or th2 > 0.95:
            idx.append(i)

    new_box = np.delete(transformed_bboxes, idx, axis=0)
    new_lab = np.delete(transformed_class_labels, idx, axis=0)

    return new_box, new_lab

def yolo_to_pascal_voc(x_center, y_center, w, h,  image_w, image_h):
    w = w * image_w
    h = h * image_h
    x1 = ((2 * x_center * image_w) - w)/2
    y1 = ((2 * y_center * image_h) - h)/2
    x2 = x1 + w
    y2 = y1 + h
    return int(x1), int(y1), int(x2), int(y2)

def yolo_to_coco(x_center, y_center, w, h,  image_w, image_h):
    w = w * image_w
    h = h * image_h
    x1 = ((2 * x_center * image_w) - w)/2
    y1 = ((2 * y_center * image_h) - h)/2
    return int(x1), int(y1), int(w), int(h)

#path: test or train which should contain 2 folder: /labels and /images
def augumentation(path,  class_aug, dict_labels):
  
  for img in os.listdir(path + "/labels"):

    img_abs = img.split(".txt")[0]
    image = cv2.imread(path + "/images/" + img_abs +".jpg", cv2.COLOR_BGR2RGB) #open image
    #image = Image.open(path + "/images/" + img_abs +".jpg")
    dh = image.shape[0]
    dw = image.shape[1]
    #dw,dh = image.size
    #print(type(image))

    with open(path + "/labels/" + img, "r") as file:
      line = [[num for num in line.replace("\n"," ").split(" ")] for line in file] #save labels
    
    #preprocessing data  
    line = np.array(line)
    annotations = line[:, :-1]
    labels = line[:,0]
    labels_filter_index = np.where(labels == str(class_aug)) #return index
    if len(list(labels_filter_index)[0]) != 0:
      labels_filter = [x for x in labels if x==str(class_aug)] #return list class
      #print(len(list(labels_filter_index)[0]))
      for i in range(len(labels_filter)):
        np.set_printoptions(suppress=True)
        id = list(labels_filter_index)[0]
        idx = id[0]
        annotations_filter = annotations[idx,:]
        boxes = np.asarray(annotations_filter, dtype=float)
        labels = np.array(int(boxes[0]))       
        labels = labels.reshape(1,1)

        #class_labels = [dict_labels[l] for l in labels]
        #boxes_ = check_bbox(boxes, dw, dh) #check
        boxes =  boxes[1:]
        boxes = boxes.reshape(1,4)

        x = boxes[0,0] 
        y = boxes[0,1] 
        w = boxes[0,2]
        h = boxes[0,3]
        #x1,y1,x2,y2 = yolo_to_pascal_voc(x,y,w,h,iw,ih)

        x1,y1,w1,h1 = yolo_to_coco(x,y,w,h,dw,dh)
        cropped = image[y1:y1+6*h1,x1:x1+w1*6, :] #image cropped

        for i in range(2):
            transformed_image, transformed_bboxes, transformed_class_labels = transformation(cropped, boxes,labels)
            if transformed_bboxes.size == 0:
                continue
            else:
                x = transformed_bboxes[0,0] 
                y = transformed_bboxes[0,1] 
                w = transformed_bboxes[0,2]
                h = transformed_bboxes[0,3]

                l = int((x - w / 2) * dw)
                r = int((x + w / 2) * dw)
                t = int((y - h / 2) * dh)
                b = int((y + h / 2) * dh)
                
                if l < 0:
                    l = 0
                if r > dw - 1:
                    r = dw - 1
                if t < 0:
                    t = 0
                if b > dh - 1:
                    b = dh - 1
                
                dt = datetime.now()
                new_image = cv2.copyMakeBorder(transformed_image, t, b, l, r, cv2.BORDER_REPLICATE)
                #resized = cv2.resize(new_image, (640,640), interpolation = cv2.INTER_AREA)
                plt.imsave(f"/usr/src/datasets/VisDrone_Augmented/VisDrone2019-DET-train/images/{dt.microsecond}.jpg",new_image)
                #new_labels_id_ = np.reshape(transformed_class_labels, (len(transformed_class_labels),1))
                new_boxes = np.concatenate((transformed_class_labels, transformed_bboxes), axis = 1)
                np.savetxt(f"/usr/src/datasets/VisDrone_Augmented/VisDrone2019-DET-train/labels/{dt.microsecond}.txt" ,new_boxes, fmt=['%d', '%f', '%f', '%f', '%f'])
                i = i+1
                print(f"saved {dt.microsecond} from {img_abs}")
    else:
      continue  

      
    
def transformation(image, boxes, labels):

    transformed = transform(image=image, bboxes=boxes, class_labels=labels) 
    transformed_image = np.array(transformed['image'])
    transformed_bboxes = np.array(transformed['bboxes'])
    transformed_class_labels = np.array(transformed['class_labels'])
    transformed_bboxes_, transformed_class_labels_ =  delete_noise(transformed_bboxes, transformed_class_labels)
    #transformed_labels_id = np.array([list(dict_labels.values()).index(i) for i in transformed_class_labels_ ])


    return transformed_image, transformed_bboxes_, transformed_class_labels


def balance_datatset(image, boxes, labels, class_aug, flag=True):
    count = 0
    end = 50
    new_image, new_boxes, new_labels, new_labels_id = transformation(image, boxes, labels)
    condition = class_aug in new_labels_id
    while (count < end) and (not condition) and (new_labels_id.shape != 0):
        count = count + 1
        new_image, new_boxes, new_labels, new_labels_id = transformation(image, boxes, labels)
        flag = True
    if count == end or new_labels_id.shape == 0:
        flag = False
        
    return flag, new_image, new_boxes, new_labels, new_labels_id






if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dir_name", type=str, help="path images and labels")
    parser.add_argument("--class_aug", type=int ,default=4, help="class to augumentation")
    args = parser.parse_args()

    #os.mkdir(args.dir_name + "/new_labels")
    #os.mkdir(args.dir_name + "/new_images")

    
    transform = A.Compose([
    A.OneOf([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            
        ], p=0.2),

    #A.RandomCrop(p=1,width=500, height=500),
    A.RandomBrightnessContrast(p=0.1),
    A.RandomSnow(p=0.1),
    #A.RandomRain(rain_type='heavy'),
    A.ChannelShuffle(p=1),
    A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2)
    #A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.15),

    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

    dict_labels = {
        0: "pedestrian",
        1: "people",
        2: "bicycle",
        3: "car",
        4: "van",
        5: "truck",
        6: "tricycle",
        7: "awning-tricycle",
        8: "bus",
        9: "motor"
    }

    
    augumentation(args.dir_name,  args.class_aug, dict_labels=dict_labels)
