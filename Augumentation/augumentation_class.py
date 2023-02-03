import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import os, zipfile
import albumentations as A
from datetime import datetime
from PIL import Image
import argparse

# Questo script effettua l'augmentation di un dataset esclusivamente sulla classe passata come argomento
# L'idea è quella di togliere le label da un immagine che non ci servono
# mantendendo la stessa foto senza le label


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




#path: test or train which should contain 2 folder: /labels and /images

def augumentation(path,  class_aug, dict_labels):
  
  for img in os.listdir(path + "/labels"):

    img_abs = img.split(".txt")[0]
    image = cv2.imread(path + "/images/" + img_abs +".jpg", cv2.COLOR_BGR2RGB) #open image
    if image.shape[0] < 500 or image.shape[1] < 500:
      continue
    with open(path + "/labels/" + img, "r") as file:
      line = [[num for num in line.replace("\n"," ").split(" ")] for line in file] #save labels
    
    #preprocessing data  
    line = np.array(line)
    line = line[:, :-1]
    opt = list(line[:,0])
    tot = filter(lambda x : x == str(class_aug), opt)
    #print(f"per {img} ho {len(list(tot))} elementi di {class_aug}")

    if str(class_aug) in opt:
      for i in range(len(list(tot))):
        np.set_printoptions(suppress=True)
        boxes = np.asarray(line, dtype=float)
        labels = [int(label) for label in boxes[:,0]]
        labels = np.array(labels)
        class_labels = [dict_labels[l] for l in labels]
        boxes =  boxes[:,1:]
        boxes_ = check_bbox(boxes, 500, 500) #check
        
        flag, new_image, new_boxes, new_labels, new_labels_id = balance_datatset(image=image, 
                                                                                  boxes=boxes_, 
                                                                                  labels=class_labels, 
                                                                                  class_aug=class_aug, 
                                                                                  flag=True)
        
        if flag:
          i = i + 1
          continue
        else:
          if len(list(new_labels_id[new_labels_id[:]==class_aug])) != 0:
            new_labels_id_ = np.reshape(new_labels_id, (len(new_labels_id),1))
            temp = np.concatenate((new_labels_id_, new_boxes), axis = 1)
            output = temp[temp[:,0] == class_aug]
            dt = datetime.now()
            file_labels = f"{path}/labels/{dt.microsecond}.txt"
            file_image = f"{path}/images/{dt.microsecond}.jpg"
            np.savetxt(file_labels ,output, fmt=['%d', '%f', '%f', '%f', '%f'])
            im = Image.fromarray(new_image)
            im.save(file_image)
            i = i + 1
          else:
            i = i + 1
          i = i + 1
      
    else:
      continue
      
    
def transformation(image, boxes, labels):

    transformed = transform(image=image, bboxes=boxes, class_labels=labels) 
    transformed_image = np.array(transformed['image'])
    transformed_bboxes = np.array(transformed['bboxes'])
    transformed_class_labels = np.array(transformed['class_labels'])
    transformed_bboxes_, transformed_class_labels_ =  delete_noise(transformed_bboxes, transformed_class_labels)
    transformed_labels_id = np.array([list(dict_labels.values()).index(i) for i in transformed_class_labels_ ])


    return transformed_image, transformed_bboxes_, transformed_class_labels, transformed_labels_id


def balance_datatset(image, boxes, labels, class_aug, flag=True):
    count = 0
    end = 100
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
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomCrop(p=1,width=500, height=500),
    A.RandomBrightnessContrast(p=0.1),
    A.RandomSnow(p=0.1),
    A.RandomRain(rain_type='heavy'),
    A.RandomRotate90(),
    A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.15),

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
