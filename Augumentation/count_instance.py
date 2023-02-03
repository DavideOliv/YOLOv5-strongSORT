import pandas as pd
import numpy as np
import os
import cv2

def count_instance(path, class_aug):
    count = 0
    for img in os.listdir(path + "/labels"):

        img_abs = img.split(".txt")[0]

        #image = cv2.imread(path + "/images/" + img_abs +".jpg", cv2.COLOR_BGR2RGB) #open image
        with open(path + "/labels/" + img, "r") as file:
            line = [[num for num in line.replace("\n"," ").split(" ")] for line in file] #save labels
        line = np.array(line)
        line = line[:, :-1] #labels
        np.set_printoptions(suppress=True)
        boxes = np.asarray(line, dtype=float)
        labels = [int(label) for label in boxes[:,0]]
        labels = np.array(labels)
        count = count + len(list(labels[labels[:]==class_aug]))
    return count

class_ug = 4
c = count_instance("/usr/src/datasets/VisDrone_Aug_2/VisDrone2019-DET-train",class_ug)
print(f"count class {class_ug} : {c} ")