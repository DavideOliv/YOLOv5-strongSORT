import numpy as np
from pathlib import Path
import warnings
import matplotlib as plt
from matplotlib.pyplot import subplots
import pandas as pd
import os
import argparse
from utils import TryExcept
import seaborn as sn
import torch
import math 



def yolo_to_pascal_voc(x_center, y_center, w, h,  image_w, image_h):
    w = w * image_w
    h = h * image_h
    x1 = ((2 * x_center * image_w) - w)/2
    y1 = ((2 * y_center * image_h) - h)/2
    x2 = x1 + w
    y2 = y1 + h
    return torch.tensor([x1, y1, x2, y2])

def box_area(box, eps=1e-7):
    # box = 4xn
    return (box[2] - box[0]) * (box[3] - box[1])

def box_iou(box1_, box2_, eps=1e-7):
    import torch
# https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
        """
    box1 = torch.empty(size=(box1_.shape))
    for i in range(box1_.shape[0]):
        box1[i] = yolo_to_pascal_voc(box1_[i,0], box1_[i,1], box1_[i,2], box1_[i,3],640,640)
    
    box2 = torch.empty(size=(box2_.shape))
    for i in range(box2_.shape[0]):
        box2[i] = yolo_to_pascal_voc(box2_[i,0], box2_[i,1], box2_[i,2], box2_[i,3],640,640)

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    (a1, a2), (b1, b2) = box1[:, None].chunk(2, 2), box2.chunk(2, 1)
    inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp(0).prod(2)

    # IoU = inter / (area1 + area2 - inter)
    iou = inter / (box_area(box1.T)[:, None] + box_area(box2.T) - inter + eps)
    return iou.cpu().detach().numpy()



class ConfusionMatrix:
    def __init__(self, nc: int, CONF_THRESHOLD=0.25, IOU_THRESHOLD=0.45):
        self.matrix = np.zeros((nc + 1, nc + 1))
        self.nc = nc
        self.CONF_THRESHOLD = CONF_THRESHOLD
        self.IOU_THRESHOLD = IOU_THRESHOLD

    def process_batch(self, detections, labels: np.ndarray):
        """
        Return intersection-over-union (Jaccard index) of boxes.
        Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
        Arguments:
            detections (Array[N, 6]), x1, y1, x2, y2, conf, class
            labels (Array[M, 5]), class, x1, y1, x2, y2
        Returns:
            None, updates confusion matrix accordingly
        """
        gt_classes = labels[:, 0].astype(np.int16)

        try:
            detections = detections[detections[:, 4] > self.CONF_THRESHOLD]
        except IndexError or TypeError:
            # detections are empty, end of process
            for i, label in enumerate(labels):
                gt_class = gt_classes[i]
                self.matrix[self.nc, gt_class] += 1
            return

        detection_classes = detections[:, 5].astype(np.int16)
        
        all_ious = box_iou(torch.from_numpy(labels[:, 1:]),torch.from_numpy(detections[:, :4]))
        want_idx = np.where(all_ious > self.IOU_THRESHOLD)

        all_matches = [[want_idx[0][i], want_idx[1][i], all_ious[want_idx[0][i], want_idx[1][i]]]
                    for i in range(want_idx[0].shape[0])]

        all_matches = np.array(all_matches)
        if all_matches.shape[0] > 0:  # if there is match
            all_matches = all_matches[all_matches[:, 2].argsort()[::-1]]

            all_matches = all_matches[np.unique(all_matches[:, 1], return_index=True)[1]]

            all_matches = all_matches[all_matches[:, 2].argsort()[::-1]]

            all_matches = all_matches[np.unique(all_matches[:, 0], return_index=True)[1]]

        for i, label in enumerate(labels):
            gt_class = gt_classes[i]
            if all_matches.shape[0] > 0 and all_matches[all_matches[:, 0] == i].shape[0] == 1:
                detection_class = detection_classes[int(all_matches[all_matches[:, 0] == i, 1][0])]
                self.matrix[detection_class, gt_class] += 1
            else:
                self.matrix[self.nc, gt_class] += 1

        for i, detection in enumerate(detections):
            if not all_matches.shape[0] or ( all_matches.shape[0] and all_matches[all_matches[:, 1] == i].shape[0] == 0 ):
                detection_class = detection_classes[i]
                self.matrix[detection_class, self.nc] += 1

    def return_matrix(self):
        return self.matrix
    
    

    @TryExcept('WARNING ⚠️ ConfusionMatrix plot failure: ')
    def plot(self, normalize=True, save_dir='', names=()):

        m = self.matrix #/ ((self.matrix.sum(0).reshape(1, -1) + 1E-9) if normalize else 1)
        tp = m.diagonal()  # true positives
        fp = m.sum(axis = 0) - tp  # false positives
        fn = m.sum(axis = 1) - tp  # false negatives (missed detections)
        p = tp / (tp + fp)
        r = tp / (tp + fn)
        import math
        for i in range(p.shape[0]):
            if math.isnan(p[i]):
                print(i)
                p[i]=0
            if math.isnan(r[i]):
                r[i]=0
            
        
        precision = np.mean(p)*100
        recall = np.mean(r)*100


        array = self.matrix / ((self.matrix.sum(0).reshape(1, -1) + 1E-9) if normalize else 1)  # normalize columns
        array[array < 0.005] = np.nan  # don't annotate (would appear as 0.00)
        fig, ax = subplots(1, 1, figsize=(12, 9), tight_layout=True)
        nc, nn = self.nc, len(names)  # number of classes, names
        sn.set(font_scale=1.0 if nc < 50 else 0.8)  # for label size
        labels = (0 < nn < 99) and (nn == nc)  # apply names to ticklabels
        ticklabels = (names + ['background']) if labels else "auto"
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # suppress empty matrix RuntimeWarning: All-NaN slice encountered
            sn.heatmap(array,
                       ax=ax,
                       annot=nc < 30,
                       annot_kws={
                           "size": 8},
                       cmap='Blues',
                       fmt='.2f',
                       square=True,
                       vmin=0.0,
                       xticklabels=ticklabels,
                       yticklabels=ticklabels).set_facecolor((1, 1, 1))
        ax.set_ylabel('True')
        ax.set_ylabel('Predicted')
        #ax.set_title('Confusion Matrix')
        ax.set_title(f"Precision: {round(precision,2)} %     Recall: {round(recall,2)} % ", fontsize=15)
        fig.savefig(Path(save_dir) / 'confusion_matrix.png', dpi=250)
        #plt.close(fig)

    def print_matrix(self):
        for i in range(self.nc + 1):
            print(' '.join(map(str, self.matrix[i])))


def swap_columns(df, col1, col2):
    col_list = list(df.columns)
    x, y = col_list.index(col1), col_list.index(col2)
    col_list[y], col_list[x] = col_list[x], col_list[y]
    df = df[col_list]
    return df

def run(dir_pred="", dir_gt="", dir_save=""):

    pg = os.listdir(dir_gt)
    pp = os.listdir(dir_pred)
    
    """
    names_traffic = ["bicycle", "bus", "car", "truck"]
    conf_mat = ConfusionMatrix(nc=4)
    new_class_traffic = [0,1,2,3]
    """
    """
    names_visdrone = ["pedestrian", "bicycle", "car", "truck", "bus"]
    conf_mat = ConfusionMatrix(nc=5)
    new_class_vis = [0,1,2,3,4]
    """
    """
    names_visdrone = ["pedestrian", "people","bicycle", "car","van", "truck", "tricycle","awning-tricycle", "bus", "motor"]
    conf_mat = ConfusionMatrix(nc=10)
    new_class_vis = [0,1,2,3,4]
    """
    names_art = ["PICKUP","SUV","BTR","BRDM2","BMP2","T72","ZSU23" ,"2S3","D20", "MTLB","BTR70"]
    conf_mat = ConfusionMatrix(nc = 11)
    #new_class_art = [0,1,2,3,4,5,6,7]
  
    for i in pp:
        if i not in pg:
            continue
        preds = pd.read_csv(dir_pred + i, sep = " ", header=None)
        preds.columns = ["class", "x1", "x2", "x3", "x4", "conf"]
        #preds.drop(preds[(preds["class"] == 0) | (preds["class"] == 8 ) | (preds["class"] == 9)].index, inplace=True)
        preds["class_l"] = preds["class"]
        preds = preds.drop(columns=["class"], axis=1)

        """
        #replace using ART
        preds["class_l"] = preds["class_l"].replace(to_replace=3, value=2)
        preds["class_l"] = preds["class_l"].replace(to_replace=4, value=3)
        preds["class_l"] = preds["class_l"].replace(to_replace=6, value=5)
        preds["class_l"] = preds["class_l"].replace(to_replace=7, value=6)
        preds["class_l"] = preds["class_l"].replace(to_replace=10, value=7)
        preds = preds[preds["class_l"].isin(new_class_art)]
        """
        """
        #replace using VisDrone classes for visdrone dataset
        preds["class_l"] = preds["class_l"].replace(to_replace=2, value=1)
        preds["class_l"] = preds["class_l"].replace(to_replace=3, value=2)
        preds["class_l"] = preds["class_l"].replace(to_replace=5, value=3)
        preds["class_l"] = preds["class_l"].replace(to_replace=8, value=4)
        preds = preds[preds["class_l"].isin(new_class_vis)]
        """

        """
        #replace preds using Coco classes for visdrone dataset
        preds["class_l"] = preds["class_l"].replace(to_replace=7, value=3)
        preds["class_l"] = preds["class_l"].replace(to_replace=5, value=4)
        preds = preds[preds["class_l"].isin(new_class_vis)]
        """

        """
        #replace pred using Coco classes for traffic dataset
        preds["class_l"] = preds["class_l"].replace(1,0)
        preds["class_l"] = preds["class_l"].replace(5,1)
        preds["class_l"] = preds["class_l"].replace(7,3)
        preds = preds[preds["class_l"].isin(new_class_traffic)]
        """

        """
        #replace pred using VisDrone classes for traffic dataset
        preds["class_l"] = preds["class_l"].replace(2,0)
        preds["class_l"] = preds["class_l"].replace(3,2)
        preds["class_l"] = preds["class_l"].replace(5,3)
        preds["class_l"] = preds["class_l"].replace(8,1)
        preds = preds[preds["class_l"].isin(new_class_traffic)]
        """


        gt = pd.read_csv(dir_gt + i, sep=" ", header=None)
        gt.columns = ["class", "x1", "x2", "x3", "x4"]
        #gt.drop(gt[(gt["class"] == 0) | (gt["class"] == 8) | (gt["class"] == 9)].index, inplace=True)

        """
        #replace ART
        gt["class"] = gt["class"].replace(to_replace=3, value=2)
        gt["class"] = gt["class"].replace(to_replace=4, value=3)
        gt["class"] = gt["class"].replace(to_replace=6, value=5)
        gt["class"] = gt["class"].replace(to_replace=7, value=6)
        gt["class"] = gt["class"].replace(to_replace=10, value=7)
        gt = gt[gt["class"].isin(new_class_art)]
        """
        """
        #replace gt using Traffic classes
        gt["class"] = gt["class"].replace(to_replace=4, value=1)
        gt["class"] = gt["class"].replace(to_replace=3, value=0)
        gt["class"] = gt["class"].replace(to_replace=5, value=2)
        gt["class"] = gt["class"].replace(to_replace=18, value=3)
        gt = gt[gt["class"].isin(new_class_traffic)]
        """

        """
        #replace gt using VisDrone classes 
        gt["class"] = gt["class"].replace(to_replace=2, value=1)
        gt["class"] = gt["class"].replace(to_replace=3, value=2)
        gt["class"] = gt["class"].replace(to_replace=5, value=3)
        gt["class"] = gt["class"].replace(to_replace=8, value=4)
        gt = gt[gt["class"].isin(new_class_vis)]
        """

        predizione = preds.to_numpy(dtype=float) 
        labels = gt.to_numpy(dtype=float)
        conf_mat.process_batch(predizione, labels)
        conf_mat.return_matrix()

    #conf_mat.return_matrix()
    #conf_mat.print_matrix()
    conf_mat.plot(save_dir=dir_save,names=names_art)
    print(f"saved in {dir_save}")
    
def opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir_gt", type=str, default="/usr/src/datasets/VisDrone/VisDrone2019-DET-test-dev/labels/",help="path list of ground truth boxes")
    parser.add_argument("--dir_pred", type=str, help="path list of predictions boxes")
    parser.add_argument("--save", type=str, help="path to save result")
    opt = parser.parse_args()
    return opt

def main(opt):
    run(opt.dir_pred, opt.dir_gt, opt.save)

if __name__ == "__main__":
    opt = opt()
    main(opt)
