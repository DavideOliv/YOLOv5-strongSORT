from argparse import ArgumentParser
import numpy as np
import torch
import torch.nn as nn
import util_trt
import glob,os,cv2

BATCH_SIZE = 16
BATCH = 100
height = 640
width = 640
CALIB_IMG_DIR = '/usr/src/datasets/VisDrone/VisDrone2019-DET-train/images'

def preprocess_v1(image_raw):
    h, w, c = image_raw.shape
    image = cv2.cvtColor(image_raw, cv2.COLOR_BGR2RGB)
    # Calculate widht and height and paddings
    r_w = width / w
    r_h = height / h
    if r_h > r_w:
        tw = width
        th = int(r_w * h)
        tx1 = tx2 = 0
        ty1 = int((height - th) / 2)
        ty2 = height - th - ty1
    else:
        tw = int(r_h * w)
        th = height
        tx1 = int((width - tw) / 2)
        tx2 = width - tw - tx1
        ty1 = ty2 = 0
    # Resize the image with long side while maintaining ratio
    image = cv2.resize(image, (tw, th))
    # Pad the short side with (128,128,128)
    image = cv2.copyMakeBorder(
        image, ty1, ty2, tx1, tx2, cv2.BORDER_CONSTANT, (128, 128, 128)
    )
    image = image.astype(np.float32) #changed
    # Normalize to [0,1]
    image /= 255.0
    # HWC to CHW format:
    image = np.transpose(image, [2, 0, 1])
    # CHW to NCHW format
    #image = np.expand_dims(image, axis=0)
    # Convert the image to row-major order, also known as "C order":
    #image = np.ascontiguousarray(image)
    return image


def preprocess(img):
    img = cv2.resize(img, (640, 640))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.transpose((2, 0, 1)).astype(np.float32) #changed
    img /= 255.0
    return img

class DataLoader:
    def __init__(self):
        self.index = 0
        self.length = BATCH
        self.batch_size = BATCH_SIZE
        # self.img_list = [i.strip() for i in open('calib.txt').readlines()]
        self.img_list = glob.glob(os.path.join(CALIB_IMG_DIR, "*.jpg"))
        assert len(self.img_list) > self.batch_size * self.length, '{} must contains more than '.format(CALIB_IMG_DIR) + str(self.batch_size * self.length) + ' images to calib'
        print('found all {} images to calib.'.format(len(self.img_list)))
        self.calibration_data = np.zeros((self.batch_size,3,height,width), dtype=np.float32) #changed

    def reset(self):
        self.index = 0

    def next_batch(self):
        if self.index < self.length:
            for i in range(self.batch_size):
                assert os.path.exists(self.img_list[i + self.index * self.batch_size]), 'not found!!'
                img = cv2.imread(self.img_list[i + self.index * self.batch_size])
                img = preprocess_v1(img)
                self.calibration_data[i] = img

            self.index += 1

            # example only
            return np.ascontiguousarray(self.calibration_data, dtype=np.float32) #changed
        else:
            return np.array([])

    def __len__(self):
        return self.length

def opt_parse():
    parser = ArgumentParser()
    parser.add_argument("--input", type=str, help="input model (ONNX)")
    parser.add_argument("--output", type=str, default="./model_int8.engine", help="output model (ENGINE TRT)")
    parser.add_argument("--cache", type=str, default="./calibration.cache", help="calibration file")
    parser.add_argument("--int8", default=False, action="store_true")
    parser.add_argument("--fp16", default=False, action="store_true")
    opt = parser.parse_args()
    return opt

def main(opt):
    print('*** onnx to tensorrt begin ***')
    calibration_stream = DataLoader()
    engine_fixed = util_trt.get_engine(BATCH_SIZE, opt.input, opt.output, fp16_mode=opt.fp16, 
        int8_mode=opt.int8, calibration_stream=calibration_stream, calibration_table_path=opt.cache, save_engine=True)
    assert engine_fixed, 'Broken engine_fixed'
    if opt.int8:
        print('*** onnx to tensorrt int8 completed ***\n')
    if opt.fp16:
        print('*** onnx to tensorrt fp16 completed ***\n')

    
if __name__ == '__main__':
    opt = opt_parse()
    main(opt)
    
