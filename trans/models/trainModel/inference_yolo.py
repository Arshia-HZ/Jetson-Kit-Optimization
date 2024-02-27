import random
import shutil
import time
import os
import torch
import numpy as np
from PIL import Image, ImageDraw
from sklearn.model_selection import train_test_split
from ultralytics import YOLO
from glob import glob
from pathlib import Path
import pdb
import cv2
from operator import itemgetter

class Yolo():
    def __init__(self, path):
        self.ckpt_path = path
        self.model = None

    def match(self, target):
        assert hasattr(self, 'model'), 'run set_templates first'
        t0 = time.time()
        results = self.model.predict(target, conf=0.1, verbose=False)
        t1 = time.time()
        model_time = t1 - t0
        # r:Results
        if len(results) < 1 or len(results[0].boxes) < 1:
            return []
        preds = []
        npreds = len(results[0].boxes)
        for i in range(min(10, npreds)):
            p = results[0].boxes.xyxy[i].tolist()
            p.append(results[0].boxes.conf[i].item())
            p.append(model_time)
            preds.append(p)

        return preds

    def set_templates(self):
        self.model = YOLO(self.ckpt_path)
        if torch.cuda.is_available():                    
            self.model = self.model.cuda()
            warmup_input = np.random.randn(700, 1200, 3).astype('uint8')
            print('Warming up the model ...')
            _ = self.model.predict(warmup_input, conf=0.0001, verbose=False)
            _ = self.model.predict(warmup_input, conf=0.0001, verbose=False)


def compute_iou(b1, b2):
    x11, y11, x12, y12 = b1
    x21, y21, x22, y22 = b2
    
    x_min = np.maximum(x11, x21)
    x_max = np.minimum(x12, x22)
    y_min = np.maximum(y11, y21)
    y_max = np.minimum(y12, y22)
        
    w = np.maximum(0, x_max - x_min + 1)
    h = np.maximum(0, y_max - y_min + 1)
    intersection = w * h
    if intersection == 0:
        return 0
        
    area1 = (x12 - x11 + 1) * (y12 - y11 + 1)
    area2 = (x22 - x21 + 1) * (y22 - y21 + 1)
    
    return intersection / (area1 + area2 - intersection)


def NMS(preds, iou_th=0.5):
    if len(preds) == 0:
        return preds

    preds.sort(reverse=True, key=itemgetter(4))
    nms_preds = []
    removed = np.zeros(len(preds))
    for i in range(len(preds)):
        if removed[i] == 0:
            nms_preds.append(preds[i])
        else:
            # print('!!!!!!!!!!!!REMOVE!!!!!!!!!!!!!!!!!!')
            continue
            
        box1 = preds[i][:4]
        for j in range(i+1, len(preds)):
            box2 = preds[j][:4]
            # print(box1)
            # print(box2)
            # print(compute_iou(box1, box2))
            if compute_iou(box1, box2) > iou_th:
                removed[j] = 1
                
        # print(removed)
            
    return nms_preds
    
def predict_multiscale(frm, mdl):
    predictions = []
    predicts = mdl.match(frm)

    r = 1

    x1_, y1_, x2_, y2_ = 0, 0, frm.shape[1], frm.shape[0]

    for p, pred in enumerate(predicts):
        x1_pre,y1_pre, x2_pre, y2_pre, score, time_ = pred
        x1 = int(((x1_pre+x1_)*r))
        y1 = int(((y1_pre+y1_)*r))
        x2 = int((x2_pre+x1_)*r)
        y2 = int((y2_pre+y1_)*r)
        # image = cv2.rectangle(tar.copy(),(x1,y1),(x2,y2),color=(255, 255, 0),thickness=5)
        # plt.imshow(image, cmap='gray')
        # plt.title(f"final box score :{score} anad x1: {x1}, y1:{y1}, w: {x2}, h:{y2}")
        # plt.show()
        predicts[p] = [x1, y1, x2, y2, score, time_]
        
    predictions += predicts

    # print(len(predictions))

    # print('Average consumed time for predicting: ', consumed_time / counter)
    predictions = NMS(predictions)
    # print(len(predictions))
    predictions.sort(reverse = True, key = itemgetter(4))
    return predictions[:10]

def plot_result_multi(raw_img, bboxes):
    d_img = raw_img.copy()
    # print(f'boxes = {len(bboxes)}')
    # print(f'shape = {raw_img.shape}')
    for elem in bboxes:
        x1, y1, x2, y2, _, _ = elem
        # print(x1, y1, x2, y2)
        cv2.rectangle(d_img, (x1, y1), (x2, y2), (255, 255, 0), 2)

    return d_img


model = Yolo('/home/user/Desktop/trans/yolo_stuff/bests_cup/catidx0_nshot1.pt')
model.set_templates()
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if ret:
        predictions = predict_multiscale(frame, model)

        d_img = plot_result_multi(frame, predictions)

        
        cv2.imshow('test', d_img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
