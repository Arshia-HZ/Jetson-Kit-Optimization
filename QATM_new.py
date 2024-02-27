import numpy as np
import imutils
import cv2
import time
from operator import itemgetter


import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
# from seaborn import color_palette
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import models, transforms, utils
import copy
# from utils import *

# from __future__ import print_function, division
import matplotlib.pyplot as plt
import math
# from sklearn.metrics import auc
import numpy as np
import cv2
import os, sys
from sklearn.preprocessing import MinMaxScaler
from scipy import signal

from torch2trt import torch2trt


image_name = '/mnt/janjan/QATM_pytorch-master/sample//sample1.jpg'

image_raw = cv2.imread(image_name)

# TODO  ##########################
TARGET_SHAPE = (1, 3, image_raw.shape[0], image_raw.shape[1])
################


class Featex():
    def __init__(self, model, use_cuda):
        self.use_cuda = use_cuda
        self.feature1 = None
        self.feature2 = None
        self.model = copy.deepcopy(model.eval())
        self.model = self.model[:17]
        for param in self.model.parameters():
            param.requires_grad = False
        if self.use_cuda:
            self.model = self.model.cuda()
        self.model[2].register_forward_hook(self.save_feature1)
        self.model[16].register_forward_hook(self.save_feature2)

    def save_feature1(self, module, input, output):
        self.feature1 = output.detach()

    def save_feature2(self, module, input, output):
        self.feature2 = output.detach()

    def __call__(self, input, mode='big'):
        if self.use_cuda:
            input = input.cuda()
        _ = self.model(input)
        if mode == 'big':
            # resize feature1 to the same size of feature2
            self.feature1 = F.interpolate(self.feature1, size=(self.feature2.size()[2], self.feature2.size()[3]),
                                          mode='bilinear', align_corners=True)
        else:
            # resize feature2 to the same size of feature1
            self.feature2 = F.interpolate(self.feature2, size=(self.feature1.size()[2], self.feature1.size()[3]),
                                          mode='bilinear', align_corners=True)
        return torch.cat((self.feature1, self.feature2), dim=1)



class MyNormLayer():
    def __call__(self, x1, x2):
        bs, _, H, W = x1.size()
        _, _, h, w = x2.size()
        x1 = x1.view(bs, -1, H * W)
        x2 = x2.view(bs, -1, h * w)
        concat = torch.cat((x1, x2), dim=2)
        x_mean = torch.mean(concat, dim=2, keepdim=True)
        x_std = torch.std(concat, dim=2, keepdim=True) + 1e-6
        # x1 = (x1 - x_mean) / x_std # TODO: Do not normalize 
        # x2 = (x2 - x_mean) / x_std
        x1 = x1.view(bs, -1, H, W)
        x2 = x2.view(bs, -1, h, w)
        return [x1, x2]


def find_keypart(template):
  candidate_size = (template.shape[0] // 2, template.shape[1] // 2)
  stride_size = (template.shape[0] // 4, template.shape[1] // 4)
  m = np.array([int(np.mean(template[...,0])), int(np.mean(template[...,1])), int(np.mean(template[...,2]))])
  start_point = [0, 0]
  min_value = 0
  min_index = [0, 0]
  for i in range(3):
      start_point[0] = i * stride_size[0]
      for j in range(3):
          start_point[1] = j * stride_size[1]
          candid = template.copy()
          candid[start_point[0]: start_point[0] + candidate_size[0], start_point[1]: start_point[1] + candidate_size[1]] = m
          i_product = sum(candid.reshape(-1)* template.reshape(-1))
          if i == 0 and j == 0:
              min_value = i_product
          if i_product < min_value:
              min_value = i_product
              min_index = [i, j]
  start_point[0] = min_index[0] * stride_size[0]
  start_point[1] = min_index[1] * stride_size[1]
  key = template.copy()
  key = key[start_point[0]: start_point[0] + candidate_size[0], start_point[1]: start_point[1] + candidate_size[1]]
  return key


def the_corr(x):
    T_feat = np.transpose(x[0].cpu().numpy(), axes=[0, 2, 3, 1])
    k_feat = np.transpose(x[1].cpu().numpy(), axes=[0, 2, 3, 1])
    scaler = MinMaxScaler()
    for i in range(T_feat.shape[3]):
       T_feat[0 ,: ,: ,i] = scaler.fit_transform(signal.correlate2d(T_feat[0 ,: ,: ,i], k_feat[0 ,: ,: ,i], boundary='fill', mode='same')) + 0.75
    T_feat = torch.tensor(np.transpose(T_feat, axes=[0, 3, 1, 2]))
    return T_feat


class CreateModel():
    def __init__(self, alpha, model, use_cuda):
        self.alpha = alpha
        self.featex = Featex(model=model, use_cuda=use_cuda)
        self.I_feat = None
        self.I_feat_name = None

    def fuse(self, target, temp_features):
        if self.I_feat is None:
            self.I_feat = self.featex(target)
        conf_maps = None

        for elem in temp_features:
            batchsize_T = elem.size()[0]
            for i in range(batchsize_T):
                T_feat_i = elem[i].unsqueeze(0)
                # T_feat_i = T_feat_i * k_feat
                I_feat_norm, T_feat_i = MyNormLayer()(self.I_feat, T_feat_i)
                dist = torch.einsum("xcab,xcde->xabde", I_feat_norm / torch.norm(I_feat_norm, dim=1, keepdim=True),
                                    T_feat_i / torch.norm(T_feat_i, dim=1, keepdim=True))

                conf_map = QATM_Model(self.alpha)(dist)

                if conf_maps is None:
                    conf_maps = conf_map
                else:
                    conf_maps = torch.cat([conf_maps, conf_map], dim=0)
        return conf_maps

    def __call__(self, template, image):
        T_feat = self.featex(template)
        if self.I_feat is None:
            self.I_feat = self.featex(image)
        conf_maps = None

        # key = find_keypart(np.transpose(template.cpu().numpy()[0], axes=[1, 2, 0]))
        # mean = np.array([0.485, 0.456, 0.406])
        # std = np.array([0.229, 0.224, 0.225])
        # key -= mean
        # key /= std
        # key = np.expand_dims(np.transpose(key, axes=[2, 0, 1]), axis=0)
        # k_feat = self.featex(torch.tensor(key))
        # k_feat = the_corr([T_feat, k_feat])

        batchsize_T = T_feat.size()[0]
        for i in range(batchsize_T):
            T_feat_i = T_feat[i].unsqueeze(0)
            # T_feat_i = T_feat_i * k_feat
            I_feat_norm, T_feat_i = MyNormLayer()(self.I_feat, T_feat_i)
            dist = torch.einsum("xcab,xcde->xabde", I_feat_norm / torch.norm(I_feat_norm, dim=1, keepdim=True),
                                T_feat_i / torch.norm(T_feat_i, dim=1, keepdim=True))

            conf_map = QATM_Model(self.alpha)(dist)

            if conf_maps is None:
                conf_maps = conf_map
            else:
                conf_maps = torch.cat([conf_maps, conf_map], dim=0)
        return conf_maps


class QATM_Model():
    def __init__(self, alpha):
        self.alpha = alpha

    def __call__(self, x):
        batch_size, ref_row, ref_col, qry_row, qry_col = x.size()
        x = x.view(batch_size, ref_row * ref_col, qry_row * qry_col)
        xm_ref = x - torch.max(x, dim=1, keepdim=True)[0]
        xm_qry = x - torch.max(x, dim=2, keepdim=True)[0]
        confidence = torch.sqrt(F.softmax(self.alpha * xm_ref, dim=1) * F.softmax(self.alpha * xm_qry, dim=2))
        conf_values, ind3 = torch.topk(confidence, 1)
        ind1, ind2 = torch.meshgrid(torch.arange(batch_size), torch.arange(ref_row * ref_col))
        ind1 = ind1.flatten()
        ind2 = ind2.flatten()
        ind3 = ind3.flatten()
        if x.is_cuda:
            ind1 = ind1.cuda()
            ind2 = ind2.cuda()

        values = confidence[ind1, ind2, ind3]
        values = torch.reshape(values, [batch_size, ref_row, ref_col, 1])
        return values

    def compute_output_shape(self, input_shape):
        bs, H, W, _, _ = input_shape
        return (bs, H, W, 1)


def compute_score(x, w, h):
    # score of response strength
    k = np.ones((h, w))
    score = cv2.filter2D(x, -1, k)
    score[:, :w // 2] = 0
    score[:, math.ceil(-w / 2):] = 0
    score[:h // 2, :] = 0
    score[math.ceil(-h / 2):, :] = 0
    return score


def nms_multi(scores, w_array, h_array, thresh_list):
    indices = np.arange(scores.shape[0])
    maxes = np.max(scores.reshape(scores.shape[0], -1), axis=1)
    # omit not-matching templates
    scores_omit = scores[maxes > 0.1 * maxes.max()]
    indices_omit = indices[maxes > 0.1 * maxes.max()]
    # extract candidate pixels from scores
    dots = None
    dos_indices = None
    for index, score in zip(indices_omit, scores_omit):
        dot = np.array(np.where(score > thresh_list[index] * score.max()))
        if dots is None:
            dots = dot
            dots_indices = np.ones(dot.shape[-1]) * index
        else:
            dots = np.concatenate([dots, dot], axis=1)
            dots_indices = np.concatenate([dots_indices, np.ones(dot.shape[-1]) * index], axis=0)
    dots_indices = dots_indices.astype(np.int)
    x1 = dots[1] - w_array[dots_indices] // 2
    x2 = x1 + w_array[dots_indices]
    y1 = dots[0] - h_array[dots_indices] // 2
    y2 = y1 + h_array[dots_indices]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    scores = scores[dots_indices, dots[0], dots[1]]
    order = scores.argsort()[::-1]
    dots_indices = dots_indices[order]
    keep = []
    keep_index = []
    keep_scores = []
    while order.size > 0:
        i = order[0]
        index = dots_indices[0]
        keep.append(i)
        keep_index.append(index)
        keep_scores.append(scores[i])
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= 0.05)[0]
        order = order[inds + 1]
        dots_indices = dots_indices[inds + 1]
    #     print("keep_scores " , keep_scores)
    boxes = np.array([[x1[keep], y1[keep]], [x2[keep], y2[keep]]]).transpose(2, 0, 1)
    return boxes, np.array(keep_index), keep_scores


def nms_multi_2(scores, w_array, h_array, thresh_list):
    indices = np.arange(scores.shape[0])
    maxes = np.max(scores.reshape(scores.shape[0], -1), axis=1)
    # omit not-matching templates
    scores_omit = scores[maxes > 0.1 * maxes.max()]
    indices_omit = indices[maxes > 0.1 * maxes.max()]
    # extract candidate pixels from scores
    dots = None
    dos_indices = None
    thresh_list *= len(indices_omit)
    for index, score in zip(indices_omit, scores_omit):
        dot = np.array(np.where(score > thresh_list[index] * score.max()))
        if dots is None:
            dots = dot
            dots_indices = np.ones(dot.shape[-1]) * index
        else:
            dots = np.concatenate([dots, dot], axis=1)
            dots_indices = np.concatenate([dots_indices, np.ones(dot.shape[-1]) * index], axis=0)
    dots_indices = dots_indices.astype(int)
    x1 = dots[1] - w_array[dots_indices] // 2
    x2 = x1 + w_array[dots_indices]
    y1 = dots[0] - h_array[dots_indices] // 2
    y2 = y1 + h_array[dots_indices]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    scores = scores[dots_indices, dots[0], dots[1]]
    order = scores.argsort()[::-1]
    dots_indices = dots_indices[order]
    keep = []
    keep_index = []
    keep_scores = []
    while order.size > 0:
        i = order[0]
        index = dots_indices[0]
        keep.append(i)
        keep_index.append(index)
        keep_scores.append(scores[i])
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= 0.05)[0]
        order = order[inds + 1]
        dots_indices = dots_indices[inds + 1]
    #     print("keep_scores " , keep_scores)
    boxes = np.array([[x1[keep], y1[keep]], [x2[keep], y2[keep]]]).transpose(2, 0, 1)
    return boxes, np.array(keep_index), keep_scores


def score_predictions(val, template_size, image_size):

    if val.is_cuda:
        val = val.cpu()
    val = val.numpy()

    val = np.log(val)

    batch_size = val.shape[0]
    scores = []

    h, w = [], []
    for element in template_size:
        h.append(element[-2])
        w.append(element[-1])

    for i in range(batch_size):
        gray = val[i, :, :, 0]
        gray = cv2.resize(gray, (image_size[-1], image_size[-2]))
        score = compute_score(gray, w[i], h[i])
        score[score > -1e-7] = score.min()
        score = np.exp(score / (h[i] * w[i]))
        scores.append(score)
    return np.array(scores)



def run_one_sample(model, template, image):
    # print('template = {} : {}'.format(np.min(template.numpy()), np.max(template.numpy())))
    # print('image = {} : {}'.format(np.min(image.numpy()), np.max(image.numpy())))
    
    val = model(template, image)

    if val.is_cuda:
        val = val.cpu()
    val = val.numpy()
    # print('val = {} : {}'.format(np.min(val), np.max(val)))

    val = np.log(val)

    batch_size = val.shape[0]
    scores = []
    for i in range(batch_size):
        # compute geometry average on score map
        gray = val[i, :, :, 0]
        # print('gray = {}'.format(np.max(gray)))
        gray = cv2.resize(gray, (image.size()[-1], image.size()[-2]))
        h = template.size()[-2]
        w = template.size()[-1]
        score = compute_score(gray, w, h)
        # print('score = {}'.format(np.max(score)))
        score[score > -1e-7] = score.min()
        score = np.exp(score / (h * w))  # reverse number range back after computing geometry average
        scores.append(score)
    return np.array(scores)


class QATM():
    def __init__(self):
        model = models.vgg19(pretrained=True).features
        self.model = CreateModel(model=model , alpha=25, use_cuda=True)

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            )
        ])
        
        self.scales = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5]
        self.template_size = None

        self.flag = False


    def extract_template_features(self, template_dataset):
        featex_list = []

        self.template_size = []

        for template_data in template_dataset:
            temp = template_data['template'] 
            h, w = temp.shape[:2]
            scale = math.sqrt((w * h) / (60 ** 2))
            dim = (int(w / scale), int(h / scale))
            temp = cv2.resize(src=temp, dsize=dim)
            temp = self.transform(temp)
            self.template_size.append(temp.size())
            r = 1
            h = temp.size()[-2]
            w = temp.size()[-1]
            # x1, y1, x2, y2 = -1, -1, -1, -1

            """
            if target_size[0] < h or target_size[1] < w:
                print("small image")
                self.flag = True
                continue
            """

            if w * h < 200 or w * h > 1000000:
                print("small target")
                self.flag = True
                continue

            temp_feat = self.model.featex(temp.unsqueeze(0))
            featex_list.append(temp_feat)
        
        if len(featex_list) == 0:
            return None
        return featex_list


    def match(self, target, temp_features):

        start = time.time()

        predictions = []
        r = 1
        if temp_features is None:
            predictions.append([-1, -1, -1, -1,
                                -1, -1])
        else:
            self.model.I_feat = None
            target = self.transform(target)
            val = self.model.fuse(target.unsqueeze(0), temp_features)
            score = score_predictions(val, self.template_size, target.size())
            
            if math.isnan(score.max()):
                return predictions.append([-1, -1, -1, -1,
                                -1, -1])
            h, w = [], []
            for element in self.template_size:
                h.append(element[-2])
                w.append(element[-1])

            boxes, indices, scores = nms_multi_2(score, np.array(w), np.array(h),
                                                   [0.5])

            for i in range(len(indices)):
                x1, y1 = boxes[i].tolist()[0]
                x2, y2 = boxes[i].tolist()[1]

                x1 = int(x1 * r)
                y1 = int(y1 * r)
                x2 = int(x2 * r)
                y2 = int(y2 * r)

                end = time.time()
                timePred = float("{:.3f}".format(end - start))

                predictions.append([x1, y1, x2, y2,
                                    float("{:.3f}".format(scores[i])), timePred])

            predictions.sort(reverse=True, key=itemgetter(4))
        
        return predictions[:10]


    def predict(self, target, templates, temp_boxes):
        predictions = []

        self.model.I_feat = None
        target = self.transform(target)
        for i, temp in enumerate(templates):
            x1, y1, x2, y2 = temp_boxes[i]
            temp = temp[y1:y2, x1:x2, :]
            
            h, w = temp.shape[:2]
            # if h * w > 10000:
            scale = math.sqrt((w * h) / (60 ** 2))
            temp = cv2.resize(temp, (int(w / scale), int(h / scale)))

            temp = self.transform(temp)
            
            start = time.time()
            
            # print('{} {}'.format(temp.shape, target.shape))

            r = 1 # target.shape[1] / float(target.shape[1])

            h = temp.size()[-2]
            w = temp.size()[-1]
            x1, y1, x2, y2 = -1, -1, -1, -1

            if target.size()[-2] < h or target.size()[-1] < w:
                print("small image")
                end = time.time()
                timePred = float("{:.3f}".format(end - start))
                continue

            if w * h < 200 or w * h > 1000000:
                print("small target")
                end = time.time()
                timePred = float("{:.3f}".format(end - start))
                continue
                
            score = run_one_sample(self.model, temp.unsqueeze(0), target.unsqueeze(0))
            # print('max score = {}'.format(score.max()))
            del temp
            
            if math.isnan(score.max()):
                end = time.time()
                timePred = float("{:.3f}".format(end - start))
                continue
            boxes, indices, scores = nms_multi(np.squeeze(np.array([score]), axis=1), np.array([w]), np.array([h]),
                                                   [0.5])
            end = time.time()
            timePred = float("{:.3f}".format(end - start))
            for i in range(len(indices)):
                x1, y1 = boxes[i].tolist()[0]
                x2, y2 = boxes[i].tolist()[1]

                x1 = int(x1 * r)
                y1 = int(y1 * r)
                x2 = int(x2 * r)
                y2 = int(y2 * r)

                predictions.append([x1, y1, x2, y2,
                                    float("{:.3f}".format(scores[i])), timePred])
                                    
                # print(x1, y1, x2, y2, scores[i])

        if len(predictions) == 0:
            predictions.append([x1, y1, x2, y2,
                                -1, timePred])
        # print("predictions : ", predictions)

        predictions.sort(reverse=True, key=itemgetter(4))
        
        #
        del target
        torch.cuda.empty_cache()

        # TODO : add a parameter for get max of predictions data e.g. in this case it is 10.
        return predictions[:10]




####################################################################################################
class TemplateDataset(torch.utils.data.Dataset):
    def __init__(self, template_dir_path, thresh_csv=None):
        self.template_path = list(Path(template_dir_path).iterdir())
        
        self.thresh_df = None
        if thresh_csv:
            self.thresh_df = pd.read_csv(thresh_csv)
        
    def __len__(self):
        return len(self.template_path)

    def __getitem__(self, idx):
        template_path = str(self.template_path[idx])
        template = cv2.imread(template_path)
        
        thresh = 0.7
        if self.thresh_df is not None:
            if self.thresh_df.path.isin([template_path]).sum() > 0:
                thresh = float(self.thresh_df[self.thresh_df.path==template_path].thresh)
        return {
                    'template': template,
                    'template_name': template_path,
                    'template_h': template.shape[-2],
                    'template_w': template.shape[-1],
                    'thresh': thresh
                }

from seaborn import color_palette

def plot_result(image_raw, boxes, show=False, save_name=None, color=(255, 0, 0)):
    # plot result
    d_img = image_raw.copy()
    for box in boxes:
        d_img = cv2.rectangle(d_img, tuple(box[0]), tuple(box[1]), color, 3)
    if show:
        plt.imshow(d_img[:,:,::-1])
    if save_name:
        cv2.imwrite(save_name, d_img)
    return d_img


def plot_result_multi(image_raw, boxes, indices, show=False, save_name=None, color_list=None):
    d_img = image_raw.copy()
    if color_list is None:
        color_list = color_palette("hls", indices.max()+1)
        color_list = list(map(lambda x: (int(x[0]*255), int(x[1]*255), int(x[2]*255)), color_list))
    for i in range(len(indices)):
        d_img = plot_result(d_img, boxes[i][None, :,:].copy(), color=color_list[indices[i]])
    if show:
        plt.imshow(d_img[:,:,::-1])
    if save_name:
        cv2.imwrite(save_name, d_img)
    return d_img




template_dir = '/mnt/janjan/QATM_pytorch-master/template/'
template_dataset = TemplateDataset(template_dir)

qatm = QATM()


start = time.time()
featex_list = qatm.extract_template_features(template_dataset=template_dataset)

predictions = qatm.match(target=image_raw, temp_features=featex_list)

end = time.time()

print("################# time: " + str((end-start)*1000) + " msec #######################")

for elem in predictions:
    x1, y1, x2, y2, _, _ = elem
    cv2.rectangle(image_raw, (x1, y1), (x2, y2), (255, 255, 0), 2)

cv2.imwrite('/home/user/Desktop/temp.png', image_raw)

print(predictions)
