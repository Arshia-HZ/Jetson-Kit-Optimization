import numpy as np
import imutils
import cv2
import time
from operator import itemgetter

from baseFewShotMatcher import BaseFewShotMatcher
# from baseFewShotMatcher_sensitivity import BaseFewShotMatcher

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
# from torch2trt import torch2trt

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

class Featex2():
    def __init__(self, model1, model2, use_cuda):
        self.use_cuda = use_cuda
        
        """
        if use_cuda:
            model1 = model1.to('cuda')
            model2 = model2.to('cuda')
        """
        
        ############################################
        
        self.input_shape_1 = (1, 3, 40, 40)
        self.input_shape_2 = (1, 3, 255, 255) ### >>> this line should be equal to original picture shape (please change this based on input shape) i.e. (1, 3, 225, 384)
        self.input_shape_3 = (1, 3, 20, 45)
        self.input_shape_4 = (1, 3, 35, 38)
        #############################################

        self.feature1 = None
        self.feature2 = None
        layer1to2 = nn.Sequential(*list(model1[:3])).to('cuda')
        layer1to16 = nn.Sequential(*list(model2[:17])).to('cuda')
        
        self.sample_data_1 = torch.zeros(self.input_shape_1).to('cuda')
        self.layer1to2_trt_1 = torch2trt(layer1to2, [self.sample_data_1])
        self.sample_data_1 = torch.zeros(self.input_shape_1).to('cuda')
        self.layer1to16_trt_1 = torch2trt(layer1to16, [self.sample_data_1])

        self.sample_data_2 = torch.zeros(self.input_shape_2).to('cuda')
        self.layer1to2_trt_2 = torch2trt(layer1to2, [self.sample_data_2])
        self.sample_data_2 = torch.zeros(self.input_shape_2).to('cuda')
        self.layer1to16_trt_2 = torch2trt(layer1to16, [self.sample_data_2])

        self.sample_data_3 = torch.zeros(self.input_shape_3).to('cuda')
        self.layer1to2_trt_3 = torch2trt(layer1to2, [self.sample_data_3])
        self.sample_data_3 = torch.zeros(self.input_shape_3).to('cuda')
        self.layer1to16_trt_3 = torch2trt(layer1to16, [self.sample_data_3])

        self.sample_data_4 = torch.zeros(self.input_shape_4).to('cuda')
        self.layer1to2_trt_4 = torch2trt(layer1to2, [self.sample_data_4])
        self.sample_data_4 = torch.zeros(self.input_shape_4).to('cuda')
        self.layer1to16_trt_4 = torch2trt(layer1to16, [self.sample_data_4])
        

    def __call__(self, target, mode='big'):
        if self.use_cuda:
            target = target.to('cuda')
        print('in the trt')
        # we replace hook_register_forward with trt module

        ###################################################
        self.model1 = self.layer1to2_trt_2
        self.model2 = self.layer1to16_trt_2

        ###################################################

        self.feature1 = self.model1(target).detach() #need to be changed
        self.feature2 = self.model2(target).detach() #need to be changed


        if mode=='big':
            # resize feature1 to the same size of feature2
            self.feature1 = F.interpolate(self.feature1, size=(self.feature2.size()[2], self.feature2.size()[3]), mode='bilinear', align_corners=True)
        else:
            # resize feature2 to the same size of feature1
            self.feature2 = F.interpolate(self.feature2, size=(self.feature1.size()[2], self.feature1.size()[3]), mode='bilinear', align_corners=True)
        
        result = torch.cat((self.feature1, self.feature2), dim=1)
        return result



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
    def __init__(self, alpha, model1, model2, model3, use_cuda, args_model):
        self.alpha = alpha
        self.featex = Featex(model1, use_cuda)
        self.featex2 = Featex2(model1=model2, model2=model3, use_cuda=use_cuda)
        self.I_feat = None
        self.I_feat_name = None
        self.temp_features = None
        self.mynormlayer = MyNormLayer()
        self.qatm_model = QATM_Model(alpha, args_model, use_cuda)
        

    def fuse(self, target, trt_or_not):
        if self.I_feat is None:
            if trt_or_not:
                self.I_feat = self.featex2(target)
            else:
                self.I_feat = self.featex(target)
                
        conf_maps = None
        for elem in self.temp_features:
            batchsize_T = elem.size()[0]
            for i in range(batchsize_T):
                T_feat_i = elem[i].unsqueeze(0)
                # T_feat_i = T_feat_i * k_feat
                # I_feat_norm, T_feat_i = self.mynormlayer(self.I_feat, T_feat_i)
                dist = torch.einsum("xcab,xcde->xabde", self.I_feat / torch.norm(self.I_feat, dim=1, keepdim=True),
                                    T_feat_i / torch.norm(T_feat_i, dim=1, keepdim=True))
                                    
                conf_map = self.qatm_model(dist)
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
    def __init__(self, alpha, args_qatm_model, use_cuda=True):
        self.alpha = alpha
        self.init_or_not = args_qatm_model[0]
        if args_qatm_model[0]:
            self.ind1, self.ind2 = torch.meshgrid(torch.arange(args_qatm_model[1]), torch.arange(args_qatm_model[2] * args_qatm_model[3]))
            self.ind1 = self.ind1.flatten()
            self.ind2 = self.ind2.flatten()
            if use_cuda:
                self.ind1 = self.ind1.cuda()
                self.ind2 = self.ind2.cuda()

    def __call__(self, x, use_cuda=True):
        batch_size, ref_row, ref_col, qry_row, qry_col = x.size()
        x = x.view(batch_size, ref_row * ref_col, qry_row * qry_col)
        xm_ref = x - torch.max(x, dim=1, keepdim=True)[0]
        xm_qry = x - torch.max(x, dim=2, keepdim=True)[0]
        confidence = torch.sqrt(F.softmax(self.alpha * xm_ref, dim=1) * F.softmax(self.alpha * xm_qry, dim=2))
        conf_values, ind3 = torch.topk(confidence, 1)
        if not self.init_or_not:
            ind1, ind2 = torch.meshgrid(torch.arange(batch_size), torch.arange(ref_row * ref_col))
            ind1 = ind1.flatten()
            ind2 = ind2.flatten()
            if use_cuda:
                ind1 = ind1.cuda()
                ind2 = ind2.cuda()
        ind3 = ind3.flatten()
        
        if self.init_or_not:
            values = confidence[self.ind1, self.ind2, ind3]
        else:    
            values = confidence[ind1, ind2, ind3]
        values = torch.reshape(values, [batch_size, ref_row, ref_col, 1])
        return values

    def compute_output_shape(self, input_shape):
        bs, H, W, _, _ = input_shape
        return (bs, H, W, 1)


def compute_score(x, w, h, reshaped, sgpu):
    if sgpu:
        k = torch.ones((1, 1, h, w), dtype=torch.float32).to('cuda')
        if reshaped:
            inputTen = torch.tensor(x.reshape(1, 1, x.shape[0], x.shape[1]), device='cuda')
        else:
            inputTen = torch.tensor(x.reshape(1, 1, x.shape[0], x.shape[1]), device='cuda')
        score = F.conv2d(inputTen, k, padding='same').cpu().detach().numpy()[0, 0, :, :]
    else:
        k = np.ones((h, w))
        score = cv2.filter2D(x, -1, k)
    score[:, :w // 2] = 0
    score[:, math.ceil(-w / 2):] = 0
    score[:h // 2, :] = 0
    score[math.ceil(-h / 2):, :] = 0
    return score


def nms_multi(scores, w_array, h_array, thresh_list, nms_count):
    indices = np.arange(scores.shape[0])
    maxes = np.max(scores.reshape(scores.shape[0], -1), axis=1)
    scores_omit = scores[maxes > 0.1 * maxes.max()]
    indices_omit = indices[maxes > 0.1 * maxes.max()]
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
    count = 0
    while order.size > 0:
        i = order[0]
        index = dots_indices[0]
        keep.append(i)
        keep_index.append(index)
        keep_scores.append(scores[i])
        if count >= nms_count:
            break
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
        count += 1
    boxes = np.array([[x1[keep], y1[keep]], [x2[keep], y2[keep]]]).transpose(2, 0, 1)
    return boxes, np.array(keep_index), keep_scores


def score_predictions(val, template_size, image_size, sGPU, numba, res):
    if val.is_cuda:
        val = val.cpu().detach()
    val = val.numpy()
    val = np.log(val)
    batch_size = val.shape[0]
    scores = []
    for i in range(batch_size):
        gray = val[i, :, :, 0]
        gray = cv2.resize(gray, (image_size[-1], image_size[-2]))
        
        score = compute_score(gray, template_size[i][-1], template_size[i][-2], res, sGPU)
            
        score[score > -1e-7] = score.min()
        
        if numba:
            score = my_exp(score / (template_size[i][-2] * template_size[i][-1]))
        else:
            score = np.exp(score / (template_size[i][-2] * template_size[i][-1]))
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


class QATM(BaseFewShotMatcher):
    def __init__(self, args_run):
        self.args = args_run
        model1 = models.vgg19(pretrained=True).features
        model2 = models.vgg19(pretrained=True).features
        model3 = models.vgg19(pretrained=True).features
        self.model = CreateModel(args_model=[args_run.init, args_run.init_batch, args_run.ref_row, args_run.ref_col], \
                                 model1=model1, model2=model2, model3=model3 , alpha=25, use_cuda=True)

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            )
        ])
        self.scales = [float(scale) for scale in args_run.scales.split(' ')]
        self.template_size = None
        self.flag = False
      


    def set_templates(self, templates, temp_boxes):
        featex_list = []

        self.template_size = []
        self.model.temp_features = None

        for i, temp in enumerate(templates):
            x1, y1, x2, y2 = temp_boxes[i]
            temp = temp[y1:y2, x1:x2, :]
            
            h, w = temp.shape[:2]
            if h * w > 10000:
                scale = math.sqrt((w * h) / (100 ** 2))
                temp = cv2.resize(temp, (int(w / scale), int(h / scale)))
            
            temp = self.transform(temp)
            
            h = temp.size()[-2]
            w = temp.size()[-1]
            
            if w * h < 200 or w * h > 1000000:
                print("small target")
                self.flag = True
                continue

            temp_feat = self.model.featex(temp.unsqueeze(0))
            featex_list.append(temp_feat)
            self.template_size.append(temp.size())
        
        self.model.temp_features = featex_list


    def match(self, target):

        start = time.time()

        predictions = []
        r = 1
        if len(self.model.temp_features)  == 0:
            predictions.append([-1, -1, -1, -1,
                                -1, -1])
        else:
            self.model.I_feat = None
            target = self.transform(target)
            val = self.model.fuse(target.unsqueeze(0), self.args.trt)
            score = score_predictions(val, self.template_size, target.size(), self.args.sGPU, self.args.numba, self.args.reshape)
            
            if math.isnan(score.max()):
                return predictions.append([-1, -1, -1, -1,
                                -1, -1])
            h, w = [], []
            for element in self.template_size:
                h.append(element[-2])
                w.append(element[-1])

            boxes, indices, scores = nms_multi(score, np.array(w), np.array(h), [0.5]*len(w), self.args.nms)

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
            if h * w > 10000:
                scale = math.sqrt((w * h) / (100 ** 2))
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
            x1, y1, x2, y2 = -1, -1, -1, -1
            timePred = -1
            predictions.append([x1, y1, x2, y2,
                                -1, timePred])
        # print("predictions : ", predictions)

        predictions.sort(reverse=True, key=itemgetter(4))
        
        #
        del target
        torch.cuda.empty_cache()

        # TODO : add a parameter for get max of predictions data e.g. in this case it is 10.
        return predictions[:10]
