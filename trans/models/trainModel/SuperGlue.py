#! /usr/bin/env python3
from pathlib import Path
import argparse
import random
import cv2
import numpy as np
import matplotlib.cm as cm
import torch
import matplotlib.pyplot as plt
import os
from easydict import EasyDict as edict
import time
import sys
sys.path.append("superGlue")

from baseFewShotMatcher import BaseFewShotMatcher
from trainModel.superGlue.models.matching import Matching
from operator import itemgetter
from trainModel.superGlue.models.utils import (compute_pose_error, compute_epipolar_error,
                                                      estimate_pose, make_matching_plot,
                                                      error_colormap, AverageTimer, pose_auc, read_image,
                                                      rotate_intrinsics, rotate_pose_inplane,
                                                      scale_intrinsics)
class SuperGlue(BaseFewShotMatcher):

    def predict(self, tar, templates, tempbox,_):
        opt = edict({
            "input_pairs":None,
            "input_dir":None,
            "output_dir":None,
            "max_length":-1,
            "resize":[-1],
            "resize_float":None,
            "superglue":"outdoor",
            # "max_keypoints":1024,
            # "keypoint_threshold":0.005,
            # "nms_radius":5,
            # "sinkhorn_iterations":20,
            # "match_threshold":0.2,
            "viz":True,
            "eval":False,
            "fast_viz":True,
            "cache":False,
            "show_keypoints":True,
            "opencv_display":False,
            "shuffle":True,
            "force_cpu":False,
            "viz_extension":"png"
        })

        for nms_radius in [5]:
            for keypoint_threshold in [0.001]:
                for max_keypoints in [1024]:
                    for sinkhorn_iterations in [20]:
                        for match_threshold in [0.2]:
                            device = 'cuda' if torch.cuda.is_available() and not opt.force_cpu else 'cpu'
                            do_match=True
                            # print('Running inference on device \"{}\"'.format(device))
                            # print(nms_radius,keypoint_threshold,max_keypoints,sinkhorn_iterations,match_threshold)
                            config = {
                                'superpoint': {
                                    'nms_radius': nms_radius,
                                    'keypoint_threshold': keypoint_threshold,
                                    'max_keypoints': max_keypoints
                                },
                                'superglue': {
                                    'weights': opt.superglue,
                                    'sinkhorn_iterations': sinkhorn_iterations,
                                    'match_threshold':match_threshold,
                                }
                            }
                            matching = Matching(config).eval().to(device)
                            predictions = []

                            for temp in templates:
                                t1 = time.time()
                                image0, inp0 = read_image(temp, device)
                                image1, inp1 = read_image(tar, device)
                                if image0 is None or image1 is None:
                                    print('Problem reading image pair')
                                if do_match:
                                    # Perform the matching.
                                    pred = matching({'image0': inp0, 'image1': inp1})
                                    pred = {k: v[0].cpu().detach().numpy() for k, v in pred.items()}
                                    kpts0, kpts1 = pred['keypoints0'], pred['keypoints1']
                                    matches, conf = pred['matches0'], pred['matching_scores0']
                                    out_matches = {'keypoints0': kpts0, 'keypoints1': kpts1,
                                                       'matches': matches, 'match_confidence': conf}
                                    valid = matches > -1
                                    mkpts0 = kpts0[valid]
                                    mkpts1 = kpts1[matches[valid]]
                                    mconf = conf[valid]
                                    Matches=np.sum(matches> -1)
                                    dst_pts = np.float32(mkpts1).reshape(-1,1,2)#target_kpt
                                    src_pts=np.float32(mkpts0).reshape(-1,1,2)#temp_kpt
                                    if  Matches<4:
                                        predictions.append([-1,-1,-1,-1,-1,-1])
                                    else:
                                        matrix, mask = cv2.findHomography(src_pts,dst_pts, cv2.RANSAC, 5.0)
                                        h,w = temp.shape
                                        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
                                        if pts is None or matrix is None:
                                            predictions.append([-1,-1,-1,-1,-1,-1])
                                        else:
                                            dst = cv2.perspectiveTransform(pts, matrix)
                                            x = int(np.min(dst[:, 0, 0]))
                                            x1 = int(np.max(dst[:, 0, 0]))
                                            y = int(np.min(dst[:, 0, 1]))
                                            y1 = int(np.max(dst[:, 0, 1]))
                                            t2 = time.time()
                                            predictions.append([x,y,x1,y1, mconf.sum()*Matches, (t2-t1)])
                                            predictions.sort(reverse = True, key=itemgetter(4))
                            return predictions
