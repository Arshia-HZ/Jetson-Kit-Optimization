verbose = -1

from pathlib import Path
import cv2
import time
import pickle
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import glob
import pandas as pd
import math
import imutils
from pandas import ExcelWriter

from operator import itemgetter


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
    
class BaseFewShotMatcher():
    def __init__(self):
        self.net = None
        self.tracker = None

    def IoU(self, box_pred, box_gt):
        if box_pred[0] != -1:
            xg1, yg1, xg2, yg2 = box_gt[0], box_gt[1], box_gt[2], box_gt[3]
            xp1, yp1, xp2, yp2 = box_pred[0], box_pred[1], box_pred[2], box_pred[3]
            intersect = np.maximum(np.minimum(xg2, xp2) - np.maximum(xg1, xp1), 0) *\
                        np.maximum(np.minimum(yg2, yp2) - np.maximum(yg1, yp1), 0)
            union = (yg2 - yg1) * (xg2 -xg1) + (yp2 - yp1) * (xp2 -xp1) - intersect
            iou = intersect / union
        else:
            iou = -1
        return iou

    def calc_aspect_ratio_orientation(self, width: int, height: int):
        if width < height:
            return "vertical"
        elif width > height:
            return "horizontal"
        else:
            return "square"

    def calc_ratio_and_slice(self, orientation, slide=1, ratio=0.1):
        if orientation == "vertical":
            slice_row, slice_col, overlap_height_ratio, overlap_width_ratio = slide, slide * 2, ratio, ratio
        elif orientation == "horizontal":
            slice_row, slice_col, overlap_height_ratio, overlap_width_ratio = slide * 2, slide, ratio, ratio
        elif orientation == "square":
            slice_row, slice_col, overlap_height_ratio, overlap_width_ratio = slide, slide, ratio, ratio

        return slice_row, slice_col, overlap_height_ratio, overlap_width_ratio  # noqa

    def calc_slice_and_overlap_params(self, resolution: str, height: int, width: int, orientation: str):
        if resolution == "medium":
            split_row, split_col, overlap_height_ratio, overlap_width_ratio = self.calc_ratio_and_slice(
                orientation, slide=1, ratio=0.8
            )

        elif resolution == "high":
            split_row, split_col, overlap_height_ratio, overlap_width_ratio = self.calc_ratio_and_slice(
                orientation, slide=2, ratio=0.4
            )

        elif resolution == "ultra-high":
            split_row, split_col, overlap_height_ratio, overlap_width_ratio = self.calc_ratio_and_slice(
                orientation, slide=4, ratio=0.4
            )
        else:  # low condition
            split_col = 1
            split_row = 1
            overlap_width_ratio = 1
            overlap_height_ratio = 1

        slice_height = height // split_col
        slice_width = width // split_row

        x_overlap = int(slice_width * overlap_width_ratio)
        y_overlap = int(slice_height * overlap_height_ratio)

        return x_overlap, y_overlap, slice_width, slice_height  # noqa

    def get_resolution_selector(self, res: str, height: int, width: int):
        orientation = self.calc_aspect_ratio_orientation(width=width, height=height)
        x_overlap, y_overlap, slice_width, slice_height = self.calc_slice_and_overlap_params(
            resolution=res, height=height, width=width, orientation=orientation)

        return x_overlap, y_overlap, slice_width, slice_height

    def calc_resolution_factor(self, resolution: int):
        expo = 0
        while np.power(2, expo) < resolution:
            expo += 1
        return expo - 1

    def get_auto_slice_params(self, height: int, width: int):
        resolution = height * width
        factor = self.calc_resolution_factor(resolution)
        if factor <= 18:
            return self.get_resolution_selector("low", height=height, width=width)
        elif 18 <= factor < 21:
            return self.get_resolution_selector("medium", height=height, width=width)
        elif 21 <= factor < 24:
            return self.get_resolution_selector("high", height=height, width=width)
        else:
            return self.get_resolution_selector("ultra-high", height=height, width=width)

    def get_slice_bboxes(self, image_height: int, image_width: int, slice_height: int = None,
                         slice_width: int = None, auto_slice_resolution: bool = True,
                         overlap_height_ratio: float = 0.2, overlap_width_ratio: float = 0.2, ):
        slice_bboxes = []
        y_max = y_min = 0
        if slice_height and slice_width:
            y_overlap = int(overlap_height_ratio * slice_height)
            x_overlap = int(overlap_width_ratio * slice_width)
        elif auto_slice_resolution:
            x_overlap, y_overlap, slice_width, slice_height = self.get_auto_slice_params(height=image_height,
                                                                                         width=image_width)
        else:
            raise ValueError("Compute type is not auto and slice width and height are not provided.")

        while y_max < image_height:
            x_min = x_max = 0
            y_max = y_min + slice_height
            while x_max < image_width:
                x_max = x_min + slice_width
                if y_max > image_height or x_max > image_width:
                    xmax = min(image_width, x_max)
                    ymax = min(image_height, y_max)
                    xmin = max(0, xmax - slice_width)
                    ymin = max(0, ymax - slice_height)
                    slice_bboxes.append([xmin, ymin, xmax, ymax])
                else:
                    slice_bboxes.append([x_min, y_min, x_max, y_max])
                x_min = x_max - x_overlap
            y_min = y_max - y_overlap
        return slice_bboxes
        
    def predict(self, target, templates, boxes_):
        predictions = []
        return predictions
    
    def set_templates(self, templates, boxes_):
        pass

    def match(self, target):
        pass

    def nms0(self, boxes, overlapThresh=0.5):
        if len(boxes) == 0:
            return []
        pick = []
        boxes = np.array(boxes)
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        idxs = np.argsort(y2)
        # list
        while len(idxs) > 0:
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)
            # for the end of the bounding box
            xx1 = np.maximum(x1[i], x1[idxs[:last]])
            yy1 = np.maximum(y1[i], y1[idxs[:last]])
            xx2 = np.minimum(x2[i], x2[idxs[:last]])
            yy2 = np.minimum(y2[i], y2[idxs[:last]])
            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)
            overlap = (w * h) / area[idxs[:last]]
            idxs = np.delete(idxs, np.concatenate(([last],
                                                   np.where(overlap > overlapThresh)[0])))
        return boxes[pick].astype("int")
        
    def NMS(self, preds, iou_th=0.5):
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

    def index_2d(self, myList, v):
        for i, x in enumerate(myList):
            if v == x[0:4]:
                return i

    def visualize(self, tar, predictions, tarGT, filename=None):
        boxs = []
        target = tar.copy()
        for pred in predictions:
            boxs.append(pred[0:4])

        nms_boxs = list(self.nms0(boxs))

        # gt
        cv2.rectangle(target.copy(), (tarGT[0], tarGT[1]),
                      (tarGT[2], tarGT[3]), (255, 0, 0), 2)
        startX = int(tarGT[0] +
                     (tarGT[2] - tarGT[0]) / 4)
        stratY = int(tarGT[1]) - 5

        if (stratY - 5) < 0:
            stratY = int(tarGT[3]) + 15
        cv2.putText(img=target, text='GT', org=(startX, stratY),
                    fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=1 / 2, color=(255, 0, 0), thickness=1)
        for nms_box in nms_boxs:
            i = self.index_2d(predictions,
                              nms_box.tolist())
            # print(i, predictions)
            if predictions[i][4] != -1:  # value = -1
                cv2.rectangle(target, (list(nms_box)[0], list(nms_box)[1]),
                              (list(nms_box)[2], list(nms_box)[3]), (0, 255, 0), 2)
                startX = int(predictions[i][0] +
                             (predictions[i][2] - predictions[i][0]) / 4)
                stratY = int(predictions[i][1]) - 5

                if (stratY - 5) < 0:
                    stratY = int(predictions[i][3]) + 15
                cv2.putText(img=target, text=str(predictions[i][4]), org=(startX, stratY),
                            fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=1 / 2, color=(0, 255, 0), thickness=1)
        
        plt.imsave(filename or "sample.png", target)
    
    def predict_multiscale(self, target, templates, temp_boxes, patch_shape=512, vers='old', ind_dr=0, tmp_ind_list=[]):
        if vers != 'old':
            self.set_templates(templates, temp_boxes, ind_dr, tmp_ind_list)

        predictions = []
        patch_shapes = []
        consumed_time = 0.0
        time_of_each_img = []
        tar = target.copy()
        counter = 0
        whole_infer_st = time.time()
        predict_start = time.time()
        for scale in self.scales:
            resized = imutils.resize(tar, width=int(tar.shape[1]*scale), height=int(tar.shape[0]*scale))
            if 1: # required fix size such as SiamSE
                if patch_shape != 0:
                    patch_shapes = [patch_shape, patch_shape]
                    if resized.shape[0] < patch_shape or resized.shape[1] < patch_shape:
                        # print(resized.shape)
                        resized = cv2.copyMakeBorder(resized, 0, np.maximum(patch_shape-resized.shape[0], 0), 0, np.maximum(patch_shape-resized.shape[1], 0), cv2.BORDER_REPLICATE)
                        # resized = imutils.resize(tar, width=patch_shape, height=patch_shape)
                        # print(resized.shape)
            
            # r = tar.shape[1] / float(resized.shape[1])
                else:
                    patch_shapes = [resized.shape[0], resized.shape[1]]
            r = 1 / scale
            max_score = 0.0
            crop_box = []
            tar_boxes = self.get_slice_bboxes(resized.shape[0], resized.shape[1], patch_shapes[0], patch_shapes[1], overlap_height_ratio = 0.25, overlap_width_ratio=0.25)
            for tar_box in tar_boxes:
                if verbose > 0:
                    print('scale = {}, box = {}'.format(scale, tar_box))

                x1_, y1_, x2_, y2_ = tar_box[0], tar_box[1], tar_box[2], tar_box[3]
                
                # print('tar_box = {}, tar_shape = {}'.format(tar_box, resized.shape))
                if x2_ - x1_ < patch_shapes[1]:
                    # print('tar_box = {}, tar_shape = {}'.format(tar_box, resized.shape))
                    x1_ = x2_ - resized.shape[1]
                    x1_ = np.maximum(x1_, 0)
                    # print(x1_, y1_, x2_, y2_)
                if y2_ - y1_ < patch_shapes[0]:
                    # print('tar_box = {}, tar_shape = {}'.format(tar_box, resized.shape))
                    y1_ = y2_ - resized.shape[0]
                    y1_ = np.maximum(y1_, 0)
                    # print(x1_, y1_, x2_, y2_)
                # print(x1_, y1_, x2_, y2_)
                
                tar_crop = resized[y1_:y2_, x1_:x2_]
                tar_ = tar_crop.copy()
                
                start = time.time()
                if vers == 'old':
                    predicts = self.predict(tar_, templates, temp_boxes)
                else:
                    predicts = self.match(tar_)
                    counter += 1
                end = time.time()
                consumed_time += end - start
                
                time_of_each_img.append(consumed_time)

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
        predict_end = time.time()

        # print(len(predictions))

        # print('Average consumed time for predicting: ', consumed_time / counter)
        nms_st = time.time()
        predictions = self.NMS(predictions)
        nms_ed = time.time()
        # print(len(predictions))
        sort_st = time.time()
        predictions.sort(reverse = True, key = itemgetter(4))
        sort_ed = time.time()
        whole_infer_ed = time.time()
        return predictions[:10], time_of_each_img, len(templates), predict_end - predict_start, nms_ed - nms_st, sort_ed - sort_st, whole_infer_ed-whole_infer_st, counter

    def evaluate(self, args, threshold=0.5, topn_n_acc=10):
        dirs_pass = os.environ.get('ARTIFICIAL_DIRS_NAME_TXT', '/media/amirreza/6E2AF8F52AF8BB63/khoursha_files/dirs.txt')
        f = open(dirs_pass, 'r')
        lines = f.readlines()
        f.close()
        dirs_index = {}
        for line in lines:
            dr, ind = line[:-1].split(' ')
            dirs_index[dr] = ind
        base_path = args.base_path
        csv_paths = args.csv_paths
        excel_path = args.excel_path
        select_shot = args.select_shot
        patch_shape = args.size_sahi
        multi_scale = args.select_run_with_scale
        sum_time = 0.0
        flag_all_sampl = 1
        iou_predictions = {}
        unique_category = {}
        unique_dataset = {}
        unique_size = {}
        count_all_samples = 0

        count_dataset = {}
        count_category = {}
        count_size = {}
        analyze_error_dict = {
            'good_trg_area' : [],
            'bad_trg_area' : [],
        }
        times_list = []
        lens_list = []
        predict_loop_time = []
        nms_times = []
        sort_times = []
        whole_times = []
        counter_avg = []
        for csv_path in csv_paths:
            csv_file = pd.read_csv(csv_path)
            # csv_file = csv_file[:1]
            for i in tqdm(range(len(csv_file))):
                path = csv_file.iloc[i][0]
                tar = cv2.imread(os.path.join(base_path, path))
                # print(base_path + path)
                # print(tar.shape)
                temps = []
                boxes_=[]
                temps_box = []
                tmp_ind_list = []
                for shot in range(0, select_shot):                        
                    path = csv_file.iloc[i][(5 * (shot + 1)) + 0]
                    x1 = int(csv_file.iloc[i][(5 * (shot + 1)) + 1])
                    x2 = int(csv_file.iloc[i][(5 * (shot + 1)) + 1]) + int(csv_file.iloc[i][(5 * (shot + 1)) + 3])
                    y1 = int(csv_file.iloc[i][(5 * (shot + 1)) + 2])
                    y2 = int(csv_file.iloc[i][(5 * (shot + 1)) + 2]) + int(csv_file.iloc[i][(5 * (shot + 1)) + 4])
                    temp_full_path = os.path.join(base_path, path)
                    temp = cv2.imread(temp_full_path)
                    drname = path.split('/')[-2]
                    ind_dr = dirs_index[drname]
                    name_tmp = int(path.split('/')[-1].split('.')[0].split('temp')[1]) - 1
                    tmp_ind_list.append(str(name_tmp))
                    if temp is not None:
                        Temp = temp.copy()
                        
                        # h, w = temp.shape[:2]
                        # if h * w > 10000:
                            # scale = math.sqrt((w * h) / 10000)
                            # temp = cv2.resize(temp, (int(w / scale), int(h / scale)))
                            # # print(h, w, (int(w / scale), int(h / scale)))

                        if 1:
                            temps.append(Temp)
                            boxes_.append([x1, y1 ,x2, y2])
                        else: # low margin
                            temp = temp[np.maximum(0, y1-10):y2+10, np.maximum(0, x1-10):x2+10]
                            temps.append(temp)
                            boxes_.append([10, 10, temp.shape[1]-10, temp.shape[0]-10])
                        
                        # cv2.imshow('temp', temp)
                        # cv2.waitKey()

                    if shot not in [0, 4]:
                        continue

                    if multi_scale:
                        predictions, time_list, temp_len, loop_time, \
                            nms_time, sort_time, whole_time, cntr = self.predict_multiscale(tar, temps, boxes_, patch_shape, args.version, ind_dr, tmp_ind_list) # 512
                        times_list.append(time_list)
                        lens_list.append(temp_len)
                        predict_loop_time.append(loop_time)
                        nms_times.append(nms_time)
                        sort_times.append(sort_time)
                        whole_times.append(whole_time)
                        counter_avg.append(cntr)
                    else:
                        predictions = self.predict(tar, temps, boxes_)
                    scale = "large"
                    size = int(csv_file.iloc[i][4]) * int(csv_file.iloc[i][3])
                    if size <= 1024:
                        scale = "small"
                    elif size > 1024 and size < 4096:
                        scale = "medium"

                    if csv_file.iloc[i]["category"] not in unique_category.keys():
                        unique_category[csv_file.iloc[i]["category"]] = len(unique_category.keys())
                        count_category[csv_file.iloc[i]["category"]] = 1
                    else:
                        if shot == 0:
                            count_category[csv_file.iloc[i]["category"]] += 1
                    if csv_file.iloc[i]["dataset"] not in unique_dataset.keys():
                        unique_dataset[csv_file.iloc[i]["dataset"]] = len(unique_dataset.keys())
                        count_dataset[csv_file.iloc[i]["dataset"]] = 1
                    else:
                        if shot == 0:
                            count_dataset[csv_file.iloc[i]["dataset"]] += 1
                    if scale not in unique_size.keys():
                        unique_size[scale] = len(unique_size.keys())
                        count_size[scale] = 1
                    else:
                        if shot == 0:
                            count_size[scale] += 1
                            

                    # print(shot, count_all_samples, len(predictions))
                    tar_x1, tar_y1, tar_x2, tar_y2 = csv_file.iloc[i][1], csv_file.iloc[i][2], csv_file.iloc[i][1] + csv_file.iloc[i][3], csv_file.iloc[i][2] + csv_file.iloc[i][4]
                    
                    internal_error_analyze_dict = {
                        'max_score_id' : -1,
                        'max_score_value' : 0,
                        'max_score_iou' : 0,
                    }

                    for p, pred in enumerate(predictions):
                        iou = self.IoU([pred[0], pred[1], pred[2], pred[3]], [tar_x1, tar_y1, tar_x2, tar_y2])
                        
                        if pred[4] > internal_error_analyze_dict['max_score_value']:
                            internal_error_analyze_dict['max_score_value'] = pred[4]
                            internal_error_analyze_dict['max_score_id'] = p
                            internal_error_analyze_dict['max_score_iou'] = iou

                        # print("\n iou : ", iou)
                        if shot not in iou_predictions.keys():
                            iou_predictions[shot] = {}
                            iou_predictions[shot][count_all_samples] = {}
                            iou_predictions[shot][count_all_samples][p] = [iou, csv_file.iloc[i]["category"], scale, csv_file.iloc[i]["dataset"]]
                        else:
                            if count_all_samples in iou_predictions[shot].keys():
                                iou_predictions[shot][count_all_samples][p] = [iou, csv_file.iloc[i]["category"], scale, csv_file.iloc[i]["dataset"]]
                            else:
                                iou_predictions[shot][count_all_samples] = {}
                                iou_predictions[shot][count_all_samples][p] = [iou, csv_file.iloc[i]["category"], scale, csv_file.iloc[i]["dataset"]]
                        if p == len(predictions) - 1 and p < 9:
                            for c in range(p + 1, 10):
                                if shot not in iou_predictions.keys():
                                    iou_predictions[shot] = {}
                                    iou_predictions[shot][count_all_samples] = {}
                                    iou_predictions[shot][count_all_samples][c] = [
                                        [iou, csv_file.iloc[i]["category"], scale, csv_file.iloc[i]["dataset"]]]
                                else:
                                    if count_all_samples in iou_predictions[shot].keys():
                                        iou_predictions[shot][count_all_samples][c] = [iou, csv_file.iloc[i]["category"], scale, csv_file.iloc[i]["dataset"]]
                                    else:
                                        iou_predictions[shot][count_all_samples] = {}
                                        iou_predictions[shot][count_all_samples][c] = [iou, csv_file.iloc[i]["category"], scale, csv_file.iloc[i]["dataset"]]
                        if flag_all_sampl == 0:
                            flag_all_sampl += 1
                        else:
                            flag_all_sampl += 1
                            sum_time += float(pred[5])

                    if args.analyze_error_path != "":
                        area = abs(tar_x2 - tar_x1) * abs(tar_y2 - tar_y1)
                        if internal_error_analyze_dict['max_score_iou'] >= threshold:
                            analyze_error_dict["good_trg_area"].append(area)
                        else:
                            analyze_error_dict["bad_trg_area"].append(area)
                        log_image = tar.copy()
                        cv2.rectangle(log_image, (tar_x1, tar_y1), (tar_x2, tar_y2), (0, 255, 0), 2)
                        best_pred = predictions[internal_error_analyze_dict["max_score_id"]]
                        cv2.rectangle(log_image, (best_pred[0], best_pred[1]), (best_pred[2], best_pred[3]), (255, 0, 0), 2)
                        log_image_path = os.path.join(args.analyze_error_path, f'{Path(temp_full_path).parent.name}_{shot}shot.jpg')
                        cv2.imwrite(log_image_path, log_image)

                count_all_samples += 1
            # with open('/mnt/400G/hajizadeh/FewShotMatching/result.pkl', 'wb') as f:
            #     pickle.dump(iou_predictions, f)
        if args.analyze_error_path != "":
            if len(analyze_error_dict['good_trg_area']) > 0:
                plt.hist(analyze_error_dict['good_trg_area'], bins=1000)
                log_image_path = os.path.join(args.analyze_error_path, f'trg_bbox_area_hist_good_pred.jpg')
                plt.savefig(log_image_path)

            plt.figure()
            if len(analyze_error_dict['bad_trg_area']) > 0:
                plt.hist(analyze_error_dict['bad_trg_area'], bins=1000)
                log_image_path = os.path.join(args.analyze_error_path, f'trg_bbox_area_hist_bad_pred.jpg')
                plt.savefig(log_image_path)
                
        all_result = np.zeros((10, 10))
        batched_tem = 0
        single_tem = 0
        batched_tem_count = 0
        single_tem_count = 0
        whole_tm = 0
        whole_tm_count = 0
        for index, tmlist in enumerate(times_list):
            for tm_ls in tmlist:
                whole_tm_count += 1
                whole_tm += tm_ls

                if lens_list[index] == 1:
                    single_tem_count += 1
                    single_tem += tm_ls
                else:
                    batched_tem_count += 1
                    batched_tem += tm_ls
        
        print('Whole time:', whole_tm / whole_tm_count)
        print('Single time:', single_tem / single_tem_count)
        print('Batched time:', batched_tem / batched_tem_count)

        print('The avegrage time of prediction loop (in sec.): ', np.array(predict_loop_time).mean())
        print('The avegrage time of NMS in predict multi-scale (in sec.): ', np.array(nms_times).mean())
        print('The avegrage time of sort in predict multi-scale (in sec.): ', np.array(sort_times).mean())
        print('The avegrage time of Whole predict multi-scale (in sec.): ', np.array(whole_times).mean())
        print('The avegrage number of match calls: ', np.array(counter_avg).mean())

        result_dataset = np.zeros((len(unique_dataset.keys()), 10, 10))
        result_category = np.zeros((len(unique_category.keys()), 10, 10))
        result_size = np.zeros((len(unique_size.keys()), 10, 10))
        for shot in iou_predictions.keys():
            for sample in range(count_all_samples):
                flag_t = False
                # import pdb
                # pdb.set_trace()
                # print(shot, sample)
                for i, preds_in_nth_top in enumerate(iou_predictions[shot][sample]):
                    index_dataset = unique_dataset[iou_predictions[shot][sample][i][3]]
                    index_category = unique_category[iou_predictions[shot][sample][i][1]]
                    index_size = unique_size[iou_predictions[shot][sample][i][2]]
                    if iou_predictions[shot][sample][i][0] >= threshold:
                        all_result[shot][i:topn_n_acc] += 1
                        result_dataset[index_dataset][shot][i:topn_n_acc] += 1
                        result_category[index_category][shot][i:topn_n_acc] += 1
                        result_size[index_size][shot][i:topn_n_acc] += 1
                        flag_t = True
                        break
        flag_sheets = 0
        # print("all avg time in algorithms: ", sum_time / (flag_all_sampl - 1))

        datasetNames = {}
        categoryNames = {}
        sizeNames = {}
        for i, dataset in enumerate(unique_dataset.keys()):
            datasetNames[i] = dataset
        for i, category in enumerate(unique_category.keys()):
            categoryNames[i] = category
        for i, size in enumerate(unique_size.keys()):
            sizeNames[i] = size
        # excel_path = base_path + excel_path
        print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
        with ExcelWriter(excel_path) as writer:
            df_result = pd.DataFrame(all_result / count_all_samples)
            print((all_result / count_all_samples)[:select_shot])
            df_result.to_excel(writer, 'ALL_RESULT')
            flag_sheets += 1
            for i in range(len(result_dataset)):
                df_result = pd.DataFrame(result_dataset[i] / count_dataset[datasetNames[i]])
                df_result.to_excel(writer, 'dataset_%s' % datasetNames[i])
                flag_sheets += 1
            for i2 in range(len(result_category)):
                df_result = pd.DataFrame(result_category[i2] / count_category[categoryNames[i2]])
                df_result.to_excel(writer, "category_%s" % categoryNames[i2])
                flag_sheets += 1
            for i3 in range(len(result_size)):
                df_result = pd.DataFrame(result_size[i3] / count_size[sizeNames[i3]])
                df_result.to_excel(writer, "size_%s" % sizeNames[i3])
                flag_sheets += 1
            print("complete")

    def checkMethod(self, checkList, args):
        print("checkList : ", checkList)

        def image_with_bb(img, x, y, w, h, x1_p, y1_p, x2_p, y2_p, topN, color=(0, 255, 0), thickness=2):
            if type(img) is str:
                img = cv2.imread(img)
            start = (int(x), int(y))
            end = (int(x) + int(w), int(y) + int(h))

            img = cv2.rectangle(img, start, end, color, thickness)  # gt
            if topN == "True":
                color = (255, 0, 0)
            else:
                color = (0, 0, 255)

            img = cv2.rectangle(img, (int(x1_p), int(y1_p)), (int(x2_p), int(y2_p)), color, thickness)
            # img = cv2.putText(img, topN, (x1_p + 5, y1_p - 15), cv2.FONT_HERSHEY_TRIPLEX, 1, (36, 255, 12), 1)
            return img

        base_path = args.base_path
        df_all = pd.DataFrame()
        # append all files together
        for file in args.csv_paths:
            df_temp = pd.read_csv(file)
            df_all = df_all.append(df_temp, ignore_index=True)
        csv_file = df_all.copy()
        # csv_file = pd.read_csv(args.csv_path)
        for i in range(2):
            if i == 0:
                topN = 'True'
            elif i == 1:
                topN = 'False'
            for j in range(10):
                t, x1_p, y1_p, x2_p, y2_p = checkList[i][j]
                print(t, not (t == 0 and x1_p == 0 and y1_p == 0 and x2_p == 0 and x2_p == 0))
                if not (t == 0 and x1_p == 0 and y1_p == 0 and x2_p == 0 and x2_p == 0):
                    path = csv_file.iloc[t][0]
                    # print(len(csv_file))
                    print("Path target : ", path)
                    tar = cv2.imread(base_path + path)

                    x, y, w, h = csv_file.iloc[t][1], csv_file.iloc[t][2], csv_file.iloc[t][3], csv_file.iloc[t][4]
                    img = image_with_bb(tar, x, y, w, h, x1_p, y1_p, x2_p, y2_p, str(topN))
                    cv2.imwrite("F:\\fewshot\\" + args.method + '_' + str(i) + '_' + str(j) + '.jpg', img)

                    fig = plt.figure()
                    for tempIndex in range(5):
                        x = csv_file.iloc[t][5 * (tempIndex + 1) + 1]
                        y = csv_file.iloc[t][5 * (tempIndex + 1) + 2]
                        w = csv_file.iloc[t][5 * (tempIndex + 1) + 3]
                        h = csv_file.iloc[t][5 * (tempIndex + 1) + 4]
                        x, y, w, h = int(x), int(y), int(w), int(h)
                        imgPath = csv_file.iloc[t][5 * (tempIndex + 1)]
                        plt.subplot(1, 5, tempIndex + 1)
                        path = base_path + imgPath
                        if not os.path.exists(path):
                            print("Exception: path not exist!")
                            return
                        # rgb
                        img = np.array(cv2.cvtColor(cv2.imread(path)[y:y + h, x:x + w], cv2.COLOR_BGR2RGB))
                        plt.imshow(img)
                        plt.tick_params(axis='both', labelsize=0)
                    # plt.gcf().set_size_inches(11, 5)
                    plt.savefig(args.method + '_' + str(i) + '_' + str(j) + '_temps' + '.jpg')

    
