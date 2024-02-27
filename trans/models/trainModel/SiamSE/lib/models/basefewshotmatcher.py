import cv2
import time
import pickle
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use('pdf')
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import glob
import pandas as pd
import math

from pandas import ExcelWriter


class BaseFewShotMatcher():
    def IoU(self, box_pred, box_gt):
        '''
            if coordinates of box_pred are -1 return -1 and not processing
       '''
        if box_pred[0] != -1:
            x1, y1, x2, y2 = box_gt[0], box_gt[1], box_gt[2], box_gt[3]
            x_prim1, y_prim1, x_prim2, y_prim2 = box_pred[0], box_pred[1], box_pred[2], box_pred[3]
            intersect = max(min(int(y2), int(y_prim2)) - max(int(y1), int(y_prim1)), 0) * max(
                min(int(x2), int(x_prim2)) - max(int(x1), int(x_prim1)), 0)
            union = (max(int(y2), int(y_prim2)) - min(int(y1), int(y_prim1))) * (
                    max(int(x2), int(x_prim2)) - min(int(x1), int(x_prim1)))
            iou = intersect / union
        else:
            iou = -1
        print("iou: ",iou)
        return iou

    # here I should set that return number of prediction
    def predict(self, target, templates,bbx):
        predictions = []
        # return list of predictions that each prediction has bounding box, score, timeو format of bounding box for returning : x1, y1, x2, y2
        # our predictions list like this :
        '''
            [
            [40,50,60,70,0.9, 0.002], [40,50,60,70,0.9, 0.002], [40,50,60,70,0.9, 0.002], [40,50,60,70,0.9, 0.002], [40,50,60,70,0.9, 0.002], [40,50,60,70,0.9, 0.002], [40,50,60,70,0.9, 0.002], [40,50,60,70,0.9, 0.002], [40,50,60,70,0.9, 0.002] , [40,50,60,70,0.9, 0.002]
            ]
            list of predictions such that format of each prediction is [x1, y1, x2, y2, score, time], maximum number of predictin is 10 sublists
            if our model can not predict any box we should return -1 instead of each element of coordinates like the second template
        '''
        return predictions

    def nms(self, boxes, overlapThresh=0.5):
        # if there are no boxes, return an empty list
        if len(boxes) == 0:
            return []
        # initialize the list of picked indexes
        pick = []
        # grab the coordinates of the bounding boxes
        boxes = np.array(boxes)
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        # compute the area of the bounding boxes and sort the bounding
        # boxes by the bottom-right y-coordinate of the bounding box
        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        idxs = np.argsort(y2)
        # keep looping while some indexes still remain in the indexes
        # list
        while len(idxs) > 0:
            # grab the last index in the indexes list and add the
            # index value to the list of picked indexes
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)
            # find the largest (x, y) coordinates for the start of
            # the bounding box and the smallest (x, y) coordinates
            # for the end of the bounding box
            xx1 = np.maximum(x1[i], x1[idxs[:last]])
            yy1 = np.maximum(y1[i], y1[idxs[:last]])
            xx2 = np.minimum(x2[i], x2[idxs[:last]])
            yy2 = np.minimum(y2[i], y2[idxs[:last]])
            # compute the width and height of the bounding box
            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)
            # compute the ratio of overlap
            overlap = (w * h) / area[idxs[:last]]
            # delete all indexes from the index list that have
            idxs = np.delete(idxs, np.concatenate(([last],
                                                   np.where(overlap > overlapThresh)[0])))
        # return only the bounding boxes that were picked using the
        # integer data type
        return boxes[pick].astype("int")

    def index_2d(self, myList, v):
        for i, x in enumerate(myList):
            if v == x[0:4]:
                return i

    def visualize(self, tar, predictions, tarGT):
        boxs = []
        target = tar.copy()
        for pred in predictions:
            boxs.append(pred[0:4])

        nms_boxs = list(self.nms(boxs))

        # gt
        cv2.rectangle(target, (tarGT[0], tarGT[1]),
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
        plt.imsave("sample.png", target)
        # print(target.shape)
        # cv2.imshow("img",target)
        # plt.imshow(target)
        # plt.show()
        # return

    def evaluate(self, args, threshold=0.5, topn_n_acc=10):
        base_path = args.base_path
        csv_paths = args.csv_paths
        excel_path = args.excel_path
        select_shot = args.select_shot
        sum_time = 0.0
        flag_all_sampl = 0
        iou_predictions = {}
        unique_category = {}
        unique_dataset = {}
        unique_size = {}
        count_all_samples = 0

        count_dataset = {}
        count_category = {}
        count_size = {}

        for csv_path in csv_paths:
            csv_file = pd.read_csv(csv_path)
            for i in tqdm(range(len(csv_file))):
                path = csv_file.iloc[i][0]
                if args.method == "superGlue":
                    tar = cv2.imread(base_path + path, cv2.IMREAD_GRAYSCALE)
                else:
                    tar = cv2.imread(base_path + path)
                temps = []
                temps_box = []
                bbx=[]
                for shot in range(0, select_shot):
                    path = csv_file.iloc[i][(5 * (shot + 1)) + 0]
                    x1 = int(csv_file.iloc[i][(5 * (shot + 1)) + 1])
                    x2 = int(csv_file.iloc[i][(5 * (shot + 1)) + 1]) + int(csv_file.iloc[i][(5 * (shot + 1)) + 3])
                    y1 = int(csv_file.iloc[i][(5 * (shot + 1)) + 2])
                    y2 = int(csv_file.iloc[i][(5 * (shot + 1)) + 2]) + int(csv_file.iloc[i][(5 * (shot + 1)) + 4])
                    bbx.append([x1,y1,x2-x1,y2-y1])
                    if args.method == 'TMOpenCv' or args.method == 'superGlue':
                        temp = cv2.imread(base_path + path, cv2.IMREAD_GRAYSCALE)
                    else:
                        temp = cv2.imread(base_path + path)
                    # print(base_path + path)
                    #temp = temp[y1:y2, x1:x2]
                    Temp=temp.copy()
                    Temp=Temp[y1:y2, x1:x2]
                    h, w = Temp.shape[:2]
                    # print(h, w)
                    # if h * w > 10000:
                    #     scale = math.sqrt((w * h) / 10000)
                    #     temp = cv2.resize(temp, (int(w / scale), int(h / scale)))
                        # print(h, w, (int(w / scale), int(h / scale)))

                    temps.append(temp)
                    predictions = self.predict(tar, temps[:shot + 1],bbx[:shot + 1])
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

                    for p, pred in enumerate(predictions):
                        iou = self.IoU([pred[0], pred[1], pred[2], pred[3]], [csv_file.iloc[i][1],
                                                                              csv_file.iloc[i][2],
                                                                              csv_file.iloc[i][1] + csv_file.iloc[i][3],
                                                                              csv_file.iloc[i][2] + csv_file.iloc[i][
                                                                                  4]])

                        if shot not in iou_predictions.keys():
                            iou_predictions[shot] = {}
                            iou_predictions[shot][count_all_samples] = {}
                            iou_predictions[shot][count_all_samples][p] = [iou, csv_file.iloc[i]["category"], scale, csv_file.iloc[i]["dataset"]]
                        else:
                            if count_all_samples in iou_predictions[shot].keys():
                                iou_predictions[shot][count_all_samples][p] = [iou, csv_file.iloc[i]["category"], scale, csv_file.iloc[i]["dataset"]]
                                # if p in iou_predictions[shot][count_all_samples].keys():
                                #     iou_predictions[shot][count_all_samples][p].append(
                                #             [iou, csv_file.iloc[i]["category"], scale, csv_file.iloc[i]["dataset"]])
                                # else:
                                #     iou_predictions[shot][count_all_samples][p] = [iou, csv_file.iloc[i]["category"], scale, csv_file.iloc[i]["dataset"]]

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
                                        # if c in iou_predictions[shot][count_all_samples].keys():
                                        #     iou_predictions[shot][count_all_samples][c].append(
                                        #         [iou, csv_file.iloc[i]["category"], scale, csv_file.iloc[i]["dataset"]])
                                        # else:
                                        #     iou_predictions[shot][count_all_samples][c] = [
                                        #         [iou, csv_file.iloc[i]["category"], scale, csv_file.iloc[i]["dataset"]]]
                                    else:
                                        iou_predictions[shot][count_all_samples] = {}
                                        iou_predictions[shot][count_all_samples][c] = [iou, csv_file.iloc[i]["category"], scale, csv_file.iloc[i]["dataset"]]
                        if flag_all_sampl == 0:
                            flag_all_sampl += 1
                        else:
                            flag_all_sampl += 1
                            sum_time += float(pred[5])
                count_all_samples += 1
        all_result = np.zeros((10, 10))
        result_dataset = np.zeros((len(unique_dataset.keys()), 10, 10))
        result_category = np.zeros((len(unique_category.keys()), 10, 10))
        result_size = np.zeros((len(unique_size.keys()), 10, 10))
        # print("*******************************************************************************")
        # print(iou_predictions)
        # print("********************************************************************************")
        for shot in iou_predictions.keys():
            # print(f"#################### shot{shot} ########################")
            # print(iou_predictions[shot])
            for sample in range(count_all_samples):
                flag_t = False
                for i, preds_in_nth_top in enumerate(iou_predictions[shot][sample]):
                    # print(iou_predictions[shot][sample][i][3])
                    index_dataset = unique_dataset[iou_predictions[shot][sample][i][3]]
                    index_category = unique_category[iou_predictions[shot][sample][i][1]]
                    index_size = unique_size[iou_predictions[shot][sample][i][2]]
                    if iou_predictions[shot][sample][i][0] >= threshold:
                        # print("Shot: ", shot, " n_th top: ", nth_topn)
                        all_result[shot][i:topn_n_acc] += 1
                        result_dataset[index_dataset][shot][i:topn_n_acc] += 1
                        result_category[index_category][shot][i:topn_n_acc] += 1
                        result_size[index_size][shot][i:topn_n_acc] += 1
                        flag_t = True
                        break
                # if flag_t :
                #     break
                    # count_dataset[index_dataset] += 1
                    # count_category[index_category] += 1
                    # count_size[index_size] += 1

        ##################################################
        # transfer to sheet
        flag_sheets = 0
        # print("all avg time in algorithms: ", sum_time / (flag_all_sampl - 1))

        datasetNames = {}
        categoryNames = {}
        sizeNames = {}
        # print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

        # print(unique_dataset)
        # print(unique_category)
        # print(unique_size)
        # print(count_dataset)
        # print(count_category)
        # print(count_size)
        # print(flag_all_sampl)
        # print(count_all_samples)
        for i, dataset in enumerate(unique_dataset.keys()):
            datasetNames[i] = dataset
        for i, category in enumerate(unique_category.keys()):
            categoryNames[i] = category
        for i, size in enumerate(unique_size.keys()):
            sizeNames[i] = size
        # excel_path = base_path + excel_path
        # print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
        with ExcelWriter(excel_path) as writer:
            # print("come here1")
            # print(all_result)
            # print(all_result / count_all_samples)
            df_result = pd.DataFrame(all_result / count_all_samples)
            print(all_result / count_all_samples)
            df_result.to_excel(writer, 'ALL_RESULT')
            flag_sheets += 1
            # print("come here2")
            # print(result_dataset)
            for i in range(len(result_dataset)):
                # name_dataset = csv_paths[i].split("\\")[len(csv_paths[i]) - 1].split(".")[0]
                df_result = pd.DataFrame(result_dataset[i] / count_dataset[datasetNames[i]])
                df_result.to_excel(writer, 'dataset_%s' % datasetNames[i])
                flag_sheets += 1
            # print("come here3")
            # print(result_category)
            for i2 in range(len(result_category)):
                df_result = pd.DataFrame(result_category[i2] / count_category[categoryNames[i2]])
                # for k in all_category.keys():size
                #     if all_category[k] == i2:
                #         sheetname = "sheet" + k
                df_result.to_excel(writer, "category_%s" % categoryNames[i2])
                flag_sheets += 1
            # print("come here4")
            # print(result_size)
            for i3 in range(len(result_size)):
                df_result = pd.DataFrame(result_size[i3] / count_size[sizeNames[i3]])
                # for k in all_size.keys():
                #     if all_size[k] == i:
                #         sheetname = "sheet" + k
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
                    plt.savefig(args.method + '_' + str(i) + '_' + str(j) + '_t')