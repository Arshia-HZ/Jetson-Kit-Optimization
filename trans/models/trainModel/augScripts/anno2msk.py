import json
import os
import numpy as np
import cv2
import glob

dirs = os.listdir('/home/amirreza/Downloads/labeld/')

for dr in dirs:
    jsons = glob.glob('/home/amirreza/Downloads/labeld/{}/*.json'.format(dr))
    for js in jsons:
        with open(js, "r") as read_file:
            data = json.load(read_file)

        pts = data['shapes'][0]['points']
        pts = [[int(pt[0]), int(pt[1])] for pt in pts]
        pts = np.array(pts).reshape((-1,1,2)).astype(np.int32)
        filename = os.path.basename(js).split('.')[0]
        imgpath = '/home/amirreza/shamgholi/tm/data/artificial/{}/{}.jpg'.format(dr, filename)
        real_img = cv2.imread(imgpath)
        img = np.zeros_like(real_img)[:, :, 0].astype(np.uint8)
        img = cv2.drawContours(img, [pts], -1, (255,255,255), -1)

        rect_json = '/home/amirreza/shamgholi/tm/data/artificial/{}/{}.json'.format(dr, filename)
        with open(rect_json, "r") as read_file:
            data = json.load(read_file)

        pts = data['shapes'][0]['points']
        pts = [[int(pt[0]), int(pt[1])] for pt in pts]
        img = img[pts[0][1]:pts[1][1], pts[0][0]:pts[1][0]]

        cv2.imwrite('/home/amirreza/shamgholi/tm/data/artificial/{}/{}.png'.format(dr, int(filename.split('p')[-1]) - 1), img)

