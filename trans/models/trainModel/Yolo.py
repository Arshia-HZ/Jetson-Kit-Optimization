"""
use set for windows
export AUGMENT_RES_DIR=/mnt/File/shamgholi/tm/yolo_stuff/temporary_dir 
export BG_IMAGES_DIR=/mnt/File/shamgholi/tm/data/VisDroneSampled/
export BASE_CKPT_PATH=/mnt/File/shamgholi/tm/yolo_stuff/yolov8m-oiv7.pt
export ARTIFCIAL_CSV_PATH=/mnt/File/shamgholi/tm/data/artificial_test_out.csv
export YOLO_MODEL_PATH=E:\CS\Work\FewshotTemplate\yolo\ckpt_archive\yolov8m-oiv7-artificial.pt
export MY_BASE_PATH=E:\CS\Work\FewshotTemplate\data
python main.py --base_path /mnt/File/shamgholi/tm/data 
"""


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
from baseFewShotMatcher import BaseFewShotMatcher
from pathlib import Path
import pdb
import cv2

yolo_project_dir = os.environ.get('YOLO_PROJECT_DIR', None) #"/mnt/File/shamgholi/tm/yolo_stuff/temporary_dir"
bg_images_dir = os.environ.get('BG_IMAGES_DIR', None)
# background_images = glob(os.path.join("/mnt/File/shamgholi/tm/data/background/", '*.jpg')) 
assert yolo_project_dir is not None
assert bg_images_dir is not None
background_images = glob(os.path.join(bg_images_dir, '*.jpg'))


def add_template_to_image(tm_image, main_image):
    image = main_image.copy()
    mask = Image.new('L', tm_image.size, 0)
    draw = ImageDraw.Draw(mask)
    z = 40
    # ValueError: x1 must be greater than or equal to x0
    try:
        draw.rectangle((z, z, mask.width - z, mask.height - z), fill=255)
    except ValueError:
        z = int(z / 2)
        draw.rectangle((z, z, mask.width - z, mask.height - z), fill=255)
        
    for i in range(z):
        alpha = int(255 * (i / z))
        draw.rectangle((i, i, mask.width - i, mask.height - i), fill=alpha)
    # Apply the mask to the template
    tm_image.putalpha(mask)
    # Paste the template onto the image
    w_max_range = image.size[0] - tm_image.size[0]
    h_max_range = image.size[1] - tm_image.size[1]
    new_x1, new_y1 = random.randint(0, w_max_range), random.randint(0, h_max_range)
    image.paste(tm_image, (new_x1, new_y1), tm_image)
    # Save the result
    return image, new_x1, new_y1
    # image.save('E:\\CS\\Work\\FewshotTemplate\\result.jpg')


def convert_to_yolo(x1, y1, x2, y2, img_width, img_height):
    x_center = (x1 + x2) / (2 * img_width)
    y_center = (y1 + y2) / (2 * img_height)
    width = (x2 - x1) / img_width
    height = (y2 - y1) / img_height
    return x_center, y_center, width, height



def freeze_layer(trainer):
    model = trainer.model
    reach_wanted_layer = False
    for k, v in model.named_parameters(): 
        if '.18' in k:
            reach_wanted_layer = True
        if not reach_wanted_layer:        
            v.requires_grad = False 
        else:
            v.requires_grad = True
        print(k, v.requires_grad)


class Yolo(BaseFewShotMatcher):
    def __init__(self, arguments):
        super().__init__()
        # self.ckpt_path = (os.getenv('YOLO_MODEL_PATH', None) or
                    #   "/mnt/File/shamgholi/tm/yolo_stuff/yolov8m-oiv7-artificial.pt")
        # self.model = YOLO(self.ckpt_path)
        self.base_ckpt_path = os.environ.get("BASE_CKPT_PATH", None) #"/mnt/File/shamgholi/tm/yolo_stuff/yolov8m-oiv7.pt"
        assert self.base_ckpt_path is not None
        self.scales = [float(scale) for scale in arguments.scales.split(' ')]
        self.offline_mode = arguments.offline_mode
        #self.scales = [float(scale) for scale in [1]]
        #self.scales = [float(scale) for scale in [0.6, 1, 1.5]]

    def match(self, target):
        assert hasattr(self, 'model'), 'run set_templates first'
        t0 = time.time()
        results = self.model.predict(target, conf=0.0001, verbose=False)
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

    
    def clear_last_run_data(self):
        imgs = glob(os.path.join(yolo_project_dir, 'images', '**', '*.jpg'), recursive=True)
        txts = glob(os.path.join(yolo_project_dir, 'labels', '**', '*.txt'), recursive=True)


        for im in imgs:
            os.remove(im)
        for txt in txts:
            os.remove(txt)


    def set_templates(self, templates, temp_boxes, ind_dr=0, tmp_ind_list=[]):
        if self.offline_mode:
            base_train_dir = os.environ.get('AUG_BASE_TRAIN_DIR','/media/amirreza/6E2AF8F52AF8BB63/khoursha_files/train/')
            base_val_dir = os.environ.get('AUG_BASE_VAL_DIR', '/media/amirreza/6E2AF8F52AF8BB63/khoursha_files/val/')
            best_weights_dir = os.environ.get("YOLO_BEST_WEIGHTS_DIR", None)
            assert best_weights_dir is not None
            print('YOLO_BEST_WEIGHTS_DIR is', best_weights_dir)
            print("base_train_dir is", base_train_dir)
            print("base_val_dir is", base_val_dir)
            nshot = len(templates)
            curr_best_weights_path = os.path.join(best_weights_dir, f"catidx{ind_dr}_nshot{nshot}.pt")
            if os.path.exists(curr_best_weights_path):
                print('best weights file exists, loading ...')
                self.model = YOLO(curr_best_weights_path)
                if torch.cuda.is_available():                    
                    self.model = self.model.cuda()
                    warmup_input = np.random.randn(700, 1200, 3).astype('uint8')
                    print('Warming up the model ...')
                    _ = self.model.predict(warmup_input, conf=0.0001, verbose=False)
                    _ = self.model.predict(warmup_input, conf=0.0001, verbose=False)
                return
            else:
                print('best weights file not exists, training ...')

            '''
            trainIMGS = glob(os.path.join(base_train_dir, '{}_{}_*.jpg'.format(ind_dr, tmp_ind_list[0])))
            trainIMGS = glob(os.path.join(base_train_dir, '{}_0{}_*.jpg'.format(ind_dr, tmp_ind_list[0])))
            trainIMGS = glob(os.path.join(base_train_dir, '{}_1{}_*.jpg'.format(ind_dr, tmp_ind_list[0])))
            trainTXTS = glob(os.path.join(base_train_dir, '{}_*.txt'.format(ind_dr)))
            valIMGS = glob(os.path.join(base_val_dir, '{}_*.jpg'.format(ind_dr)))
            valTXTS = glob(os.path.join(base_val_dir, '{}_*.txt'.format(ind_dr)))
            '''
        
            self.clear_last_run_data()
            os.makedirs(os.path.join(yolo_project_dir, 'images', 'train'), exist_ok=True)
            os.makedirs(os.path.join(yolo_project_dir, 'images', 'val'), exist_ok=True)
            os.makedirs(os.path.join(yolo_project_dir, 'labels', 'train'), exist_ok=True)
            os.makedirs(os.path.join(yolo_project_dir, 'labels', 'val'), exist_ok=True)

            train_images = []
            val_images = []
            for tmp_ind in tmp_ind_list:
                # pdb.set_trace()
                for itr in ['', 0, 1]:
                    if len(glob(os.path.join(base_train_dir, '{}_{}{}_*.jpg'.format(ind_dr, itr, tmp_ind)))) == 0:
                        tmp_ind = 7
                    train_images += glob(os.path.join(base_train_dir, '{}_{}{}_*.jpg'.format(ind_dr, itr, tmp_ind)))
                    #train_images += glob(os.path.join(base_train_dir, '{}_0{}_*.jpg'.format(ind_dr, tmp_ind)))
                    #train_images += glob(os.path.join(base_train_dir, '{}_1{}_*.jpg'.format(ind_dr, tmp_ind)))
                    val_images += glob(os.path.join(base_val_dir, '{}_{}{}_*.jpg'.format(ind_dr, itr, tmp_ind)))
                    #val_images += glob(os.path.join(base_val_dir, '{}_0{}_*.jpg'.format(ind_dr, tmp_ind)))
                    #val_images += glob(os.path.join(base_val_dir, '{}_1{}_*.jpg'.format(ind_dr, tmp_ind)))

            print('%'*20)
            print('>>> len train_images', len(train_images))
            print('>>> len val_images', len(val_images))
            print('%'*20)

            def make_data(image_list, mode):
                for path in image_list:
                    img_name = Path(path).name
                    img_dest = os.path.join(yolo_project_dir, 'images', mode, img_name)
                    txt_dest = os.path.join(yolo_project_dir, 'labels', mode, img_name.replace('.jpg', '.txt'))
                    os.symlink(path, img_dest)
                    os.symlink(path.replace('.jpg', '.txt'), txt_dest)
        
            make_data(train_images, 'train')
            make_data(val_images, 'val')

            train_conf_content = f"path: {yolo_project_dir}\ntrain: images/train\nval: images/val\n\nnc: 1\n"
            train_conf_path = os.path.join(yolo_project_dir, 'config.yaml')
            with open(train_conf_path, 'w') as f:
                f.write(train_conf_content)
            self.model = YOLO(self.base_ckpt_path)
            if torch.cuda.is_available():
                self.model = self.model.cuda()
            self.model.add_callback("on_train_start", freeze_layer)
            #self.model.train(data=train_conf_path, epochs=15, batch=16, project=augmented_dir, close_mosaic=2, exist_ok=True)
            self.model.train(data=train_conf_path, epochs=100, batch=16, project=yolo_project_dir,
                              close_mosaic=2, exist_ok=True, patience=15)
            self.model = YOLO(os.path.join(yolo_project_dir, 'train', 'weights', 'best.pt'))
            #self.model.eval()
            if torch.cuda.is_available():
                self.model = self.model.cuda()

            Path(curr_best_weights_path).parent.mkdir(exist_ok=True)
            shutil.copyfile(os.path.join(yolo_project_dir, 'train', 'weights', 'best.pt'), curr_best_weights_path)
        else:
            random.seed(100)
            bg_path = '/home/user/Desktop/trans/VisDroneSampled/'
            bgs = glob(os.path.join(bg_path, '*.jpg'))
            random.shuffle(bgs)
            train_bgs, val_bgs = bgs[:int(0.8*len(bgs))], bgs[int(0.8*len(bgs)):]

            augmented_dir = '/home/user/Desktop/trans/yolo_stuff/temporary_dir'
            if os.path.isdir(os.path.join(augmented_dir, 'images/train')):
                os.system('rm -rf {}'.format(os.path.join(augmented_dir, 'images/train')))
            os.system('mkdir -p {}'.format(os.path.join(augmented_dir, 'images/train')))

            if os.path.isdir(os.path.join(augmented_dir, 'images/val')):
                os.system('rm -rf {}'.format(os.path.join(augmented_dir, 'images/val')))
            os.system('mkdir -p {}'.format(os.path.join(augmented_dir, 'images/val')))

            if os.path.isdir(os.path.join(augmented_dir, 'labels/val')):
                os.system('rm -rf {}'.format(os.path.join(augmented_dir, 'labels/val')))
            os.system('mkdir -p {}'.format(os.path.join(augmented_dir, 'labels/val')))

            if os.path.isdir(os.path.join(augmented_dir, 'labels/train')):
                os.system('rm -rf {}'.format(os.path.join(augmented_dir, 'labels/train')))
            os.system('mkdir -p {}'.format(os.path.join(augmented_dir, 'labels/train')))

            def make_data(tmpl, bg_list, mode: str):

                def calc_hist(image):
                    hist = np.zeros(256,dtype=int)
                    for i in range(256):
                        hist[i] = np.sum(image == i)
                    return hist

                def hist_matching(src_image, ref_image):
                    output_image = src_image.copy()
                    def channel_wise_matching(src_channel, ref_channel):
                        output_channel = src_channel.copy()
                        src_hist = calc_hist(src_channel)
                        ref_hist = calc_hist(ref_channel)
                        def most_similar(rcdf, elem):
                            dis = np.abs(rcdf - elem)
                            return np.argmin(dis)
                        def cdf(hist):
                            cdf_hist = np.zeros(256,dtype=int)
                            for i in range(256):
                                cdf_hist[i] = np.sum(hist[:i+1])
                            return cdf_hist
                        src_cdf = cdf(src_hist) / (src_channel.shape[0] * src_channel.shape[1])
                        ref_cdf = cdf(ref_hist) / (ref_channel.shape[0] * ref_channel.shape[1])
                        hist_matching_dict = {}
                        for index in range(256):
                            hist_matching_dict[index] = most_similar(ref_cdf, src_cdf[index])
                        for row in range(src_channel.shape[0]):
                            for col in range(src_channel.shape[1]):
                                output_channel[row, col] = hist_matching_dict[src_channel[row, col]]
                        return output_channel
                    for i in range(src_image.shape[2]):
                        output_image[:, :, i] = channel_wise_matching(src_image[:, :, i], ref_image[:, :, i])
                    return output_image

                def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
                    dim = None
                    (h, w) = image.shape[:2]
                    if width is None and height is None:
                        return image
                    if width is None:
                        r = height / float(h)
                        dim = (int(w * r), height)
                    else:
                        r = width / float(w)
                        dim = (width, int(h * r))
                    resized = cv2.resize(image, dim, interpolation = inter)
                    return resized

                def is_valid_position(position, image, placed_images):
                    x, y = position
                    h, w = image.shape[:2]
                    for placed_image in placed_images:
                        px, py = placed_image['position']
                        ph, pw = placed_image['image'].shape[:2]
                        if (x < px + pw and x + w > px and y < py + ph and y + h > py):
                            return False  # Overlapping
                    return True

                def place_image(position, image, reference_imagein, top_):
                    f = open(top_, 'a')
                    f.write('0 {} {} {} {}\n'.format(position[0] / reference_imagein.shape[1], position[1] / reference_imagein.shape[0], \
                            image.shape[1] / reference_imagein.shape[1], image.shape[0] / reference_imagein.shape[0]))
                    f.close()
                    src_mask = np.ones_like(image) * 255
                    result = cv2.seamlessClone(image, reference_imagein, src_mask, position, cv2.NORMAL_CLONE)
                    return result

                def prob_rot(want_rot, rot_small_photo):
                    if want_rot == 1:
                        rot_wrap = random.randint(0, 10)
                        rotation_angle = random.randint(0, 10)
                        M = cv2.getRotationMatrix2D((rot_small_photo.shape[1] // 2, rot_small_photo.shape[0] // 2), rotation_angle, 1)
                        rot_small_photo = cv2.warpAffine(rot_small_photo, M, (rot_small_photo.shape[1], rot_small_photo.shape[0]))
                    return rot_small_photo

                def prob_resize(want_resize, rs_small_photo, bg_shape):
                    if want_resize == 1:
                        if rs_small_photo.shape[0] < bg_shape[0] and rs_small_photo.shape[1] < bg_shape[1]:
                            if random.randint(0, 10) % 2 == 0:
                                hei = random.randint(rs_small_photo.shape[0] // 3, int(1 * rs_small_photo.shape[0]))
                                if random.randint(0, 10) % 2 == 0:
                                    hei = None
                                wid = None
                            else:
                                wid = random.randint(rs_small_photo.shape[1] // 3, int(1 * rs_small_photo.shape[1]))
                                if random.randint(0, 10) % 2 == 0:
                                    wid = None
                                hei = None
                        else:
                            wid = random.randint(bg_shape[1] // 3, int(1 * bg_shape[1]))
                            hei = random.randint(bg_shape[0] // 3, int(1 * bg_shape[0]))
                        rs_small_photo = image_resize(rs_small_photo, width = wid, height = hei)
                    return rs_small_photo

                def prob_flip(want_flip, fp_small_photo):
                    if want_flip == 1:
                        if random.randint(0, 10) % 2 == 0:
                            fp_small_photo = cv2.flip(fp_small_photo, 1)
                    return fp_small_photo

                want_resize = 1
                want_rot = 0
                want_flip = 0
                want_more_obj = 0
                want_more_variance = 1
                hist_match_want = 0
                want_histmatch_in_mor_var = 1

                dest_img = os.path.join(augmented_dir, 'images/{}'.format(mode))
                dest_lbl = os.path.join(augmented_dir, 'labels/{}'.format(mode))

                for bg in bg_list:
                    tpname, bgname = int(round(time.time() * 1000)), bg.split('/')[-1].split('.')[0]
                    out_path = os.path.join(dest_img, '{}_{}.jpg'.format(tpname, bgname))
                    txt_out_path = os.path.join(dest_lbl, '{}_{}.txt'.format(tpname, bgname))
                    orig_small_photo = tmpl
                    large_photo = cv2.imread(bg)
                    potential_positions = []

                    placed_images = []

                    image1 = orig_small_photo.copy()
                    image1 = prob_rot(want_rot, image1)
                    image1 = prob_resize(want_resize, image1, large_photo.shape)
                    image1 = prob_flip(want_flip, image1)
                    # try:
                    for _ in range(10000):
                        x = random.randint(image1.shape[1] // 2 + 1, (large_photo.shape[1] - image1.shape[1] // 2) - 1)
                        y = random.randint(image1.shape[0] // 2 + 1, (large_photo.shape[0] - image1.shape[0] // 2) - 1)
                        potential_positions.append((x, y))

                    position = potential_positions.pop(random.randint(0, len(potential_positions) - 1))
                    placed_images.append({'image': image1, 'position': position})

                    if hist_match_want == 1:
                        image1 = hist_matching(image1, large_photo)

                    reference_image = place_image(position, image1, large_photo, txt_out_path)

                    if want_more_variance == 1:
                        if want_histmatch_in_mor_var == 1:
                            orig_matched = hist_matching(orig_small_photo, large_photo)
                        else:
                            orig_matched = orig_small_photo.copy()

                        for index in range(0, 2):
                            tempo = orig_matched.copy()
                            tempo = prob_rot(want_rot, tempo)
                            tempo = prob_resize(want_resize, tempo, large_photo.shape)
                            tempo = prob_flip(want_flip, tempo)
                            x = random.randint(tempo.shape[1] // 2 + 1, (large_photo.shape[1] - tempo.shape[1] // 2) - 1)
                            y = random.randint(tempo.shape[0] // 2 + 1, (large_photo.shape[0] - tempo.shape[0] // 2) - 1)
                            txtoutpath = txt_out_path.split('/')
                            txtoutpath[-1] = str(index)+txtoutpath[-1].split('.')[0]+'.txt'
                            txtoutpath = '/'.join(txtoutpath)
                            reference_image1 = place_image((x, y), tempo, large_photo, txtoutpath)
                            outpath = out_path.split('/')
                            outpath[-1] = str(index)+outpath[-1].split('.')[0]+'.jpg'
                            outpath = '/'.join(outpath)
                            cv2.imwrite(outpath, reference_image1)

                    if want_more_obj == 1:
                        image2 = orig_small_photo.copy()
                        image2 = prob_rot(want_rot, image2)
                        image2 = prob_resize(want_resize, image2, large_photo.shape)
                        image2 = prob_flip(want_flip, image2)

                        if len(potential_positions) > 0:
                            x = random.randint(image2.shape[1] // 2 + 1, (reference_image.shape[1] - image2.shape[1] // 2) - 1)
                            y = random.randint(image2.shape[0] // 2 + 1, (reference_image.shape[0] - image2.shape[0] // 2) - 1)
                            position = (x, y)
                            if is_valid_position(position, image2, placed_images):
                                placed_images.append({'image': image2, 'position': position})

                                reference_image = place_image(position, image2, reference_image, txt_out_path)

                        image3 = orig_small_photo.copy()
                        image3 = prob_rot(want_rot, image3)
                        image3 = prob_resize(want_resize, image3, large_photo.shape)
                        image3 = prob_flip(want_flip, image3)

                        if len(potential_positions) > 0:
                            x = random.randint(image3.shape[1] // 2 + 1, (reference_image.shape[1] - image3.shape[1] // 2) - 1)
                            y = random.randint(image3.shape[0] // 2 + 1, (reference_image.shape[0] - image3.shape[0] // 2) - 1)
                            position = (x, y)
                            if is_valid_position(position, image3, placed_images):
                                placed_images.append({'image': image3, 'position': position})

                                reference_image = place_image(position, image3, reference_image, txt_out_path)


                    cv2.imwrite(out_path, reference_image)

            for i, tmp_img in enumerate(templates):
                make_data(tmp_img[temp_boxes[i][1]:temp_boxes[i][3], temp_boxes[i][0]:temp_boxes[i][2], :], train_bgs, 'train')
                make_data(tmp_img[temp_boxes[i][1]:temp_boxes[i][3], temp_boxes[i][0]:temp_boxes[i][2], :], val_bgs, 'val')

            train_conf_content = f"path: {augmented_dir}\ntrain: images/train\nval: images/val\n\nnc: 1\n"
            train_conf_path = os.path.join(augmented_dir, 'config.yaml')
            with open(train_conf_path, 'w') as f:
                f.write(train_conf_content)
            self.model = YOLO(self.base_ckpt_path)
            if torch.cuda.is_available():
                self.model = self.model.cuda()
            self.model.add_callback("on_train_start", freeze_layer)
            self.model.train(data=train_conf_path, epochs=15, batch=16, project=augmented_dir, close_mosaic=2, exist_ok=True)
            self.model = YOLO(os.path.join(augmented_dir, 'train', 'weights', 'best.pt'))
            #self.model.eval()
            if torch.cuda.is_available():
                self.model = self.model.cuda()
