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
import functools
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.data import build_dataloader, build_yolo_dataset

TMDIR = os.environ.get('TM_PROJECT_DIR', "/home/amirreza/shamgholi/tm")
augmented_dir = os.path.join(TMDIR, "yolo_stuff", "temporary_dir")
background_images = glob(os.path.join(TMDIR, "data", "background", '*.jpg')) 
background_images += glob(os.path.join(TMDIR, "data", "VisDroneSampled", '*.jpg')) 
print(">>>> background list size", len(background_images))

def add_template_to_image(tm_image, main_image):
    image = main_image.copy()
    mask = Image.new('L', tm_image.size, 0)
    draw = ImageDraw.Draw(mask)
    z = random.randint(10, 40) #40
    # ValueError: x1 must be greater than or equal to x0
    try:
        draw.rectangle((z, z, mask.width - z, mask.height - z), fill=255)
    except ValueError:
        # z = int(z / 2)
        z = min(mask.width - z, mask.height - z)
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
        if '.21' in k:
            reach_wanted_layer = True
        if not reach_wanted_layer:        
            v.requires_grad = False 
        else:
            v.requires_grad = True
        print(k, v.requires_grad) 



class MyDetectionTrainer(DetectionTrainer):

    def build_dataset(self, img_path, mode='train', batch=None):
        """
        Build YOLO Dataset.

        Args:
            img_path (str): Path to the folder containing images.
            mode (str): `train` mode or `val` mode, users are able to customize different augmentations for each mode.
            batch (int, optional): Size of batches, this is for `rect`. Defaults to None.
        """
        gs = max(int(de_parallel(self.model).stride.max() if self.model else 0), 32)
        return build_yolo_dataset(self.args, img_path, batch, self.data, mode=mode, rect=mode == 'val', stride=gs)

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode='train'):
        """Construct and return dataloader."""
        assert mode in ['train', 'val']
        with torch_distributed_zero_first(rank):  # init dataset *.cache only once if DDP
            dataset = self.build_dataset(dataset_path, mode, batch_size)
        pdb.set_trace()
        shuffle = mode == 'train'
        if getattr(dataset, 'rect', False) and shuffle:
            LOGGER.warning("WARNING ⚠️ 'rect=True' is incompatible with DataLoader shuffle, setting shuffle=False")
            shuffle = False
        workers = self.args.workers if mode == 'train' else self.args.workers * 2
        return build_dataloader(dataset, batch_size, workers, shuffle, rank)  # return dataloader



def make_data(templates, temp_boxes, bg_list, mode: str, trainer=None):
    print(">>>>", f"make {mode} data")
    for i, tmpl in enumerate(templates):
        x1, y1, x2, y2 = temp_boxes[i]
        
        clear_last_run_data(mode)
        os.makedirs(os.path.join(augmented_dir, 'images', mode), exist_ok=True)
        os.makedirs(os.path.join(augmented_dir, 'labels', mode), exist_ok=True)
        
        tmpl_crop = Image.fromarray(np.array(tmpl)[y1:y2, x1:x2, :])
        # augment tmpl size
        random_size = random.uniform(0.2, 2)
        tmpl_crop = tmpl_crop.resize((int(tmpl_crop.size[0] * random_size), int(tmpl_crop.size[1] * random_size)))
        x2, y2 = x1 + tmpl_crop.size[0], y1 + tmpl_crop.size[1]  
        random.shuffle(bg_list)
        for ii, path in enumerate(bg_list):
            back_image = Image.open(path).convert("RGB")
            bg_area = back_image.size[0] * back_image.size[1]
            tmpl_crop_area = tmpl_crop.size[0] * tmpl_crop.size[1]
            # resize template if its size is bigger than quarter of bg image size
            if tmpl_crop_area > bg_area / 4:
                tmpl_crop = tmpl_crop.resize((int(tmpl_crop.size[0]/2), int(tmpl_crop.size[1]/2)))
                x2, y2 = x1 + tmpl_crop.size[0], y1 + tmpl_crop.size[1]
            try:
                aug, new_x1, new_y1 = add_template_to_image(tmpl_crop, back_image)
            except ValueError:
                continue
            new_x2, new_y2 = new_x1 + tmpl_crop.size[0], new_y1 + tmpl_crop.size[1]
            aug_name = f'{ii}.jpg'
            aug_path = os.path.join(augmented_dir, 'images', mode, aug_name)
            aug.save(aug_path)
            bg_width, bg_height = back_image.size
            xc, yc, w, h = convert_to_yolo(new_x1, new_y1, new_x2, new_y2, bg_width, bg_height)
            with open(os.path.join(augmented_dir, 'labels', mode, aug_name.replace('.jpg', '.txt')), 'w') as f:
                f.write(f'0 {xc} {yc} {w} {h}\n')



    
def clear_last_run_data(mode):
    imgs = glob(os.path.join(augmented_dir, 'images', mode, '*.jpg'), recursive=True)
    txts = glob(os.path.join(augmented_dir, 'labels', mode, '*.txt'), recursive=True)
    for im in imgs:
        os.remove(im)
    for txt in txts:
        os.remove(txt)


class Yolo(BaseFewShotMatcher):
    def __init__(self):
        super().__init__()
        # self.ckpt_path = (os.getenv('YOLO_MODEL_PATH', None) or
                    #   "/mnt/File/shamgholi/tm/yolo_stuff/yolov8m-oiv7-artificial.pt")
        # self.model = YOLO(self.ckpt_path)
        self.base_ckpt_path = "/home/amirreza/shamgholi/tm/yolo_stuff/yolov8m-oiv7.pt"
        self.scales = [float(scale) for scale in [0.1, 0.2, 0.4, 0.6, 0.8, 1, 1.5]]
        # self.scales = [float(scale) for scale in [0.6, 1, 1.5]]

    def match(self, target):
        assert hasattr(self, 'model'), 'run set_templates first'
        t0 = time.time()
        results = self.model.predict(target, conf=0.001, verbose=False)
        t1 = time.time()
        model_time = t1 - t0
        # r:Results
        if len(results) < 1 or len(results[0].boxes) < 1:
            return []
        preds = []
        p = results[0].boxes.xyxy[0].tolist()
        p.append(results[0].boxes.conf[0].item())
        p.append(model_time)
        preds.append(p)
        return preds




    def set_templates(self, templates, temp_boxes):

        train_bg, valid_bg = train_test_split(background_images, test_size=0.2)
        # we run this part in every epoch
        make_data(templates, temp_boxes, train_bg, 'train')
        make_data(templates, temp_boxes, valid_bg, 'val')

        train_conf_content = f"path: {augmented_dir}\ntrain: images/train\nval: images/val\n\nnc: 1\n"
        train_conf_path = os.path.join(augmented_dir, 'config.yaml')
        with open(train_conf_path, 'w') as f:
            f.write(train_conf_content)
        self.model = YOLO(self.base_ckpt_path)
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        self.model.add_callback("on_train_start", freeze_layer)
        # partial_make_data = functools.partial(make_data, templates, temp_boxes, train_bg, 'train')
        # self.model.add_callback("on_train_epoch_start", partial_make_data)
        self.model.train(data=train_conf_path, epochs=15, batch=16, project=augmented_dir, 
                         close_mosaic=2, exist_ok=True, single_cls=True, degrees=0.0)
        self.model = YOLO(os.path.join(augmented_dir, 'train', 'weights', 'best.pt'))
        #self.model.eval()
        if torch.cuda.is_available():
            self.model = self.model.cuda()
