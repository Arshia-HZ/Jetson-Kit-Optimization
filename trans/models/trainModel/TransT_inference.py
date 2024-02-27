import torch.nn as nn
from trainModel.TransT.ltr import model_constructor

import torch
import torch.nn.functional as F
from trainModel.TransT.util import box_ops
from trainModel.TransT.util.misc import (NestedTensor, nested_tensor_from_tensor,
                       nested_tensor_from_tensor_2,
                       accuracy)

from trainModel.TransT.ltr.models.backbone.transt_backbone import build_backbone
from trainModel.TransT.ltr.models.loss.matcher import build_matcher
from trainModel.TransT.ltr.models.neck.featurefusion_network import build_featurefusion_network
from PIL import Image
from torchvision import transforms
from torchvision.transforms import ToTensor, Resize
import numpy as np
import matplotlib.pyplot as plt
import cv2
from baseFewShotMatcher import BaseFewShotMatcher
import time

class TransT(nn.Module):
    def __init__(self, backbone, featurefusion_network, num_classes):
        super().__init__()
        self.featurefusion_network = featurefusion_network
        hidden_dim = featurefusion_network.d_model
        self.class_embed = MLP(hidden_dim, hidden_dim, num_classes + 1, 3)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.backbone = backbone

    def forward(self, search, template):
        if not isinstance(search, NestedTensor):
            search = nested_tensor_from_tensor(search)
        if not isinstance(template, NestedTensor):
            template = nested_tensor_from_tensor(template)
        feature_search, pos_search = self.backbone(search)
        feature_template, pos_template = self.backbone(template)
        src_search, mask_search= feature_search[-1].decompose()
        assert mask_search is not None
        src_template, mask_template = feature_template[-1].decompose()
        assert mask_template is not None
        hs = self.featurefusion_network(self.input_proj(src_template), mask_template, self.input_proj(src_search), mask_search, pos_template[-1], pos_search[-1])

        outputs_class = self.class_embed(hs)
        outputs_coord = self.bbox_embed(hs).sigmoid()
        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        return out

    def match(self, search):
        if not isinstance(search, NestedTensor):
            search = nested_tensor_from_tensor_2(search)
        features_search, pos_search = self.backbone(search)
        feature_template = self.zf
        pos_template = self.pos_template
        src_search, mask_search= features_search[-1].decompose()
        assert mask_search is not None
        src_template, mask_template = feature_template[-1].decompose()
        assert mask_template is not None
        hs = self.featurefusion_network(self.input_proj(src_template), mask_template, self.input_proj(src_search), mask_search, pos_template[-1], pos_search[-1])

        outputs_class = self.class_embed(hs)
        outputs_coord = self.bbox_embed(hs).sigmoid()
        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        return out

    def template(self, z):
        if not isinstance(z, NestedTensor):
            z = nested_tensor_from_tensor_2(z)
        zf, pos_template = self.backbone(z)
        self.zf = zf
        self.pos_template = pos_template

class Settings:
    def __init__(self):
        self.device = 'cpu'
        self.description = 'TransT with default settings.'
        self.batch_size = 1
        self.num_workers = 8
        self.multi_gpu = False
        self.print_interval = 1
        self.normalize_mean = [0.485, 0.456, 0.406]
        self.normalize_std = [0.229, 0.224, 0.225]
        self.search_area_factor = 4.0
        self.template_area_factor = 2.0 
        self.search_feature_sz = 32
        self.template_feature_sz = 16
        self.search_sz = self.search_feature_sz * 8
        self.temp_sz = self.template_feature_sz * 8
        self.center_jitter_factor = {'search': 3, 'template': 0}
        self.scale_jitter_factor = {'search': 0.25, 'template': 0}

        # Transformer
        self.position_embedding = 'sine'
        self.hidden_dim = 256
        self.dropout = 0.1
        self.nheads = 8
        self.dim_feedforward = 2048
        self.featurefusion_layers = 4

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class transt_model(BaseFewShotMatcher):
    def __init__(self, settings=None):
        super().__init__()
        if settings is None:
            settings = Settings()
        num_classes = 1
        backbone_net = build_backbone(settings, backbone_pretrained=True)
        featurefusion_network = build_featurefusion_network(settings)
        self.model = TransT(
            backbone_net,
            featurefusion_network,
            num_classes=num_classes
        )
        self.device = torch.device(settings.device)
        self.model.to(self.device)
        self.model.load_state_dict(torch.load('/home/slipknot/Downloads/transt.pth', map_location=torch.device('cpu'))['net'])
        self.temp_transform = transforms.Compose([
            transforms.ToTensor()
        ])
        self.search_transform = transforms.Compose([
            transforms.ToTensor()
        ])
        self.scales = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5]

    def predict(self, target, templates, tembox):
        
        # target_orig = target.copy()
        orig_shape = target.shape

        def iou(box1, box2):
            x1, y1, w1, h1 = box1
            x2, y2, w2, h2 = box2
            x_intersection = max(0, min(x1 + w1 / 2, x2 + w2 / 2) - max(x1 - w1 / 2, x2 - w2 / 2))
            y_intersection = max(0, min(y1 + h1 / 2, y2 + h2 / 2) - max(y1 - h1 / 2, y2 - h2 / 2))
            intersection_area = x_intersection * y_intersection
            area_box1 = w1 * h1
            area_box2 = w2 * h2
            union_area = area_box1 + area_box2 - intersection_area
            iou_value = intersection_area / union_area
            return iou_value

        def nms(predictions, iou_threshold=0.5, top_k=10):
            predictions = sorted(predictions, key=lambda x: x[4], reverse=True)
            selected_predictions = []

            while len(predictions) > 0 and len(selected_predictions) < top_k:
                current_prediction = predictions[0]
                selected_predictions.append(current_prediction)
                predictions = predictions[1:]
                predictions = [p for p in predictions if iou(p[:4], current_prediction[:4]) < iou_threshold]
            return selected_predictions[:top_k]

        predictions = []

        target = self.search_transform(Image.fromarray(target).resize((256, 256)))

        if len(target.shape) != 4:
            target = target.unsqueeze(0)

        for i, temp1 in enumerate(templates):

            # target_to_draw = target_orig.copy()
            
            # 2* of the exact size of target

            # box_height = abs(tembox[i][2]-tembox[i][0])
            # box_width = abs(tembox[i][3]-tembox[i][1])
            
            # if box_height > box_width:
            #     crop_factor = box_height
            # else:
            #     crop_factor = box_width
            
            # temp_center_coor = (tembox[i][1] + box_width // 2, tembox[i][0] + box_height // 2)

            # temp1 = temp1[max(temp_center_coor[0] - crop_factor, 0):min(temp_center_coor[0] + crop_factor, temp1.shape[0]),
            #               max(temp_center_coor[1] - crop_factor, 0):min(temp_center_coor[1] + crop_factor, temp1.shape[1])]

            # using the FSRCNN model for upscaling

            temp1 = temp1[tembox[i][1]:tembox[i][3], tembox[i][0]:tembox[i][2]]

            # cv2.imwrite(f'/home/slipknot/Desktop/trant_proj/temp_cropped_{i}.png', temp1)

            temp1 = cv2.resize(temp1, (64, 64))

            tmp_zeros = np.zeros((128, 128), dtype='uint8')
            tmp_zeros[32:96, 32:96, :] = temp1
            temp1 = tmp_zeros

            # cv2.imwrite(f'/home/slipknot/Desktop/trant_proj/temp_resized_{i}.png', temp1)

            # sr = cv2.dnn_superres.DnnSuperResImpl_create()
            
            # path = "/home/slipknot/Desktop/FSRCNN_x2.pb"
            
            # sr.readModel(path)
            
            # sr.setModel("fsrcnn", 2)
            
            # temp1 = sr.upsample(temp1)

            # cv2.imwrite(f'/home/slipknot/Desktop/trant_proj/temp_upsampled_{i}.png', temp1)

            # temp1 = self.temp_transform(Image.fromarray(temp1).resize((128, 128)))
            temp1 = self.temp_transform(Image.fromarray(temp1))

            if len(temp1.shape) != 4:
                temp1 = temp1.unsqueeze(0)

            temp1 = temp1.to(self.device)
            target = target.to(self.device)
            
            start = round(time.time() * 1000)
            self.model.template(temp1)

            output = self.model.match(target)
            end = round(time.time() * 1000)

            pred_logits = output['pred_logits']
            pred_boxes = output['pred_boxes']

            pred_boxes = pred_boxes.squeeze(0).detach().cpu().numpy()
            pred_logits = pred_logits.squeeze(0).detach().cpu().numpy()

            processed_pred_box = nms(np.concatenate([pred_boxes, pred_logits[:, 0].reshape((-1, 1))], axis=1))
            final_processed_boxes = []

            for box in [processed_pred_box[0]]:
                center_x, center_y, height, width, score = box
                x_min = int((center_x - height / 2) * orig_shape[1])
                y_min = int((center_y - width / 2) * orig_shape[0])
                x_max = int((center_x + height / 2) * orig_shape[1])
                y_max = int((center_y + width / 2) * orig_shape[0])
            #     cv2.rectangle(target_to_draw, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            #     cv2.imwrite(f'/home/slipknot/Desktop/trant_proj/pred_{i}.png', target_to_draw)


                final_processed_boxes.append((x_min, y_min, x_max, y_max, 
                                            float("{:.3f}".format(score)), end - start))


            predictions.extend(final_processed_boxes)
        return predictions




            