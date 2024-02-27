from trainModel.mainLib import *
import trainModel.SiamSe.lib.models.models as models
from baseFewShotMatcher import BaseFewShotMatcher
Corner = namedtuple('Corner', 'x1 y1 x2 y2')
BBox = Corner
Center = namedtuple('Center', 'x y w h')

# ---------------------------------
# Functions for FC tracking tools
# ---------------------------------

verbose = 0
def convert_color_RGB(img):
    if len(img.shape) == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    else:
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def to_torch(ndarray):
    return torch.from_numpy(ndarray)


def im_to_torch(img):
    img = np.transpose(img, (2, 0, 1))  # C*H*W
    img = to_torch(img).float()
    return img


# ORIGINAL
def get_subwindow_tracking(im, pos, model_sz, original_sz, avg_chans, out_mode='torch'):
    """
    SiamFC type cropping
    """
    if isinstance(pos, float):
        pos = [pos, pos]
    sz = original_sz
    im_sz = im.shape
    c = (original_sz + 1) / 2
    context_xmin = round(pos[0] - c)
    context_xmax = context_xmin + sz - 1
    context_ymin = round(pos[1] - c)
    context_ymax = context_ymin + sz - 1
    left_pad = int(max(0., -context_xmin))
    top_pad = int(max(0., -context_ymin))
    right_pad = int(max(0., context_xmax - im_sz[1] + 1))
    bottom_pad = int(max(0., context_ymax - im_sz[0] + 1))

    context_xmin = context_xmin + left_pad
    context_xmax = context_xmax + left_pad
    context_ymin = context_ymin + top_pad
    context_ymax = context_ymax + top_pad
    r, c, k = im.shape
    if any([top_pad, bottom_pad, left_pad, right_pad]):
        te_im = np.zeros((r + top_pad + bottom_pad, c + left_pad + right_pad, k), np.uint8)
        te_im[top_pad:top_pad + r, left_pad:left_pad + c, :] = im
        if top_pad:
            te_im[0:top_pad, left_pad:left_pad + c, :] = avg_chans
        if bottom_pad:
            te_im[r + top_pad:, left_pad:left_pad + c, :] = avg_chans
        if left_pad:
            te_im[:, 0:left_pad, :] = avg_chans
        if right_pad:
            te_im[:, c + left_pad:, :] = avg_chans
        im_patch_original = te_im[int(context_ymin):int(
            context_ymax + 1), int(context_xmin):int(context_xmax + 1), :]

    else:
        im_patch_original = im[int(context_ymin):int(context_ymax + 1),
                               int(context_xmin):int(context_xmax + 1), :]

    scale = model_sz/im_patch_original.shape[0]
    if not np.array_equal(model_sz, original_sz):
        im_patch = cv2.resize(im_patch_original, (model_sz, model_sz))

    else:
        im_patch = im_patch_original
    return im_to_torch(im_patch.copy()) if out_mode in 'torch' else im_patch, scale

# from google.colab.patches import cv2_imshow

def make_scale_pyramid(im, pos, in_side_scaled, out_side, avg_chans):
    in_side_scaled = [round(x) for x in in_side_scaled]
    num_scale = len(in_side_scaled)
    pyramid = torch.zeros(num_scale, 3, out_side, out_side)
    max_target_side = in_side_scaled[-1]
    min_target_side = in_side_scaled[0]
    beta = out_side / min_target_side

    search_side = round(beta * max_target_side)
    search_region,_ = get_subwindow_tracking(im, pos, int(
        search_side), int(max_target_side), avg_chans, out_mode='np')

    for s, temp in enumerate(in_side_scaled):
        target_side = round(beta * temp)
        pyramid[s, :],_ = get_subwindow_tracking(
            search_region, (1 + search_side) / 2, out_side, target_side, avg_chans)

    return pyramid

# -----------------------------------
# Functions for benchmark and others
# -----------------------------------


def load_dataset(dataset, root=None):
    info = {}

    if 'OTB' in dataset:
        if root:
            base_path = join(root, dataset)
            json_path = join(root, dataset + '.json')
        else:
            raise FileNotFoundError

        info = json.load(open(json_path, 'r'))
        for v in info.keys():
            path_name = info[v]['name']
            info[v]['image_files'] = [join(base_path, path_name, 'img', im_f)
                                      for im_f in info[v]['image_files']]
            info[v]['gt'] = np.array(info[v]['gt_rect']) - [1, 1, 0, 0]
            info[v]['name'] = v

    elif 'VOT' in dataset:
        if root:
            base_path = join(root, dataset)
        else:
            raise FileNotFoundError
        list_path = join(base_path, 'list.txt')
        with open(list_path) as f:
            videos = [v.strip() for v in f.readlines()]
        videos = sorted(videos)
        for video in videos:
            video_path = join(base_path, video)
            image_path = join(video_path, '*.jpg')
            image_files = sorted(glob.glob(image_path))
            if len(image_files) == 0:  # VOT2018
                image_path = join(video_path, 'color', '*.jpg')
                image_files = sorted(glob.glob(image_path))
            gt_path = join(video_path, 'groundtruth.txt')
            gt = np.loadtxt(gt_path, delimiter=',').astype(np.float64)
            info[video] = {'image_files': image_files, 'gt': gt, 'name': video}

    else:
        raise ValueError("Dataset not support now, edit for other dataset youself...")

    return info


def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys

    print('missing keys:{}'.format(missing_keys))
    print('unused checkpoint keys:{}'.format(unused_pretrained_keys))
    # print('used keys:{}'.format(used_pretrained_keys))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    '''
    Old style model is stored with all names of parameters share common prefix 'module.'
    '''
    print('remove prefix "{}"'.format(prefix))
    def f(x): return x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_pretrain(model, pretrained_path):
    print('load pretrained model from {}'.format(pretrained_path))

    device = torch.cuda.current_device()
    pretrained_dict = torch.load(
        pretrained_path, map_location=lambda storage, loc: storage.cuda(device))

    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model


def corner2center(corner):
    """
    [x1, y1, x2, y2] --> [cx, cy, w, h]
    """
    if isinstance(corner, Corner):
        x1, y1, x2, y2 = corner
        return Center((x1 + x2) * 0.5, (y1 + y2) * 0.5, (x2 - x1), (y2 - y1))
    else:
        x1, y1, x2, y2 = corner[0], corner[1], corner[2], corner[3]
        x = (x1 + x2) * 0.5
        y = (y1 + y2) * 0.5
        w = x2 - x1
        h = y2 - y1
        return x, y, w, h


def center2corner(center):
    """
    [cx, cy, w, h] --> [x1, y1, x2, y2]
    """
    if isinstance(center, Center):
        x, y, w, h = center
        return Corner(x - w * 0.5, y - h * 0.5, x + w * 0.5, y + h * 0.5)
    else:
        x, y, w, h = center[0], center[1], center[2], center[3]
        x1 = x - w * 0.5
        y1 = y - h * 0.5
        x2 = x + w * 0.5
        y2 = y + h * 0.5
        return x1, y1, x2, y2


def aug_apply(bbox, param, shape, inv=False, rd=False):
    """
    apply augmentation
    :param bbox: original bbox in image
    :param param: augmentation param, shift/scale
    :param shape: image shape, h, w, (c)
    :param inv: inverse
    :param rd: round bbox
    :return: bbox(, param)
        bbox: augmented bbox
        param: real augmentation param
    """
    if not inv:
        center = corner2center(bbox)
        original_center = center

        real_param = {}
        if 'scale' in param:
            scale_x, scale_y = param['scale']
            imh, imw = shape[:2]
            h, w = center.h, center.w

            scale_x = min(scale_x, float(imw) / w)
            scale_y = min(scale_y, float(imh) / h)
            center = Center(center.x, center.y, center.w * scale_x, center.h * scale_y)

        bbox = center2corner(center)

        if 'shift' in param:
            tx, ty = param['shift']
            x1, y1, x2, y2 = bbox
            imh, imw = shape[:2]

            tx = max(-x1, min(imw - 1 - x2, tx))
            ty = max(-y1, min(imh - 1 - y2, ty))

            bbox = Corner(x1 + tx, y1 + ty, x2 + tx, y2 + ty)

        if rd:
            bbox = Corner(*map(round, bbox))

        current_center = corner2center(bbox)

        real_param['scale'] = current_center.w / \
            original_center.w, current_center.h / original_center.h
        real_param['shift'] = current_center.x - \
            original_center.x, current_center.y - original_center.y

        return bbox, real_param
    else:
        if 'scale' in param:
            scale_x, scale_y = param['scale']
        else:
            scale_x, scale_y = 1., 1.

        if 'shift' in param:
            tx, ty = param['shift']
        else:
            tx, ty = 0, 0

        center = corner2center(bbox)
        center = Center(center.x - tx, center.y - ty, center.w / scale_x, center.h / scale_y)
        return center2corner(center)


# others
def cxy_wh_2_rect(pos, sz):
    # 0-index
    return [float(max(float(0), pos[0] - sz[0] / 2)), float(max(float(0), pos[1] - sz[1] / 2)), float(sz[0]), float(sz[1])]


def get_axis_aligned_bbox(region):
    x = int(region[0])
    y = int(region[1])
    w = int(region[2])
    h = int(region[3])
    cx = x + w / 2
    cy = y + h / 2

    return cx, cy, w, h


def poly_iou(polys1, polys2, bound=None):
    r"""Intersection over union of polygons.

    Args:
        polys1 (numpy.ndarray): An N x 4 numpy array, each line represent a rectangle
            (left, top, width, height); or an N x 8 numpy array, each line represent
            the coordinates (x1, y1, x2, y2, x3, y3, x4, y4) of 4 corners.
        polys2 (numpy.ndarray): An N x 4 numpy array, each line represent a rectangle
            (left, top, width, height); or an N x 8 numpy array, each line represent
            the coordinates (x1, y1, x2, y2, x3, y3, x4, y4) of 4 corners.
    """
    assert polys1.ndim in [1, 2]
    if polys1.ndim == 1:
        polys1 = np.array([polys1])
        polys2 = np.array([polys2])
    assert len(polys1) == len(polys2)

    polys1 = _to_polygon(polys1)
    polys2 = _to_polygon(polys2)
    if bound is not None:
        bound = box(0, 0, bound[0], bound[1])
        polys1 = [p.intersection(bound) for p in polys1]
        polys2 = [p.intersection(bound) for p in polys2]

    eps = np.finfo(float).eps
    ious = []
    for poly1, poly2 in zip(polys1, polys2):
        area_inter = poly1.intersection(poly2).area
        area_union = poly1.union(poly2).area
        ious.append(area_inter / (area_union + eps))
    ious = np.clip(ious, 0.0, 1.0)

    return ious


def _to_polygon(polys):
    r"""Convert 4 or 8 dimensional array to Polygons

    Args:
        polys (numpy.ndarray): An N x 4 numpy array, each line represent a rectangle
            (left, top, width, height); or an N x 8 numpy array, each line represent
            the coordinates (x1, y1, x2, y2, x3, y3, x4, y4) of 4 corners.
    """

    def to_polygon(x):
        assert len(x) in [4, 8]
        if len(x) == 4:
            return box(x[0], x[1], x[0] + x[2], x[1] + x[3])
        elif len(x) == 8:
            return Polygon([(x[2 * i], x[2 * i + 1]) for i in range(4)])

    if polys.ndim == 1:
        return to_polygon(polys)
    else:
        return [to_polygon(t) for t in polys]


def print_speed(i, i_time, n):
    average_time = i_time
    remaining_time = (n - i) * average_time
    remaining_day = math.floor(remaining_time / 86400)
    remaining_hour = math.floor(remaining_time / 3600 - remaining_day * 24)
    remaining_min = math.floor(remaining_time / 60 - remaining_day * 1440 - remaining_hour * 60)
    print('Progress: %d / %d [%d%%], Speed: %.3f s/iter, ETA %d:%02d:%02d (D:H:M)\n' %
          (i, n, i / n * 100, average_time, remaining_day, remaining_hour, remaining_min))
    print('\nPROGRESS: {:.2f}%\n'.format(100 * i / n))


def save_model(model, epoch, optimizer, model_name, cfg):
    if not exists(cfg.CHECKPOINT_DIR):
        os.makedirs(cfg.CHECKPOINT_DIR)

    path = os.path.join(cfg.CHECKPOINT_DIR, 'checkpoint_e{}.pth'.format(epoch + 1))
    state_dict = {
        'epoch': epoch + 1,
        'arch': model_name,
        'state_dict': model.module.state_dict(),
        'optimizer': optimizer.state_dict()
    }

    torch.save(state_dict, path)



class SESiamFCTracker(object):
    def __init__(self, net, num_scales=5, scale_step=1.0375, scale_penalty=0.9745,
                 scale_lr=0.590, response_up=16, w_influence=0.350, exemplar_size=127,
                 instance_size=255, score_size=17, total_stride=8, context_amount=0.5, **kwargs):
        super(SESiamFCTracker, self).__init__()
        self.net = net
        # config parameters
        self.num_scales = num_scales
        self.scale_step = scale_step
        self.scale_penalty = scale_penalty
        self.scale_lr = scale_lr
        self.response_up = response_up
        self.w_influence = w_influence
        self.exemplar_size = exemplar_size
        self.instance_size = instance_size
        self.score_size = score_size
        self.total_stride = total_stride
        self.context_amount = context_amount

        # constant extra parameteres
        window = np.outer(np.hanning(int(self.score_size) * int(self.response_up)),
                          np.hanning(int(self.score_size) * int(self.response_up)))
        self.window = window / window.sum()

        self.scales = self.scale_step ** (range(self.num_scales) - np.ceil(self.num_scales // 2))

        # runnning stats
        self.target_pos = None
        self.target_sz = None
        self.avg_chans = None
        self.im_h = None
        self.im_w = None
        self.s_x = None
        self.min_s_x = None
        self.max_s_x = None

    @torch.no_grad()
    def init(self, im, target_pos, target_sz):
        self.target_pos = target_pos
        self.target_sz = target_sz
        self.avg_chans = np.mean(im, axis=(0, 1))
        self.im_h = im.shape[0]
        self.im_w = im.shape[1]

        # prep
        wc_z = target_sz[0] + self.context_amount * sum(target_sz)
        hc_z = target_sz[1] + self.context_amount * sum(target_sz)
        s_z = round(np.sqrt(wc_z * hc_z))
        scale_z = self.exemplar_size / s_z

        d_search = (self.instance_size - self.exemplar_size) / 2
        pad = d_search / scale_z

        self.s_x = s_z + 2 * pad
        self.min_s_x = 0.2 * self.s_x
        self.max_s_x = 5 * self.s_x

        z_crop, scale = get_subwindow_tracking(im, target_pos, self.exemplar_size, s_z, self.avg_chans)
        cropped = torch.clone(z_crop)
        self.target_sz *= scale  # we added
        self.target_sz *= 2.5  # we added
        self.net.template(z_crop.unsqueeze(0).cuda())

    @torch.no_grad()
    def track(self, im):
        scaled_instance = self.s_x * self.scales
        scaled_target = [[self.target_sz[0] * self.scales], [self.target_sz[1] * self.scales]]
        tar_pos = np.array([im.shape[1] / 2, im.shape[0] / 2])
        tar_sz = np.array([im.shape[1], im.shape[0]])
        if verbose == 0:
            plt.imshow(im)
            plt.title("x_crops")
            plt.show()
            print("\ntar_sz : ", tar_sz)
        x_crops = make_scale_pyramid(im, tar_pos, scaled_instance,
                                     self.instance_size, self.avg_chans).cuda()

        response_map, _ = self.net.track(x_crops)

        # score=np.mean(np.array(response_map.cpu()))
        score = np.max(np.array(response_map.cpu()))

        up_size = self.response_up * response_map.shape[-1]
        response_map_up = F.interpolate(response_map, size=(up_size, up_size), mode='bicubic')
        response_map_up = response_map_up.squeeze(1).detach().cpu().data.numpy().transpose(1, 2, 0)

        s_penaltys = np.array([self.scale_penalty ** (abs(i - self.num_scales // 2))
                               for i in range(self.num_scales)])
        temp_max = np.max(response_map_up, axis=(0, 1))
        temp_max *= s_penaltys
        best_scale = np.argmax(temp_max)
        response_map = response_map_up[:, :, best_scale]

        response_map = response_map - response_map.min()
        response_map = response_map / response_map.sum()
        # response_map = (1 - self.w_influence) * response_map + self.w_influence * self.window
        # response_map = cv2.copyMakeBorder(response_map, 27, 27, 27, 27, cv2.BORDER_CONSTANT)#we added
        r_max, c_max = np.unravel_index(response_map.argmax(), response_map.shape)
        p_corr = [c_max, r_max]

        disp_instance_final = p_corr - np.ceil(self.score_size * self.response_up / 2)
        disp_instance_input = disp_instance_final * self.total_stride / self.response_up
        disp_instance_frame = disp_instance_input * self.s_x / self.instance_size
        target_pos = tar_pos + disp_instance_frame
        self.s_x = max(self.min_s_x, min(self.max_s_x, (1 - self.scale_lr) *
                                         self.s_x + self.scale_lr * scaled_instance[best_scale]))
        target_sz = [((1 - self.scale_lr) * self.target_sz[0] + self.scale_lr * scaled_target[0][0][best_scale]),
                     ((1 - self.scale_lr) * self.target_sz[1] + self.scale_lr * scaled_target[1][0][best_scale])]

        self.target_pos = target_pos
        self.target_sz = target_sz
        return target_pos, target_sz, score

    def __repr__(self):
        s = self.__class__.__name__ + ':\n'
        s += '  num_scales={num_scales}\n'
        s += '  scale_step={scale_step}\n'
        s += '  scale_lr={scale_lr}\n'
        s += '  response_up={response_up}\n'
        s += '  w_influence={w_influence}\n'
        s += '  exemplar_size={exemplar_size}\n'
        s += '  instance_size={instance_size}\n'
        s += '  score_size={score_size}\n'
        s += '  total_stride={total_stride}\n'
        s += '  context_amount={context_amount}\n'
        s += '  scales={scales}\n'
        return s.format(**self.__dict__)







class SiamSE(BaseFewShotMatcher):
    def __init__(self):
        opt = edict({
            "checkpoint": 'pretrain/checkpoint_otb.pth',
            "cfg": "configs/test.yaml",
        })
        with open(opt.cfg, 'r') as f:
            tracker_config = yaml.load(f, Loader=yaml.FullLoader)

        # prepare model
        net = models.__dict__[tracker_config['MODEL']](padding_mode='constant')
        net = load_pretrain(net, opt.checkpoint)
        net = net.eval().cuda()
        # prepare tracker
        tracker_config = tracker_config['TRACKER']["VOT2017"]
        tracker = SESiamFCTracker(net, **tracker_config)
        self.tracker = tracker

    def predict(self, tar, templates, bbx, scale):
        predictions = []
        tar = convert_color_RGB(tar)
        for i, temp in enumerate(templates):
            t1 = time.time()
            shot = i
            temp = convert_color_RGB(temp)
            cx, cy, w, h = get_axis_aligned_bbox(bbx[shot])
            x0, y0, x00, y00 = bbx[shot][0], bbx[shot][1], bbx[shot][0] + bbx[shot][2], bbx[shot][1] + bbx[shot][3]

            if verbose > 0:
                Temp = cv2.rectangle(temp, (int(x0), int(y0)), (int(x00), int(y00)), (0, 0, 255), 5)
                plt.imshow(Temp)
                plt.title("cropped template")
                plt.show()

            i += 1
            target_pos = np.array([cx, cy])
            target_sz = np.array([float(w), float(h)])
            # print("\ntarget_sz in predict base : ", target_sz)
            self.tracker.init(temp, target_pos, target_sz)  # init tracker
            # print("\ntar shape in predict : ", tar.shape)
            target_pos, target_sz, score = self.tracker.track(tar)  # tracking

            location = cxy_wh_2_rect(target_pos, target_sz)
            if location:
                t2 = time.time()
                Tar = tar.copy()
                x1, y1, x2, y2 = int(location[0]), int(location[1]), int(location[0] + location[2]), int(
                    location[1] + location[3])
                cv2.rectangle(Tar, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255))
                predictions.append([x1, y1, x2, y2, score, (t2 - t1) / i])
            else:
                t2 = time.time()
                predictions.append([-1, -1, -1, -1, score, (t2 - t1) / i])

        predictions.sort(reverse=True, key=itemgetter(4))
        return predictions[:10]


opt = edict({
    "checkpoint": 'pretrain/checkpoint_otb.pth',
    "cfg": "configs/test.yaml",
})
with open(opt.cfg, 'r') as f:
    tracker_config = yaml.load(f, Loader=yaml.FullLoader)

# prepare model
net = models.__dict__[tracker_config['MODEL']](padding_mode='constant')
net = load_pretrain(net, opt.checkpoint)
net = net.eval().cuda()
# prepare tracker
tracker_config = tracker_config['TRACKER']["VOT2017"]
tracker = SESiamFCTracker(net, **tracker_config)
