import cv2
import numpy as np
import random
import glob
import os
import argparse

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

def add_transparent_image(background, foreground, x_offset=None, y_offset=None):
    bg_h, bg_w, bg_channels = background.shape
    fg_h, fg_w, fg_channels = foreground.shape

    assert bg_channels == 3, f'background image should have exactly 3 channels (RGB). found:{bg_channels}'
    assert fg_channels == 4, f'foreground image should have exactly 4 channels (RGBA). found:{fg_channels}'
    if x_offset is None: x_offset = (bg_w - fg_w) // 2
    if y_offset is None: y_offset = (bg_h - fg_h) // 2

    w = min(fg_w, bg_w, fg_w + x_offset, bg_w - x_offset)
    h = min(fg_h, bg_h, fg_h + y_offset, bg_h - y_offset)

    if w < 1 or h < 1: return
    bg_x = max(0, x_offset)
    bg_y = max(0, y_offset)
    fg_x = max(0, x_offset * -1)
    fg_y = max(0, y_offset * -1)
    foreground = foreground[fg_y:fg_y + h, fg_x:fg_x + w]
    background_subsection = background[bg_y:bg_y + h, bg_x:bg_x + w]
    foreground_colors = foreground[:, :, :3]
    alpha_channel = foreground[:, :, 3] / 255
    alpha_mask = np.dstack((alpha_channel, alpha_channel, alpha_channel))
    composite = background_subsection * (1 - alpha_mask) + foreground_colors * alpha_mask
    # width, height, channels = background_subsection.shape
    # center = (height//2, width//2)
    # composite = cv2.seamlessClone(foreground_colors, background_subsection, foreground[:, :, 3], center, cv2.NORMAL_CLONE)
    return composite

def place_image(position, image, reference_imagein1, top_, msk_img, segmsk):
    reference_imagein = reference_imagein1.copy()
    f = open(top_, 'a')
    f.write('0 {} {} {} {}\n'.format(position[0] / reference_imagein.shape[1], position[1] / reference_imagein.shape[0], \
            image.shape[1] / reference_imagein.shape[1], image.shape[0] / reference_imagein.shape[0]))
    f.close()
    if segmsk % 2 != 0:
        src_mask = np.ones_like(image) * 255
        reference_imagein = cv2.seamlessClone(image, reference_imagein, src_mask, position, cv2.NORMAL_CLONE)
    else:
        fg_img = np.concatenate([image, np.expand_dims(msk_img, axis=2)], axis=2)

        if image.shape[0] % 2 == 0:
            x_st, x_ed = image.shape[0] // 2, image.shape[0] // 2
        else:
            x_st, x_ed = image.shape[0] // 2, (image.shape[0] // 2) + 1

        if image.shape[1] % 2 == 0:
            y_st, y_ed = image.shape[1] // 2, image.shape[1] // 2
        else:
            y_st, y_ed = image.shape[1] // 2, (image.shape[1] // 2) + 1
        to_be_added = reference_imagein[position[1] - x_st:position[1] + x_ed, position[0] - y_st:position[0] + y_ed, :].copy()
        to_be_added = add_transparent_image(to_be_added, fg_img)
        reference_imagein[position[1] - x_st:position[1] + x_ed, position[0] - y_st:position[0] + y_ed, :] = to_be_added
    return reference_imagein

def prob_rot(want_rot, rot_small_photo):
    if want_rot == 1:
        rot_wrap = random.randint(0, 10)
        rotation_angle = random.randint(0, 10)
        M = cv2.getRotationMatrix2D((rot_small_photo.shape[1] // 2, rot_small_photo.shape[0] // 2), rotation_angle, 1)
        rot_small_photo = cv2.warpAffine(rot_small_photo, M, (rot_small_photo.shape[1], rot_small_photo.shape[0]))
    return rot_small_photo

def prob_resize(want_resize, rs_small_photo, bg_shape, msk_img_file, segmsk):
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
        if segmsk % 2 == 0:
            resized_msk = image_resize(msk_img_file, width = wid, height = hei)
        else:
            resized_msk = None
    return rs_small_photo, resized_msk

def prob_flip(want_flip, fp_small_photo):
    if want_flip == 1:
        if random.randint(0, 10) % 2 == 0:
            fp_small_photo = cv2.flip(fp_small_photo, 1)
    return fp_small_photo


def augment_data(dirname, tp, bg_path, resize_want, rotate_want, flip_want, more_obj_want, more_var_want, hist_match_want):
    # tempPaths = '/home/slipknot/Desktop/unchanged/'
    # outputDir = '/home/slipknot/Desktop/tm_bg/'

    want_resize = resize_want
    want_rot = rotate_want
    want_flip = flip_want
    want_more_obj = more_obj_want
    want_more_variance = more_var_want

    segMask = 2

    # dirs = os.listdir(tempPaths)

    if not os.path.isdir(dirname):
        dest = dirname
        os.system('mkdir -p {}'.format(dest))
    else:
        dest = dirname
    
    bgs = glob.glob(os.path.join(bg_path, '*.jpg'))
    for bg in bgs:
        tpname, bgname = tp.split('/')[-1].split('.')[0], bg.split('/')[-1].split('.')[0]
        out_path = os.path.join(dest, '{}_{}.jpg'.format(tpname, bgname))
        txt_out_path = os.path.join(dest, '{}_{}.txt'.format(tpname, bgname))
        orig_small_photo = cv2.imread(tp)

        if segMask % 2 == 0:
            kernel = np.ones((10, 10), np.uint8)
            msk_tp = cv2.imread('/home/amirreza/shamgholi/tm/data/artificial/{}/{}'.format(tp.split('/')[-2], tp.split('/')[-1]), 0)
            msk_tp = cv2.dilate(msk_tp, kernel, iterations=1)
        else:
            msk_tp = np.zeros_like(orig_small_photo)

        large_photo = cv2.imread(bg)
        potential_positions = []

        placed_images = []

        image1 = orig_small_photo.copy()
        image1 = prob_rot(want_rot, image1)
        image1, msk_tp_1 = prob_resize(want_resize, image1, large_photo.shape, msk_tp.copy(), segMask)
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

        reference_image = place_image(position, image1, large_photo, txt_out_path, msk_tp_1, segMask)

        if want_more_variance == 1:
            #orig_matched = hist_matching(orig_small_photo, large_photo)
            
            for index in range(0, 2):
                tempo = orig_small_photo.copy()
                tempo = prob_rot(want_rot, tempo)
                tempo, msk_tempo = prob_resize(want_resize, tempo, large_photo.shape, msk_tp.copy(), segMask)
                tempo = prob_flip(want_flip, tempo)
                x = random.randint(tempo.shape[1] // 2 + 1, (large_photo.shape[1] - tempo.shape[1] // 2) - 1)
                y = random.randint(tempo.shape[0] // 2 + 1, (large_photo.shape[0] - tempo.shape[0] // 2) - 1)
                txtoutpath = txt_out_path.split('/')
                txtoutpath[-1] = str(index)+txtoutpath[-1].split('.')[0]+'.txt'
                txtoutpath = '/'.join(txtoutpath)
                reference_image1 = place_image((x, y), tempo, large_photo, txtoutpath, msk_tempo, segMask)
                outpath = txtoutpath.split('/')
                outpath[-1] = outpath[-1].split('.')[0]+'.jpg'
                outpath = '/'.join(outpath)
                cv2.imwrite(outpath, reference_image1)

        if want_more_obj == 1:
            image2 = orig_small_photo.copy()
            image2 = prob_rot(want_rot, image2)
            image2i, msk_tp_2 = prob_resize(want_resize, image2, large_photo.shape, msk_tp.copy())
            image2 = prob_flip(want_flip, image2)

            if len(potential_positions) > 0:
                x = random.randint(image2.shape[1] // 2 + 1, (reference_image.shape[1] - image2.shape[1] // 2) - 1)
                y = random.randint(image2.shape[0] // 2 + 1, (reference_image.shape[0] - image2.shape[0] // 2) - 1)
                position = (x, y)                
                if is_valid_position(position, image2, placed_images):
                    placed_images.append({'image': image2, 'position': position})

                    if hist_match_want == 1:
                        image2 = hist_matching(image2, large_photo)

                    reference_image = place_image(position, image2, reference_image, txt_out_path, msk_tp_2)

            image3 = orig_small_photo.copy()
            image3 = prob_rot(want_rot, image3)
            image3, msk_tp_3 = prob_resize(want_resize, image3, large_photo.shape, msk_tp.copy())
            image3 = prob_flip(want_flip, image3)

            if len(potential_positions) > 0:
                x = random.randint(image3.shape[1] // 2 + 1, (reference_image.shape[1] - image3.shape[1] // 2) - 1)
                y = random.randint(image3.shape[0] // 2 + 1, (reference_image.shape[0] - image3.shape[0] // 2) - 1)
                position = (x, y)
                if is_valid_position(position, image3, placed_images):
                    placed_images.append({'image': image3, 'position': position})

                    if hist_match_want == 1:
                        image3 = hist_matching(image3, large_photo)

                    reference_image = place_image(position, image3, reference_image, txt_out_path, msk_tp_3)

        cv2.imwrite(out_path, reference_image)
                # except Exception as error:
                #     print('Error', error)



if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-b', '--base', default='/media/amirreza/6E2AF8F52AF8BB63/khoursha_files/tms', help='base to the tms directory')
    ap.add_argument('-d', '--dest', required=True, help='destination path to store results')
    ap.add_argument('-g', '--background', default='/media/amirreza/6E2AF8F52AF8BB63/khoursha_files/VisDroneSampled/', help='path to the backgrounds directory')
    ap.add_argument('-r', '--resize', default=1, type=int, help='want resize (1) or not (0)')
    ap.add_argument('-o', '--rotate', default=0, type=int, help='want rotate (1) or not (0)')
    ap.add_argument('-f', '--flip', default=0, type=int, help='want flip (1) or not (0)')
    ap.add_argument('-m', '--more_obj', default=0, type=int, help='want more objects in a background image (1) or not (0)')
    ap.add_argument('-v', '--more_var', default=1, type=int, help='want more variances of an augmentation method of a tm on a bg image (1) or not (0)')
    ap.add_argument('-z', '--hist_match', default=0, type=int, help='want histogram matching (1) or not (0)')

    args = vars(ap.parse_args())

    base_path = args['base']
    dest_path = args['dest']
    bg_path = args['background']
    resize_want = args['resize']
    rotate_want = args['rotate']
    flip_want = args['flip']
    more_obj_want = args['more_obj']
    more_var_want = args['more_var']
    hist_match_want = args['hist_match']

    dirs = os.listdir(base_path)
    for dr in dirs:
        dirname_ = os.path.join(dest_path, dr)
        tms = glob.glob(os.path.join(os.path.join(base_path, dr), '*.png'))
        for tm in tms:
            print('Augmenting for {} in {}'.format(tm.split('/')[-1].split('.')[0], dr))
            augment_data(dirname_, tm, bg_path, resize_want, rotate_want, flip_want, more_obj_want, more_var_want, hist_match_want)
