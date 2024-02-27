import glob
import os
import argparse


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-b', '--base', default='/media/amirreza/6E2AF8F52AF8BB63/khoursha_files/dilation_trans_resize_morevar', help='base to the augmented images directory')
    ap.add_argument('-d', '--dest', default='/media/amirreza/6E2AF8F52AF8BB63/khoursha_files/', help='path of destination directory')

    args = vars(ap.parse_args())

    base_path = args['base']
    path = args['dest']

    res_dir_name = base_path.split('/')[-1]

    dirs = os.listdir(base_path)

    train = os.path.join(path, 'train')
    val = os.path.join(path, 'val')

    for index, dr in enumerate(dirs):
        for i in range(10):
            for number in ['', 0, 1]:
                imgs = glob.glob(os.path.join(os.path.join(base_path, dr), '{}{}_*.jpg'.format(number, i)))
                for img in imgs[:int(0.8*len(imgs))]:
                    os.system('cp {} {}'.format(img, os.path.join(train, str(index)+'_'+img.split('/')[-1])))
                    os.system('cp {} {}'.format(os.path.join(os.path.join(base_path, dr), img.split('/')[-1].split('.')[0])+'.txt', os.path.join(train, str(index)+'_'+img.split('/')[-1].split('.')[0])+'.txt'))
                for img in imgs[int(0.8*len(imgs)):]:
                    os.system('cp {} {}'.format(img, os.path.join(val, str(index)+'_'+img.split('/')[-1])))
                    os.system('cp {} {}'.format(os.path.join(os.path.join(base_path, dr), img.split('/')[-1].split('.')[0])+'.txt', os.path.join(val, str(index)+'_'+img.split('/')[-1].split('.')[0])+'.txt'))
