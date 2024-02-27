import os
import glob
import argparse
import cv2

def plot_bbox_on_img(rect, img, save_path):
    imgfile = cv2.imread(img)
    f = open(rect, "r")
    lines = f.readlines()
    f.close()
    for index, line in enumerate(lines):
        xyhw = [float(elem)*imgfile.shape[1] if index % 2 == 0 else float(elem)*imgfile.shape[0] for index, elem in enumerate(line[3:-1].split(" "))]
        if line.startswith("1"):
            cv2.rectangle(imgfile, (int(xyhw[0] - (xyhw[2] / 2)), int(xyhw[1] - (xyhw[-1] / 2))),
                          (int(xyhw[0] + (xyhw[2] / 2)), int(xyhw[1] + (xyhw[-1] / 2))), (0, 0, 255), thickness=2)
        elif line.startswith("0"):
            cv2.rectangle(imgfile, (int(xyhw[0] - (xyhw[2] / 2)), int(xyhw[1] - (xyhw[-1] / 2))),
                          (int(xyhw[0] + (xyhw[2] / 2)), int(xyhw[1] + (xyhw[-1] / 2))), (0, 255, 0), thickness=2)
        else:
            cv2.rectangle(imgfile, (int(xyhw[0] - (xyhw[2] / 2)), int(xyhw[1] - (xyhw[-1] / 2))),
                          (int(xyhw[0] + (xyhw[2] / 2)), int(xyhw[1] + (xyhw[-1] / 2))), (255, 0, 0), thickness=2)
    cv2.imwrite(save_path, imgfile)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--data", required=True,
                    help="data path contains images")
    ap.add_argument("-l", "--labels", required=True,
                    help="data path contains txt files")
    ap.add_argument("-o", "--output", required=True,
                    help="path to the output directory")

    args = vars(ap.parse_args())

    base_path = args["data"]
    txtsPath = args["labels"]

    output_path = args["output"]
    if not os.path.isdir(output_path):
        os.mkdir(output_path)



    txts = glob.glob(os.path.join(txtsPath, '*.txt'))

    for txt in txts:
        save_path = os.path.join(output_path, '{}.png'.format(txt.split('/')[-1].split('.')[0]))
        plot_bbox_on_img(txt, os.path.join(base_path, '{}.png'.format(txt.split('/')[-1].split('.')[0])), save_path)
