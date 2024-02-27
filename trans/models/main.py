import argparse
from trainModel.TMOpenCv import TMOpenCv
from trainModel.QATM import QATM
from trainModel.deep_DIM import DeepDIM
from trainModel.SIFT import sift
# from trainModel.SiamFc import SiamFCModel
from trainModel.Yolo import Yolo
import os
from pathlib import Path
import sys
cwd = Path(__file__).parent

# sys.path.append(os.path.join(cwd, 'trainModel', 'SiamSE'))
# sys.path.append(os.path.join(cwd, 'trainModel'))
# from SiamSE import SiamSE
#
# sys.path.append(os.path.join(cwd, 'trainModel', 'superGlue'))
# sys.path.append(os.path.join(cwd, 'trainModel'))
# from SuperGlue import SuperGlue
#
# sys.path.append(os.path.join(cwd, 'trainModel', 'TransT'))
# from trainModel.TransT_inference import transt_model

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def main(argparse):
    parser = argparse.ArgumentParser(description='Run a model')
    parser.add_argument('--method', default='Yolo',  # TMOpenCV QATM DeepDIM SIFT SuperGlue SiamFC SiamSE TransT
                        help='Name of model')
    #     '/home/FewShotMatching/test/csvs/artificial_test_out.csv',
    # parser.add_argument('--csv_paths', default=[
                                                # '//172.17.9.133/sambashare/400G/FewShotMathing/csvs/csv_test_after_refine/artificial_test_out.csv',
                                               # ],
                        # help='The path of dataset csv file')
    parser.add_argument('--csv_paths', default=[
                                                # 'D:/Projects/FewShot/Dataset/test_VSAIv1Labeled.csv',
                                                # 'D:/Projects/FewShot/Dataset/train_VSAIv1Labeled.csv',
                                                #"E:\\CS\\Work\\FewshotTemplate\\data\\artificial_test_out.csv",
                                                os.environ.get("ARTIFCIAL_CSV_PATH", "/home/amirreza/shamgholi/tm/data/new_artificial.csv")
                                                # "/mnt/File/shamgholi/tm/data/artificial_test_out_small.csv"
                                                # 'D:/Projects/FewShot/Dataset/Test/uav123_test.csv',
                                                # 'D:/Projects/FewShot/Dataset/Test/visdrone_test_out.csv',
                                                # '//172.17.9.133/sambashare/400G/FewShotMathing/csvs/csv_VSAIv1Labeled_test_after_refine/test_VSAIv1Labeled.csv',
                                                # '//172.17.9.133/sambashare/400G/FewShotMathing/csvs/csv_VSAIv1Labeled_train_after_refine/train_VSAIv1Labeled.csv',
                                                # '//172.17.9.133/sambashare/400G/FewShotMathing/csvs/csv_VSAIv1Labeled_train_before_refine/train_VSAIv1Labeled_sharifi.csv',
                                                # '//172.17.9.133/sambashare/400G/FewShotMathing/csvs/csv_test_after_refine/artificial_test_out.csv',
                                                # '//172.17.9.133/sambashare/400G/FewShotMathing/csvs/csv_test_after_refine/GOT-10k_test_out.csv',
                                                # '//172.17.9.133/sambashare/400G/FewShotMathing/csvs/csv_test_after_refine/oxuva_c_test_out.csv',
                                                # '//172.17.9.133/sambashare/400G/FewShotMathing/csvs/csv_test_after_refine/uav123_test.csv',
                                                # '//172.17.9.133/sambashare/400G/FewShotMathing/csvs/csv_test_after_refine/visdrone_test_out.csv',
                                               ],
                        help='The path of dataset csv file')
    # parser.add_argument('--base_path', default='//172.17.9.133/sambashare/400G/FewShotMathing/Dataset/',
    parser.add_argument('--base_path', default=os.environ.get('MY_BASE_PATH', '/mnt/File/shamgholi/tm/data'),
                        help='The root path of images')
    parser.add_argument('--check', default=False, type=bool,
                        help='Save some samples for checking methad')
    parser.add_argument('--excel_path', default='results_SIFT_test.xlsx',
                        help='Path for save final result in excel')
    parser.add_argument('--select_shot', default=5, type=int,
                        help='selection shots for template matching')
    parser.add_argument('--select_sensivity', default="contrast",
                        help='selection sensitivity checking options : contrast, rotation,scale, None')
    parser.add_argument('--select_run_with_scale', default=True,
                        help='select run with scaling or no')
    parser.add_argument('--size_sahi', default=640, type=int,
                        help='set size of slicing in sahi')
    parser.add_argument('--version', default='sep',
                        help='an argument for controling the prediction mode between separate and one_sample\
                        use sep for separate and old for one_sample')
    parser.add_argument('--init', default=True, type=bool,
                        help='Do init ind1 and ind2 in QATM model or not')
    parser.add_argument('--sGPU', default=True, type=bool,
                        help='compute score on GPU or not')
    parser.add_argument('--reshape', default=True, type=bool,
                        help='Do reshape or not')
    parser.add_argument('--numba', default=False, type=bool,
                        help='have numba or not')
    parser.add_argument('--trt', default=False, type=bool,
                        help='have trt or not')
    parser.add_argument('--nms', default=0, type=int,
                        help='Count of nms iterations')
    parser.add_argument('--init_batch', default=1, type=int,
                        help='batch size for initializing the ind1 and ind2 in QATM model')
    parser.add_argument('--ref_col', default=63, type=int,
                        help='referene columns for initializing the ind1 and ind2 in QATM model')
    parser.add_argument('--ref_row', default=63, type=int,
                        help='reference rows for initializing the ind1 and ind2 in QATM model')  
    parser.add_argument('--scales', default='0.1 0.2 0.4 0.6 0.8 1 1.5',
                        help='scales for multi scale inference. Please seperate digits with space and \
                        if you have 1 scale enter with no space')
    parser.add_argument('--analyze_error_path', default='', type=str,
                        help="specify the path where preds and gts bbox going to save. if it equal to '' (empty) it doesn't visualize")
    parser.add_argument('--offline_mode', default=False, type=bool,
                        help="distinguish between mode of augmentation while training (False) or before training (True) of YOLO")


    args = parser.parse_args()
    print('>>>> SAHI SIZE is', args.size_sahi)

    if type(args.csv_paths) != list:
        print('Error! csv_paths is not a list.')
        return
    else:
        for csv_path in args.csv_paths:
            if not os.path.exists(csv_path):
                print('Error! %s is not exist.' % csv_path)
                return
    if not os.path.exists(args.base_path):
        print('Error! %s is not exist.' % args.base_path)
        return
    elif args.excel_path[-4:] != 'xlsx':
        print('Error! %s is not an exel file path.' % args.excel_path, args.excel_path[-4:])
        return

    args.excel_path = 'results_' + args.method + '.xlsx'
    print('--------------------* ' + args.method + ' on ' + str(args.csv_paths) + ' *--------------------')
    if args.method == 'TMOpenCV':
        model = TMOpenCv()
    elif args.method == 'QATM':
        model = QATM(args)
    elif args.method == 'SIFT':
        model = sift()
    elif args.method == 'SuperGlue':
        model = SuperGlue()
    elif args.method == 'DeepDIM':
        model = DeepDIM()
    elif args.method == 'SiamFC':
        model = SiamFCModel()
    elif args.method == 'SiamSE':
        model = SiamSE()
    elif args.method == 'TransT':
        model = transt_model()
    elif args.method == 'Yolo':
        model = Yolo(args)

    model.evaluate(args, threshold=0.5)


if __name__ == '__main__':
    main(argparse)

