import argparse
from trainModel.TMOpenCv import TMOpenCv
from trainModel.QATM import QATM
from trainModel.deep_DIM import DeepDIM
from trainModel.SIFT import sift
from trainModel.SuperGlue.SuperGlue import SuperGlue
import os

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def main(argparse):
    parser = argparse.ArgumentParser(description='Run a model')
    parser.add_argument('--method', default='SIFT',
                        help='Name of model')
    #     '/home/FewShotMatching/test/csvs/artificial_test_out.csv',
    parser.add_argument('--csv_paths', default=['H:\\fewshot\\all\\all_test\\artificial\\artificial.csv'
                                               ],
                        help='The path of dataset csv file')
    parser.add_argument('--base_path', default='H:\\fewshot\\all\\all_test\\',
                        help='The root path of images')
    parser.add_argument('--check', default=False, type=bool,
                        help='Save some samples for checking methad')
    parser.add_argument('--excel_path', default='results_SIFT_test.xlsx',
                        help='Path for save final result in excel')
    parser.add_argument('--select_shot', default=10,
                        help='selection shots for template matching')

    args = parser.parse_args()

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
    if args.method == 'TMOpenCv':
        model = TMOpenCv()
    elif args.method == 'QATM':
        model = QATM()
    elif args.method == 'SIFT':
        model = sift()
    elif args.method == 'superGlue':
        model = SuperGlue()
    elif args.method == 'DeepDIM':
        model = DeepDIM()

    model.evaluate(args)


if __name__ == '__main__':
    main(argparse)

