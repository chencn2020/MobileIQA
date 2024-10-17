import os
import argparse
import random
import sys
import time
import shutil

from utils import log_writer
from utils import iqa_solver
import numpy as np
import torch


def setup_seed(seed):
    '''
        Fix the random seed for result reproduction.
    '''
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def printArgs(args, savePath):
    with open(os.path.join(savePath, 'args_info.log'), 'w') as f:
        print('--------------args----------------')
        f.write('--------------args----------------\n')
        for arg in vars(args):
            print(format(arg, '<20'), format(str(getattr(args, arg)), '<'))  # str, arg_type
            f.write('{}\t{}\n'.format(format(arg, '<20'), format(str(getattr(args, arg)), '<')))  # str, arg_type

        print('----------------------------------')
        f.write('----------------------------------')


def init(config):
    loger_path = os.path.join(config.save_path, 'log')
    if not os.path.isdir(loger_path):
        os.makedirs(loger_path)
    sys.stdout = log_writer.Logger(os.path.join(loger_path, 'training_logs.log'))
    print('All train and test data will be saved in: ', config.save_path)
    print('----------------------------------')
    print('Begin Time: ', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    printArgs(config, loger_path)
    os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu_id
    setup_seed(config.seed)

    # Save the traning files.
    file_backup = os.path.join(config.save_path, 'training_files')
    if not os.path.isdir(file_backup):
        os.makedirs(file_backup)
    shutil.copy(os.path.basename(__file__), os.path.join(file_backup, os.path.basename(__file__)))

    save_folder_list = ['models', 'utils']
    for save_folder in save_folder_list:
        save_folder_path = os.path.join(file_backup, save_folder)
        if os.path.exists(save_folder_path):
            shutil.rmtree(save_folder_path)
        shutil.copytree(save_folder, save_folder_path)

import json
def get_data(dataset, data_path='./utils/dataset/dataset_info.json'):
    '''
        Load dataset information from the json file.
    '''
    with open(data_path, 'r') as data_info:
        data_info = json.load(data_info)
    path, img_num = data_info[dataset]
    img_num = list(range(img_num))
    random.shuffle(img_num)
    
    train_index = img_num[0:int(round(0.8 * len(img_num)))]
    test_index = img_num[int(round(0.8 * len(img_num))):len(img_num)]

    return path, train_index, test_index

def main(config):
    init(config)
        
    # Begin Traning.
    path, train_index, test_index = get_data(dataset=config.dataset)
    print('Train idx: ', train_index[:10])
    print('Test idx: ', test_index[:10])
    solver = iqa_solver.Solver(config, path, train_index, test_index)
    krocc, srocc, plcc = solver.train()

    print('KROCC: {}, SROCC: {}, PLCC: {}'.format(krocc, srocc, plcc))
    print('End Time: ', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', dest='seed', type=int, default=570908, help='Random seeds for result reproduction.')
    parser.add_argument('--gpu_id', dest='gpu_id', type=str, default='0', help='GPU Id for traning.')
    parser.add_argument('--model', dest='model', type=str, default='MobileVit_IQA')

    # model related
    parser.add_argument('--backbone', dest='backbone', type=str, default='vit_base_patch8_224', help='The backbone for MoNet.')
    parser.add_argument('--mal_num', dest='mal_num', type=int, default=3, help='The number of the MAL modules.')

    # data related
    parser.add_argument('--dataset', dest='dataset', type=str, default='uhdiqa', help='Support datasets: livec|koniq10k|bid|spaq|uhdiqa')

    # training related
    parser.add_argument('--loss', dest='loss', type=str, default='MSE', help='Loss function to use. Support losses: MAE|MSE.')
    parser.add_argument('--lr', dest='lr', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--weight_decay', dest='weight_decay', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=11, help='Batch size')
    parser.add_argument('--epochs', dest='epochs', type=int, default=10, help='Epochs for training')
    parser.add_argument('--T_max', dest='T_max', type=int, default=50, help='Hyper-parameter for CosineAnnealingLR')
    parser.add_argument('--eta_min', dest='eta_min', type=int, default=0, help='Hyper-parameter for CosineAnnealingLR')

    # result related
    parser.add_argument('--save_path', dest='save_path', type=str, default='./Running_Test', help='The path where the model and logs will be saved.')
    parser.add_argument('--teacher_pkl', dest='teacher_pkl', type=str, default=None)

    config = parser.parse_args()
    main(config)
