import os 
import time 
import torch 
import argparse

from utils import print_cz

data_dir_split = './data/'
csv_folder = './csv_folder/'

def get_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('--network', type = str, default='vgg_nb', help = 'classification model')
    # training settings
    parser.add_argument('--optim', type=str, default='sgd')
    parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')
    parser.add_argument("--lr_step", type=int, default=5)
    parser.add_argument("--lr_gamma", type=float, default=0.5)
    parser.add_argument("--wd", type=float, default=1e-6)
    parser.add_argument('--batch', type = int, default= 16, help ='batch size')
    parser.add_argument('--iters', type = int, default=50, help = 'iterations for communication')
    parser.add_argument('--wk_iters', type = int, default=5, help = 'optimization iters in local worker between communication')
    # fedprox and fedbn
    # parser.add_argument('--weight', type = bool, default=True, help='class imbalance weight')
    # parser.add_argument('--mu', type=float, default=1e-2, help='The hyper parameter for fedprox')
    # PFA
    parser.add_argument("--l_rate", type=float, default=0.7)
    # DET
    parser.add_argument('--alpha1', type = float, default= 0.7, help = 'alpha1')
    parser.add_argument('--alpha2', type = float, default= 0.9, help = 'alpha2')
    
    parser.add_argument('--seed', type = int, default=1, help = 'seed')
    parser.add_argument('--theme', type = str, default='', help='comments for training')
    parser.add_argument('--save_path', type = str, default='./experiment/', help='path to save the checkpoint')
    
    args = parser.parse_args()
    return args 

def args_info(args, device, logfile=None):
    print_cz(os.getcwd(), f=logfile)
    print_cz('Device: {}'.format(device), f=logfile)
    print_cz('==={}==='.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())), f=logfile)
    print_cz('===Setting===', f=logfile)
    print_cz('    l_rate: {}'.format(args.l_rate), f=logfile)
    print_cz('    optim: {}'.format(args.optim), f=logfile)
    print_cz('    lr: {}'.format(args.lr), f=logfile)
    print_cz('    lr_step: {}'.format(args.lr_step), f=logfile)
    print_cz('    iters: {}'.format(args.iters), f=logfile)
    print_cz('    wk_iters: {}'.format(args.wk_iters), f=logfile)
    print_cz('    network: {}'.format(args.network), f=logfile)
    print_cz('    alpha1: {}'.format(args.alpha1), f=logfile)
    print_cz('    alpha2: {}'.format(args.alpha2), f=logfile)
    

