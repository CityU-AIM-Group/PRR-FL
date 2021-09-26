"""
federated learning with different aggregation strategy on benchmark exp.
"""
import sys, os
from albumentations.core.composition import OneOf
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)
print(os.getcwd())


import torch
from torch import nn, optim
import torch.nn.functional as F
import time
import copy
import argparse
import numpy as np
import torchvision
import torchvision.transforms as transforms
from nets import vgg_checked
from torch.autograd import Variable
import random 
import matplotlib
matplotlib.use('agg') 
import matplotlib.pyplot as plt
import json

from utils import init_dict, save_dict, curve_save, time_mark, print_cz, update_lr, remove_oldfile
from utils import metric_macro, overall_performance
from dataset import dataset_skin
from local_train import DET, test 
from server import communication
import config


if __name__ == '__main__':
   
    args = config.get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # specific seed
    seed= args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)     
    torch.cuda.manual_seed_all(seed) 

    log_path = args.save_path + time_mark() + '_'+ args.theme + '_' + str(args.l_rate) + '_' +args.optim + '_lr' + str(args.lr) + '_step'+str(args.lr_step) + '_seed'+str(args.seed) + '_wd'+str(args.wd) +'_iters'+str(args.iters)
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    logfile = open(os.path.join(log_path,'log.txt'), 'a')
    # print args info
    config.args_info(args, device=device, logfile=logfile)
 
    SAVE_PTH_NAME = 'save'
   
    # model initialization at server
    if args.network == 'vgg_nb': # no_bias, default
        server_model = vgg_checked.VGG16_Slim_Checked(n_classes=3).to(device)
    elif args.network == 'vgg_b': # bias
        server_model = vgg_checked.VGG16_Slim_Checked_Biased(n_classes=3).to(device)
       
    
    # prepare the data    
    train_loaders, valid_loaders, test_loaders = dataset_skin.prepare_data_client(
        batch_size=args.batch, 
        data_dir = config.data_dir_split, #####
        fine_task=True,
        low_resolution=True
        )
    train_len = [len(loader) for loader in train_loaders]
    valid_len  = [len(loader) for loader in valid_loaders]
    test_len  = [len(loader) for loader in test_loaders]
    print_cz('Train loader len:  {}'.format(train_len), f=logfile)
    print_cz('Valid loader len:  {}'.format(valid_len), f=logfile)
    print_cz('Test  loader len:  {}'.format(test_len), f=logfile)

    # name of each client dataset
    datasets = ['A', 'B', 'C', 'D']

    ############ record curve #####################
    info_keys = ['test_epochs', 'test_f1', 'test_auc']
    info_dicts = {
        'A': init_dict(keys=info_keys), 
        'B': init_dict(keys=info_keys), 
        'C': init_dict(keys=info_keys), 
        'D': init_dict(keys=info_keys),
        'Average': init_dict(keys=info_keys)}

    loss_fun = nn.CrossEntropyLoss() ## loss

    client_num = len(datasets)
    client_weights = [1/client_num for i in range(client_num)] # client importance, 1/5
    models = [copy.deepcopy(server_model).to(device) for idx in range(client_num)] # personalized model list 
    models_deputy = [copy.deepcopy(server_model).to(device) for idx in range(client_num)] # deputy model list 

    start_time = time.time()
    concurrent_best_f1 = 0
    concurrent_best_iter = 0
    # start training
    for a_iter in range(args.iters): #
        iter_start_time = time.time()
        # update lr
        lr_current = update_lr(lr=args.lr, epoch=a_iter, lr_step=args.lr_step, lr_gamma=args.lr_gamma)
        # optimizer
        if (args.optim).lower() == 'sgd':
            optimizers = [optim.SGD(params=models[idx].parameters(), lr=lr_current, weight_decay=args.wd) for idx in range(client_num)]
            optimizers_deputy = [optim.SGD(params=models_deputy[idx].parameters(), lr=lr_current, weight_decay=args.wd) for idx in range(client_num)]
        elif (args.optim).lower() == 'adam':
            optimizers = [optim.Adam(params=models[idx].parameters(), lr=lr_current, weight_decay=args.wd) for idx in range(client_num)]
            optimizers_deputy = [optim.Adam(params=models_deputy[idx].parameters(), lr=lr_current, weight_decay=args.wd) for idx in range(client_num)]
        #
        DET_stages = [0 for i in range(client_num)] # DET status initialization
        for wi in range(args.wk_iters):
            print_cz("============ Train epoch {} ============".format(wi + a_iter * args.wk_iters), f=logfile)
            print_cz("=== lr_current:  {:.4e} ===".format(lr_current), f=logfile)
            # local traininig for each client
            for client_idx in range(client_num):
                model_deputy, model, train_loader, test_loader, optimizer_deputy, optimizer, DET_stage = models_deputy[client_idx], models[client_idx], train_loaders[client_idx], test_loaders[client_idx], optimizers_deputy[client_idx], optimizers[client_idx], DET_stages[client_idx]
                # DET
                DET_stages[client_idx], train_loss_, train_loss_deputy_, train_acc_, train_f1_, train_auc_ = DET(
                    args, 
                    model_deputy, 
                    model, 
                    train_loader, 
                    optimizer_deputy, 
                    optimizer, 
                    loss_fun,  
                    device, 
                    DET_stage, 
                    logfile=logfile
                    )
                
                print_cz(' {:<5s}| Train_Loss: {:.4f} | Acc: {:.2f}  F1: {:.2f}  AUC: {:.2f}'.format(datasets[client_idx] ,train_loss_, train_acc_, train_f1_, train_auc_), f=logfile)
            # test after local train
            test_average = []
            for test_idx, test_loader in enumerate(test_loaders):
                test_loss, test_acc, test_f1, test_auc  = test(models[test_idx], test_loader, loss_fun, device)
                test_average.append([test_loss, test_acc, test_f1, test_auc])
                print_cz(' {:<11s}| Test  Loss: {:.4f} | Acc: {:.2f}  F1: {:.2f}  AUC: {:.2f}'.format(datasets[test_idx], test_loss, test_acc, test_f1, test_auc), f=logfile)
                ############ record curve #####################
                if wi < args.wk_iters-1:
                    info_dicts[datasets[test_idx]]['test_epochs'].append(wi+a_iter*args.wk_iters)
                    info_dicts[datasets[test_idx]]['test_f1'].append(test_f1)
                    info_dicts[datasets[test_idx]]['test_auc'].append(test_auc)
            test_mean = np.mean(np.array(test_average), axis=0)
            print_cz(' {:<11s}| Test  Loss: {:.4f} | Acc: {:.2f}  F1: {:.2f}  AUC: {:.2f}'.format('Average', test_mean[0], test_mean[1], test_mean[2], test_mean[3]), f=logfile)
        
        # print(client_weights)
        # aggregation
        print_cz(' Aggregation ', f=logfile)
        server_model, models_deputy = communication(args, server_model, models_deputy, models, client_weights, a_iter)
            
        # start test
        test_average = []
        for test_idx, test_loader in enumerate(test_loaders):
            test_loss, test_acc, test_f1, test_auc  = test(models[test_idx], test_loader, loss_fun, device)
            test_average.append([test_loss, test_acc, test_f1, test_auc])
            print_cz(' {:<11s}| Test  Loss: {:.4f} | Acc: {:.2f}  F1: {:.2f}  AUC: {:.2f}'.format(datasets[test_idx], test_loss, test_acc, test_f1, test_auc), f=logfile)
            ############ record curve #####################
            info_dicts[datasets[test_idx]]['test_epochs'].append(wi+a_iter*args.wk_iters)
            info_dicts[datasets[test_idx]]['test_f1'].append(test_f1)
            info_dicts[datasets[test_idx]]['test_auc'].append(test_auc)
        test_mean = np.mean(np.array(test_average), axis=0)
        print_cz(' {:<11s}| Test  Loss: {:.4f} | Acc: {:.2f}  F1: {:.2f}  AUC: {:.2f}'.format('Average', test_mean[0], test_mean[1], test_mean[2], test_mean[3]), f=logfile)
        # model selection on valid set
        valid_average = []
        for valid_idx, valid_loader in enumerate(valid_loaders):
            valid_loss, valid_acc, valid_f1, valid_auc  = test(models[valid_idx], valid_loader, loss_fun, device)
            valid_average.append([valid_loss, valid_acc, valid_f1, valid_auc])
            print_cz(' {:<11s}| Valid  Loss: {:.4f} | Acc: {:.2f}  F1: {:.2f}  AUC: {:.2f}'.format(datasets[valid_idx], valid_loss, valid_acc, valid_f1, valid_auc), f=logfile)
        valid_mean = np.mean(np.array(valid_average), axis=0)
        print_cz(' {:<11s}| Valid  Loss: {:.4f} | Acc: {:.2f}  F1: {:.2f}  AUC: {:.2f}'.format('Average', valid_mean[0], valid_mean[1], valid_mean[2], valid_mean[3]), f=logfile)
        if valid_mean[2] > concurrent_best_f1 and a_iter >= 0.8*args.iters:
            concurrent_best_f1 = valid_mean[2]
            concurrent_best_iter = wi+a_iter*args.wk_iters
            # valid save
            for i in range(len(models)):
                remove_oldfile(dirname=log_path, file_keyword='_valid_client{:d}'.format(i))
                torch.save(
                    models[i].state_dict(), 
                    os.path.join(
                        log_path, 
                        SAVE_PTH_NAME+'_valid_client{:d}_model-F1-{:.2f}-AUC-{:.2f}-iters-{:d}.pth'.format(i, test_average[i][2], test_average[i][3], a_iter)
                    )
                )
            #
            remove_oldfile(dirname=log_path, file_keyword='_valid_overall')
            F1_clients = np.array(test_average)[:, 2] # F1
            AUC_clients = np.array(test_average)[:, 3] # AUC
            overall_performance(
                dirname=log_path,
                tag=SAVE_PTH_NAME+'_valid_overall',
                F1_m=metric_macro(F1_clients),
                AUC_m=metric_macro(AUC_clients),
                iters=a_iter 
            )
            #
        #
        curve_save(x=info_dicts[datasets[0]]['test_epochs'], y=[info_dicts[datasets[0]]['test_auc'], info_dicts[datasets[1]]['test_auc'], info_dicts[datasets[2]]['test_auc'], info_dicts[datasets[3]]['test_auc']], tag=['client_A', 'client_B', 'client_C', 'client_D'], yaxis='Performance', theme='Test-AUC-all-client', save_dir=log_path)
        curve_save(x=info_dicts[datasets[0]]['test_epochs'], y=[info_dicts[datasets[0]]['test_f1'], info_dicts[datasets[1]]['test_f1'], info_dicts[datasets[2]]['test_f1'], info_dicts[datasets[3]]['test_f1']], tag=['client_A', 'client_B', 'client_C', 'client_D'], yaxis='Performance', theme='Test-F1-all-client', save_dir=log_path)
        print_cz(' Iter time:  {:.1f} min'.format((time.time()-iter_start_time)/60.0), f=logfile)
    # end of FL
    print_cz(' Total time:  {:.2f} h'.format((time.time()-start_time)/3600.0), f=logfile)
    # summary
    print_cz(' Saving the checkpoint to {}'.format(log_path), f=logfile)
    print_cz(' concurrent best iter {}'.format(str(concurrent_best_iter)), f=logfile)
    torch.save(
        server_model.state_dict(), 
        os.path.join(
            log_path, 
            SAVE_PTH_NAME+'_end_server_model.pth'
        )
    )
    # final save
    for i in range(len(models)):
        torch.save(
            models[i].state_dict(), 
            os.path.join(
                log_path, 
                SAVE_PTH_NAME+'_end_client{:d}_model-F1-{:.2f}-AUC-{:.2f}.pth'.format(i, test_average[i][2], test_average[i][3])
            )
        )
    #
    remove_oldfile(dirname=log_path, file_keyword='_end_overall')
    F1_clients = np.array(test_average)[:, 2] # F1
    AUC_clients = np.array(test_average)[:, 3] # AUC
    overall_performance(
        dirname=log_path,
        tag=SAVE_PTH_NAME+'_end_overall',
        F1_m=metric_macro(F1_clients),
        AUC_m=metric_macro(AUC_clients),
        iters=a_iter 
    )
    #         
    logfile.flush()
    logfile.close()


