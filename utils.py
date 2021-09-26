import os 
import time 
import random
import json 
import numpy as np 
from collections import OrderedDict

import matplotlib
matplotlib.use('agg') 
import matplotlib.pyplot as plt

############ record curve #####################
def init_dict(keys):
    d = {}
    for key in keys:
        d[key] = []
    return d

############ record curve #####################
def save_dict(info_dict, theme, save_dir):
    
    with open(os.path.join(save_dir, 'infodict-{}.json'.format(theme)), 'w') as f:
        f.write(json.dumps(info_dict))

############ record curve #####################
def curve_save(x, y, tag, yaxis, theme, save_dir):
    color = ['r', 'b', 'g', 'c', 'orange', 'lightsteelblue', 'cornflowerblue', 'indianred']
    fig = plt.figure()
    # ax = plt.subplot()
    plt.grid(linestyle='-', linewidth=0.5, alpha=0.5)
    if isinstance(tag, list):
        for i, (y_term, tag_term) in enumerate(zip(y, tag)):
            plt.plot(x, y_term, color[i], label=tag_term, alpha=0.7)
    else:
        plt.plot(x, y, color[0], label=tag, alpha=0.7)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel(yaxis, fontsize=12)
    plt.title('curve-{}'.format(theme), fontsize=14)
    plt.legend()

    fig.savefig(os.path.join(save_dir,'curve-{}.png'.format(theme)), dpi=300)
    plt.close('all') ####

def time_mark():
    time_now = int(time.time())
    time_local = time.localtime(time_now)

    dt = time.strftime('%Y%m%d-%H%M%S', time_local)
    return(dt)

def print_cz(str, f=None):
    if f is not None:
        print(str, file=f)
        if random.randint(0, 20) < 3:
            f.flush()
    print(str)

def update_lr(lr, epoch, lr_step=20, lr_gamma=0.5):
    """Sets the learning rate to the initial LR decayed by 0.5 every 20 epochs"""
    lr = lr * (lr_gamma ** (epoch // lr_step)) 
    return lr

#########
def remove_oldfile(dirname, file_keyword):
    for filename in os.listdir(dirname):
        if file_keyword in filename:
            os.remove(os.path.join(dirname, filename))
#########
def metric_macro(metric_clients):
    result = np.mean(metric_clients)
    return result

def overall_performance(dirname, F1_m, AUC_m, iters, tag='_overall_'):
    f = open(
        os.path.join(
            dirname, 
            '{}-F1-m{:.2f}-AUC-m{:.2f}-iters{:d}.txt'.format(
                    tag,
                    F1_m,
                    AUC_m,
                    iters 
                )
        ), 
        'w'
    )
    f.close()
