import torch 
from torch import nn, optim
import torch.nn.functional as F
import time
import copy
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
import time
import copy
import numpy as np

from utils import init_dict, save_dict, curve_save, time_mark, print_cz, update_lr


def avg_freq(
    weights, 
    L=0.1, 
    is_conv=True
    ):
    client_num = len(weights)
    
    if is_conv:
        N, C, D1, D2 = weights[0].size()
    else:
        N = 1
        C = 1
        D1, D2 = weights[0].size()
    #print(N, C, D1, D2)
    temp_low = np.zeros((C*D1, D2*N), dtype=float)
    for i in range(client_num):
        # N, C, D1, D2 = weights[i].size()
        #weights[i] = weights[i].cpu().numpy()
        if is_conv:
            weights[i] = weights[i].permute(1, 2, 3, 0).reshape((C*D1, D2*N))
        weights[i] = weights[i].cpu().numpy()

        client_fft = np.fft.fft2(weights[i], axes=(-2, -1))
        amp_fft, pha_fft = np.abs(client_fft), np.angle(client_fft) # FFT
        low_part = np.fft.fftshift(amp_fft, axes=(-2, -1))
        temp_low += low_part
    temp_low = temp_low / 4 # avg the low-frequency

    for i in range(client_num):
        client_fft = np.fft.fft2(weights[i], axes=(-2, -1))
        amp_fft, pha_fft = np.abs(client_fft), np.angle(client_fft)
        low_part = np.fft.fftshift(amp_fft, axes=(-2, -1))

        h, w = low_part.shape
        b_h = (np.floor(h *L / 2)).astype(int)
        b_w = (np.floor(w *L / 2)).astype(int)
        c_h = np.floor(h/2.0).astype(int)
        c_w = np.floor(w/2.0).astype(int)

        h1 = c_h-b_h
        h2 = c_h+b_h
        w1 = c_w-b_w
        w2 = c_w+b_w
        low_part[h1:h2,w1:w2] = temp_low[h1:h2,w1:w2] # averaged low-freq + individual high-freq
        low_part = np.fft.ifftshift(low_part, axes=(-2, -1))

        fft_back_ = low_part * np.exp(1j * pha_fft) # 
        # get the mutated image
        fft_back_ = np.fft.ifft2(fft_back_, axes=(-2, -1))
        weights[i] = torch.FloatTensor(np.real(fft_back_))
        if is_conv:
            weights[i] = weights[i].reshape(C, D1, D2, N).permute(3, 0, 1, 2)
    return weights


def PFA(
    weights, 
    L,
    is_conv
    ):
    return avg_freq(weights=weights, L=L, is_conv=is_conv)



################# Key Function ########################
def communication(
    args, 
    server_model, 
    models, 
    original_models, 
    client_weights, 
    a_iter
    ):
    pfa_rate = args.l_rate + (a_iter / args.iters) * (0.95 - args.l_rate)
    client_num = len(client_weights) # 
    with torch.no_grad():
        # aggregate params
        for key in server_model.state_dict().keys():
            if 'bn' not in key: #not bn
                if 'conv' in key and 'weight' in key:
                    temp_weights = PFA( 
                                        [
                                            models[0].state_dict()[key].data,
                                            models[1].state_dict()[key].data,
                                            models[2].state_dict()[key].data,
                                            models[3].state_dict()[key].data
                                        ], 
                                        L=pfa_rate, 
                                        is_conv=True
                    )
                    for client_idx in range(client_num): # copy from server to each client
                        models[client_idx].state_dict()[key].data.copy_(temp_weights[client_idx])
                elif 'linear' in key and 'weight' in key:
                    temp_weights = PFA(
                                        [
                                            models[0].state_dict()[key].data,
                                            models[1].state_dict()[key].data,
                                            models[2].state_dict()[key].data,
                                            models[3].state_dict()[key].data
                                        ], 
                                        L=pfa_rate, 
                                        is_conv=False
                    )
                    for client_idx in range(client_num): # 
                        models[client_idx].state_dict()[key].data.copy_(temp_weights[client_idx])
                else:
                    print(key, '\t not bn, conv, fc layer, with param!')
                    temp = torch.zeros_like(server_model.state_dict()[key], dtype=torch.float32)
                    for client_idx in range(client_num):
                        temp += client_weights[client_idx] * models[client_idx].state_dict()[key]
                    server_model.state_dict()[key].data.copy_(temp) # non-bn layer，update the server model
                    for client_idx in range(client_num): # non-bn layer, from server to each client
                        models[client_idx].state_dict()[key].data.copy_(server_model.state_dict()[key])
 
    return server_model, models

