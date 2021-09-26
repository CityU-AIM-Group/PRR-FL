import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.nn.functional as func
from collections import OrderedDict


def conv_layer(chann_in, chann_out, k_size, p_size):
    layer = nn.Sequential(
        OrderedDict([
            ('conv', nn.Conv2d(chann_in, chann_out, kernel_size=k_size, padding=p_size, bias=False)),
            ('bn', nn.BatchNorm2d(chann_out)),
            ('relu', nn.ReLU(inplace=True))
        ])
    )
    return layer

def vgg_conv_block(in_list, out_list, k_list, p_list, pooling_k, pooling_s):
    block = [ conv_layer(in_list[i], out_list[i], k_list[i], p_list[i]) for i in range(len(in_list)) ]
    block += [ nn.MaxPool2d(kernel_size = pooling_k, stride = pooling_s)]
    return nn.Sequential(*block)

def vgg_fc_layer(size_in, size_out):
    layer = nn.Sequential(
        OrderedDict([
            ('linear', nn.Linear(size_in, size_out, bias=False)),
            ('bn', nn.BatchNorm1d(size_out)),
            ('relu', nn.ReLU(inplace=True))
        ])
    )
    return layer


class VGG16_Slim_Checked(nn.Module):
    def __init__(self, n_classes=3):
        super(VGG16_Slim_Checked, self).__init__()
        #
        # Conv blocks (BatchNorm + ReLU activation added in each block)
        self.layer1 = vgg_conv_block([3,32], [32,32], [3,3], [1,1], 2, 2)
        self.layer2 = vgg_conv_block([32,64], [64,64], [3,3], [1,1], 2, 2)
        self.layer3 = vgg_conv_block([64,128,128], [128,128,128], [3,3,3], [1,1,1], 2, 2)
        self.layer4 = vgg_conv_block([128,128,128], [128,128,128], [3,3,3], [1,1,1], 2, 2)
        self.layer5 = vgg_conv_block([128,256,256], [256,256,256], [3,3,3], [1,1,1], 2, 2)
        #
        self.avgpool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)))
        # FC layers
        self.fc1 = vgg_fc_layer(256, 64)

        # Final layer
        self.fc2 = nn.Sequential(
            OrderedDict([
                ('linear', nn.Linear(64, n_classes, bias=False)) #no_biased
            ])
        )

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        vgg16_features = self.avgpool(
            self.layer5(
                out
                )
            )
        out = vgg16_features.view(out.size(0), -1)

        out = self.fc1(out)
        out = self.fc2(out)

        return out


class VGG16_Slim_Checked_Biased(nn.Module):
    def __init__(self, n_classes=3):
        super(VGG16_Slim_Checked_Biased, self).__init__()
        #
        # Conv blocks (BatchNorm + ReLU activation added in each block)
        self.layer1 = vgg_conv_block([3,32], [32,32], [3,3], [1,1], 2, 2)
        self.layer2 = vgg_conv_block([32,64], [64,64], [3,3], [1,1], 2, 2)
        self.layer3 = vgg_conv_block([64,128,128], [128,128,128], [3,3,3], [1,1,1], 2, 2)
        self.layer4 = vgg_conv_block([128,128,128], [128,128,128], [3,3,3], [1,1,1], 2, 2)
        self.layer5 = vgg_conv_block([128,256,256], [256,256,256], [3,3,3], [1,1,1], 2, 2)
        #
        self.avgpool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)))
        # FC layers
        self.fc1 = vgg_fc_layer(256, 64)

        # Final layer
        self.fc2 = nn.Sequential(
            OrderedDict([
                ('linear', nn.Linear(64, n_classes)) #### biased
            ])
        )

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        vgg16_features = self.avgpool(
            self.layer5(
                out
                )
            )
        out = vgg16_features.view(out.size(0), -1)

        out = self.fc1(out)
        out = self.fc2(out)

        return out



if __name__ == '__main__':
    
    model = VGG16_Slim_Checked()
    print(model)
    print('---'*10)
    for key in model.state_dict().keys():
        print(key)
