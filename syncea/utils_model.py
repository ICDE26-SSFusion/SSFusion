import hashlib
import time
import os
import numpy as np
import scipy.stats as stats
import os
import sys
import re
import datetime



import numpy

import torch
from torch.optim.lr_scheduler import _LRScheduler
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# import horovod
from random import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np   
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import time
import random
import scipy as sp
from scipy import stats
from scipy.stats import norm
import matplotlib.mlab as mlab
import torchvision.models as models
import torch
import torch.nn.functional as F
import scipy.stats
import torchvision.models as models

from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import make_interp_spline

import matplotlib.pyplot as plt 
import numpy as np
# from utils import get_network

def gen_random_id():
    id_ = hashlib.sha256()
    id_.update(str(time.time()))
    return id_.hexdigest()

def create_path(relative_path):
    dirname = os.path.dirname(__file__)
    filename = os.path.join(dirname, relative_path)
    if not os.path.isdir(filename):
        try:
            #os.mkdir(filename)
            os.makedirs(filename)
        except:
            pass

def update_fontsize(ax, fontsize=12.):
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                             ax.get_xticklabels() + ax.get_yticklabels()):
                item.set_fontsize(fontsize)

def autolabel(rects, ax, label, rotation=90):
    """
    Attach a text label above each bar displaying its height
    """
    for rect in rects:
        height = rect.get_y() + rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.03*height,
            label,
            ha='center', va='bottom', rotation=rotation)

def topk(tensor, k):
    indexes = np.abs(tensor).argsort()[-k:]
    return indexes, tensor[indexes]

def get_approximate_sigma_scale(density):
    sigma_scale = 1
    if density > 0.7:
        sigma_scale = 0.5
    elif density <= 0.7 and density > 0.05:
        sigma_scale = 1.5
    elif density <= 0.05 and density > 0.01:
        sigma_scale = 2.0
    else:
        sigma_scale = 3.0
    return sigma_scale



def force_insert_item(d, key, val):
    if key not in d:
        d[key] = []
    d[key].append(val)


s=2.18896957e-10 #P102-100

GbE_multi_p_ab_small = {
        2: (1.6e-3, 1.0e-8),
        4: (2.7e-3, 1.3e-8),
        8: (4.0e-3, 1.5e-8),
        
        16: (1.7e-3, 1.7e-8) #  ImageNet
        
        }


GbE_multi_p_ab_large = {
        2: (4.4e-3, 5.8e-9),
        4: (5.6e-3, 7.4e-9),
        8: (7.68e-3, 8.2e-9),
        16: (2.1e-3, 1.7e-8) # good for imagenet
        }

tenGbE_multi_p_ab = {
        2: (1.5e-5, 5.7e-11),
        4: (3.6e-5, 1.1e-10),
        8: (8.5e-5, 1.4e-10),
        16: (1.4e-4, 2.0e-10)
        }
# Aliyun cloud

# Startup, BackwardTime, CommunicationTime, 
_10k_time = {
        2: (1.5e-5, 5.7e-11),
        4: (3.6e-5, 1.1e-10),
        8: (8.5e-5, 1.4e-10),
        16: (1.4e-4, 2.0e-10),
        32: (1.4e-4, 2.0e-10),
        64: (1.4e-4, 2.0e-10)
    }

_100k_time = {
        2: (1.5e-5, 5.7e-11),
        4: (3.6e-5, 1.1e-10),
        8: (8.5e-5, 1.4e-10),
        16: (1.4e-4, 2.0e-10),
        32: (1.4e-4, 2.0e-10),
        64: (1.4e-4, 2.0e-10)
    }

_100k_time = {
        2: (1.5e-5, 5.7e-11),
        4: (3.6e-5, 1.1e-10),
        8: (8.5e-5, 1.4e-10),
        16: (1.4e-4, 2.0e-10),
        32: (1.4e-4, 2.0e-10),
        64: (1.4e-4, 2.0e-10)
    }

def calculation_backward_time_local_8_nodes(sum_numel, numel, net_name):
    if net_name=='resnet50':
        # ResNet-50
        # backward_time = 0.0403665948358382
        backward_time = 0.03523650461313676
    elif net_name=='resnet152':
        # ResNet-152
        backward_time = 0.07931486319522468
    elif net_name=='resnet101':
        # ResNet-101
        backward_time= 0.05694400899264277   
 
    backward_time_per = backward_time/sum_numel    
    backward_time_numel = numel * backward_time_per    
    return backward_time_numel


def calculation_communication_time_local_8_nodes(numel, density):

    from scipy.interpolate import interp1d
    # kB
    x_size =int(numel*density*4/1024)
    
    
    x = [0.1, 1, 10, 100, 1000, 10000, 100000, 150000]
    y = [0.005115201187133789, 0.005463216590881347, 0.0055778751373291016, 0.005975676536560059, 0.019652485847473145, 0.11236395835876464, 1.005709457397461, 1.5160596132278443]
    
    
    if x_size< x[0]:
        return y[0]
    
    f = interp1d(x, y)
 
    new_points =x_size
    result = f(new_points) 

    
    return result


def calculation_compression_time_local_8_nodes(numel):

    from scipy.interpolate import interp1d
    x_size =max(1, int(numel*1.0*4/1024)) 
    
    
    x = [10, 100, 1000, 10000, 100000, 1000000]
    dgc_array_time_y =  [ 0.00034046173095703125, 0.0003764629364013672, 0.000993967056274414, 0.00259234619140625, 0.010454635620117188, 0.0390236186981201]
    topk_array_time_y =  [ 7.677078247070312e-05, 0.0001761913299560547, 0.0003170967102050781, 0.00024127960205078125, 0.0003452301025390625, 0.0015995502471923828]
    # gaussiank_array_time_y =  [ 0.0005277561187744141, 0.0005738735198974609, 0.0008003711700439453, 0.0009584426879882812, 0.0021195411682128906, 0.014198064804077148]
    redsync_array_time_y =  [ 0.00041556358337402344, 0.00039386749267578125, 0.0007512569427490234, 0.0009708404541015625, 0.0031561851501464844, 0.026698827743530273]
    randomk_array_time_y =   [ 0.00025153160095214844, 0.0001316070556640625, 0.0003561973571777344, 0.00078582763671875, 0.0019142627716064453, 0.013434171676635742]
    sidcoexp_array_time_y = [ 0.00034332275390625, 0.0002334117889404297, 0.00045180320739746094, 0.0004067420959472656, 0.0013599395751953125, 0.011028528213500977]
    
    if x_size< x[0]:
        return topk_array_time_y[0]
     
    f = interp1d(x, topk_array_time_y)

    new_points = x_size
    
    result = f(new_points) 

    
    return result

startup_time= 0.003780956268310547


gradient_size_array= [100, 204800, 2048, 2048, 1048576, 512, 512, 2359296, 512, 512, 1048576, 2048, 2048, 1048576, 512, 512, 2359296, 512, 512, 1048576, 2048, 2048, 2097152, 2048, 2048, 1048576, 512, 512, 2359296, 512, 512, 524288, 1024, 1024, 262144, 256, 256, 589824, 256, 256, 262144, 1024, 1024, 262144, 256, 256, 589824, 256, 256, 262144, 1024, 1024, 262144, 256, 256, 589824, 256, 256, 262144, 1024, 1024, 262144, 256, 256, 589824, 256, 256, 262144, 1024, 1024, 262144, 256, 256, 589824, 256, 256, 262144, 1024, 1024, 524288, 1024, 1024, 262144, 256, 256, 589824, 256, 256, 131072, 512, 512, 65536, 128, 128, 147456, 128, 128, 65536, 512, 512, 65536, 128, 128, 147456, 128, 128, 65536, 512, 512, 65536, 128, 128, 147456, 128, 128, 65536, 512, 512, 131072, 512, 512, 65536, 128, 128, 147456, 128, 128, 32768, 256, 256, 16384, 64, 64, 36864, 64, 64, 16384, 256, 256, 16384, 64, 64, 36864, 64, 64, 16384, 256, 256, 16384, 256, 256, 16384, 64, 64, 36864, 64, 64, 4096, 64, 64, 1728]



# Optimal gradient merging scheme
def optimal_gradient_merging_0101(gradient_size_array, net_name, density = 0.1):
    density_all=1.0
    
    len_gradient_size = len(gradient_size_array)
    gradient_size_sum = sum(gradient_size_array)    
      
    group =[]
    group_len =0
    group_size =0
    
    min_diff = 1
    min_group_len =0
    min_group_len_array= []
    merging_time_compression_array = []
    # merging_communication_time_array = []
    group_backward_time_size_array = []    
    group_compression_time_array = []
    print(gradient_size_array)
    flag= False
    # Step-1: Greedy algorithm generates suboptimal merging schemes
    for i, s in enumerate(gradient_size_array):
        group.append(s)
        group_len = len(group)
        group_size = sum(group)
        
        merging_time_size_compression = calculation_communication_time_local_8_nodes(group_size, density) + calculation_compression_time_local_8_nodes(group_size)
        merging_time_wait_size_nocompression = calculation_communication_time_local_8_nodes(group_size, density_all) + calculation_backward_time_local_8_nodes(gradient_size_sum, group_size, net_name)         
        group_compression_time = calculation_compression_time_local_8_nodes(group_size)
        
        group_layer_wise_time_nocompression = 0
        group_layer_wise_time_nocompression = (group_len-1) * startup_time +calculation_communication_time_local_8_nodes(group_size, density_all) 
        group_backward_time_size =  calculation_backward_time_local_8_nodes(gradient_size_sum, group_size, net_name) 

        diff = merging_time_wait_size_nocompression -group_layer_wise_time_nocompression
        # Greedy algorithm generates suboptimal merging schemes
        if diff < 0:
            abs_diff = abs(diff)
            abs_diff = diff            
            # if abs_diff < min_diff:
            if True:
                       
                min_group_len = group_len
                min_group_len_array.append(min_group_len)
            
                merging_time_compression_array.append(merging_time_size_compression)
                group_compression_time_array.append(group_compression_time)
                
                # merging_communication_time_array.append(merging_communication_time_size)
                group_backward_time_size_array.append(group_backward_time_size)

                group = []
                group_len = 0
                group_size = 0
                min_diff = 1
                min_group_len = 0
            
                merging_communication_time_array_sum =sum(merging_time_compression_array)                
                sub_backward_time= calculation_backward_time_local_8_nodes(gradient_size_sum, sum(gradient_size_array[min_group_len_array[0]:]), net_name) 
                sub_backward_time = sub_backward_time+sum(group_compression_time_array[1:])                    
                pre_merging_time_compression =  merging_time_size_compression
                

    print('min_group_len_array = ', min_group_len_array)
    print('min_group_len_array_sum = ', sum(min_group_len_array))
    print('len(min_group_len_array) = ', len(min_group_len_array))
    print('len(group_backward_time_size_array) = ', len(group_backward_time_size_array))
    # print('len(merging_communication_time_array) = ', len(merging_communication_time_array))
    
    print('sum(merging_time_compression_array)= ', sum(merging_time_compression_array))
    print('sum(group_compression_time_array)= ', sum(group_compression_time_array))
    print('sum(group_backward_time_size_array)= ', sum(group_backward_time_size_array))


    # Step-2: Further merge small buffers 
    groups = []
    groups_new = []
    buffer_backward_time = 0
    buffer_backward_compression_time_array = []
    buffer_backward_compression_time_array_new =[]        
    merging_time_compression_array_new = []    
    groups_new.append(min_group_len_array[0])
    
    i=1
    merging_time_compression_array_new.append(merging_time_compression_array[0])
    print('len(min_group_len_array)= ', len(min_group_len_array))
    while i < len(min_group_len_array):
        
        current_merging_time = merging_time_compression_array_new[-1]
        flag_ =True
        count= 0
        while flag_ :
            if  count+i< len(min_group_len_array):
                buffer_backward_time = group_backward_time_size_array[count+i] + group_compression_time_array[count+i]
                # buffer_backward_time = group_backward_time_size_array[count+i] 
                buffer_backward_compression_time_array.append(buffer_backward_time)
            else:                
                flag_ = False
                count_= count
                # print('count__ = ', count, 'i__ = ', i)
                for j in range(0, count_):                    
                    merging_time_compression_array_new.append(merging_time_compression_array[i+j])
                    groups_new.append(min_group_len_array[i+j])
                i = i+count
                # print('i__ = ', i)
                break 
            if sum(buffer_backward_compression_time_array)> current_merging_time:
                if count>0:
                    merging_time_compression_array_new.append(sum(merging_time_compression_array[i:count+i]))
                    groups_new.append(sum(min_group_len_array[i:count+i]))
      
                i= count+i                
                flag_ = False
            else:
                count = count +1
               

        if not flag_ and count == 0:   
            # print('i =', i)       
            merging_time_compression_array_new.append(merging_time_compression_array[i])
            groups_new.append(min_group_len_array[i])
            i =i+1
        
        
            
    print('groups_new= ', groups_new)  
    print('sum(groups_new)= ', sum(groups_new))  
     
    
    # Step-3: Reduce the number of non overlapping buffers
    groups_new_=[]
    group_backward_time_sum = sum(group_compression_time_array[1:])+ sum(group_backward_time_size_array[1:])
    merging_communication_time_array_sum =0 
    sum_x= 0
    for i, g in enumerate(groups_new):
        groups_new_.append(g)

        new_buffer_size = sum(gradient_size_array[sum(groups_new[:i]): sum(groups_new[:i+1])])
        new_buffer_size_ = sum(gradient_size_array[: sum(groups_new[:i+1])])
        # merging_communication_time_array_sum = merging_communication_time_array_sum+ calculation_communication_time_local_8_nodes(new_buffer_size, density)        
        merging_communication_time_array_sum = calculation_communication_time_local_8_nodes(new_buffer_size_, density) 

        # Reduce the number of non overlapping buffers
        if merging_communication_time_array_sum > group_backward_time_sum:
            last_min_group_len = len_gradient_size - sum(groups_new_) 
            groups_new_.append(last_min_group_len)
            break
    groups_new_[0] =groups_new_[0] +1
    print('groups_new_= ', groups_new_)  
    print('sum(groups_new_)= ', sum(groups_new_)) 
    

    if density<0.1:
        # groups_new_= [3, 5, 9, 17, 60, 68]
        groups_new_ = [10, 15, 25, 28, 28, 28, 28]
    
    return groups_new_



def topk_perf_model(x, s=s):
    """
    x is the number of parameters
    Return: s * x * log2(x)
    """
    if x == 0.0:
        return 0.0
    return s * x * np.log2(x)


def allgather_perf_model(x, P, density=0.001, eth='GbE'):
    """
    x is the number of parameters
    Return: t = a + b * x
    """
    if x == 0:
        return 0.0
    size = x * P * 4 * density
    if size >= 1024*1024:
        multi_p_ab = GbE_multi_p_ab_large
    else:
        multi_p_ab = GbE_multi_p_ab_small
    a, b = multi_p_ab[P]
    return (a + b * size) * 2

def predict_density_with_size_and_computation(m, comp_time, P):
    alpha = 4*0.436e-3
    beta =  4*9e-6*1e-3
    def _denseallreduce_model(P, m):
        return 2*(P-1)*alpha + 2* (P-1)/P * m * beta

    def _sparseallreduce_model(P, m, rho=0.001):
        return np.log2(P) + 2 * (P - 1) * rho * m * beta

    def _proper_rho_with_sparse_allreduce(P, m, comp_time):
        rho = 0.001
        t = comp_time - np.log2(P) * alpha 
        if t <= 0:
            return rho 
        rho = t/ (2*(P-1)*beta*m)
        if rho > 1.0:
            rho = 0.05
        rho = max(rho, 0.001)
        return rho
    return 0.001


def predict_allreduce_time_with_size(alpha, beta, size, P):
    if size == 0:
        return 0.0
    return alpha + beta * size 

def gen_threshold_from_normal_distribution(p_value, mu, sigma):
    zvalue = stats.norm.ppf((1-p_value)/2)
    return mu+zvalue*sigma, mu-zvalue*sigma

def check_unique(l):
    d = {}
    for k in l:
        if k in d:
            print('element: %s is duplicate in %s' % (k, l))
            return False
        d[k] = 1
    return True




def get_network(args):
    """ return given network
    """
    if args.model_net == 'resnet18':
        
        from models.resnet import resnet18
        net = resnet18()
    elif args.model_net == 'resnet34':
        from models.resnet import resnet34
        net = resnet34()
    elif args.model_net == 'resnet50':
        from models.resnet import resnet50
        net = resnet50()
    elif args.model_net == 'resnet101':
        from models.resnet import resnet101
        net = resnet101()
    elif args.model_net == 'resnet152':
        from models.resnet import resnet152
        net = resnet152()
    
    elif args.model_net == 'vgg16':
        from models.vgg import vgg16_bn
        net = vgg16_bn()
    elif args.model_net == 'vgg13':
        from models.vgg import vgg13_bn
        net = vgg13_bn()
    elif args.model_net == 'vgg11':
        from models.vgg import vgg11_bn
        net = vgg11_bn()
    elif args.model_net == 'vgg19':
        from models.vgg import vgg19_bn
        net = vgg19_bn()
    elif args.model_net == 'densenet121':
        from models.densenet import densenet121
        net = densenet121()
    elif args.model_net == 'densenet161':
        from models.densenet import densenet161
        net = densenet161()
    elif args.model_net == 'densenet169':
        from models.densenet import densenet169
        net = densenet169()
    elif args.model_net == 'densenet201':
        from models.densenet import densenet201
        net = densenet201()
    elif args.model_net == 'googlenet':
        from models.googlenet import googlenet
        net = googlenet()
    elif args.model_net == 'inceptionv3':
        from models.inceptionv3 import inceptionv3
        net = inceptionv3()
    elif args.model_net == 'inceptionv4':
        from models.inceptionv4 import inceptionv4
        net = inceptionv4()
    elif args.model_net == 'inceptionresnetv2':
        from models.inceptionv4 import inception_resnet_v2
        net = inception_resnet_v2()
    elif args.model_net == 'xception':
        from models.xception import xception
        net = xception()
        
    elif args.model_net == 'preactresnet18':
        from models.preactresnet import preactresnet18
        net = preactresnet18()
    elif args.model_net == 'preactresnet34':
        from models.preactresnet import preactresnet34
        net = preactresnet34()
    elif args.model_net == 'preactresnet50':
        from models.preactresnet import preactresnet50
        net = preactresnet50()
    elif args.model_net == 'preactresnet101':
        from models.preactresnet import preactresnet101
        net = preactresnet101()
    elif args.model_net == 'preactresnet152':
        from models.preactresnet import preactresnet152
        net = preactresnet152()
    elif args.model_net == 'resnext50':
        from models.resnext import resnext50
        net = resnext50()
    elif args.model_net == 'resnext101':
        from models.resnext import resnext101
        net = resnext101()
    elif args.model_net == 'resnext152':
        from models.resnext import resnext152
        net = resnext152()
    elif args.model_net == 'shufflenet':
        from models.shufflenet import shufflenet
        net = shufflenet()
    elif args.model_net == 'shufflenetv2':
        from models.shufflenetv2 import shufflenetv2
        net = shufflenetv2()
    elif args.model_net == 'squeezenet':
        from models.squeezenet import squeezenet
        net = squeezenet()
    elif args.model_net == 'mobilenet':
        from models.mobilenet import mobilenet
        net = mobilenet()
    elif args.model_net == 'mobilenetv2':
        from models.mobilenetv2 import mobilenetv2
        net = mobilenetv2()
    elif args.model_net == 'nasnet':
        from models.nasnet import nasnet
        net = nasnet()
    elif args.model_net == 'attention56':
        from models.attention import attention56
        net = attention56()
    elif args.model_net == 'attention92':
        from models.attention import attention92
        net = attention92()
    elif args.model_net == 'seresnet18':
        from models.senet import seresnet18
        net = seresnet18()
    elif args.model_net == 'seresnet34':
        from models.senet import seresnet34
        net = seresnet34()
    elif args.model_net == 'seresnet50':
        from models.senet import seresnet50
        net = seresnet50()
    elif args.model_net == 'seresnet101':
        from models.senet import seresnet101
        net = seresnet101()
    elif args.model_net == 'seresnet152':
        from models.senet import seresnet152
        net = seresnet152()
    elif args.model_net == 'wideresnet':
        from models.wideresidual import wideresnet
        net = wideresnet()
    elif args.model_net == 'stochasticdepth18':
        from models.stochasticdepth import stochastic_depth_resnet18
        net = stochastic_depth_resnet18()
    elif args.model_net == 'stochasticdepth34':
        from models.stochasticdepth import stochastic_depth_resnet34
        net = stochastic_depth_resnet34()
    elif args.model_net == 'stochasticdepth50':
        from models.stochasticdepth import stochastic_depth_resnet50
        net = stochastic_depth_resnet50()
    elif args.model_net == 'stochasticdepth101':
        from models.stochasticdepth import stochastic_depth_resnet101
        net = stochastic_depth_resnet101()

    else:
        print('the network name you have entered is not supported yet')
        sys.exit()

    if args.gpu: #use_gpu
        net = net.cuda()

    return net


def get_training_dataloader(mean, std, batch_size=16, num_workers=2, shuffle=True):
    """ return training dataloader
    Args:
        mean: mean of cifar100 training dataset
        std: std of cifar100 training dataset
        path: path to cifar100 training python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
    Returns: train_data_loader:torch dataloader object
    """

    transform_train = transforms.Compose([
        #transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    #cifar100_training = CIFAR100Train(path, transform=transform_train)
    cifar100_training = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    cifar100_training_loader = DataLoader(
        cifar100_training, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return cifar100_training_loader

def get_test_dataloader(mean, std, batch_size=16, num_workers=2, shuffle=True):
    """ return training dataloader
    Args:
        mean: mean of cifar100 test dataset
        std: std of cifar100 test dataset
        path: path to cifar100 test python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
    Returns: cifar100_test_loader:torch dataloader object
    """

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    #cifar100_test = CIFAR100Test(path, transform=transform_test)
    cifar100_test = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    cifar100_test_loader = DataLoader(
        cifar100_test, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return cifar100_test_loader

def compute_mean_std(cifar100_dataset):
    """compute the mean and std of cifar100 dataset
    Args:
        cifar100_training_dataset or cifar100_test_dataset
        witch derived from class torch.utils.data

    Returns:
        a tuple contains mean, std value of entire dataset
    """

    data_r = numpy.dstack([cifar100_dataset[i][1][:, :, 0] for i in range(len(cifar100_dataset))])
    data_g = numpy.dstack([cifar100_dataset[i][1][:, :, 1] for i in range(len(cifar100_dataset))])
    data_b = numpy.dstack([cifar100_dataset[i][1][:, :, 2] for i in range(len(cifar100_dataset))])
    mean = numpy.mean(data_r), numpy.mean(data_g), numpy.mean(data_b)
    std = numpy.std(data_r), numpy.std(data_g), numpy.std(data_b)

    return mean, std

class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """
    def __init__(self, optimizer, total_iters, last_epoch=-1):

        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]


def most_recent_folder(net_weights, fmt):
    """
        return most recent created folder under net_weights
        if no none-empty folder were found, return empty folder
    """
    # get subfolders in net_weights
    folders = os.listdir(net_weights)

    # filter out empty folders
    folders = [f for f in folders if len(os.listdir(os.path.join(net_weights, f)))]
    if len(folders) == 0:
        return ''

    # sort folders by folder created time
    folders = sorted(folders, key=lambda f: datetime.datetime.strptime(f, fmt))
    return folders[-1]

def most_recent_weights(weights_folder):
    """
        return most recent created weights file
        if folder is empty return empty string
    """
    weight_files = os.listdir(weights_folder)
    if len(weights_folder) == 0:
        return ''

    regex_str = r'([A-Za-z0-9]+)-([0-9]+)-(regular|best)'

    # sort files by epoch
    weight_files = sorted(weight_files, key=lambda w: int(re.search(regex_str, w).groups()[1]))

    return weight_files[-1]

def last_epoch(weights_folder):
    weight_file = most_recent_weights(weights_folder)
    if not weight_file:
       raise Exception('no recent weights were found')
    resume_epoch = int(weight_file.split('-')[1])

    return resume_epoch

def best_acc_weights(weights_folder):
    """
        return the best acc .pth file in given folder, if no
        best acc weights file were found, return empty string
    """
    files = os.listdir(weights_folder)
    if len(files) == 0:
        return ''

    regex_str = r'([A-Za-z0-9]+)-([0-9]+)-(regular|best)'
    best_files = [w for w in files if re.search(regex_str, w).groups()[2] == 'best']
    if len(best_files) == 0:
        return ''

    best_files = sorted(best_files, key=lambda w: int(re.search(regex_str, w).groups()[1]))
    return best_files[-1]
