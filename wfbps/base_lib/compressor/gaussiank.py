import torch

from grace_lib import Compressor
import random
import wfbp.torch as hvd
import numpy as np
import scipy.stats as stats


class GaussiankCompressor(Compressor):

    def __init__(self, compress_ratio, rank, epoch=0):
        super().__init__()

        self.compress_ratio = compress_ratio
        self.rank = rank
        self.epoch=0


    def gen_threshold_from_normal_distribution(self, p_value, mu, sigma):
        zvalue = stats.norm.ppf((1-p_value)/2)
        return mu+zvalue*sigma, mu-zvalue*sigma


  
    def desparsify(self, tensors, numel, shape):
        values, indices = tensors
        # if True:
        if values.numel()==numel:
            return values
        else:
            tensor_decompressed = torch.zeros(
                numel, dtype=values.dtype, layout=values.layout, device=values.device).cuda()
            tensor_decompressed.scatter_(0, indices, values)

            return tensor_decompressed


    
    def compress(self, tensor, name):
        numel = tensor.numel()
        shape = tensor.size()
        
        
        k = max(int(numel * self.compress_ratio), 1)

        std = torch.std(tensor)
        mean = torch.mean(tensor)
        left_thres, thr = self.gen_threshold_from_normal_distribution(1- self.compress_ratio, float(mean), float(std))

        tensor_flatten = tensor.flatten().cuda()
        # abs_tensor_tensor_flatten = torch.abs(tensor_flatten)
        
        mask = tensor_flatten.abs() >= thr
        selected = mask.sum()

        for _ in range(5):
            if selected > 1.2 * k:
                thr = 1.2 * thr
            elif selected < 0.8 * numel * k:
                thr = 0.8 * thr
            else:
                break
            mask = tensor_flatten.abs() >= thr
            selected = mask.sum()
            
        # indexes = indexes[0:k]
        indices, = torch.where(mask)
        
        values = tensor_flatten[indices]

        tensors = values, indices
        ctx = numel, shape
        return tensors, ctx
    

    def decompress(self, tensors, ctx, name):
        """Decompress by filling empty slots with zeros and reshape back using the original shape"""
        if ctx==None:
            tensor, = tensors
            return tensor
        numel, shape = ctx
        tensor_decompressed =self.desparsify(tensors, numel, shape) 

        return tensor_decompressed.view(shape)
    

    def decompress_add(self, tensors, ctx, name):
        
        
        if ctx==None:
            tensor, = tensors
            return tensor
        numel, shape = ctx
        values, indices = tensors
        # if values.numel()==numel:
        #     return values
        tensor_decompressed = torch.zeros(
            numel, dtype=values.dtype, layout=values.layout, device=values.device).cuda()
        tensor_decompressed = tensor_decompressed.scatter_add(0, indices, values)
        return tensor_decompressed.view(shape)
