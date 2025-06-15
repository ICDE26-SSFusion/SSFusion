import torch

from grace_lib import Compressor
import wfbp.torch as hvd
import numpy as np

class RedSyncCompressor(Compressor):

    def __init__(self, compress_ratio, rank, epoch=0):
        super().__init__()

        self.compress_ratio = compress_ratio

        self.rank = rank
        self.epoch=0


  
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

        tensor_flatten = tensor.flatten().cuda()

        l = 0.0
        r = 1.0
        thres = 0.0
        eps = 0.2
        abs_tensor = torch.abs(tensor_flatten)
        mean_val = torch.mean(abs_tensor)
        max_val = torch.max(abs_tensor)

        one_indexes = abs_tensor > thres
        while r - l > eps:
            tmp_ratio = l + (r-l)/2
            thres = mean_val + tmp_ratio * (max_val - mean_val)
            one_indexes = abs_tensor > thres
            # indexes = one_indexes.nonzero().data.squeeze().view(-1)
            # nnz = indexes.numel()
            nnz = one_indexes.sum()

            if nnz > k and 2*k > nnz:
                break
            elif nnz < k/2:
                r = tmp_ratio
            else:
                l = tmp_ratio
        # indexes = indexes 
        # values = tensor.data[indexes]

        indices, = torch.where(one_indexes)
        indices = indices.cuda(tensor.device)
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
        numel, shape = ctx
        values, indices = tensors
        if values.numel()==numel:
            return values
        tensor_decompressed = torch.zeros(
            numel, dtype=values.dtype, layout=values.layout, device=values.device).cuda()
        # if hvd.rank() == 0:
        #     print('values: ', values, 'indices: ', indices)
        # [a,b,    c,d]  [0,1,    0,2]
        # [c, b ,d ][a+c, b,d ]
        tensor_decompressed = tensor_decompressed.scatter_add(0, indices, values)
        return tensor_decompressed.view(shape)
    

class RedSyncTrimCompressor(RedSyncCompressor):

    def compress(self, tensor, name):
        numel = tensor.numel()
        shape = tensor.size()
        k = max(int(numel * self.compress_ratio), 1)

        tensor_flatten = tensor.flatten().cuda()

        abs_tensor = torch.abs(tensor_flatten)
        mean_val = torch.mean(abs_tensor)
        max_val = torch.max(abs_tensor)
        eps = 0.2
        tmp_ratio = 1 - eps

        thres = mean_val + tmp_ratio * (max_val - mean_val)
        one_indexes = abs_tensor > thres
        # indexes = one_indexes.nonzero().data.squeeze().view(-1)
        # nnz = indexes.numel()
        nnz = one_indexes.sum()

        while nnz < k:
            thres = mean_val + tmp_ratio * (max_val - mean_val)
            one_indexes = abs_tensor > thres
            # indexes = one_indexes.nonzero().data.squeeze().view(-1)
            # nnz = indexes.numel()
            nnz = one_indexes.sum()
            tmp_ratio = tmp_ratio - eps
            
        # indexes = indexes 
        # values = tensor.data[indexes] 
        indices, = torch.where(one_indexes)
        indices = indices.cuda(tensor.device)
        values = tensor_flatten[indices]

        tensors = values, indices
        ctx = numel, shape
        return tensors, ctx
