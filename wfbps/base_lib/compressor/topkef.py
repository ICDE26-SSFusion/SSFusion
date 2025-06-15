import torch

from grace_lib import Compressor
import random
import numpy as np
import wfbp.torch as hvd
import math




class TopKEFCompressor(Compressor):

    def __init__(self, compress_ratio, rank):
        super().__init__()

        self.compress_ratio = compress_ratio
        self.rank = rank
        self.epoch=0
        self.iteration=0
        self.index=0
        self.layernumels={}
        self.thres_mean_arr=[]

        # self.sample_ratio = min(max(sample_ratio, 0.01), 1.0)
        # self.strided_sample = strided_sample
        # self.compress_upper_bound = compress_upper_bound
        # self.compress_lower_bound = compress_lower_bound
        # self.max_adaptation_iters = max_adaptation_iters
        # self.resample = resample

        self.attributes = {}
        self.tensor={}

        # self.residuals={{}}
        # for i in range(hvd.size()):
        #     self.residuals[str(i)]={}
   
    def initialize(self, named_parameters):
        # if hvd.rank() == 0:
        #     print("=> initializing dgc compressor")
        for name, param in named_parameters:
            if torch.is_tensor(param):
                numel = param.numel()
                shape = list(param.size())
            else:
                assert isinstance(param, (list, tuple))
                numel, shape = param[0], param[1]
            
    def sparsify(self,tensor, compress_ratio,epoch, name):
        tensor_flatten = tensor.flatten()
        numel = tensor.numel()

        if self.compress_ratio<1:
            k= max(1, int(numel * self.compress_ratio))
            _, indices_flatten_global = torch.topk(tensor_flatten.abs(), k, sorted=False,)
            values_flatten_global = torch.gather(tensor_flatten, 0, indices_flatten_global)
        
            return values_flatten_global, indices_flatten_global
        
        tensor = tensor.flatten().cuda()
        numel = tensor.numel()
        values=tensor
        indices=torch.arange(0,numel).cuda(tensor.device)
        return values, indices


  
    def desparsify(self,tensors, numel,shape,name):
        values, indices = tensors
        if values.numel()==numel:
            return values

        else:
            tensor_decompressed = torch.zeros(
                    numel, dtype=values.dtype, layout=values.layout, device=values.device).cuda()
            tensor_decompressed.scatter_(0, indices, values)


        return tensor_decompressed

    def compress(self, tensor, name):


        tensors = self.sparsify(tensor, self.compress_ratio,self.epoch, name)
        # ctx = tensor.numel(), tensor.size(),tensors[0].size()
        ctx = tensor.numel(), tensor.size()
        return tensors, ctx


    def decompress(self, tensors, ctx, name):
        """Decompress by filling empty slots with zeros and reshape back using the original shape"""
        if ctx==None:
            tensor, = tensors
            return tensor
        numel, shape = ctx
        
        tensor_decompressed = self.desparsify(tensors, numel,shape,name)
        # if self.rank==0:
        return tensor_decompressed.view(shape)

    def decompress_add(self, tensors, ctx, name):
        numel, shape = ctx
        values, indices = tensors
        if values.numel()==numel:
            return values
        tensor_decompressed = torch.zeros(
            numel, dtype=values.dtype, layout=values.layout, device=values.device).cuda()

        tensor_decompressed = tensor_decompressed.scatter_add(0, indices, values)
        return tensor_decompressed.view(shape)
