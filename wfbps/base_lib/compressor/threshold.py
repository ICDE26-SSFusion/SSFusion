import torch

from grace_lib import Compressor

class ThresholdCompressor(Compressor):

    def __init__(self, compress_ratio, rank, epoch):
        super().__init__(tensors_size_are_same=True)
        
        # self.threshold = threshold
        
        self.compress_ratio = compress_ratio
        self.rank = rank
        self.epoch=0
        self.iteration=0
        self.index=0
        self.layernumels={}
        self.thres_mean_arr=[]

        self.tensor_flatten_np_arr=[]
        self.values_flatten_global_np_arr=[]
        self.values_flatten_channel_np_arr=[]

    def compress(self, tensor, name):
        
        
        shape = tensor.size()
        tensor = tensor.flatten()
        numel = tensor.numel()
        
        tensor_flatten=tensor.flatten().cuda()
        k= max(1, int(numel * self.compress_ratio))
        values_global_abs, indices_flatten_global = torch.topk(tensor_flatten.abs(), k, sorted=False,)
        threshold = values_global_abs.min()
        
        
        indices, = torch.where(tensor.abs() > threshold)
        values = tensor[indices]
        ctx = shape, numel
        return [values, indices], ctx

    def decompress(self, tensor_compressed, ctx):
        if ctx==None:
            tensor, = tensor_compressed
            return tensor
        
        
        shape, numel = ctx
        values, indices = tensor_compressed
        tensor_decompressed = torch.zeros(numel, dtype=values.dtype, layout=values.layout, device=values.device)
        tensor_decompressed.scatter_(0, indices, values)
        return tensor_decompressed.view(shape)
