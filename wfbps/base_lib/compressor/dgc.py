import torch

from grace_lib import Compressor


class DgcCompressor(Compressor):

    def __init__(self, compress_ratio):
        super().__init__(tensors_size_are_same=True)
        self.compress_ratio = compress_ratio
        
    

    def compress(self, tensor, name):
        shape = tensor.size()
        tensor = tensor.flatten()
        numel = tensor.numel()

        sample_shape = [max(1, int(numel * self.compress_ratio))]
        sample_index = torch.empty(sample_shape).uniform_(0, numel).type(torch.long)
        sample_tensor = tensor[sample_index]

        k = max(1, int(numel * self.compress_ratio ))
        vals, indices = torch.topk(sample_tensor.abs(), k)

        thr = vals.min()
        # thr = vals.max()
        mask = tensor.abs() >= thr
        selected = mask.sum()

        for _ in range(2):
            if selected > 1.3 * numel * self.compress_ratio:
                thr = 1.3 * thr
            elif selected < 0.7 * numel * self.compress_ratio:
                thr = 0.7 * thr
            else:
                break
            mask = tensor.abs() >= thr
            selected = mask.sum()

        indices, = torch.where(mask)
        values = tensor[indices]

        tensor_compressed = values, indices
        # ctx = shape, mask, numel

        ctx = shape, numel
        return tensor_compressed, ctx

    def decompress(self, tensor_compressed, ctx, name):
        if ctx==None:
            tensor, = tensor_compressed
            return tensor
        values, indices = tensor_compressed
        # shape, _, numel = ctx
        shape, numel = ctx
        tensor_decompressed = torch.zeros(numel, dtype=values.dtype, layout=values.layout, device=values.device)
        tensor_decompressed.scatter_(0, indices, values)
        return tensor_decompressed.view(shape)
    
    def decompress_add(self, tensors, ctx, name):
     
        shape, numel = ctx
        values, indices = tensors
        if values.numel()==numel:
            return values

        tensor_decompressed = torch.zeros(
            numel, dtype=values.dtype, layout=values.layout, device=values.device).cuda()
        
    
        tensor_decompressed = tensor_decompressed.scatter_add(0, indices, values)
        return tensor_decompressed.view(shape)
