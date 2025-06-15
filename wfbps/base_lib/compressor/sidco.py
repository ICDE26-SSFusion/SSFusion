import torch

from grace_lib import Compressor
import numpy as np
import grace_lib.sidcosettings as settings
import math
import sys

class NoneCompressor(Compressor):
    norm = 1.0
    sum_ratio = 0.0
    iter = 0
    last_estimate = 0.0
    cur_stages = 1
    last_stages = 1
    update = 1
    count_stupdates=0
    first_ratio=settings.FIRST_RATIO
    fr_update=settings.FR_UPDATE
    count_frupdates=0

class MultStageSparseCompressor(NoneCompressor):
    @staticmethod
    def adapt_stages(actual_ratio=0.0, ratio=0.0, stages=0):
        NoneCompressor.sum_ratio += actual_ratio / ratio
        NoneCompressor.iter += 1
        if NoneCompressor.iter == settings.UPDATE_ITER:
            cur_estimate = 1.0 * NoneCompressor.sum_ratio / NoneCompressor.iter

            if stages == -1:
                if cur_estimate > 1 + settings.RES_DIFF_UP:
                    NoneCompressor.cur_stages -= 1
                    NoneCompressor.count_stupdates += 1

                elif cur_estimate < 1 - settings.RES_DIFF_DOWN:
                    NoneCompressor.cur_stages += 1
                    NoneCompressor.count_stupdates += 1

            if stages == -2:
                if NoneCompressor.last_estimate > 0 and abs(cur_estimate - 1) > abs(NoneCompressor.last_estimate - 1):
                    NoneCompressor.update *= -1

                if cur_estimate > 1 + settings.RES_DIFF_UP:
                    NoneCompressor.cur_stages -= NoneCompressor.update
                    NoneCompressor.count_stupdates += 1

                if cur_estimate < 1 - settings.RES_DIFF_DOWN:
                    NoneCompressor.cur_stages += NoneCompressor.update
                    NoneCompressor.count_stupdates += 1

            if stages == -3:
                if NoneCompressor.last_estimate > 0 and not (
                        cur_estimate > NoneCompressor.last_estimate and NoneCompressor.last_estimate < 1 - settings.RES_DIFF_DOWN
                        or cur_estimate < NoneCompressor.last_estimate and NoneCompressor.last_estimate > 1 + settings.RES_DIFF_UP):
                    NoneCompressor.update *= -1

                if cur_estimate > 1 + settings.RES_DIFF_UP:
                    NoneCompressor.cur_stages -= NoneCompressor.update
                    NoneCompressor.count_stupdates += 1

                if cur_estimate < 1 - settings.RES_DIFF_DOWN:
                    NoneCompressor.cur_stages += NoneCompressor.update
                    NoneCompressor.count_stupdates += 1

            NoneCompressor.cur_stages = max(min(NoneCompressor.cur_stages, math.ceil(math.log(ratio) / math.log(0.5))), 1)

            #adjust the initial ratio if the changes to the stages can not fix the drift
            if settings.ADJUST_FR and NoneCompressor.count_stupdates >= math.ceil(math.log(ratio) / math.log(0.5) / 2):
                if NoneCompressor.last_estimate > 0 and abs(cur_estimate - 1) > abs(NoneCompressor.last_estimate - 1):
                    if NoneCompressor.first_ratio == 0.5 or NoneCompressor.first_ratio == 0.05:
                        NoneCompressor.first_ratio = 0.25
                        NoneCompressor.fr_update *= -1
                        NoneCompressor.count_frupdates = 0

                    elif cur_estimate > 1 + (settings.RES_DIFF_UP / 2):
                        NoneCompressor.first_ratio -= NoneCompressor.fr_update
                        NoneCompressor.count_frupdates += 1

                        #reset stages to middle point of MAX and re-search
                        NoneCompressor.cur_stages = math.ceil(math.log(ratio) / math.log(0.5) / 2) #1
                        NoneCompressor.count_stupdates = 0

                    elif cur_estimate < 1 - (settings.RES_DIFF_DOWN / 2):
                        NoneCompressor.first_ratio += NoneCompressor.fr_update
                        NoneCompressor.count_frupdates += 1

                        # reset stages to middle point of MAX and re-search
                        NoneCompressor.cur_stages = math.ceil(math.log(ratio) / math.log(0.5) / 2) #1
                        NoneCompressor.count_stupdates = 0

                #Bound the adjustment on the first ratio
                NoneCompressor.first_ratio = max(min(NoneCompressor.first_ratio, 0.5), 0.05)

            NoneCompressor.last_stages = NoneCompressor.cur_stages
            NoneCompressor.last_estimate = cur_estimate
            NoneCompressor.sum_ratio = 0
            NoneCompressor.iter = 0

  
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
    

class ExpCompressor(MultStageSparseCompressor):
    def __init__(self, compress_ratio=0.05, i_ratio=0.25, stages=1):
        super().__init__()
        self.compress_ratio = compress_ratio
        self.i_ratio = i_ratio
        self.stages = stages

    
    def compress(self, tensor, name):
        numel = tensor.numel()
        shape = tensor.size()
        k = max(int(numel * self.compress_ratio), 1) 
        tensor_flatten = tensor.flatten().cuda()

        # -------------EXP compress-------------
        ratio = self.compress_ratio
        i_ratio = self.i_ratio
        stages = self.stages

        ada_stages = 0
        if stages < 0 or i_ratio == 0.0:
            ada_stages = stages
            stages = ExpCompressor.cur_stages

        t_norm = tensor.norm(2)
        ExpCompressor.norm = t_norm
        # abs_norm_tensor = tensor.abs() / t_norm
        abs_norm_tensor = tensor_flatten.abs() / t_norm
        abs_norm_tensor_cpy = abs_norm_tensor.clone()

        t_mean = torch.mean(abs_norm_tensor)

        # if stages == 1 or ratio >= NoneCompressor.first_ratio:
        if stages == 1 or ratio >= NoneCompressor.first_ratio:
            threshold = -t_mean * math.log(ratio)
        else:
            # threshold = -t_mean * math.log(NoneCompressor.first_ratio)
            threshold = -t_mean * math.log(NoneCompressor.first_ratio)

        # r_ratio = ratio / NoneCompressor.first_ratio
        r_ratio = ratio / NoneCompressor.first_ratio
        if stages > 1 or stages == 0:
            if stages == 0:
                loop = math.ceil(math.log(r_ratio) / math.log(i_ratio))
            else:
                i_ratio = math.pow(r_ratio, 1.0 / (stages - 1))
                loop = stages - 1
            i = loop
            while i > 0:
                one_indexes = abs_norm_tensor > threshold
                # indexes = one_indexes.nonzero().data.squeeze().view(-1)
                indexes, = torch.where(one_indexes)
                abs_norm_tensor = abs_norm_tensor.data[indexes]

                t_min = abs_norm_tensor.min()
                t_mean = torch.mean(abs_norm_tensor)

                threshold = -(t_mean - t_min) * math.log(i_ratio) + t_min
                if i == 1 and stages == 0:
                    threshold = -(t_mean - t_min) * math.log(r_ratio / math.pow(i_ratio, loop - 1)) + t_min
                i -= 1

        one_indexes = abs_norm_tensor_cpy > threshold
        # indexes = one_indexes.nonzero().data.squeeze().view(-1)
        indexes, = torch.where(one_indexes)

        if ada_stages:
            actual_ratio = (1.0 * values.numel() / numel)
            ExpCompressor.adapt_stages(actual_ratio, ratio, ada_stages)

        indices = indexes.cuda(tensor.device)
        values = tensor_flatten[indices]

        tensors = values, indices
        ctx = numel, shape
        return tensors, ctx


class GParetoCompressor(MultStageSparseCompressor):
    def __init__(self, compress_ratio=0.05, i_ratio=0.25, stages=1):
        super().__init__()
        self.compress_ratio = compress_ratio
        self.i_ratio = i_ratio
        self.stages = stages

    
    def compress(self, tensor, name):
        numel = tensor.numel()
        shape = tensor.size()
        k = max(int(numel * self.compress_ratio), 1) 
        tensor_flatten = tensor.flatten().cuda()

        # -------------GPareto compress-------------
        ratio = self.compress_ratio
        i_ratio = self.i_ratio
        stages = self.stages

        t_norm = tensor.norm(2)
        GParetoCompressor.norm = t_norm

        ada_stages = 0
        if stages < 0 or i_ratio == 0.0:
            ada_stages = stages
            stages = GParetoCompressor.cur_stages

        # abs_norm_tensor = tensor.abs() / tensor.norm(2)
        abs_norm_tensor = tensor_flatten.abs() / tensor.norm(2)
        abs_norm_tensor_cpy = abs_norm_tensor.clone()

        if torch.__version__ < '1.3.0':
            t_var = torch.var(abs_norm_tensor)
            t_mean = torch.mean(abs_norm_tensor)
        else:
            t_var, t_mean = torch.var_mean(abs_norm_tensor)

        alpha = 0.5 * t_mean * ((torch.pow(t_mean, 2) / t_var) + 1)
        k = 0.5 * ((torch.pow(t_mean, 2) / t_var) - 1)

        if stages == 1 or ratio >= NoneCompressor.first_ratio:
            threshold = alpha * (1.0 - torch.exp(k * math.log(ratio))) / k
        else:
            threshold = alpha * (1.0 - torch.exp(k * math.log(NoneCompressor.first_ratio))) / k

        r_ratio = ratio / NoneCompressor.first_ratio
        if stages > 1 or stages == 0:
            if stages == 0:
                loop = math.ceil(math.log(r_ratio) / math.log(i_ratio))
            else:
                i_ratio = math.pow(r_ratio, 1.0 / (stages - 1))
                loop = stages - 1
            i = loop
            while i > 0:
                one_indexes = abs_norm_tensor > threshold
                # indexes = one_indexes.nonzero().data.squeeze().view(-1)
                indexes, = torch.where(one_indexes)
                abs_norm_tensor = abs_norm_tensor.data[indexes]

                t_min = abs_norm_tensor.min()
                abs_norm_tensor_min = abs_norm_tensor - t_min

                if torch.__version__ < '1.3.0':
                    t_var = torch.var(abs_norm_tensor_min)
                    t_mean = torch.mean(abs_norm_tensor_min)
                else:
                    t_var, t_mean = torch.var_mean(abs_norm_tensor_min)

                alpha = 0.5 * t_mean * ((torch.pow(t_mean, 2) / t_var) + 1)
                k = 0.5 * ((torch.pow(t_mean, 2) / t_var) - 1)

                threshold = alpha * (1.0 - torch.exp(k * math.log(i_ratio))) / k + t_min
                if i == 1 and stages == 0:
                    threshold = alpha * (
                            1.0 - torch.exp(k * math.log(r_ratio / math.pow(i_ratio, loop - 1)))) / k + t_min
                i -= 1

        one_indexes = abs_norm_tensor_cpy > threshold
        # indexes = one_indexes.nonzero().data.squeeze().view(-1)
        # values = tensor.data[indexes]
        indexes, = torch.where(one_indexes)

        if ada_stages:
            actual_ratio = (1.0 * values.numel() / numel)
            GParetoCompressor.adapt_stages(actual_ratio, ratio, ada_stages)

        indices = indexes.cuda(tensor.device)
        values = tensor_flatten[indices]

        tensors = values, indices
        ctx = numel, shape
        return tensors, ctx
    

class GammaGParetoCompressor(MultStageSparseCompressor):
    def __init__(self, compress_ratio=0.05, i_ratio=0.25, stages=1):
        super().__init__()
        self.compress_ratio = compress_ratio
        self.i_ratio = i_ratio
        self.stages = stages

    
    def compress(self, tensor, name):
        numel = tensor.numel()
        shape = tensor.size()
        k = max(int(numel * self.compress_ratio), 1) 
        tensor_flatten = tensor.flatten().cuda()

        # -------------GammaGPareto compress-------------
        ratio = self.compress_ratio
        i_ratio = self.i_ratio
        stages = self.stages

        t_norm = tensor.norm(2)
        GammaGParetoCompressor.norm = t_norm

        ada_stages = 0
        if stages < 0 or i_ratio == 0.0:
            ada_stages = stages
            stages = GammaGParetoCompressor.cur_stages

        abs_norm_tensor = tensor_flatten.abs() / tensor.norm(2)
        abs_norm_tensor_cpy = abs_norm_tensor.clone()

        t_mean = torch.mean(abs_norm_tensor)
        s = torch.log(t_mean) - torch.mean(torch.log(abs_norm_tensor + sys.float_info.epsilon))

        alpha = (3 - s + torch.sqrt(torch.pow(s - 3, 2) + 24 * s)) / (12 * s)
        beta = t_mean / alpha

        if stages == 1 or ratio >= NoneCompressor.first_ratio:
            threshold = -beta * (math.log(ratio) + torch.lgamma(alpha))
        else:
            threshold = -beta * (math.log(NoneCompressor.first_ratio) + torch.lgamma(alpha))

        r_ratio = ratio / NoneCompressor.first_ratio
        if stages > 1 or stages == 0:
            if stages == 0:
                loop = math.ceil(math.log(r_ratio) / math.log(i_ratio))
            else:
                i_ratio = math.pow(r_ratio, 1.0 / (stages - 1))
                loop = stages - 1
            i = loop
            while i > 0:
                one_indexes = abs_norm_tensor > threshold
                # indexes = one_indexes.nonzero().data.squeeze().view(-1)
                indexes, = torch.where(one_indexes)
                abs_norm_tensor = abs_norm_tensor.data[indexes]

                t_min = abs_norm_tensor.min()
                abs_norm_tensor_min = abs_norm_tensor - t_min

                if torch.__version__ < '1.3.0':
                    t_var = torch.var(abs_norm_tensor_min)
                    t_mean = torch.mean(abs_norm_tensor_min)
                else:
                    t_var, t_mean = torch.var_mean(abs_norm_tensor_min)

                alpha = 0.5 * t_mean * ((torch.pow(t_mean, 2) / t_var) + 1)
                k = 0.5 * ((torch.pow(t_mean, 2) / t_var) - 1)

                threshold = alpha * (1.0 - torch.exp(k * math.log(i_ratio))) / k + t_min
                if i == 1 and stages == 0:
                    threshold = alpha * (
                                1.0 - torch.exp(k * math.log(r_ratio / math.pow(i_ratio, loop - 1)))) / k + t_min
                i -= 1

        one_indexes = abs_norm_tensor_cpy > threshold
        # indexes = one_indexes.nonzero().data.squeeze().view(-1)
        indexes, = torch.where(one_indexes)
        # values = tensor.data[indexes]

        if ada_stages:
            actual_ratio = (1.0 * values.numel() / numel)
            GammaGParetoCompressor.adapt_stages(actual_ratio, ratio, ada_stages)


        indices = indexes.cuda(tensor.device)
        values = tensor_flatten[indices]

        tensors = values, indices
        ctx = numel, shape
        return tensors, ctx