# Copyright 2020 HKBU. All Rights Reserved.
# Copyright 2018 Uber Technologies, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from horovod.common.util import check_extension

_MPI_LIB_AVAILABLE = True

# try:
#     check_extension('horovod.torch', 'HOROVOD_WITH_PYTORCH',
#                     __file__, 'mpi_lib_v2')
# except Exception as e:
#     # MPI libs are missing, but python applications are still available.
#     print(e)
#     print("Warning! MPI libs are missing, but python applications are still available.")
#     _MPI_LIB_AVAILABLE = False
# 

# 
# only import following function when mpi is available.
if _MPI_LIB_AVAILABLE:
    from horovod.torch import elastic
    from horovod.torch.compression import Compression
    from horovod.torch.functions import allgather_object, broadcast_object, broadcast_optimizer_state, broadcast_parameters
    from horovod.torch.mpi_ops import allreduce, allreduce_async, allreduce_, allreduce_async_
    from horovod.torch.mpi_ops import grouped_allreduce, grouped_allreduce_async, grouped_allreduce_, grouped_allreduce_async_
    from horovod.torch.mpi_ops import sparse_allreduce_async
    from horovod.torch.mpi_ops import allgather, allgather_async
    from horovod.torch.mpi_ops import grouped_allgather, grouped_allgather_async
    from horovod.torch.mpi_ops import broadcast, broadcast_async, broadcast_, broadcast_async_
    from horovod.torch.mpi_ops import alltoall, alltoall_async
    from horovod.torch.mpi_ops import reducescatter, reducescatter_async
    from horovod.torch.mpi_ops import grouped_reducescatter, grouped_reducescatter_async
    from horovod.torch.mpi_ops import join
    from horovod.torch.mpi_ops import barrier
    from horovod.torch.mpi_ops import poll, synchronize
    from horovod.torch.mpi_ops import init, shutdown
    from horovod.torch.mpi_ops import is_initialized, start_timeline, stop_timeline
    from horovod.torch.mpi_ops import size, local_size, cross_size, rank, local_rank, cross_rank
    from horovod.torch.mpi_ops import mpi_threads_supported, mpi_enabled, mpi_built
    from horovod.torch.mpi_ops import gloo_enabled, gloo_built
    from horovod.torch.mpi_ops import nccl_built, ddl_built, ccl_built, cuda_built, rocm_built
    from horovod.torch.mpi_ops import ProcessSet, global_process_set, add_process_set, remove_process_set
    from horovod.torch.mpi_ops import Average, Sum, Adasum, Min, Max, Product
    from horovod.torch.mpi_ops import HorovodInternalError
    from horovod.torch.optimizer import DistributedOptimizer
    from horovod.torch.sync_batch_norm import SyncBatchNorm


from horovod.torch.mpi_ops import allreduce_async_
from horovod.torch.mpi_ops import allgather_async
from horovod.torch.mpi_ops import broadcast_async_
from horovod.torch.mpi_ops import synchronize
from horovod.torch.mpi_ops import size, local_size, rank, local_rank
from horovod.torch.mpi_ops import init, broadcast
from horovod.torch.mpi_ops import allreduce

from utils_optimizer import calculation_backward_time_local_8_nodes
from utils_optimizer import calculation_communication_time_local_8_nodes
from utils_optimizer import calculation_compression_time_local_8_nodes
from utils_optimizer import calculation_compression_time_local_8_nodes_cpu


import time
import torch
import numpy as np
import utils_optimizer
from profiling import CommunicationProfiler




import collections
ADAPTIVE_SPARSE = False
DEBUG = False

# 
# from profiling import CommunicationProfiler
from sklearn.linear_model import LinearRegression
import logging


import threading
from queue import Queue


# 
# THRESHOLD = 163840 #4*8192 
p_alpha_beta_56Gbps = {
        64: (0.00080632079996292579, 1.8*3.2713239529771973e-10),
        32: (0.00040632079996292579, 1.5*3.2713239529771973e-10),
        16: (0.00023583677659915685*3, 4.0594787739537565e-10),
        8: (9.75367204301171e-05, 3.0568230536676206e-10),
        4: (4.204298980348825e-05, 2.0589360830118177e-10),
        2: (2.554691138304671e-06, 9.837548167872609e-11)
}

p_alpha_beta_10Gbps = {
        64: (0.0023476410788581382*3, 9.643300782166769e-10),
        48: (0.0018476410788581382*3, 9.643300782166769e-10),
        32: (0.0013476410788581382*3, 8.643300782166769e-10),
        16: (0.0009080981007148093, 7.395651186836712e-10),
        8: (0.0005230272768511732, 8.570746975492128e-10),
        #8: (1.4e-3, 1.7e-9), # one GPU per node
        4: (4.204298980348825e-05, 2.0589360830118177e-10),
        2: (2.554691138304671e-06, 9.837548167872609e-11)
}


class _DistributedOptimizer(torch.optim.Optimizer):
    def __init__(self, model_net_name, params, named_parameters, compression, is_sparse=False, density=0.001, seq_layernames=None, layerwise_times=None, norm_clip=None, threshold=0, writer=None, gradient_path=None, momentum_correction=False, fp16=False, mgwfbp=False, rdma=False, asc=False):
        super(self.__class__, self).__init__(params)
        self._model_net_name = model_net_name
        
        self._compression = compression
        self._sparse = is_sparse
        self._density = density
        self._profiling = False
        self._seq_layernames = seq_layernames
        self._layerwise_times = layerwise_times 
        self._original_layerwise_times_kv = None
        self._norm_clip = norm_clip
        self._threshold = threshold
        self._writer = writer
        self._gradient_path = gradient_path
        self._fp16 = fp16
        self._mgwfbp = mgwfbp
        self._asc = asc
        self._rdma = rdma
        self.alpha = None
        self.beta = None
        if self._layerwise_times is not None and self._seq_layernames is not None:
            self._original_layerwise_times_kv = dict(zip(self._seq_layernames, self._layerwise_times))
        self._compression_timers = {} # compression
        self._allreduce_timers = {} # allreduce times
        self._update_times = {} # allreduce times
        self.train_epoch = 0
        self.train_iter = 0
        self.momentum_correction = momentum_correction
        
        self.handle_synchronize_time = []
        self.synchronize_time = []
        self.para_update_time = []
        self._compressed_tensor_list = None
        self.hook_time = 0
        self.hook_communication_time = 0
        
        self.offset_compress = 0


        self._offload = True
        if self._offload:
            # 
            # 
            self._download_queue = Queue()
            self._upload_queue = Queue()
            self._queue_len = 0
            self._async_thread = threading.Thread(target=_DistributedOptimizer.async_compression,args=(self._download_queue,self._upload_queue,self._compression))
            self._async_thread.start()
        

        if density < 1:
            # self._dynamic_densities = [0.015625, 0.004, 0.001]
            self._layerwise_compressors= {}
            # self._dynamic_densities = [0.25, 0.0625, 0.015625, 0.004, 0.001] # the setting used in DGC
            self._dynamic_densities = None 
        else:
            self._dynamic_densities = None 
            self._layerwise_compressors= None
        
        if named_parameters is not None:
            named_parameters = list(named_parameters)
        else:
            named_parameters = []

        self._named_parameters = {k: v for k, v
                                in named_parameters}
        if self._seq_layernames is not None:
            self._sequential_keys = self._seq_layernames
        else:
            self._sequential_keys = [k for k, v in named_parameters]

        self.size_commtime_dict = None
        if self._mgwfbp and self._layerwise_times is None:
            self._benchmark_communication()

        self._debug_seq_keys = []

        # make sure that named_parameters are tuples
        if any([not isinstance(p, tuple) for p in named_parameters]):
            raise ValueError('named_parameters should be a sequence of '
                             'tuples (name, parameter), usually produced by '
                             'model.named_parameters().')

        if len(named_parameters) > 0:
            self._parameter_names = {v: k for k, v
                                     in sorted(named_parameters)}
            #print('Sorted named_parameters')
        else:
            self._parameter_names = {v: 'allreduce.noname.%s' % i
                                     for param_group in self.param_groups
                                     for i, v in enumerate(param_group['params'])}
        self._generate_merged_parameters()

        self._handles = {}
        self._grad_accs = []
        self._requires_update = set()
        self.local = False
        self._hook_checked_idx = 0
        if size() > 1:
            self._register_hooks()

        self.generate_threshold()
        #  
        self._layerwise = 'layerwise' in type(self._compression).__name__.lower()
        print(f'layerwise_setting: {self._layerwise}')
        if self._layerwise:
            self.generate_threshold()
        else:
            self._offload = False

        # logger.info('layerwise compressors: %s', self._layerwise_compressors)

    def _benchmark_communication(self):
        # logger.info('Benchmarking communication performance...')
        comm_profiler = CommunicationProfiler(allreduce_async_, synchronize)
        sizes, times = comm_profiler.benchmark(num_iters=10)
        def _fit_linear_function(x, y):
            X = np.array(x).reshape((-1, 1)) * 4
            Y = np.array(y)
            model = LinearRegression()
            model.fit(X, Y)
            alpha = model.intercept_
            beta = model.coef_[0]
            #A = np.vstack([X, np.ones(len(X))]).T
            #beta, alpha = np.linalg.lstsq(A, Y, rcond=None)[0]
            return alpha, beta
        alpha, beta = _fit_linear_function(sizes, times)
        self.alpha = alpha
        self.beta = beta
        alpha_tensor = torch.ones(1) * alpha 
        beta_tensor = torch.ones(1) * beta 
        alpha_tensor = broadcast(alpha_tensor, root_rank=0)
        beta_tensor = broadcast(beta_tensor, root_rank=0)
        if rank() != 0:
            self.alpha = float(alpha_tensor[0])
            self.beta = float(beta_tensor[0])
        #logger.info('[rank:{}] Communication performance fitted with f(p)=a+b*p, where a={} and b={}'.format(rank(), self.alpha, self.beta))

    def _benchmark_communication2(self):
        #logger.info('Benchmarking communication performance for the current DL model')
        sizes = [self._named_parameters[k].data.numel() for k in self._sequential_keys][::-1] # reverse from L to 1
        all_combined_sizes = []
        for i in range(len(sizes)):
            s = sizes[i]
            all_combined_sizes.append(s)
            for j in range(i+1, len(sizes)):
                s += sizes[j]
                all_combined_sizes.append(s)
        comm_profiler = CommunicationProfiler(allreduce_async_, synchronize, all_combined_sizes)
        sizes, times = comm_profiler.benchmark(num_iters=10)
        size_commtime_dict = {}
        for s, t in zip(sizes, times):
            if s not in size_commtime_dict:
                size_commtime_dict[s] = t
            else:
                if t > size_commtime_dict[s]:
                    size_commtime_dict[s] = t
        self.size_commtime_dict = size_commtime_dict
    
    
    def _register_hooks(self):
        for param_group in self.param_groups:
            for p in param_group['params']:
                if p.requires_grad:
                    p.grad = p.data.new(p.size()).zero_()
                    self._requires_update.add(p)
                    p_tmp = p.expand_as(p)
                    grad_acc = p_tmp.grad_fn.next_functions[0][0]
                    grad_acc.register_hook(self._make_hook(p))
                    self._grad_accs.append(grad_acc)

    
    # 
    def _generate_groups_with_number(self, number_layers):
        sizes = [self._named_parameters[k].data.numel() for k in self._sequential_keys][::-1] # reverse order
        self._sizes = sizes
        group_sizes=[]
        group_size=[]
        
        group_dims=[]
        group_dim=[]
        
        sub_size = 0
        groups = []
        group = []
        key_groupidx_maps = {}
        idx = 0
        for k in self._sequential_keys[::-1]:

            # BERT-base Debug, mingzq
            if 'pooler' in k:
                continue

            numel = self._named_parameters[k].data.numel()
            sub_size += 1
            key_groupidx_maps[k] = idx
            
            if sub_size < number_layers:
                group.append(k)
                group_size.append(numel)
                group_dim.append(self._named_parameters[k].dim())
            else:
                idx += 1
                group.append(k)
                groups.append(group)
                group = []
                
                group_size.append(numel)
                group_sizes.append(group_size)
                group_size=[]
                
                group_dim.append(self._named_parameters[k].dim())
                group_dims.append(group_dim)
                group_dim=[]
                
                sub_size = 0
        
        if len(group) > 0:
            groups.append(group)
            group_sizes.append(group_size)
            
            group_dims.append(group_dim)
            
        return groups, key_groupidx_maps, group_sizes, group_dims

    

    def get_current_density(self, name=None):
        density = self._density
        if self._dynamic_densities is not None:
            if self.train_epoch >= len(self._dynamic_densities):
                density = self._dynamic_densities[-1]
            else:
                density = self._dynamic_densities[self.train_epoch]
        
        if name is not None and self._layerwise_compressors is not None:
            if name not in self._layerwise_compressors:
                errstr = 'compressor density not found at layer: %s' % name
                # logger.error(errstr)
                raise Exception(errstr)
            ld = self._layerwise_compressors[name]
            density = max(ld, density)
        return density

    def _generate_groups_mgwfbp(self):
        num_of_workers = size()
        group_sizes=[]
        group_size=[]
        
        if self.alpha is not None:
            alpha, beta = self.alpha, self.beta
        else:
            if self._rdma:
                alpha, beta = p_alpha_beta_56Gbps[num_of_workers]
            else:
                alpha, beta = p_alpha_beta_10Gbps[num_of_workers]
        nbytes = 2 if self._fp16 else 4
        def __calculate_comm_start(tc, tb, taob, L):
            taoc = [0] * L 
            taoc[L-1] = taob[L-1] + tb[L-1]
            for l in range(L-1)[::-1]:
                taoc[l] = max(taoc[l+1] + tc[l+1], taob[l] + tb[l])
            return taoc
        def __merge(taob, tc, p, l):
            tc[l] = 0
            p[l-1] = p[l-1]+p[l]
            p[l] = 0
            if self.size_commtime_dict is not None:
                tc[l-1] = self.size_commtime_dict[l-1]
            else:
                tc[l-1] = utils_optimizer.predict_allreduce_time_with_size(alpha, beta, p[l-1]*nbytes, num_of_workers)
        sizes = [self._named_parameters[k].data.numel() for k in self._seq_layernames]
        seq_layernames = self._seq_layernames
        if not utils_optimizer.check_unique(seq_layernames):
            raise ValueError
        self._sizes = sizes
        p = sizes[:]
        L = len(sizes)
        if self.size_commtime_dict is not None:
            tc = [self.size_commtime_dict[s] for s in sizes]
        else:
            tc = [utils_optimizer.predict_allreduce_time_with_size(alpha, beta, s*nbytes, num_of_workers) for s in sizes]
        tb = list(self._layerwise_times)
        taob = [0]*L
        for l in range(0,L-1)[::-1]:
            taob[l] = taob[l+1] + tb[l+1]
        taoc = __calculate_comm_start(tc, tb, taob, L)
        if rank() == 0:
            # logger.debug('seq_layernames: %s', seq_layernames)
            # logger.debug('tb: %s', tb)
            # logger.debug('taob: %s', taob)
            # logger.debug('sizes: %s', p)
            # logger.warn('tc sum: %f', np.sum(tc))
            pass
            # logger.warn('tc: %s', tc)
            # logger.warn('taoc: %s', taoc)
        groups = []
        group = []
        idx = 0
        key_groupidx_maps = {}
        l = L-1
        key = seq_layernames[l] 
        key_groupidx_maps[key] = idx
        
        # 
        for l in range(1, L)[::-1]:
            key = seq_layernames[l]
            # 
            numel = self._named_parameters[key].data.numel()
            group_size.append(numel)

            group.append(key)
            key_groupidx_maps[key] = idx
            current_taob = taob[l-1] + tb[l-1]
            merged=False
            if current_taob < taoc[l]+tc[l]:
                if taoc[l] > current_taob:
                    __merge(taob, tc, p, l)
                    taoc = __calculate_comm_start(tc, tb, taob, L)
                    merged=True
                else:
                    t_wait = current_taob - taoc[l]
                    t_saved = alpha
                    if t_wait < t_saved:
                        __merge(taob, tc, p, l)
                        taoc = __calculate_comm_start(tc, tb, taob, L)
                        merged=True
            
            # if not merged and (key.find('bn') >= 0 or key.find('bias') >= 0):
            if not merged and p[l] < 8192: 
                __merge(taob, tc, p, l)
                taoc = __calculate_comm_start(tc, tb, taob, L)
                merged=True
            
            if not merged:
                idx += 1
                groups.append(group)
                group = []
                
                group_sizes.append(group_size)
                group_size=[]
            
            #elif current_taob > taoc[l+1]+tc[l+1] and current_taob < taoc[l]+tc[l] and taoc[l]+alpha > current_taob:
            #    __merge(taob, tc, p, l)
            #    taoc = __calculate_comm_start(tc, tb, taob, L)
            #else:
            #    idx += 1
            #    groups.append(group)
            #    group = []
        
        l = 0
        key = seq_layernames[l]
        key_groupidx_maps[key] = idx
        group.append(key)
        if len(group) > 0:
            groups.append(group)
            
            group_sizes.append(group_size)
            
        if rank() == 0:
            #logger.debug('seq_layernames: %s', seq_layernames)
            #pass
            #logger.info('Merged tc sum: %f', np.sum(tc))
            print('Merged sizes: ', p[::-1])
            print('# of parameters: ', np.sum(p[::-1]))
            #logger.info('Merged tb: %s', tb[::-1])
            #logger.info('Merged taob: %s', taob[::-1])
            #logger.info('Merged tc: %s', tc[::-1])
            #logger.info('Merged taoc: %s', taoc[::-1])

        return groups, key_groupidx_maps, group_sizes

    def _generate_groups_asc(self):
        num_of_workers = size()

        if self.alpha is not None:
            alpha, beta = self.alpha, self.beta
        else:
            if self._rdma:
                alpha, beta = p_alpha_beta_56Gbps[num_of_workers]
            else:
                alpha, beta = p_alpha_beta_10Gbps[num_of_workers]
        nbytes = 2 if self._fp16 else 4
        def __calculate_comm_start(tc, tb, taob, L):
            taoc = [0] * L 
            taoc[L-1] = taob[L-1] + tb[L-1]
            for l in range(L-1)[::-1]:
                taoc[l] = max(taoc[l+1] + tc[l+1], taob[l] + tb[l])
            return taoc
        def __merge(taob, tc, p, l):
            tc[l] = 0
            p[l-1] = p[l-1]+p[l]
            p[l] = 0
            if self.size_commtime_dict is not None:
                tc[l-1] = self.size_commtime_dict[l-1]
            else:
                tc[l-1] = utils_optimizer.predict_allreduce_time_with_size(alpha, beta, p[l-1]*nbytes, num_of_workers)
        sizes = [self._named_parameters[k].data.numel() for k in self._seq_layernames]
        seq_layernames = self._seq_layernames
        if not utils_optimizer.check_unique(seq_layernames):
            raise ValueError
        self._sizes = sizes
        p = sizes[:]
        L = len(sizes)
        if self.size_commtime_dict is not None:
            tc = [self.size_commtime_dict[s] for s in sizes]
        else:
            tc = [utils_optimizer.predict_allreduce_time_with_size(alpha, beta, s*nbytes, num_of_workers) for s in sizes]
        tb = list(self._layerwise_times)
        taob = [0]*L
        for l in range(0,L-1)[::-1]:
            taob[l] = taob[l+1] + tb[l+1]
        taoc = __calculate_comm_start(tc, tb, taob, L)
        groups = []
        group = []
        idx = 0
        key_groupidx_maps = {}
        l = L-1
        key = seq_layernames[l] 
        key_groupidx_maps[key] = idx
        for l in range(1, L)[::-1]:
            key = seq_layernames[l]
            group.append(key)
            key_groupidx_maps[key] = idx
            current_taob = taob[l-1] + tb[l-1]
            merged=False
            if current_taob < taoc[l]+tc[l]:
                if taoc[l] > current_taob:
                    __merge(taob, tc, p, l)
                    taoc = __calculate_comm_start(tc, tb, taob, L)
                    merged=True
            if not merged:
                idx += 1
                groups.append(group)
                group = []
        l = 0
        key = seq_layernames[l]
        key_groupidx_maps[key] = idx
        group.append(key)
        if len(group) > 0:
            groups.append(group)

        #if rank() == 0:
        #    logger.info('Predicted non-overlapped time: %f', taoc[0]+tc[0]-(taob[0]+tb[0]))
        #    logger.info('Predicted tb+tc= %f', taoc[0]+tc[0])
        #    logger.info('Merged tc sum: %f', np.sum(tc))
        return groups, key_groupidx_maps
    

    def _generate_groups_mgs(self):
        P = size() # number of wokers

        def __calculate_sparse_and_backward_start(tb, sizes, L, start=0):
            taos = [start] * L 
            ts = [utils_optimizer.topk_perf_model(s) for s in sizes]
            taob = [start] * L 
            taob[L-1] = start 
            taos[L-1] = taob[L-1] + tb[L-1]
            for l in range(L-1)[::-1]:
                taob[l] = taos[l+1] + ts[l+1]
                taos[l] = taob[l] + tb[l]
            return taob, taos, ts

        def __calculate_comm_start(ts, taos, sizes, L):
            taoc = [0] * L 
            tc = [utils_optimizer.allgather_perf_model(s, P, self._density) for s in sizes]
            taoc[L-1] = taos[L-1] + ts[L-1]
            for l in range(L-1)[::-1]:
                taoc[l] = max(taoc[l+1] + tc[l+1], taos[l] + ts[l])
            return taoc, tc

        def __merge(tb, ts, tc, p, l):
            tb[l-1] += tb[l]
            tb[l] = 0

            p[l-1] = p[l-1]+p[l]
            p[l] = 0

            tc[l-1] = utils_optimizer.allgather_perf_model(p[l-1], P, self._density) 
            tc[l] = 0

            ts[l-1] = utils_optimizer.topk_perf_model(p[l-1])
            ts[l] = 0

        sizes = [self._named_parameters[k].data.numel() for k in self._seq_layernames]
        seq_layernames = self._seq_layernames
        self._sizes = sizes
        p = sizes[:]
        L = len(sizes)
        tb = list(self._layerwise_times)
        taob, taos, ts = __calculate_sparse_and_backward_start(tb, p, L)
        taoc, tc = __calculate_comm_start(ts, taos, p, L)

        groups = []
        group = []
        idx = 0
        key_groupidx_maps = {}
        l = L-1
        key = seq_layernames[l] 
        key_groupidx_maps[key] = idx
        group.append(key)
        for l in range(1, L-1)[::-1]:
            key = seq_layernames[l]
            group.append(key)
            key_groupidx_maps[key] = idx

            tw = tb[l-1]+utils_optimizer.topk_perf_model(p[l]+p[l-1])\
                - utils_optimizer.topk_perf_model(p[l]) - utils_optimizer.topk_perf_model(p[l-1])\
                - (taoc[l] - (taos[l]+ts[l]))
            tsave = utils_optimizer.allgather_perf_model(p[l], P, self._density)+utils_optimizer.allgather_perf_model(p[l-1], P, self._density)-\
                    utils_optimizer.allgather_perf_model((p[l]+p[l-1]), P, self._density)
            if tw < tsave:
                __merge(tb, ts, tc, p, l)
                taob2, taos2, ts2 = __calculate_sparse_and_backward_start(tb[:l], p[:l], l, start=taob[l]+tb[l])
                taob[:l] = taob2
                taos[:l] = taos2
                taoc, tc = __calculate_comm_start(ts, taos, p, L)
            else:
                idx += 1
                groups.append(group)
                group = []
        l = 0
        key = seq_layernames[l]
        key_groupidx_maps[key] = idx
        group.append(key)
        if len(group) > 0:
            groups.append(group)
        return groups, key_groupidx_maps

    def _generate_merged_parameters(self):
        self._merged_parameters = {}
        self._merged_parameter_names = {}
        self._group_sizes=[]
        
        if self._offload:
            self._merged_parameters_indicates = mp.Manager().dict()
            self._merged_parameters_values = mp.Manager().dict()
        else:
            self._merged_parameters_indicates = {}
            self._merged_parameters_values = {}
        
        
        sizes = [self._named_parameters[k].data.numel() for k in self._sequential_keys][::-1] # reverse order
        self._sizes = sizes
        
        if self._mgwfbp and self._layerwise_times is not None:
            if self._density < 1: # MGS 
                groups, key_groupidx_maps = self._generate_groups_mgs()
            else:
                if self._asc:
                    groups, key_groupidx_maps = self._generate_groups_asc()
                else:
                    groups, key_groupidx_maps, group_sizes  = self._generate_groups_mgwfbp()
        else:
                        
            buffers = 20
            buffer_length = len(self._sizes)//buffers
            groups, key_groupidx_maps, group_sizes, group_dims = self._generate_groups_with_number(buffer_length)


      
        self._group_sizes = group_sizes
        self._group_dims = group_dims
        
        
        print('Number of parameters: %d' % np.sum(self._sizes))
        print('Total number of tensors: %s' % len(self._sizes))
        print('Merged number of groups: %s' % len(groups))
        
        # 
        if rank()==0:
            print('self._group_sizes: ',len(self._group_sizes))
            print('self._group_sizes: ',self._group_sizes)
            arr_=[]
            for gz in self._group_sizes:
                print(len(gz))
                arr_.append(len(gz))
                
            print('sum(len(gz))=', sum(arr_))
            print('groups_len: ', len(groups))
            print('groups: ', groups)
            print('group_dims: ', group_dims)

        
        # 
        self._end_of_buffer = set()
        for group in groups:
            if len(group) > 0:
                self._end_of_buffer.add(group[-1])
            else:
                raise ValueError('group is empty')

        new_keys = []
        self._merged_parameter_offsets = {}
        self._layerwise_compressors = None
        self._layerwise_compressors = {}
        self._merged_parameters_group_sizes={}
        self._merged_parameters_group_ids={}
        
        self._merged_parameters_group_dims={}
        num_of_workers = size()
        for i, g in enumerate(groups):
            sub_size = 0
            offsets = []
            computation_time = 0
            for k in g:
                offsets.append(sub_size)
                numel = self._named_parameters[k].data.numel()
                sub_size += numel
                if self._original_layerwise_times_kv is not None and k in self._original_layerwise_times_kv and ADAPTIVE_SPARSE:
                    computation_time += self._original_layerwise_times_kv[k]
            new_key = ':'.join(g)
            new_keys.append(new_key)
            # t = torch.zeros(sub_size, device=self._named_parameters[g[0]].device, dtype=self._named_parameters[g[0]].dtype, requires_grad=False).share_memory_()
            t = torch.zeros(sub_size, device=self._named_parameters[g[0]].device, dtype=self._named_parameters[g[0]].dtype, requires_grad=False)

            
            self._merged_parameters[new_key] = t
            self._merged_parameters_indicates[new_key] =t
            self._merged_parameters_values[new_key] =t



            self._merged_parameter_names[t] = new_key
            self._merged_parameter_offsets[new_key] = offsets
            if self._density < 1 and ADAPTIVE_SPARSE:
                _density = utils_optimizer.predict_density_with_size_and_computation(sub_size, computation_time, num_of_workers)
                density = max(_density, self._density)
            else:
                density = self._density
            if self._layerwise_compressors is not None:
                self._layerwise_compressors[new_key] = density
            
            self._merged_parameters_group_sizes[new_key]= self._group_sizes[i]
            self._merged_parameters_group_dims[new_key]= self._group_dims[i]
            
            self._merged_parameters_group_ids[new_key]= i


        self._groups = groups
        self._key_groupidx_maps = key_groupidx_maps
        self._groups_flags = []
        for g in self._groups:
            flags = []
            for k in g:
                flags.append(0)
            self._groups_flags.append(flags)


        # Bert Bug
        # if self._model_net_name=='bert_base' and len(self._groups)==1:
        #     self._groups_flags[0][2] = 1
        #     self._groups_flags[0][3] = 1
        

        # logger.info('offsets: ', self._merged_parameter_offsets)
        # logger.info('_layerwise_compressors: %s', self._layerwise_compressors)        
        # if rank()==0:
        #     # print('self._merged_parameter_offsets: ',self._merged_parameter_offsets)
        #     # print('self._layerwise_compressors: ', self._layerwise_compressors)            
        #     # print('self._merged_parameters: ', self._merged_parameters)
        #     # print('self._merged_parameter_names: ', self._merged_parameter_names)
        #     # print('self._merged_parameter_offsets: ', self._merged_parameter_offsets)            
        #     print('self._groups: ', self._groups)
        #     print('self._key_groupidx_maps: ', self._key_groupidx_maps)                   
        #     print('self._merged_parameters_group_sizes: ', self._merged_parameters_group_sizes)
            
        #     for k in self._merged_parameters_group_sizes.keys():
        #         print(sum(self._merged_parameters_group_sizes[k]))
    
        #     print('self._merged_parameters_group_ids: ', self._merged_parameters_group_ids)


    def _push_to_buffer(self, name, tensor):
        with torch.no_grad():
            if len(self._groups) == len(self._sequential_keys):
                new_tensor = tensor.data.view(-1)
                return name, new_tensor
            
            # group_idx=0
            group_idx = self._key_groupidx_maps[name]
            g = self._groups[group_idx]
            new_key = ':'.join(g)
            layer_idx = g.index(name)
            offset = self._merged_parameter_offsets[new_key][layer_idx]
            numel = tensor.data.numel()
            
            # compression
            
            self._merged_parameters[new_key].data[offset:offset+numel].copy_(tensor.view(numel))

            self._groups_flags[group_idx][layer_idx] = 1
            
            # if self._model_net_name=='bert_base' and len()
            
            # if rank()==0:   
            #     print('self._groups_flags[group_idx] =',self._groups_flags[group_idx])
            #     print('len(self._groups_flags[group_idx]) =', len(self._groups_flags[group_idx]) )
            #     print('group_idx=',group_idx, 'layer_idx=', layer_idx)
                             
                # if layer_idx==2 or layer_idx==3:
                #     print('self._groups_flags[group_idx] =',self._groups_flags[group_idx])
                #     print('len(self._groups_flags[group_idx]) =', len(self._groups_flags[group_idx]) )
                #     print('group_idx=',group_idx, 'layer_idx=', layer_idx)
                # if layer_idx==0:                    
                #     print('name =', name)

            # 
            for i, idx in enumerate(self._groups_flags[group_idx]):
                # Debug bert_base mingzq
                # if self._model_net_name=='bert_base' and (i==2 or i==3):
                #     continue
                
                if idx == 0:
                    return name, None

            return new_key, self._merged_parameters[new_key]
    
    
    def _push_to_buffer_and_check_full(self,name):
        group_idx = self._key_groupidx_maps[name]
        layer_idx = self._groups[group_idx].index(name)
        self._groups_flags[group_idx][layer_idx] = 1
        return all(self._groups_flags[group_idx])



    @staticmethod
    def async_compression(download_queue,upload_queue,compressor):
        while True:
            kwargs,new_key,offset,layer_idx = download_queue.get()
            if kwargs is None:
                break
            result = compressor.compress(**kwargs)
            upload_queue.put(((new_key,result[1],result[2],offset),layer_idx))
    
    # 
    def _push_to_buffer_and_compression_back(self, name, tensor, density):
        self._iter_time = 0
        debug = False
        with torch.no_grad():
            if debug:
                print("begin pushing")
            if len(self._groups) == len(self._sequential_keys):
                new_tensor = tensor.data.view(-1)
                self.offset_compress = 0
                return name, new_tensor, self._merged_parameters_indicates[new_key], self._merged_parameters_values[new_key]

            # group_idx = 0
            group_idx = self._key_groupidx_maps[name]
            g = self._groups[group_idx]
            new_key = ':'.join(g)
            layer_idx = g.index(name)
            if self._compressed_tensor_list is None:
                self._compressed_tensor_list = [None] * len(g)

            offset = self._merged_parameter_offsets[new_key][layer_idx]
            numel = tensor.data.numel()

            
            self._merged_parameters[new_key].data[offset:offset+numel].copy_(tensor.to(self._merged_parameters[new_key].data.device).view(numel))
            self._groups_flags[group_idx][layer_idx] = 1
            compress_ratio = density
            if hasattr(self,'_dict') and name in self._dict.keys():
                compress_ratio = self._dict[name]
                # if self._offload:
                #     print(f'{name}: offload? {name in self.on_cpu}')
                # 
            
            if self._offload and name in self.on_cpu:
                # 
                kwargs = {'tensor':tensor.to('cpu'),'name':name,'group_size':None,'ratio':compress_ratio}
                self._download_queue.put((kwargs,new_key,offset,layer_idx))
                self._queue_len += 1
            else:
                # 
                # if self._offload:
                #     while self._queue_len > 0:
                #         self._queue_len -= 1
                #         _new_key,s_indicates,s_values,offset = self._upload_queue.get()
                #         numel_compress = s_indicates.numel()
                #         self._merged_parameters_indicates[_new_key].data[self.offset_compress:self.offset_compress+numel_compress].copy_(s_indicates.view(numel_compress)+offset)
                #         self._merged_parameters_values[_new_key].data[self.offset_compress:self.offset_compress+numel_compress].copy_(s_values.view(numel_compress))
                #         self.offset_compress += numel_compress
                # 

                result = self._compression.compress(tensor=tensor,name=name,ratio=compress_ratio)
                self._compressed_tensor_list[layer_idx] = (new_key,result[1],result[2],offset)
                # 
                # numel_compress = result[1].numel()
                # self._merged_parameters_indicates[new_key].data[self.offset_compress:self.offset_compress+numel_compress].copy_(result[1].view(numel_compress)+offset)
                # self._merged_parameters_values[new_key].data[self.offset_compress:self.offset_compress+numel_compress].copy_(result[2].view(numel_compress))
                # self.offset_compress += numel_compress

            # tensor_compressed, selected_indicates, selected_values = self._compression.compress(tensor, name, group_size=None, ratio=density)


            # 
            # 
            for i, idx in enumerate(self._groups_flags[group_idx]):
                # 
                # Debug bert_base mingzq
                # if self._model_net_name=='bert_base' and (i==2 or i==3):
                #     continue
                # 

                if idx == 0:
                    if debug:
                        print("end pushing")
                    return name, None, None, None
            
            bt = time.time()
            # 
            if self._offload:
                while self._queue_len > 0:
                    self._queue_len -= 1
                    res,layer_idx = self._upload_queue.get()
                    self._compressed_tensor_list[layer_idx] = res
            # 
            # print(f'self._compressed_tensor_list: {[type(elem) for elem in self._compressed_tensor_list]}')
            for i,res in enumerate(self._compressed_tensor_list):
                if i == 0:
                    continue
                _new_key,s_indicates,s_values,offset = res
                numel_compress = s_indicates.numel()
                self._merged_parameters_indicates[_new_key].data[self.offset_compress:self.offset_compress+numel_compress].copy_(s_indicates.view(numel_compress)+offset)
                self._merged_parameters_values[_new_key].data[self.offset_compress:self.offset_compress+numel_compress].copy_(s_values.view(numel_compress))
                self.offset_compress += numel_compress
            self._compressed_tensor_list = None
            # self.offset_compress = 0
            
            if debug:
                et = time.time()
                print(f"synchronize Time: {et - bt}")
                self._iter_time += et - bt

            # arg1 = new_key
            # arg2 = self._merged_parameters[new_key].clone()
            # arg3 = self._merged_parameters_indicates[new_key].clone()
            # arg4 = self._merged_parameters_values[new_key].clone()

            # return arg1,arg2,arg3,arg4
            if debug:
                # print("end pushing for last layer")
                print(f"accumulated time: {self._iter_time}")
                
            merged_parameters_indicates = self._merged_parameters_indicates[new_key][:self.offset_compress]
            merged_parameters_values = self._merged_parameters_values[new_key][:self.offset_compress]
            
            self.offset_compress = 0


            return new_key, self._merged_parameters[new_key], merged_parameters_indicates, merged_parameters_values
    
    
    
    def _push_to_buffer_and_compression(self, name, tensor, density):
        self._iter_time = 0
        debug = False
        with torch.no_grad():
            if debug:
                print("begin pushing")
            if len(self._groups) == len(self._sequential_keys):
                new_tensor = tensor.data.view(-1)
                self.offset_compress = 0
                return name, new_tensor, self._merged_parameters_indicates[new_key], self._merged_parameters_values[new_key]
            
            # 
            # group_idx = 0
            group_idx = self._key_groupidx_maps[name]
            g = self._groups[group_idx]
            new_key = ':'.join(g)
            layer_idx = g.index(name)
            if self._compressed_tensor_list is None:
                self._compressed_tensor_list = [None] * len(g)

            offset = self._merged_parameter_offsets[new_key][layer_idx]
            numel = tensor.data.numel()


            
            self._merged_parameters[new_key].data[offset:offset+numel].copy_(tensor.to(self._merged_parameters[new_key].data.device).view(numel))
            self._groups_flags[group_idx][layer_idx] = 1
            compress_ratio = density
            if hasattr(self,'_dict') and name in self._dict.keys():
                compress_ratio = self._dict[name]
                
            if layer_idx <= self.partion_idx[group_idx] and compress_ratio == 1:
                # 
                # SSFusion-I: Buffer All-Reduce
                # 
                try:
                    self._merged_parameters_noncompress[new_key].data[self.offset_noncompress:self.offset_noncompress + numel].copy_(tensor.to(self._merged_parameters_noncompress[new_key].data.device).view(numel))
                    self.offset_noncompress += numel
                except Exception as e:
                    pass
            else:
                # SSFusion-I: Buffer All-Reduce
                # 
                if self._offload and name in self.on_cpu:
                    #  async_compression 
                    kwargs = {'tensor':tensor.to('cpu'),'name':name,'group_size':None,'ratio':compress_ratio}
                    self._download_queue.put((kwargs,new_key,offset,layer_idx))
                    self._queue_len += 1
                else:

                    result = self._compression.compress(tensor=tensor,name=name,ratio=compress_ratio)
                    self._compressed_tensor_list[layer_idx] = (new_key,result[1],result[2],offset)
                    # 
                    # numel_compress = result[1].numel()
                    # self._merged_parameters_indicates[new_key].data[self.offset_compress:self.offset_compress+numel_compress].copy_(result[1].view(numel_compress)+offset)
                    # self._merged_parameters_values[new_key].data[self.offset_compress:self.offset_compress+numel_compress].copy_(result[2].view(numel_compress))
                    # self.offset_compress += numel_compress
                    # 
        
            # 
            for i, idx in enumerate(self._groups_flags[group_idx]):
                # 
                # if self._model_net_name=='bert_base' and (i==2 or i==3):
                #     continue
                # 

                if idx == 0:
                    if debug:
                        print("end pushing")
                    return name, None, None, None, None
            
            bt = time.time()
            # If there are still uninstallation results in the queue, they need to be retrieved
            if self._offload:
                while self._queue_len > 0:
                    self._queue_len -= 1
                    res,layer_idx = self._upload_queue.get()
                    self._compressed_tensor_list[layer_idx] = res
            # 
            # print(f'self._compressed_tensor_list: {[type(elem) for elem in self._compressed_tensor_list]}')
            # 
            for i,res in enumerate(self._compressed_tensor_list):
                # if i == 0:
                    # continue
                if res is None:
                    continue
                _new_key,s_indicates,s_values,offset = res
                numel_compress = s_indicates.numel()
                self._merged_parameters_indicates[_new_key].data[self.offset_compress:self.offset_compress+numel_compress].copy_(s_indicates.view(numel_compress)+offset)
                self._merged_parameters_values[_new_key].data[self.offset_compress:self.offset_compress+numel_compress].copy_(s_values.view(numel_compress))
                self.offset_compress += numel_compress
            self._compressed_tensor_list = None
            # self.offset_compress = 0
            
            if debug:
                et = time.time()
                print(f"Synchronization time: {et - bt}")
                self._iter_time += et - bt

            # arg1 = new_key
            # arg2 = self._merged_parameters[new_key].clone()
            # arg3 = self._merged_parameters_indicates[new_key].clone()
            # arg4 = self._merged_parameters_values[new_key].clone()

            # return arg1,arg2,arg3,arg4
            if debug:
                # print("end pushing for last layer")
                print(f"Accumulated time: {self._iter_time}")
                
            merged_parameters_indicates = self._merged_parameters_indicates[new_key][:self.offset_compress]
            merged_parameters_values = self._merged_parameters_values[new_key][:self.offset_compress]
            self._merged_parameters_noncompress[new_key] = self._merged_parameters_noncompress[new_key][:self.offset_noncompress]
            
            self.offset_compress = 0
            self.offset_noncompress = 0


            return new_key, self._merged_parameters[new_key], merged_parameters_indicates, merged_parameters_values,group_idx
    

    
    
    
    def _pull_from_buffer(self, name, merged_tensor):
        if len(self._groups) == len(self._sequential_keys):
            shape = self._named_parameters[name].data.shape
            return {name: merged_tensor.view(shape)} 
        offsets = self._merged_parameter_offsets[name]
        g = name.split(':')
        group_idx = self._key_groupidx_maps[g[0]]
        self._groups_flags[group_idx] = [0]*len(self._groups_flags[group_idx])
        
        # Bert Bug
        # if self._model_net_name=='bert_base' and len(self._groups)==1:
        #     self._groups_flags[0][2] = 1
        #     self._groups_flags[0][3] = 1
        
        tensors = {}
        for i, k in enumerate(g):
            offset = offsets[i]
            original_tensor = self._named_parameters[k]
            numel = original_tensor.numel()
            tensors[k] = merged_tensor.data[offset:offset+numel].view(original_tensor.shape)
        return tensors


    def _allreduce_grad_async(self, p, name):
        tensor = p.data.view(-1)
        if False and rank() == 0 and self.train_iter % 200 == 0 and self.train_iter < 3000:
            grads = tensor.cpu().numpy()
            layer_idx = self._sequential_keys.index(name)
            np.save('%s/r%d_gradients_iter_%d::%s::%d' % (self._gradient_path, rank(), self.train_iter, name, layer_idx), grads)
        
        allreduce_name = name
        if len(name) > 200:
            allreduce_name = name[0:100]+'...'+name[-100:]
        handle = allreduce_async_(tensor, average=True, name=allreduce_name)
        return handle, None

    def _sparse_allreduce_async(self, p, name, density):
        stime = time.time()
        tensor = p.data.view(-1)        

        merged_parameters_group_size = self._merged_parameters_group_sizes[name]
        merged_parameters_group_dim  = self._merged_parameters_group_dims[name]

        
        group_idx = self._merged_parameters_group_ids[name] 
        
        
        # print(name,':',group_idx)
        tensor_compressed, ctx, selected_values = self._compression.compress(tensor, name, group_size=merged_parameters_group_size, ratio=density)


        if False and rank() == 0 and self.train_iter % 200 == 0 and self.train_iter < 3000:
            grads = tensor.cpu().numpy()
            layer_idx = self._sequential_keys.index(name)
            np.save('%s/r%d_gradients_iter_%d::%s::%d' % (self._gradient_path, rank(), self.train_iter, name, layer_idx), grads)
        
        
        indexes = ctx
        if indexes is None:
            handle = allgather_async(tensor_compressed, name)
            handle_idx = None # quantization uses all indices
        else:
            handle = allgather_async(selected_values, name)
            handle_idx = allgather_async(indexes.int(), name+'_indexes')
        if self._profiling:
            utils_optimizer.force_insert_item(self._compression_timers, name, time.time()-stime)

        return (handle, handle_idx), ctx 


    def _sparse_allreduce_async_compress(self, p, name, indices, values):
        stime = time.time()
        tensor = p.data.view(-1)        
        
        merged_parameters_group_size= self._merged_parameters_group_sizes[name]
        merged_parameters_group_dim= self._merged_parameters_group_dims[name]
        
        group_idx = self._merged_parameters_group_ids[name] 

        if False and rank() == 0 and self.train_iter % 200 == 0 and self.train_iter < 3000:
            grads = tensor.cpu().numpy()
            layer_idx = self._sequential_keys.index(name)
            np.save('%s/r%d_gradients_iter_%d::%s::%d' % (self._gradient_path, rank(), self.train_iter, name, layer_idx), grads)
        
        
        # indices = ctx
        if indices is None:
            handle = allgather_async(p, name)
            handle_idx = None # quantization uses all indices
        else:
            handle = allgather_async(values, name)
            handle_idx = allgather_async(indices.int(), name+'_indices')
        if self._profiling:
            utils_optimizer.force_insert_item(self._compression_timers, name, time.time()-stime)
            
        
        # print('_sparse_allreduce_async_compress')

        return (handle, handle_idx), indices 




    def check_hooked_tensor_sequence(self, name):
        if self._seq_layernames is None:
            return
        ntensors = len(self._seq_layernames)
        idx = self._seq_layernames.index(name)
        if idx == ntensors-self._hook_checked_idx-1:
            self._hook_checked_idx += 1
            if idx == 0:
                self._hook_checked_idx = 0
        else:
            raise

    def _make_hook(self, p):

        def hook(*ignore):
            e_time=time.time()
            # nonlocal p
            assert p not in self._handles
            assert not p.grad.requires_grad
            if not self.local:
                name = self._parameter_names.get(p)
                self.check_hooked_tensor_sequence(name)
                d_p = p.grad.data

                if self.momentum_correction and self._sparse:
                    param_state = self.state[p]
                    momentum = 0.9
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
                    buf = param_state['momentum_buffer']
                    buf.mul_(momentum).add_(d_p)
                    d_p = buf

                # new_name, new_tensor = self._push_to_buffer(name, d_p)
                # 
                # if rank()==0 and 'bert.pooler' in name:
                #     print('new_tensor.numle = ', new_tensor)
                # 
                
                density = self._density
                


                new_name, new_tensor, indices, values = self._push_to_buffer_and_compression(name, d_p, density)
                # assert new_tensor is not None and new_name in self._end_of_buffer or\
                    # new_tensor is None and new_name not in self._end_of_buffer

                # if name in self._end_of_buffer:
                c_time = time.time()
                
                # if self._push_to_buffer_and_check_full(name):
                if new_tensor is not None:
                    density = self._density
                    if self._mgwfbp:
                        current_stream = torch.cuda.current_stream()
                        current_stream.synchronize()
                    
                    if self._sparse and density < 1:

                        # handle, ctx = self._sparse_allreduce_async(new_tensor, new_name, density)
                        # self._handles[new_tensor] = (handle, ctx, density)
                        
                        handle, ctx = self._sparse_allreduce_async_compress(new_tensor, new_name, indices, values)
                        self._handles[new_tensor] = (handle, ctx, density)
                        

                    else:
                        handle, ctx = self._allreduce_grad_async(new_tensor, new_name)
                        self._handles[new_tensor] = (handle, ctx, 1)
                    
                self.hook_communication_time += time.time() - c_time
            # print(f'self.hook_time:{self.hook_time}')
            self.hook_time += time.time()-e_time
            
        return hook



    def synchronize(self):
        s_time=time.time()        
        num_of_workers = size()
        # key(p)=new_tensor, value=(handle, ctx, 1)
        for p, value in self._handles.items():
            # torch.cuda.synchronize()
            # handle initialization time
            handle_time = time.time() 
            
            name = self._merged_parameter_names.get(p)
            handle, ctx, density = value
            # handle=(handle, handle_idx)
            if self._sparse and density < 1:
                stime = time.time()
                handle_idx = None
                all_indexes = None
                if type(handle) is tuple:
                    handle, handle_idx = handle[0], handle[1]
                output = synchronize(handle)
                if handle_idx is not None:
                    all_indexes = synchronize(handle_idx)

                if self._profiling:
                    utils_optimizer.force_insert_item(self._allreduce_timers, name, time.time()-stime)

                stime = time.time()
                new_grad = p.data.view(-1)
                new_grad.fill_(0.0)
                numel = output.size(0)
                real_num_values = numel//num_of_workers

                for i in range(num_of_workers):
                    values_and_indexes = output.data[i*real_num_values:(i+1)*real_num_values]
                    if all_indexes is None:
                        values = values_and_indexes
                        indexes = None
                        per_values = values
                        
                        per_values = self._compression.decompress(per_values, p.size())
                        # Decompression gradient

                        new_grad += per_values.view(-1)
                    else:
                        values = values_and_indexes
                        indexes = all_indexes.data[i*real_num_values:(i+1)*real_num_values].long()

                        per_values = values[0:indexes.numel()]

                        per_values = self._compression.decompress(per_values, p.size())

                        # 
                        # Decompression gradient
                        new_grad[indexes[0:indexes.numel()]] += per_values


                new_grad /= num_of_workers

                if self._profiling:
                    utils_optimizer.force_insert_item(self._update_times, name, time.time()-stime)
            else:
                stime = time.time()
                output = synchronize(handle)
                if self._profiling:
                    utils_optimizer.force_insert_item(self._allreduce_timers, name, time.time()-stime)
                stime = time.time()

                if self._norm_clip is not None:
                    norm_clip = np.sqrt(1.0/size()) * self._norm_clip
                    norm_type = 2.0
                    param_norm = output.norm(norm_type)
                    total_norm = param_norm.item() 
                    clip_coef = norm_clip / (total_norm + 1e-6)
                    if clip_coef < 1:
                        output.mul_(clip_coef)
                if self._compression:
                    output = self._compression.decompress(output, p.size())
                    
                    
                p.set_(output)
                if self._profiling:
                    utils_optimizer.force_insert_item(self._update_times, name, time.time()-stime)
            # torch.cuda.synchronize()
            self.handle_synchronize_time.append(time.time()-handle_time) 

        self.synchronize_time.append(time.time()-s_time)


        p_time=time.time()
        if len(self._groups) != len(self._sequential_keys):
            for merged_p, value in self._handles.items():
                new_name = self._merged_parameter_names.get(merged_p)
                tensors = self._pull_from_buffer(new_name, merged_p)
                for n in tensors:
                    p = self._named_parameters.get(n)
                    if self._fp16:
                        p.grad.set_(tensors[n].data.type(p.grad.type()))
                    else:
                        # Debug bert_base mingzq
                        # if p.grad is not None:
                        #     p.grad.set_(tensors[n].data)
                        p.grad.set_(tensors[n].data)
        self.train_iter += 1
        self._handles.clear()
        self._print_profiling()
        self.para_update_time.append(time.time()-p_time)
        
        


    def _print_profiling(self):
        if self._profiling and rank() == 0 and len(self._allreduce_timers.keys()) > 0 and len(self._allreduce_timers.get(self._allreduce_timers.keys()[0], [])) ==  40:
            cps = self._compression_timers # compression
            ars = self._allreduce_timers # allreduce times
            ups = self._update_times # update times
            r = rank()
            tcp = 0.0; tar = 0.0; tup = 0.0; total=0.0
            for k in cps:
                acp = np.mean(cps[k])
                tcp += acp
                aar = np.mean(ars[k])
                tar += aar
                aup = np.mean(ups[k])
                tup += aup
                # logger.info('[%d][%s]: %f, %f, %f', r, k, acp, aar, aup)
            total = tcp+tar+tup
            cps.clear()
            ars.clear()
            ups.clear()


    def _step_with_mc(self, closure=None):
        """Performs a single optimization step.
            Arguments:
                closure (callable, optional): A closure that reevaluates the model
                    and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()
    
        offset = 0
        density = self.get_current_density()
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                name = self._parameter_names.get(p)
                if weight_decay != 0:
                    wd = p.data
                    d_p.add_(weight_decay, wd)
                if momentum != 0 and not self.momentum_correction:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        param_state['momentum_buffer'] = torch.zeros_like(p.data)
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(d_p)
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf
                p.data.add_(-group['lr'], d_p)
                if momentum != 0 and self.momentum_correction and density < 1:
                    param_state = self.state[p]
                    buf = param_state['momentum_buffer']
                    if self._compression.zc is not None:
                        buf.view(-1).mul_(self._compression.zc[offset:offset+d_p.numel()])
                        offset += d_p.numel()
        return loss
    
    
    # 
    # 
    def step(self, closure=None):
        if not self.local:
            self.synchronize()
        if self.momentum_correction and self._sparse:
            return self._step_with_mc(closure)
        
        return super(self.__class__, self).step(closure)
    


    def get_threshold(self,arr:list):
        arr = sorted(arr)
        elems = max(1,sum(arr) * self._density)
        idx = 0
        while idx < len(arr):
            if arr[idx] >= elems:
                return arr[idx] - 1
            elems -= arr[idx]
            idx += 1
        raise ValueError(f'Inexpected error while calculating threshold, with compression ratio {self._density} and arr {arr}')
        

    @property
    def valid_threshold(self):
        """
        get the edge value of the threshold s.t. threshold > value is not applicable and threshold <= value is ok.
        """
        return min([self.get_threshold(group_size) for group_size in self._group_sizes])
    
    
    def valid_threshold_by_group(self,group_idx:int):
        return self.get_threshold(self._group_sizes[group_idx])



    def generate_threshold(self):
        # 
        # 
        candidate_vals = set([0])
        candidate_vals.update([val for buffer in self._group_sizes for val in buffer])
        candidate_vals = sorted(candidate_vals)
        print(f'candidate_vals = {candidate_vals}')
        group_threshold = []
        for i,group_size in enumerate(self._group_sizes):
            # 
            expected_val = 0
            total_time = float('inf')
            valid_threshold = self.valid_threshold_by_group(i)
            # print(f"valid threshold for group {i} is {valid_threshold}")
            for val in candidate_vals:
                if val > valid_threshold:
                    break
                current_time = self.cal_total_time(val,i)
                if current_time < total_time:
                    total_time = current_time
                    expected_val = val
            group_threshold.append(expected_val)
            
            
        print(f'the best threshold for each group is {group_threshold}\n\
              and valid_threshold is {self.valid_threshold}')
        # self.update_threshold([2048 for _ in range(len(self._group_sizes))])
        self.update_threshold(group_threshold)
    
    
    def cal_total_time(self,threshold:int,group_idx:int):
        """
        calculate the total time of an iteration, returning the total time of this iteration, float.
        """
        # if(threshold > self.valid_threshold):
        #     import warnings
        #     warnings.warn(f'threshold {threshold} in cal_total_time for group {group_idx} is too large, this will cause an improper total compression ratio.')
        calculation_end = .0
        communication_end = .0
        sum_numel = sum(sum(group_size) for group_size in self._group_sizes)
        group_size = self._group_sizes[group_idx]
        communication_numel = 0
        calculation_end += calculation_backward_time_local_8_nodes(sum_numel,sum(group_size),self._model_net_name)
        for layer_size in group_size:
            if layer_size > threshold:
                calculation_end += calculation_compression_time_local_8_nodes(layer_size)
                communication_numel += max(1,int(layer_size * self._density))
            else:
                communication_numel += layer_size
        communication_end = max(communication_end,calculation_end) + calculation_communication_time_local_8_nodes(communication_numel,self._density)
        return communication_end
    
    
    def update_threshold(self,threshold_arr:list):
        """
        calculate the sparsity of each layer, and pass the corresponding dict to compressor.
        """
        layer_sparsity = {}
        on_cpu = set()
        all_group_size = sum([sum(group_size) for group_size in self._group_sizes])
        total_tensors = sum([sum(len(group_size) for group_size in self._group_sizes)])
        total_sparse = 0


        # 
        # Iterate through all buffers
        for i,group_size in enumerate(self._group_sizes):
            # 
            # For each buffer, first calculate whether the tensors in each layer of the buffer need to be compressed, corresponding to layer_Sparsity
            # 
            total_numel = sum(group_size);total_count = len(group_size)
            fixed_numel = 0;fixed_count = 0
            total_computation_time = calculation_backward_time_local_8_nodes(all_group_size,total_numel,self._model_net_name)
            for j,layer_size in enumerate(group_size):
                if layer_size <= threshold_arr[i]:
                    fixed_numel += layer_size
                    fixed_count += 1
                    layer_sparsity[self._groups[i][j]] = 1.0
            if fixed_count != total_count:
                assert fixed_count < total_count
                other_sparsity = max(self._density,(total_numel * self._density - fixed_numel) / (total_numel - fixed_numel))
                for j,layer_size in enumerate(group_size):
                    if layer_size > threshold_arr[i]:
                        total_sparse += 1
                        layer_sparsity[self._groups[i][j]] = other_sparsity
                        total_computation_time += calculation_compression_time_local_8_nodes(layer_size)
            
            
            # 
            # =====================================================================
            # 
            cpu_timestamp = 0
            now_time = 0
            last_name = ''

            for j,layer_size in enumerate(group_size):
                now_time += calculation_backward_time_local_8_nodes(all_group_size,layer_size,self._model_net_name)
                if layer_size <= threshold_arr[i]:
                    # Not compressed, therefore it is not necessary and impossible to unload to CPU for compression
                    pass
                elif cpu_timestamp > now_time:
                    # Because there is still a CPU unloading compression task ahead, in order to avoid conflicts, compression will not be unloaded here
                    # 
                    now_time += calculation_compression_time_local_8_nodes(layer_size)
                    pass
                else:
                    # 
                    last_name = self._groups[i][j]
                    on_cpu.add(last_name)
                    cpu_timestamp = now_time + calculation_compression_time_local_8_nodes_cpu(layer_size)

            if cpu_timestamp > now_time:
                # Note that the last tensor will delay the subsequent fusion time, so it should not be unloaded
                on_cpu.remove(last_name)
            # 
            # =====================================================================
            # 
        print(f'tensors that to be offloaded: {len(on_cpu)}')
        print(f'sparsed tensor / total tensors: {total_sparse} / {total_tensors} = {total_sparse/total_tensors}')
        print(on_cpu)
        # raise ValueError(f'{len(on_cpu)} and {len(on_cpu_)} and {on_cpu == on_cpu_}')
        # 


        print("successfully upated the layer sparsity")
        # 
        # You should directly attach layer_sparsity and on_cpu to Optimizer.
        # 
        self._dict = layer_sparsity
        self.on_cpu = on_cpu
        # 
        # self._compression.update_threshold(layer_sparsity,on_cpu)
        # 
        # 


def DistributedOptimizer(model_net_name, optimizer, named_parameters=None, compression=None, is_sparse=False, density=0.001, seq_layernames=None, layerwise_times=None, norm_clip=None, threshold=0, writer=None, gradient_path=None, momentum_correction=False, fp16=False, mgwfbp=False, rdma=False, asc=False):
    """
    An optimizer that wraps another torch.optim.Optimizer, using an allreduce to
    average gradient values before applying gradients to model weights.

    Allreduce operations are executed after each gradient is computed by `loss.backward()`
    in parallel with each other. The `step()` method ensures that all allreduce operations are
    finished before applying gradients to the model.

    DistributedOptimizer exposes the `synchronize()` method, which forces allreduce operations
    to finish before continuing the execution. It's useful in conjunction with gradient
    clipping, or other operations that modify gradients in place before `step()` is executed.

    Example of gradient clipping:
    ```
    output = model(data)
    loss = F.nll_loss(output, target)
    loss.backward()
    optimizer.synchronize()
    torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)
    optimizer.step()
    ```

    Arguments:
        optimizer: Optimizer to use for computing gradients and applying updates.
        named_parameters: A mapping between parameter names and values. Used for naming of
                          allreduce operations. Typically just `model.named_parameters()`.
        compression: Compression algorithm used during allreduce to reduce the amount
                     of data sent during the each parameter update step.  Defaults to
                     not using compression.
    """
    # We dynamically create a new class that inherits from the optimizer that was passed in.
    # The goal is to override the `step()` method with an allreduce implementation.
    cls = type(optimizer.__class__.__name__, (optimizer.__class__,),
               dict(_DistributedOptimizer.__dict__))

    return cls(model_net_name, optimizer.param_groups, named_parameters, compression, is_sparse, density, seq_layernames=seq_layernames, layerwise_times=layerwise_times, norm_clip=None, threshold=threshold, writer=writer, gradient_path=gradient_path, momentum_correction=momentum_correction, fp16=fp16, mgwfbp=mgwfbp,rdma=rdma,asc=asc)


def broadcast_parameters(params, root_rank):
    """
    Broadcasts the parameters from root rank to all other processes.
    Typical usage is to broadcast the `model.state_dict()`,
    `model.named_parameters()`, or `model.parameters()`.

    Arguments:
        params: One of the following:
            - list of parameters to broadcast
            - dict of parameters to broadcast
        root_rank: The rank of the process from which parameters will be
                   broadcasted to all other processes.
    """
    if isinstance(params, dict):
        params = sorted(params.items())
    elif isinstance(params, list):
        # support both named_parameters() and regular parameters()
        params = [p if isinstance(p, tuple) else (None, p) for p in params]
    else:
        raise ValueError('invalid params of type: %s' % type(params))

    # Run asynchronous broadcasts.
    handles = []
    for name, p in params:
        if p is not None:
            handle = broadcast_async_(p, root_rank, name)
            handles.append(handle)

    # Wait for completion.
    for handle in handles:
        synchronize(handle)


def broadcast_optimizer_state(optimizer, root_rank,model=None):
    """
    Broadcasts an optimizer state from root rank to all other processes.

    Arguments:
        optimizer: An optimizer.
        root_rank: The rank of the process from which the optimizer will be
                   broadcasted to all other processes.
    """
    from horovod.torch.optimizer import DistributedOptimizer
    if isinstance(optimizer, torch.optim.LBFGS):
        # TODO(travis): L-BFGS cannot be easily supported without serializing
        # the entire state_dict, as its structure is deeply nested and contains
        # None type parameter values
        raise ValueError('cannot broadcast torch.optim.LBFGS state')

    state_dict = optimizer.state_dict()
    
    # identify sparse parameters
    sparse_params = []
    if model:
        for m in model.modules():
            if isinstance(m, torch.nn.modules.sparse.Embedding) and m.sparse:
                for p in m.parameters():
                    sparse_params.append(id(p))

    # Newly created optimizers will not have their state initialized, so
    # do that initialization here
    if len(state_dict['state']) == 0:
        for group in optimizer.param_groups:
            for p in group['params']:
                p.grad = p.data.new(p.size()).zero_()
        # This function accepts a torch.optim.Optimizer or a DistributedOptimizer
        # wrapped around a torch optimizer. Calling step() with a DistributedOptimizer
        # forces allreduce on all model parameters, which will result in deadlock
        # unless every rank calls step(). Therefore, to finish state initialization
        # only call optimizer.step() with a torch.optim.Optimizer.
        if optimizer.__module__ == DistributedOptimizer.__module__:
            super(optimizer.__class__, optimizer).step()
        else:
            optimizer.step()
        state_dict = optimizer.state_dict()

    # If the state_dict is still empty after initialization, then
    # the optimizer is stateless, and there is nothing to broadcast.
    # Furthermore, attempting to access the state dict would result in
    # an error.
    if len(state_dict['state']) == 0:
        return

    params = []
    callbacks = {}
    occurrences = collections.defaultdict(int)

    # Returns the full type structure of the possibly nested objects for recursive casting back
    def _get_types(x):
        if isinstance(x, collections.Iterable):
            return type(x), [_get_types(xi) for xi in x]
        else:
            return type(x)

    # Casts an object encoded in a tensor back into its original type and subtypes
    def _recursive_cast(x, dtype):
        if isinstance(dtype, tuple):
            t, dtypes = dtype
            x = t(x)
            return t([_recursive_cast(x[i], dtypes[i]) for i in range(len(x))])
        else:
            return dtype(x)
    
    # 
    # 
    # Some optimizer parameters may be represented as scalars instead of
    # tensors.  In such cases, we need to wrap the scalar in a tensor, then
    # broadcast, then update the appropriate value in the state_dict with the
    # new unwrapped scalar value via a callback.
    # def _create_callback(pid, name, t, p):
    #     def _from_tensor():
    #         state_dict['state'][pid][name] = t(p.numpy()[0])
    #     return _from_tensor
    # 
    # 
    
    
    # 
    # def _create_option_callback(index, option_key, option_tensor, dtypes):
    #     def _from_tensor():
    #         optimizer.param_groups[index][option_key] = _recursive_cast(option_tensor.numpy()[0], dtypes)
    #     return _from_tensor
    def _create_state_callback(pid, name):
        def _assign_state(v):
            state_dict['state'][pid][name] = v
        return _assign_state

    def _create_option_callback(index, option_key):
        def _assign_option(v):
            optimizer.param_groups[index][option_key] = v
        return _assign_option
    scalars = {}
    
    
    # Param groups are an ordered list, normally there is only one per model,
    # but users can add additional param groups for example to train
    # previously frozen layers
    for index, group in enumerate(state_dict['param_groups']):
        # Broadcast options like learning rate
        for option_key, option_value in group.items():
            if option_key == 'params':
                continue

            # Options like the learning rate are scalar, and need to be wrapped in tensors
            key = '%s.%d' % (option_key, index)
            # dtypes = _get_types(option_value)
            # option_tensor = torch.Tensor([option_value])
            
            scalars[key] = option_value
            callbacks[key] = _create_option_callback(index, option_key)
            # params.append((key, option_tensor))

        # The params list here is ordered by the layers in the model
        for pid in group['params']:
            param_state = state_dict['state'][pid]
            for name, p in param_state.items():
                # Some parameter names may appear more than once, in which
                # case we ensure they have a unique identifier defined by
                # their order
                occurrences[name] += 1
                key = '%s.%d' % (str(name), occurrences[name])

                if p is not None and not torch.is_tensor(p):
                    # Wrap the scalar in a FloatTensor, and remember its type
                    # so we can cast it back after unwrapping
                    # t = type(p)
                    # p = torch.Tensor([p])
                    callbacks[key] = _create_state_callback(pid, name)

                params.append((key, p))

    # Synchronized broadcast of all parameters
    broadcast_parameters(params, root_rank)

    # Post-broadcast clenaup for non-tensor parameters
    for key, p in params:
        if key in callbacks:
            callbacks[key]()














