import torch
import pprint
from grace_lib import Communicator
from wfbp.torch import allgather, allgather_async, synchronize
import wfbp.torch as hvd
import time


class Allgather(Communicator):
    def __init__(self, compressor, memory, world_size):
        super().__init__(compressor, memory)
        self.world_size = world_size
        
    
    def async_send(self, tensors_compressed, name):
        """
        :param tensors_compressed: list of flat tensors to communicate
        :param name: for the all_gather operation
        :return: handles to synchronize, tensor sizes per rank
        """
        # list of tensor size for this rank, allgather on the 1st dimension of the tensor\
        # 这个rank的张量大小的列表,全部聚集在张量的第一维上
        tensors_size = []
        for t in tensors_compressed:
            size_dim0 = t.size()[0] if len(t.size())>0 else t.numel()
            tensors_size.append(size_dim0)
        
        # 判断是否要求压缩的tensor维度大小相同
        if self.compressor.tensors_size_are_same:
            tensors_size_ag = [tensors_size] * self.world_size  # list of tensor sizes per rank
            tensor_sizes = zip(*tensors_size_ag)  # transpose
        else:
            tensors_size = torch.tensor(tensors_size)  # TODO: set device
            gathered = allgather(tensors_size)  # tensor of tensor sizes per rank
            tensor_sizes = gathered.view([self.world_size, -1]).t().tolist()  # transpose, to list
        
        handles = []
        for tensor_compressed in tensors_compressed:
            handle = allgather_async(tensor_compressed)
            handles.append(handle)

        return handles, tensor_sizes


    def wait_receive(self, result, ctx, name):

        if ctx != None:
            return self.block_wait_receive(result, ctx, name)
        
        handles, tensor_sizes = result
        tensors_ag = []
        gathered_list = []
        
        time_start=time.time()
        
        # 2 times: val and idx
        for handle, sizes in zip(handles, tensor_sizes):
            gathered = synchronize(handle) if not callable(handle) else handle()
            gathered_list.append(gathered)
            tensors_ag.append(gathered.split(sizes))
        
        list_tensor_decompressed = []
        
        time_end=time.time()
        synchronize_time=time_end-time_start
        
        # n times: n is the number of nodes
        for tensor_compressed in zip(*tensors_ag):
            # tensor_decompressed = self.compressor.decompress(tensor_compressed, ctx, name)
            tensor_decompressed = self.compressor.decompress_add(tensor_compressed, ctx, name)

            list_tensor_decompressed.append(tensor_decompressed)
        decompression_time=time.time()-time_end
        
        tensors_aggregated = self.compressor.aggregate(list_tensor_decompressed)

        return (tensors_aggregated / self.world_size) if self.compressor.average else tensors_aggregated, synchronize_time, decompression_time



    def block_wait_receive(self, result, ctx, name):
        handles, tensor_sizes = result
        tensors_ag = []
        
        # 2 times
        # for handle, sizes in zip(handles, tensor_sizes):
        #     gathered = synchronize(handle)
        #     tensors_ag.append(gathered)
        # list_tensor_decompressed = []
        
        time_start=time.time()
        # 1 times 
        for handle in handles:
            gathered = synchronize(handle)
            tensors_ag.append(gathered)


        # 处理压缩的梯度，len(tensors_ag)==2
        # 1 times, use 
        tensor_compressed = tensors_ag[0], tensors_ag[1]
        
        time_end=time.time()
        synchronize_time=time_end-time_start
        
        # if hvd.rank()==0:
        #     print(name,'_synchronize_time:',synchronize_time)

        tensor_decompressed = self.compressor.decompress_add(tensor_compressed, ctx, name)

        decompression_time=time.time()-time_end
        
        return tensor_decompressed / self.world_size, synchronize_time, decompression_time

    # def test_block_wait_receive(self, result, ctx, gathered_list):
    #     handles, tensor_sizes = result
    #     tensors_ag = gathered_list

    #     # 1 times, use
    #     tensor_compressed = tensors_ag[0], tensors_ag[1]
    #     if hvd.rank() == 0:
    #         print(tensor_compressed)
    #     tensor_decompressed = self.compressor.decompress_add(tensor_compressed, ctx)
    #     if hvd.rank() == 0 and tensor_decompressed.numel() < 100:
    #         print('my: ', tensor_decompressed)
    #     return (tensor_decompressed / self.world_size) if self.compressor.average else tensors_aggregated

