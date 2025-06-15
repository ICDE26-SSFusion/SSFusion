"""
This code uses portions of code from GRACE

Hang Xu, Chen-Yu Ho, Ahmed M Abdelmoniem, Aritra Dutta, El Houcine Bergou, Konstantinos Karatsenidis, Marco Canini, and Panos Kalnis. GRACE: A Compressed Communication Framework for Distributed Machine Learning. In Proc. of ICDCS, 2021.

"""

from abc import ABC, abstractmethod
import time


class Memory(ABC):
    @abstractmethod
    def compensate(self, tensor, name):
        """Update the tensor with the residuals."""
        raise NotImplemented("compensate was not implemented.")

    def update(self, tensor, name, compressor, tensor_compressed, ctx):
        """Update the residuals."""
        pass

class Compressor(ABC):
    """Interface for compressing and decompressing a given tensor."""

    def __init__(self, average=True, tensors_size_are_same=True):
        self.average = average

        self.tensors_size_are_same = tensors_size_are_same
        # self.tensors_size_are_same = False

    @abstractmethod
    def compress(self, tensor, name):
        """Compresses a tensor and returns it with the context needed to decompress it."""
        raise NotImplemented("compress was not implemented.")

    @abstractmethod
    def decompress(self, tensors, ctx):
        """Decompress the tensor with the given context."""
        raise NotImplemented("decompress was not implemented.")

    def aggregate(self, tensors):
        """Aggregate a list of tensors."""
        return sum(tensors)

class Communicator(ABC):
    @abstractmethod
    def async_send(self, tensors, name):
        raise NotImplemented("async_send was not implemented.")

    @abstractmethod
    def wait_receive(self, handles, ctx, name=None):
        raise NotImplemented("wait_receive was not implemented.")

    def __init__(self, compressor, memory):
        self.compressor = compressor
        self.memory = memory
        self.tensor = {}
        self.tensors_compressed=None
        self.epoch = 500
        
        # compression+ef
        self.compression_time_array=[]
        self.decompression_time_array=[]
        self.send_time_array=[]
        self.receive_time_array=[]
        self.synchronize_time_array=[]
        

    def send_step(self, tensor, name):        
        
        if tensor.dim()==1 : 
            tensors_compressed,ctx =[tensor], None
            handles = self.async_send(tensors_compressed, name)
            return handles, ctx

        # self.epoch for warm-up training
        # if tensor.dim()==1 or 'rnn.weight_hh' in name: 
        #     tensors_compressed,ctx =[tensor], None
        #     handles = self.async_send(tensors_compressed, name)
        #     return handles, ctx

        # Resnet-50 ACTop-k, All Channel Top-k
        # if tensor.dim()==1:
        # # # if tensor.dim()==1 or tensor.numel()<30000: 
        # # if tensor.numel()<5000 :
        #     tensors_compressed,ctx =[tensor], None
        #     handles = self.async_send(tensors_compressed, name)
        #     return handles, ctx

        # elif 'fc' in name:
        #     tensors_compressed, ctx = self.compressor.compress(tensor, name)
        #     handles = self.async_send(tensors_compressed, name)
        #     return handles, ctx
        
        # VGG-16
        # if tensor.dim()==1 :


        # if tensor.dim()==1 or 'features.0' in name:
        #     tensors_compressed,ctx =[tensor], None
        #     handles = self.async_send(tensors_compressed, name)
        #     return handles, ctx        
        # elif 'classifier.6' in name:
        #     tensors_compressed, ctx = self.compressor.compress(tensor, name)
        #     handles = self.async_send(tensors_compressed, name)
        #     return handles, ctx


        # all compression and all ef
        
        time_start=time.time()
        tensor = self.memory.compensate(tensor, name)
        self.tensor[name]=tensor
        tensors_compressed, ctx = self.compressor.compress(tensor, name)
        self.tensor_decompressed=self.memory.update(tensor, name, self.compressor,
                           tensors_compressed, ctx)
        
        time_end_compression=time.time()
        self.compression_time_array.append(time_end_compression - time_start)
        
        handles = self.async_send(tensors_compressed, name)
        time_end_send=time.time()
        self.send_time_array.append(time_end_send - time_end_compression)
        
        return handles, ctx

    def receive_step(self, handles, ctx,name,tensor):
        time_start=time.time()
        # return self.block_wait_receive(handles, ctx, name)
        tensors_aggregated_avg, synchronize_time,decompression_time=self.wait_receive(handles, ctx, name)
        time_end=time.time()- time_start
        self.decompression_time_array.append(decompression_time)
        self.synchronize_time_array.append(synchronize_time)
        
        self.receive_time_array.append(time_end-decompression_time)

        return tensors_aggregated_avg




