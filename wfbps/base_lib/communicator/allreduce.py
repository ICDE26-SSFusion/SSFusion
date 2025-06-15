from grace_lib import Communicator
from wfbp.torch import allreduce_async_, synchronize
import time


class Allreduce(Communicator):

    def async_send(self, tensors_compressed, name):
        handles = []
        for i, tensor_compressed in enumerate(tensors_compressed):
            # handles.append(allreduce_async_(tensor_compressed, self.compressor.average, name + str(i)))
            # handles.append(allreduce_async_(tensor_compressed, None, name + str(i), op='Sum'))
            handles.append(allreduce_async_(tensor_compressed, None, name + str(i)))
        return handles




    def wait_receive(self, handles, ctx, name):
        
        time_start=time.time()
        output = [synchronize(h) for h in handles]
        time_end=time.time()
        synchronize_time=time_end-time_start
        
        # if hvd.rank()==0:
        #     print(name,'_synchronize_time:',synchronize_time)
        
        # tensor_decompressed = self.compressor.decompress_add(tensor_compressed, ctx, name)

        decompression_time=time.time()-time_end
        
        return self.compressor.decompress(output, ctx, name), synchronize_time, decompression_time
        
        # return self.compressor.decompress(output, ctx, name)
