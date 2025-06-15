import wfbp.torch as hvd 

def get_compressor(params):
    comp = params.get('compressor', 'none')
    world_size = hvd.size()     
     
    cur_rank=hvd.rank()

    if comp == 'dgc':
        from grace_lib.compressor.dgc import DgcCompressor
        density = params.get('density', 0.3)
        compressor = DgcCompressor(density)
    elif comp == 'none':
        from grace_lib.compressor.none import NoneCompressor
        compressor = NoneCompressor()
    elif comp == 'dgc':
        from grace_lib.compressor.topk import TopKCompressor
        density = params.get('density', 0.01)
        compressor = TopKCompressor(density,rank=cur_rank)
    
    elif comp == 'gaussiank':
        from grace_lib.compressor.gaussiank import GaussiankCompressor
        density = params.get('density', 0.01)
        compressor = GaussiankCompressor(density,rank=cur_rank)
    
    elif comp == 'redsync':
        from grace_lib.compressor.redsync import RedSyncCompressor
        density = params.get('density', 0.01)
        compressor = RedSyncCompressor(density,rank=cur_rank)
    elif comp == 'redsynctrim':
        from grace_lib.compressor.redsync import RedSyncTrimCompressor
        density = params.get('density', 0.01)
        compressor = RedSyncTrimCompressor(density,rank=cur_rank)
    
    elif comp == 'sidcoexp':
        from grace_lib.compressor.sidco import ExpCompressor
        density = params.get('density', 0.01)
        compressor = ExpCompressor(density)
    elif comp == 'sidcogp':
        from grace_lib.compressor.sidco import GParetoCompressor
        density = params.get('density', 0.01)
        compressor = GParetoCompressor(density)
    elif comp == 'sidcogam':
        from grace_lib.compressor.sidco import GammaGParetoCompressor
        density = params.get('density', 0.01)
        compressor = GammaGParetoCompressor(density)

    elif comp == 'topkef':
        from grace_lib.compressor.topkef import TopKEFCompressor
        # density = params.get('density', 0.3)
        density = params.get('density', 0.1)
        model_named_parameters = params.get('model_named_parameters')
        # density = params.get('density', 0.001)
        compressor = TopKEFCompressor(density,rank=cur_rank)
        compressor.initialize(model_named_parameters)
    

  
    
    elif comp == 'randomk':
        from grace_lib.compressor.randomk import RandomKCompressor
        # density = params.get('density', 0.3)
        density = params.get('density', 0.01)
        model_named_parameters = params.get('model_named_parameters')
        # density = params.get('density', 0.001)
        compressor = RandomKCompressor(density,rank=cur_rank)
        # compressor.initialize(model_named_parameters)

    

    
    elif comp == 'imbalancetopktime':
        from grace_lib.compressor.imbalancetopktime import ImbalanceTopkTimeCompressor
        density = params.get('density', 0.01)
        model_named_parameters = params.get('model_named_parameters')
        compressor = ImbalanceTopkTimeCompressor(density, rank=hvd.rank())
        compressor.initialize(model_named_parameters)
        

    else:
        raise NotImplementedError(compressor)
    
    return compressor

def get_memory(params):
    
    mem = params.get('memory', 'none') 
    if mem == 'dgc':
        from grace_lib.memory.dgc import DgcMemory
        momentum = params.get('momentum', 0.9)
        gradient_clipping = params.get('gradient_clipping', False)
        memory = DgcMemory(momentum, gradient_clipping)
    elif mem == 'none':
        from grace_lib.memory.none import NoneMemory
        memory = NoneMemory()
   
    elif mem == 'residual':
        from grace_lib.memory.residual import ResidualMemory
        memory = ResidualMemory()
    elif mem == 'residualgtopk':
        from grace_lib.memory.residualgtopk import ResidualGlobalTopkMemory
        memory = ResidualGlobalTopkMemory()
    else:
        raise NotImplementedError(mem)

    return memory



def get_communicator(params):
   
    world_size = hvd.size()
 
    cur_rank=hvd.rank()
 
    rank=params.get('rank', 0)     
    cur_epoch=params.get('cur_epoch')
    
    comm = params.get('communicator', 'allreduce')
    
    compressor = get_compressor(params)
    memory = get_memory(params)

    if comm == 'allreduce':
        from grace_lib.communicator.allreduce import Allreduce
        return Allreduce(compressor, memory)
    elif comm == 'allgather':
        from grace_lib.communicator.allgather import Allgather
        return Allgather(compressor, memory, world_size)
    else:
        raise NotImplementedError(comm)


    