# coding: utf-8
import argparse
import time
import math
import os
import torch
import torch.nn as nn
import torch.onnx
import torch.optim as optim
import wfbp.torch as hvd

import datahelper
import model
from torch.optim import lr_scheduler
import numpy as np 
torch.backends.cudnn.benchmark = True

os.environ['HOROVOD_FUSION_THRESHOLD'] = '0'
os.environ['HOROVOD_CACHE_CAPACITY'] = '0'
os.environ['HOROVOD_CYCLE_TIME'] = '0'


import example_syncea.hv_distributed_optimizer_synea as hvd
from compression import compressors
# from utils_model import get_network



# same hyperparameter scheme as word-language-model
parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 RNN/LSTM Language Model')
# parser.add_argument('--data', type=str, default='/home/user/nlp/data/wikitext-2',
parser.add_argument('--data', type=str, default='/data/dataset/nlp/lstm/wikitext-2',help='location of the data corpus')

# Docker
# parser.add_argument('--data', type=str, default='/horovod/dataset/nlp/lstm/wikitext-2',
#                     help='location of the data corpus')

parser.add_argument('--model-net', default='lstm',type=str, help='net type')

parser.add_argument('--model', type=str, default='Transformer',
                    help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)')


parser.add_argument('--emsize', type=int, default=400,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=400,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=8,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=0.002,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.1,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=100,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=100,

                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--tied', action='store_true', default=True,
                    help='tie the word embedding and softmax weights')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--cudnn', action='store_true',
                    help='use cudnn optimized version. i.e. use RNN instead of RNNCell with for loop')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str, default='model.pt',
                    help='path to save the final model')
parser.add_argument('--onnx-export', type=str, default='',
                    help='path to export the final model in onnx format')
parser.add_argument('--resume', type=int, default=None,
                    help='if specified with the 1-indexed global epoch, loads the checkpoint and resumes training')
parser.add_argument('--nhead', type=int, default=10,
                    help='the number of heads in the encoder/decoder of the transformer model')

# parameters for adaptive softmax
parser.add_argument('--adaptivesoftmax', action='store_true',
                    help='use adaptive softmax during hidden state to output logits.'
                         'it uses less memory by approximating softmax of large vocabulary.')
parser.add_argument('--cutoffs', nargs="*", type=int, default=[10000, 50000, 100000],
                    help='cutoff values for adaptive softmax. list of integers.'
                         'optimal values are based on word frequencey and vocabulary size of the dataset.')

# experiment name for this run
parser.add_argument('--name', type=str, default=None,
                    help='name for this experiment. generates folder with the name if specified.')


# MGradient Merging
parser.add_argument('--fp16', action='store_true', default=False,
                    help='use fp16 compression during allreduce')

parser.add_argument('--num-warmup-batches', type=int, default=20,
                    help='number of warm-up batches that don\'t count towards benchmark')
parser.add_argument('--num-batches-per-iter', type=int, default=10,
                    help='number of batches per benchmark iteration')
parser.add_argument('--num-iters', type=int, default=50,
                    help='number of benchmark iterations')

parser.add_argument('--mgwfbp', action='store_true', default=False, help='Use MG-WFBP')
parser.add_argument('--asc', action='store_true', default=False, help='Use MG-WFBP')
parser.add_argument('--nstreams', type=int, default=1, help='Number of communication streams')

parser.add_argument('--threshold', type=int, default=34015396, help='Set threshold if mgwfbp is False')
parser.add_argument('--rdma', action='store_true', default=False, help='Use RDMA')



parser.add_argument('--compressor', type=str, default = 'dgc', help='Specify the compressors if density < 1.0')
parser.add_argument('--memory', type=str, default = 'residual', help='Error-feedback')
parser.add_argument('--density', type=float, default=0.1, help='Density for sparsification')








args = parser.parse_args()

ppl_list = []
time_list = []

# horovod 0
hvd.init()

# horovod 1
if torch.cuda.is_available():
    torch.cuda.set_device(hvd.local_rank())
    # print(hvd.local_rank())
    torch.cuda.manual_seed(args.seed)

# if args.cuda:
device = torch.device("cuda")


###############################################################################
# Load data
###############################################################################

corpus = datahelper.Corpus(args.data) 


# Starting from sequential data, batchify arranges the dataset into columns.
# For instance, with the alphabet as the sequence and batch size 4, we'd get
# ┌ a g m s ┐
# │ b h n t │
# │ c i o u │
# │ d j p v │
# │ e k q w │
# └ f l r x ┘.
# These columns are treated as independent by the model, which means that the
# dependence of e. g. 'g' on 'f' can not be learned, but allows more efficient
# batch processing.

def batchify(data, bsz):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)

size = hvd.size()

eval_batch_size = 10

train_datas = batchify(corpus.train, args.batch_size*size)



num_columns = train_datas.shape[1] 
column_indices = np.arange(num_columns)  

val_data = batchify(corpus.valid, eval_batch_size)
test_data = batchify(corpus.test, eval_batch_size)

ntokens = len(corpus.dictionary)
if args.model == 'Transformer':
    # model = model.TransformerModel(ntokens, args.emsize, args.nhead, args.nhid, args.nlayers, args.dropout).to(device)
    model = model.TransformerModel(ntokens, args.emsize, args.nhead, args.nhid, args.nlayers, args.dropout).to(device)
else:
    model = model.RNNModel(args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.dropout, args.tied).to(device)

criterion = nn.NLLLoss()

if hvd.rank() == 0:
    for name, param in model.named_parameters():
        print(name,':',param.size())

def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


# get_batch subdivides the source data into chunks of length args.bptt.
# If source is equal to the example output of the batchify function, with
# a bptt-limit of 2, we'd get the following two Variables for i = 0:
# ┌ a g m s ┐ ┌ b h n t ┐
# └ b h n t ┘ └ c i o u ┘
# Note that despite the name of the function, the subdivison of data is not
# done along the batch dimension (i.e. dimension 1), since that was handled
# by the batchify function. The chunks are along dimension 0, corresponding
# to the seq_len dimension in the LSTM.


def get_batch(source, i):
    seq_len = min(args.bptt, len(source) - 1 - i)
    data = source[i:i + seq_len]
    target = source[i + 1:i + 1 + seq_len].contiguous().view(-1)
    return data, target


def evaluate(data_source):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0.
    ntokens = len(corpus.dictionary)
    if args.model != 'Transformer':
        hidden = model.init_hidden(eval_batch_size)
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, args.bptt):
            data, targets = get_batch(data_source, i)
            if args.model == 'Transformer':
                output = model(data)
                output = output.view(-1, ntokens)
            else:
                output, hidden = model(data, hidden)
                hidden = repackage_hidden(hidden)
            total_loss += len(data) * criterion(output, targets).item()
    return total_loss / (len(data_source) - 1)

# 备份
def train(optimizer, train_data):
    
    optimizer._compression.compress_time=[]
    optimizer._compression.threshold_time=[]
    
    optimizer.synchronize_time= []
    optimizer.para_update_time= []
    optimizer.hook_time= []

    io_time_array= []
    forward_backforward_time_array= []
    forward_time_array= []
    backward_time_array= []
    step_time_array= []
    update_time_array= []
    
    optimizer.handle_synchronize_time= []
    optimizer_synchronize_time_array= []

    # Turn on training mode which enables dropout.
    model.train()
    total_loss = 0.
    start_time = time.time()
    ntokens = len(corpus.dictionary)
    if args.model != 'Transformer':
        hidden = model.init_hidden(args.batch_size)
    for batch, i in enumerate(range(0, train_data.size(0) - 1, args.bptt)):
        s_time=time.time()
        data, targets = get_batch(train_data, i)
        io_time_array.append(time.time()-s_time)
        
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        e_time=time.time()
        model.zero_grad()
        if args.model == 'Transformer':
            output = model(data)
            output = output.view(-1, ntokens)
        else:
            hidden = repackage_hidden(hidden)
            output, hidden = model(data, hidden)
        loss = criterion(output, targets)
        forward_time_array.append(time.time()-e_time)
        
        b_time=time.time()
        loss.backward()
        backward_time_array.append(time.time()-b_time)

        s_time=time.time()  
        optimizer.step()
        step_time_array.append(time.time()-s_time)

        u_time=time.time()
        total_loss += loss.item()
        
        
        # if batch % args.log_interval == 0 and batch > 0 and hvd.local_rank()==0:
        if batch % 10 == 0 and batch > 0 and hvd.rank()==0:
            cur_loss = total_loss / args.log_interval
            elapsed = time.time() - start_time
            # batches = train_data[0] // bptt
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.7f} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, batch, len(train_data) // args.bptt, optimizer.param_groups[0]['lr'],
                elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()
        update_time_array.append(time.time()-u_time)
        
        optimizer_synchronize_time_array.append(optimizer.handle_synchronize_time)
        optimizer.handle_synchronize_time= []
    
    # end_time_epoch = time.time()
    # x_train_epoch_time.append(end_time_epoch - modified_time)


# Loop over epochs.
best_val_loss = None

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr*hvd.size(),weight_decay=1.2e-5) 
# optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1.2e-6) 

from torch.optim.lr_scheduler import _LRScheduler 
class WarmupLR(_LRScheduler):
    """The WarmupLR scheduler

    This scheduler is almost same as NoamLR Scheduler except for following
    difference:

    NoamLR:
        lr = optimizer.lr * model_size ** -0.5
             * min(step ** -0.5, step * warmup_step ** -1.5)
    WarmupLR:
        lr = optimizer.lr * warmup_step ** 0.5
             * min(step ** -0.5, step * warmup_step ** -1.5)

    Note that the maximum lr equals to optimizer.lr in this scheduler.

    """
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: 5,
        last_epoch: int = -1,
    ):
        self.warmup_steps = warmup_steps
        # __init__() must be invoked before setting field
        # because step() is also invoked in __init__()
        super().__init__(optimizer, last_epoch)

    def __repr__(self):
        return f"{self.__class__.__name__}(warmup_steps={self.warmup_steps})"

    def get_lr(self):
        step_num = self.last_epoch + 1
        return [
            lr
            * self.warmup_steps ** 0.5
            * min(step_num ** -0.5, step_num * self.warmup_steps ** -1.5)
            for lr in self.base_lrs
        ]

    def set_step(self, step: int):
        self.last_epoch = step

scheduler_warmup = WarmupLR(optimizer, warmup_steps=100, last_epoch = -1)

# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=1)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)


from grace_lib.helper import get_communicator

if args.density<1:
    communicator_str = 'allgather'
else:
    communicator_str = 'allreduce'
    
params = {'compressor': args.compressor, 'memory': args.memory, 'density': args.density,'communicator': 'allgather','model_named_parameters':model.named_parameters()}


seq_layernames, layerwise_times = None, None
optimizer = hvd.DistributedOptimizer(args.model_net, optimizer, 
                                         named_parameters=model.named_parameters(), compression=compressors[args.compressor](), is_sparse=args.density<1, density=args.density, seq_layernames=seq_layernames, layerwise_times=layerwise_times, norm_clip=None, threshold=args.threshold, writer=None, gradient_path='./', momentum_correction=False, fp16=args.fp16, mgwfbp=args.mgwfbp, rdma=args.rdma, asc=args.asc)

np.random.seed(0)
np.random.shuffle(column_indices) 
train_data = train_datas[:, column_indices]

train_data = torch.chunk(train_data, size , dim=1)
train_data = train_data[hvd.rank()]


def train_step(optimizer, train_data, batch):

    # Turn on training mode which enables dropout.
    model.train()
    total_loss = 0.
    start_time = time.time()
    ntokens = len(corpus.dictionary)
    if args.model != 'Transformer':
        hidden = model.init_hidden(args.batch_size)
    model.zero_grad()
    if args.model == 'Transformer':
        output = model(data)
        output = output.view(-1, ntokens)
    else:
        hidden = repackage_hidden(hidden)
        output, hidden = model(data, hidden)
    
    loss = criterion(output, targets)
    loss.backward()
    optimizer.step()
    total_loss += loss.item()
    if batch > 0 and hvd.rank()==0:
        cur_loss = total_loss
        elapsed = time.time() - start_time
        # batches = train_data[0] // bptt
        print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.7f} | ms/batch {:5.2f} | '
                'loss {:5.2f} | ppl {:8.2f}'.format(
            epoch, batch, len(train_data) // args.bptt, optimizer.param_groups[0]['lr'],
            elapsed * 1000, cur_loss, math.exp(cur_loss)))
        total_loss = 0
        start_time = time.time()




try:
    start_time= time.time()  
    start_time_epochs= time.time() 
    for epoch in range(1, args.epochs+1):         
        optimizer._compression.epoch=epoch
        optimizer._compression.compress_time=[]
        optimizer._compression.threshold_time=[]
    
        optimizer.synchronize_time= []
        optimizer.para_update_time= []
        optimizer.hook_time= []
        
        # optimizer._communicator.compressor.bias_gaussiank=[]
        # optimizer._communicator.compressor.bias_dgc=[]
        # optimizer._communicator.compressor.bias_redsync=[]
    
        # optimizer._communicator.compression_time_array=[]
        # optimizer._communicator.decompression_time_array=[]
        # optimizer._communicator.send_time_array=[]
        # optimizer._communicator.receive_time_array=[]
        # optimizer._communicator.synchronize_time_array=[]

        io_time_array= []
        forward_backforward_time_array= []
        forward_time_array= []
        backward_time_array= []
        step_time_array= []
        update_time_array= []
    
        optimizer.handle_synchronize_time= []
        optimizer_synchronize_time_array= []
        epoch_start_time = time.time() 
        for batch, i in enumerate(range(0, train_data.size(0) - 1, args.bptt)):         
            s_time=time.time()
            data, targets = get_batch(train_data, i) 
            io_time_array.append(time.time()-s_time)
            
            # train_step(optimizer, train_data, batch)
            model.train()
            total_loss = 0.
            start_time = time.time()
            
            e_time=time.time()
            ntokens = len(corpus.dictionary)
            if args.model != 'Transformer':
                hidden = model.init_hidden(args.batch_size)
            model.zero_grad()
            if args.model == 'Transformer':
                output = model(data)
                output = output.view(-1, ntokens)
            else:
                hidden = repackage_hidden(hidden)
                output, hidden = model(data, hidden)
    
            loss = criterion(output, targets)
            forward_time_array.append(time.time()-e_time)
            
            b_time=time.time()
            loss.backward()
            backward_time_array.append(time.time()-b_time)
            
            s_time=time.time()  
            optimizer.step()
            step_time_array.append(time.time()-s_time)
            
            u_time=time.time()
            total_loss += loss.item()
            update_time_array.append(time.time()-u_time)

            if batch %10== 0 and hvd.rank()==0:
                cur_loss = total_loss
                elapsed = time.time() - start_time
                # batches = train_data[0] // bptt
                print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.7f} | ms/batch {:5.2f} | '
                        'loss {:5.2f} | ppl {:8.2f}'.format(
                    epoch, batch, len(train_data) // args.bptt, optimizer.param_groups[0]['lr'],
                    elapsed * 1000, cur_loss, math.exp(cur_loss)))
                total_loss = 0
                start_time = time.time()
            
            update_time_array.append(time.time()-u_time)
            optimizer_synchronize_time_array.append(optimizer.handle_synchronize_time)
            optimizer.handle_synchronize_time= []
            
            if  hvd.rank() == 0:
                scheduler_warmup.step(batch/15) 
            
            
            
        if  hvd.rank() == 0:
            # scheduler_warmup.step(batch/15) 
            val_loss = evaluate(val_data) 
            ppl_list.append(math.exp(val_loss))
            
            # ppl_test = [math.exp(test_loss)]
                          
            print('-' * 89)
            tmp = time.time() - epoch_start_time           
            print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.4f} | ' 'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time), val_loss, math.exp(val_loss)))
            time_list.append(tmp)            
            print('-' * 89) 
        
        # end_time_epoch = time.time()
        # x_train_epoch_time.append(end_time_epoch - modified_time)
    
    if hvd.rank() == 0:
         # torch.cuda.synchronize()
        end_time = time.time()
        end_time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        print('end_time_str = ', end_time_str)
        print('end_time - start_time = ', end_time - start_time_epochs)


except KeyboardInterrupt:     
    print('-' * 89)     
    print('Exiting from training early') 



if hvd.rank() == 0:     
    test_loss = evaluate(test_data)     
    print('=' * 89)     
    print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(test_loss, math.exp(test_loss)))     
    print('=' * 89) 
    ppl_test = [math.exp(test_loss)]


if hvd.rank() == 0:
    import numpy as np           
    time_arr = np.array(time_list)
    
    # val loss
    ppl_arr = np.array(ppl_list)
    # train loss
    ppl_test = np.array(ppl_test)
