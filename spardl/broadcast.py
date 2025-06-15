from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

print(rank)

if rank == 0:
    data = np.arange(4.0)
    

else:
    data = None

data = comm.bcast(data, root=0)

print(data)

if rank == 0:
    print('Process {} broadcast data:'.format(rank), data)
else:
    print('Process {} received data:'.format(rank), data)