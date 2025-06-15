density="${density:-0.01}"
threshold="${threshold:-8192}"
compressor="${compressor:-gaussian}"
max_epochs="${max_epochs:-80}"
nwpernode=4
nstepsupdate=1
PY=python

HOROVOD_GPU_OPERATIONS=NCCL HOROVOD_CACHE_CAPACITY=0 horovodrun -np 8 -H node15:1,node16:1,node17:1,node18:1,node19:1,node20:1,node21:1,node22:1   python  pytorch_cifar100_resnet50.py  --epochs $max_epochs --density 0.05 --compressor gaussian --threshold $threshold