#!/usr/bin/env bash

GPUS=$1
CONFIG=$2
PORT=${PORT:-29500}

python3 -m torch.distributed.run --nproc_per_node $GPUS --master_port=$PORT \
    train.py --config $CONFIG ${@:3}
