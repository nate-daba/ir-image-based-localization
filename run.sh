#!/usr/bin/env bash

python train.py \
--workers=8 \
--epochs=200 \
--batch-size=16 \
--learning-rate=0.001 \
--momentum=0.9 \
--weight-decay=1e-4 \
--print-freq=200 \
--save_path='./results' \
--world-size=1 \
--rank=0 \
--dist-url='tcp://localhost:10001' \
--gpu=0 \
--multiprocessing-distributed \
--evaluate \
--dim=1024 \
--cos \
--mining
