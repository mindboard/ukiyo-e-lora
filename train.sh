#!/bin/bash
python train_lora.py \
  --data_root dataset \
  --output_dir output/lora_ukiyoe \
  --batch_size 2 \
  --grad_accum 2 \
  --lora_rank 64 \
  --lora_alpha 64 \
  --max_steps 1500 \
  --save_every 100

