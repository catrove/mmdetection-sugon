srun -p caspra -N1 -n1 -c8 --gres=dcu:4 python -u ../tools/train.py gloo_32x4_centripetalnet_mask_hg104.py --gpus=1 | tee single_card.log
