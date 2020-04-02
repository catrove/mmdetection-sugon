srun -p caspra -N1 -n1 -c8 --gres=dcu:1 python -u tools/train.py centripetalnet_mask_hg104.py --gpus=1 | tee c2mh.log
