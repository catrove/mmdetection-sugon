#!/bin/bash
#SBATCH -p caspra
#SBATCH -N 8
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --gres=dcu:4
#SBATCH -J cnt
module load compiler/devtoolset/7.3.1 
module load compiler/rocm/2.9
module load mpi/hpcx/2.4.1/gcc-7.3.1
module load apps/PyTorch/1.3.0a0/hpcx-2.4.1-gcc-7.3.1
which python3
which mpirun
hostfile=./$SLURM_JOB_ID
scontrol show hostnames $SLURM_JOB_NODELIST > ${hostfile}
num_node=$(cat $hostfile|sort|uniq |wc -l)
 
num_DCU=$(($num_node*4))
nodename=$(cat $hostfile |sed -n "1p")
dist_url=`echo $nodename | awk '{print $1}'`
 
rm `pwd`/hostfile-dl -f
cat $hostfile|sort|uniq >`pwd`/tmp
 
for i in `cat ./tmp`
do
    echo ${i} slots=4 >> `pwd`/hostfile-dl
done

export MIOPEN_DISABLE_CACHE=1
##multi-node multi-gpu
mpirun -np $num_DCU --allow-run-as-root --hostfile `pwd`/hostfile-dl --bind-to none `pwd`/single_process_eval.sh $dist_url

#single-node multi-gpu 
#mpirun -np $num_DCU --allow-run-as-root --bind-to none `pwd`/single_process.sh $dist_url

#single-node multi-gpu
#python3 main.py --batch-size=128  --arch=resnet50 --workers 24 --epochs=1 /public/software/apps/DeepLearning/Data/ImageNet-pytorch/

#sing-node sing-gpu
#python3 main.py --batch-size=32  --arch=resnet50 --workers 6 --epochs=1 --gpu=0  /public/software/apps/DeepLearning/Data/ImageNet-pytorch/  
