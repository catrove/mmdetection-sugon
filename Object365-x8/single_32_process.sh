#!/bin/bash
export MIOPEN_DEBUG_DISABLE_FIND_DB=1
export GLOO_SOCKET_IFNAME=ib0,ib1,ib2,ib3
export MIOPEN_USER_DB_PATH=/tmp/pytorch-miopen-2.8
export HSA_USERPTR_FOR_PAGED_MEM=0

 
lrank=$OMPI_COMM_WORLD_LOCAL_RANK
comm_rank=$OMPI_COMM_WORLD_RANK
comm_size=$OMPI_COMM_WORLD_SIZE
module load compiler/devtoolset/7.3.1
module load compiler/rocm/2.9
module load mpi/hpcx/2.4.1/gcc-7.3.1
module load apps/PyTorch/1.3.0a0/hpcx-2.4.1-gcc-7.3.1
## default gloo backend
APP="python3 -u ../tools/gloo_train.py gloo_32x4_centripetalnet_mask_hg104.py --dist_url tcp://${1}:34567 --world_size=${comm_size} --rank=${comm_rank} --launcher=gloo "
case ${lrank} in
[0])
  export HIP_VISIBLE_DEVICES=0
  export UCX_NET_DEVICES=mlx5_0:1
  export UCX_IB_PCI_BW=mlx5_0:50Gbs
  GLOO_SOCKET_IFNAME=ib0 numactl --cpunodebind=0 --membind=0 ${APP}
  ;;
[1])
  export HIP_VISIBLE_DEVICES=1
  export UCX_NET_DEVICES=mlx5_1:1
  export UCX_IB_PCI_BW=mlx5_1:50Gbs
  GLOO_SOCKET_IFNAME=ib1 numactl --cpunodebind=1 --membind=1 ${APP}
  ;;
[2])
  export HIP_VISIBLE_DEVICES=2
  export UCX_NET_DEVICES=mlx5_2:1
  export UCX_IB_PCI_BW=mlx5_2:50Gbs
  GLOO_SOCKET_IFNAME=ib2 numactl --cpunodebind=2 --membind=2 ${APP}
  ;;
[3])
  export HIP_VISIBLE_DEVICES=3
  export UCX_NET_DEVICES=mlx5_3:1
  export UCX_IB_PCI_BW=mlx5_3:50Gbs
  GLOO_SOCKET_IFNAME=ib3 numactl --cpunodebind=3 --membind=3 ${APP}
  ;;
esac

