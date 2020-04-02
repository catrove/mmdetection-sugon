import torch
import deform_conv_cuda

device = torch.device("cuda:0")
a_list = [ torch.zeros([10,10],device=device) for _ in range(6)]
b_list = [int(0)] * 15
#print(deform_conv_cuda.deform_conv_forward_cuda(*a_list, *b_list))
print(deform_conv_cuda.deformable_im2col(*a_list[:2],*b_list[:13],a_list[3]))
