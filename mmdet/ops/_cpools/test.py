import torch 
import right_pool
device = torch.device("cuda:0")
a = torch.zeros([10,10,10,10],device=device)
b = torch.zeros([10,10,10,10],device=device)
c = right_pool.backward(a,b)
print(c)
