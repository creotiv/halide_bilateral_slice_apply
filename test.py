import torch
import bilateral_slice_apply
from ops import BilateralSliceApply
import time

grid = torch.rand((2,2,2,1,1),dtype=torch.float32).cuda()
guide = torch.ones((5,5,1),dtype=torch.float32).cuda()
img = torch.rand((5,5,3,1),dtype=torch.float32).cuda()
res = torch.zeros((5,5,3,1),dtype=torch.float32).cuda()
st = time.time()
bilateral_slice_apply.bilateral_slice_apply_cuda_float32(grid, guide, img,res)
print(res)
print(time.time()-st)
st = time.time()
res2 = BilateralSliceApply()(grid, guide, img)
print(res2)
print(time.time()-st)
res2.backward()
print("grid_grad", grid.grad)
print("guide_grad", guide.grad)
