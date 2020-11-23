import torch
import bilateral_slice_apply
from ops import BilateralSliceApply
import time
import torch.nn.functional as F
torch.manual_seed(0)

grid = torch.rand((2,2,2,12,1),dtype=torch.float32, requires_grad=True)
gridx = grid.to('cuda')
guide = torch.rand((5,5,1),dtype=torch.float32, requires_grad=True)
guidex = guide.to('cuda')
img = torch.rand((5,5,3,1),dtype=torch.float32).cuda()
res = torch.zeros((5,5,3,1),dtype=torch.float32).cuda()
# st = time.time()
# a = bilateral_slice_apply.bilateral_slice_apply_cuda_float32(gridx, guidex, img,res)
# print(time.time()-st)
# print(res)
st = time.time()
res2 = BilateralSliceApply()(gridx, guidex, img)
print(res2)
print(time.time()-st)
st = time.time()
res2.backward(torch.ones((5,5,3,1),dtype=torch.float32).cuda())
# print("grid_grad", grid.grad)
# print("guide_grad", guide.grad)
print(time.time()-st)
# print('====')
# grid = grid.permute(4,3,2,1,0).contiguous()
# guide = guide.permute(2,1,0).unsqueeze(3).contiguous()
# print(grid.shape)
# hg, wg = torch.meshgrid([torch.arange(0, 3), torch.arange(0, 3)]) # [0,511] HxW
# hg = hg.to('cuda')
# wg = wg.to('cuda')
# hg = hg.float().repeat(1, 1, 1).unsqueeze(3) / (3-1) * 2 - 1 # norm to [-1,1] NxHxWx1
# wg = wg.float().repeat(1, 1, 1).unsqueeze(3) / (3-1) * 2 - 1 # norm to [-1,1] NxHxWx1
# guidemap_guide = torch.cat([wg, hg, guide ], dim=3).unsqueeze(1) # Nx1xHxWx3

# st = time.time()
# coeff = F.grid_sample(grid, guidemap_guide, 'bilinear', align_corners=True)
# coeff = coeff.squeeze(2)
# coeff = coeff.permute(3,2,1,0).contiguous()
# print(coeff.shape)
# print(coeff[0,:,0,0])
