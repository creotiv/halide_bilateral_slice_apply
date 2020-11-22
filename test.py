# import torch
# import torch.nn.functional as fn

# # N x C x D x H x W image
# image = torch.arange(0, 8, dtype=torch.float32).reshape([1, 1, 2, 2, 2])

# # sampling positions with (W, H, D) coordinates between 0 and 1
# w_coord = 1/5
# h_coord = 3/4
# d_coord = 2/3
# sampling_position = torch.tensor([w_coord, h_coord, d_coord], dtype=torch.float32)
# # bring the sampling coordinates into the range -1..1
# sampling_position = 2 * sampling_position - 1
# print(sampling_position.shape)
# # add the N, D, H, W dimensions
# sampling_position = torch.reshape(sampling_position, [1, 1, 1, 1, *sampling_position.shape])
# print(sampling_position.shape)

# sample = fn.grid_sample(image, sampling_position, 'bilinear', align_corners=True)
# print(sample)

# # Manual computation
# w_00 = image[0, 0, 0, 0, 0] * (1 - w_coord) + image[0, 0, 0, 0, 1] * w_coord
# w_01 = image[0, 0, 0, 1, 0] * (1 - w_coord) + image[0, 0, 0, 1, 1] * w_coord
# w_10 = image[0, 0, 1, 0, 0] * (1 - w_coord) + image[0, 0, 1, 0, 1] * w_coord
# w_11 = image[0, 0, 1, 1, 0] * (1 - w_coord) + image[0, 0, 1, 1, 1] * w_coord

# w_0 = w_00 * (1 - h_coord) + w_01 * h_coord
# w_1 = w_10 * (1 - h_coord) + w_11 * h_coord

# w = w_0 * (1 - d_coord) + w_1 * d_coord

# print(w)

# a = torch.Tensor([1.,2.,3.]).reshape(1,3,1,1)
# c = torch.Tensor([0.5,0.5,0.5]).reshape(1,3,1,1)
# R = torch.sum((a*c), dim=1, keepdim=True)
# print(R, R.shape)

import torch
import bilateral_slice_apply
from ops import BilateralSliceApply
import time
import torch.nn.functional as F
torch.manual_seed(0)

grid = torch.tensor([0.2,0.7],dtype=torch.float32, requires_grad=True)
gridx = grid.to('cuda').reshape(1,1,2,1,1)
guide = torch.tensor([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],dtype=torch.float32, requires_grad=True)
guidex = guide.to('cuda').reshape((3,3,1))
# guide = torch.rand((3,3,1),dtype=torch.float32, requires_grad=True).cuda()
img = torch.rand((2,2,3,1),dtype=torch.float32).cuda()
res = torch.zeros((2,2,3,1),dtype=torch.float32).cuda()
st = time.time()
a = bilateral_slice_apply.bilateral_slice_apply_cuda_float32(gridx, guidex, img,res)
print(time.time()-st)
print(res)
st = time.time()
res2 = BilateralSliceApply()(gridx, guidex, img)
print(res2)
print(time.time()-st)
res2.backward(torch.ones((2,2,3,1),dtype=torch.float32).cuda())
print("grid_grad", grid.grad)
print("guide_grad", guide.grad)
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
