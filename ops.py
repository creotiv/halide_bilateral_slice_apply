"""Wrap our operator (and gradient) in autograd."""

# We need to import torch before loading the custom modules
import torch
import bilateral_slice_apply as ops



class BilateralSliceApplyFunction(torch.autograd.Function):
    # grid (x,y,z,c,n)
    # guide (x,y,n)
    # input (x,y,c,n)

    @staticmethod
    def forward(ctx, grid, guide, image):
        out = image.new()
        out.resize_(image.shape)
        ops.bilateral_slice_apply_cuda_float32(grid, guide, image, out)

        return out

    @staticmethod
    def backward(ctx, grad_output):
        outputs = ops.bilateral_slice_apply_cuda_float32_grad(grid, guide, image, grad_output, out)
        d_grid = grid.new()
        d_grid.resize_(grid.shape)
        d_guide = guide.new()
        d_guide.resize_(guide.shape)
        return d_grid, d_guide


class BilateralSliceApply(torch.nn.Module):
    def __init__(self):
        super(BilateralSliceApply, self).__init__()

    def forward(self, grid, guide, image):
        return BilateralSliceApplyFunction.apply(grid, guide, image)
