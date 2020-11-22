#include "Halide.h"

#include <map>
#include <string>

using namespace Halide;

Var x("x"), y("y"), z("z"), ci("ci"), co("co"), c("c"), n("n");
// grid (x,y,z,c,n)
// guide (x,y,n)
// input (x,y,c,n)
template <typename Input>
Func bilateral_slice_apply(
        const Input &grid,
        const Input &guide,
        const Input &input) {
    Func f_grid = BoundaryConditions::repeat_edge(grid);
    Func f_guide = BoundaryConditions::repeat_edge(guide);
    Func f_input = BoundaryConditions::repeat_edge(input);

    Expr gw = grid.dim(0).extent();
    Expr gh = grid.dim(1).extent();
    Expr gd = grid.dim(2).extent();
    Expr nci = input.dim(2).extent();

    Expr w = guide.dim(0).extent();
    Expr h = guide.dim(1).extent();

    // Enclosing voxel
    Expr gx = ((x+0.5f) * gw) / w;
    Expr gy = ((y+0.5f) * gh) / h;
    Expr gz = clamp(f_guide(x, y, n), 0.0f, 1.0f)*gd;

    Expr fx = cast<int>(floor(gx-0.5f));
    Expr fy = cast<int>(floor(gy-0.5f));
    Expr fz = cast<int>(floor(gz-0.5f));

    // Expr wx = max(1.0f - abs(gx-0.5f - fx), 0.0f);
    // Expr wy = max(1.0f - abs(gy-0.5f - fy), 0.0f);
    // Expr wz = max(1.0f - abs(gz-0.5f - fz), 0.0f);

    // Func affine_coeffs("affine_coeffs");
    // affine_coeffs(x, y, c, n) = f_grid(fx  , fy  , fz  , c, n)*wx*wy*wz;

    Expr wx = abs(gx-0.5f - fx);
    Expr wy = abs(gy-0.5f - fy);
    Expr wz = abs(gz-0.5f - fz);

    // Slice affine coeffs
    Func affine_coeffs("affine_coeffs");
    affine_coeffs(x, y, c, n) =
         f_grid(fx  , fy  , fz  , c, n)*(1.f - wx)*(1.f - wy)*(1.f - wz)
       + f_grid(fx  , fy  , fz+1, c, n)*(1.f - wx)*(1.f - wy)*(      wz)
       + f_grid(fx  , fy+1, fz  , c, n)*(1.f - wx)*(      wy)*(1.f - wz)
       + f_grid(fx  , fy+1, fz+1, c, n)*(1.f - wx)*(      wy)*(      wz)
       + f_grid(fx+1, fy  , fz  , c, n)*(      wx)*(1.f - wy)*(1.f - wz)
       + f_grid(fx+1, fy  , fz+1, c, n)*(      wx)*(1.f - wy)*(      wz)
       + f_grid(fx+1, fy+1, fz  , c, n)*(      wx)*(      wy)*(1.f - wz)
       + f_grid(fx+1, fy+1, fz+1, c, n)*(      wx)*(      wy)*(      wz);

    // Apply them to the input
    Func f_output("f_output");
    RDom r(0, nci);
    f_output(x, y, co, n) = affine_coeffs(x, y, co*(nci+1) + nci, n);
    f_output(x, y, co, n) += 
      affine_coeffs(x, y, co*(nci+1) + r, n)*f_input(x, y, r, n);

    return affine_coeffs;
}

namespace creotiv {
class BilateralSliceApplyGenerator : public Halide::Generator<BilateralSliceApplyGenerator> {
public:

    Input<Buffer<float>> grid{"grid", 5};  
    Input<Buffer<float>> guide{"guide", 3};  
    Input<Buffer<float>> input{"input", 4};  

    Output<Buffer<float>> output{"output", 4};

   void generate() {
        output(x,y,c,n) = bilateral_slice_apply(grid, guide, input)(x, y, c, n);

        // const int kEdge = 1024;
        // grid.set_estimates({{0, kEdge}, {0, kEdge}, {0, kEdge}, {0, 64}, {0, 64}});
        // guide.set_estimates({{0, kEdge}, {0, kEdge}, {0, 1}});
        // input.set_estimates({{0, kEdge}, {0, kEdge}, {0, 3}, {0, 10}});
        // output.set_estimates({{0, kEdge}, {0, kEdge}, {0, 200}, {0, 10}});

        // Schedule
        if (!auto_schedule) {
            Var tx("tx"), xy("xy"), xyc("xyc"), xycn("xycn");
            if (get_target().has_gpu_feature()) {
                output
                    .fuse(x, y, xy)
                    .fuse(xy, c, xyc)
                    .fuse(xyc, n, xycn)
                    .gpu_tile(xycn, tx, 1);

        //  Func(l["abs_selection"])
        //     .compute_root()
        //     .reorder(k, x, y, n)
        //     .fuse(x, y, xy)
        //     .fuse(xy, n, xyn)
        //     .gpu_tile(xyn, xi, ts)
        //     ;
            } else {
                // output
                //     .compute_root()
                //     .fuse(c, n, cn)
                //     .fuse(y, cn, allvars)
                //     .parallel(allvars, 8)
                //     .vectorize(x, 8);
            }
        }

   }
};

class BilateralSliceApplyGradGenerator : public Halide::Generator<BilateralSliceApplyGradGenerator> {
public:

    Input<Buffer<float>> grid{"grid", 5};  
    Input<Buffer<float>> guide{"guide", 3};  
    Input<Buffer<float>> input{"input", 4};  
    Input<Buffer<float>> d_output{"d_output", 4};  

    Output<Buffer<float>> d_grid{"d_grid", 5};  
    Output<Buffer<float>> d_guide{"d_guide", 3};  
    // Output<Buffer<float>> d_input{"d_input", 4};  

    void generate() {
        // Algorithm
        Func f_output = bilateral_slice_apply(grid, guide, input);

        Func adjoint_func = BoundaryConditions::constant_exterior(d_output, Halide::Internal::make_zero(d_output.type()));
        // NOTE: the output_bounds argument is technically supposed to
        // be the shape of f_output; we'll use the bounds of input_a since it
        // is equivalent and easier to access.
        auto d = propagate_adjoints(f_output, adjoint_func,
                                          {{0, input.dim(0).extent()},
                                           {0, input.dim(1).extent()},
                                           {0, input.dim(2).extent()},
                                           {0, input.dim(3).extent()}});

        d_grid(x, y, z, c, n) = d(grid)(x, y, z, c, n);
        d_guide(x, y, n) = d(guide)(x, y, n);
        // d_input(x, y, c, n) = d(input)(x, y, c, n);

        // const int kEdge = 1024;
        // grid.set_estimates({{0, kEdge}, {0, kEdge}, {0, kEdge}, {0, 64}, {0, 64}});
        // guide.set_estimates({{0, kEdge}, {0, kEdge}, {0, 1}});
        // input.set_estimates({{0, kEdge}, {0, kEdge}, {0, 3}, {0, 10}});
        // d_output.set_estimates({{0, kEdge}, {0, kEdge}, {0, 3}, {0, 10}});
        // d_guide.set_estimates({{0, kEdge}, {0, kEdge}, {0, 1}});
        // d_grid.set_estimates({{0, kEdge}, {0, kEdge}, {0, kEdge}, {0, 64}, {0, 64}});

          // Schedule
        if (!auto_schedule) {
            Var tx("tx"), xy("xy"), xyn("xyn"), xyc("xyc"), xycn("xycn"),xyz("xyz"), xyzc("xyzc"), xyzcn("xyzcn");

            if (get_target().has_gpu_feature()) {
                d_grid
                    .fuse(x, y, xy)
                    .fuse(xy, z, xyz)
                    .fuse(xyz, c, xyzc)
                    .fuse(xyzc, n, xyzcn)
                    .gpu_tile(xyzcn, tx, 1);
                d_guide
                    .fuse(x, y, xy)
                    .fuse(xy, n, xyn)
                    .gpu_tile(xyn, tx, 1);
                // d_input
                //     .fuse(x, y, xy)
                //     .fuse(xy, c, xyc)
                //     .fuse(xyc, n, xycn)
                //     .gpu_tile(xycn, tx, 1);
                
            } else {
                // d_grid
                //     .compute_root()
                //     .fuse(c, n, cn)
                //     .fuse(y, cn, allvars)
                //     .parallel(allvars, 8)
                //     .vectorize(x, 8);
                // d_guide
                //     .compute_root()
                //     .fuse(c, n, cn)
                //     .fuse(y, cn, allvars)
                //     .parallel(allvars, 8)
                //     .vectorize(x, 8);
            }
        }
    }
};

}

HALIDE_REGISTER_GENERATOR(creotiv::BilateralSliceApplyGenerator, bsa);
HALIDE_REGISTER_GENERATOR(creotiv::BilateralSliceApplyGradGenerator, bsa_grad)