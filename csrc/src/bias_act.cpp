// Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

//#define LANTERN_BUILD
//#include "lantern/lantern.h"
//#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/core/MemoryFormat.h>
#include "lantern_ob.h"
#include "bias_act.h"
#include "styleganr/styleganr.h"
#include <torch/torch.h>
//#include "../../utils.hpp"

using namespace torch::autograd;

//------------------------------------------------------------------------

static bool has_same_layout(torch::Tensor x, torch::Tensor y)
{
    if (x.dim() != y.dim())
        return false;
    for (int64_t i = 0; i < x.dim(); i++)
    {
        if (x.size(i) != y.size(i))
            return false;
        if (x.size(i) >= 2 && x.stride(i) != y.stride(i))
            return false;
    }
    return true;
}

//------------------------------------------------------------------------

static torch::Tensor bias_act(torch::Tensor x, torch::Tensor b, torch::Tensor xref, torch::Tensor yref, torch::Tensor dy, int grad, int dim, int act, float alpha, float gain, float clamp)
{
    // Validate arguments.
    TORCH_CHECK(x.is_cuda(), "x must reside on CUDA device");
    TORCH_CHECK(b.numel() == 0 || (b.dtype() == x.dtype() && b.device() == x.device()), "b must have the same dtype and device as x");
    TORCH_CHECK(xref.numel() == 0 || (xref.sizes() == x.sizes() && xref.dtype() == x.dtype() && xref.device() == x.device()), "xref must have the same shape, dtype, and device as x");
    TORCH_CHECK(yref.numel() == 0 || (yref.sizes() == x.sizes() && yref.dtype() == x.dtype() && yref.device() == x.device()), "yref must have the same shape, dtype, and device as x");
    TORCH_CHECK(dy.numel() == 0 || (dy.sizes() == x.sizes() && dy.dtype() == x.dtype() && dy.device() == x.device()), "dy must have the same dtype and device as x");
    TORCH_CHECK(x.numel() <= INT_MAX, "x is too large");
    TORCH_CHECK(b.dim() == 1, "b must have rank 1");
    TORCH_CHECK(b.numel() == 0 || (dim >= 0 && dim < x.dim()), "dim is out of bounds");
    TORCH_CHECK(b.numel() == 0 || b.numel() == x.size(dim), "b has wrong number of elements");
    TORCH_CHECK(grad >= 0, "grad must be non-negative");

    // Validate layout.
    TORCH_CHECK(x.is_non_overlapping_and_dense(), "x must be non-overlapping and dense");
    TORCH_CHECK(b.is_contiguous(), "b must be contiguous");
    TORCH_CHECK(xref.numel() == 0 || has_same_layout(xref, x), "xref must have the same layout as x");
    TORCH_CHECK(yref.numel() == 0 || has_same_layout(yref, x), "yref must have the same layout as x");
    TORCH_CHECK(dy.numel() == 0 || has_same_layout(dy, x), "dy must have the same layout as x");

    // Create output tensor.
    const at::cuda::OptionalCUDAGuard device_guard(device_of(x));
    torch::Tensor y = torch::empty_like(x);
    TORCH_CHECK(has_same_layout(y, x), "y must have the same layout as x");

    // Initialize CUDA kernel parameters.
    bias_act_kernel_params p;
    p.x     = x.data_ptr();
    p.b     = (b.numel()) ? b.data_ptr() : NULL;
    p.xref  = (xref.numel()) ? xref.data_ptr() : NULL;
    p.yref  = (yref.numel()) ? yref.data_ptr() : NULL;
    p.dy    = (dy.numel()) ? dy.data_ptr() : NULL;
    p.y     = y.data_ptr();
    p.grad  = grad;
    p.act   = act;
    p.alpha = alpha;
    p.gain  = gain;
    p.clamp = clamp;
    p.sizeX = (int)x.numel();
    p.sizeB = (int)b.numel();
    p.stepB = (b.numel()) ? (int)x.stride(dim) : 1;

    // Choose CUDA kernel.
    void* kernel;
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.scalar_type(), "upfirdn2d_cuda", [&]
    {
        kernel = choose_bias_act_kernel<scalar_t>(p);
    });
    TORCH_CHECK(kernel, "no CUDA kernel found for the specified activation func");

    // Launch CUDA kernel.
    p.loopX = 4;
    int blockSize = 4 * 32;
    int gridSize = (p.sizeX - 1) / (p.loopX * blockSize) + 1;
    void* args[] = {&p};
    AT_CUDA_CHECK(cudaLaunchKernel(kernel, gridSize, blockSize, args, 0, at::cuda::getCurrentCUDAStream()));
    return y;
}

class BiasActFunction : public Function<BiasActFunction> {
public:
    
    static torch::Tensor forward(AutogradContext *ctx, torch::Tensor x, torch::Tensor b, int cuda_idx, bool has_2nd, MemoryFormat memory_format, bool xref, int dim, float alpha, float gain, float clamp) {
        
        _null_tensor = torch::Tensor();
        
        auto x = x.contiguous(memory_format = memory_format);
        auto b = b.contiguous();
            
        auto y = bias_act(x, b, _null_tensor, _null_tensor, _null_tensor, 0, dim, cuda_idx, alpha, gain, clamp);

        // Save context
        ctx->save_for_backward({xref && has_2nd ? x : _null_tensor,
                                xref && has_2nd ? b : _null_tensor,
                                !xref ? y : _null_tensor,
                                _null_tensor,
                                _null_tensor,
                                _null_tensor,
                                _null_tensor,
                                _null_tensor,
                                _null_tensor,
                                _null_tensor,
                                _null_tensor});
        ctx->saved_data["needs_reshaping"] = needs_reshaping;
        ctx->saved_data["dim"] = dim;
        
        if (needs_reshaping)
        {
            // Tranpose flattened dim to last dim, nth dim to 0th dim
            output = output.transpose(0, 1);
            
            // Reshape to original size
            output = output.reshape(original_size);
            
            // Swap batch dim and nth dim
            output = output.transpose(0, dim);
        }
        
        return output;
    }
    
    static tensor_list backward(AutogradContext *ctx, tensor_list grad_outputs) {
        auto saved = ctx->get_saved_variables();
        auto output = saved[0];
        auto grad_output = grad_outputs[0];
        
        bool needs_reshaping = ctx->saved_data["needs_reshaping"].toBool();
        int dim = ctx->saved_data["dim"].toInt();
        auto original_size = grad_output.sizes().vec();
        
        if (needs_reshaping)
        {
            // transpose batch and nth dim
            grad_output = grad_output.transpose(0, dim);
            
            // Flatten all dimensions except nth dim
            grad_output = grad_output.reshape({grad_output.size(0), -1});
            
            // Transpose flattened dimensions to 0th dim, nth dim to last dim
            grad_output = grad_output.transpose(0, -1);
        }
        
        // Compute gradient
        auto nonzeros = torch::ne(output, 0);
        auto num_nonzeros = nonzeros.sum(-1, true);
        auto sum = (grad_output * nonzeros).sum(-1, true) / num_nonzeros;
        auto grad_input = nonzeros * (grad_output - sum.expand_as(grad_output));
        
        if (needs_reshaping)
        {
            // Tranpose flattened dim to last dim, nth dim to 0th dim
            grad_input = grad_input.transpose(0, 1);
            
            // Reshape to original size
            grad_input = grad_input.reshape(original_size);
            
            // Swap batch dim and nth dim
            grad_input = grad_input.transpose(0, dim);
        }
        
        auto o = torch::autograd::variable_list(2);
        o[0] = grad_input;
        
        return o;
    }
};

//------------------------------------------------------------------------

STYLEGANR_API void * c_styleganr_bias_act (void* x, void* b, void* xref, void* yref, void* dy, int grad, int dim, int act, float alpha, float gain, float clamp)
{
    //LANTERN_FUNCTION_START
    torch::Tensor result = bias_act(
        reinterpret_cast<LanternObject<torch::Tensor>*>(x)->get(), 
        reinterpret_cast<LanternObject<torch::Tensor>*>(b)->get(), 
        reinterpret_cast<LanternObject<torch::Tensor>*>(xref)->get(), 
        reinterpret_cast<LanternObject<torch::Tensor>*>(yref)->get(), 
        reinterpret_cast<LanternObject<torch::Tensor>*>(dy)->get(), 
        grad, 
        dim, 
        act, 
        alpha, 
        gain, 
        clamp
    );
    return (void*) new LanternObject<torch::Tensor>(result);
    //LANTERN_FUNCTION_END
}
