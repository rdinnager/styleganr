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
#include <vector>
#include <numeric>
#include <algorithm>
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

class BiasActCudaGrad : public Function<BiasActCudaGrad> {
public:
    static torch::Tensor forward(AutogradContext *ctx, torch::Tensor dy, torch::Tensor x, torch::Tensor b, torch::Tensor y, bool has_2nd, int cuda_idx, int dim, float alpha, float gain, float clamp) {
        
        auto _null_tensor = torch::Tensor();
        c10::MemoryFormat memory_format = dy.dim() > 2 && dy.stride(1) == 1 ? c10::MemoryFormat::ChannelsLast : c10::MemoryFormat::Contiguous;
        
        auto dx = bias_act(dy, b, x, y, _null_tensor, 1, dim, cuda_idx, alpha, gain, clamp);
        ctx->save_for_backward(
            {has_2nd ? dy : _null_tensor,
            x, b, y}
        );
        ctx->saved_data["memory_format"] = memory_format;
        ctx->saved_data["dim"] = dim;
        ctx->saved_data["alpha"] = alpha;
        ctx->saved_data["gain"] = gain;
        ctx->saved_data["clamp"] = clamp;
        ctx->saved_data["cuda_idx"] = cuda_idx;
        
        return {dx};
    }
    
    static tensor_list backward(AutogradContext *ctx, tensor_list grad_outputs) {
        
        auto _null_tensor = torch::Tensor();
        auto saved = ctx->get_saved_variables();
        auto d_dx = grad_outputs[0];
        c10::MemoryFormat memory_format = ctx->saved_data["memory_format"].toMemoryFormat();
        d_dx = d_dx.contiguous();
        int dim = ctx->saved_data["dim"].toInt();
        int cuda_idx = ctx->saved_data["cuda_idx"].toInt();
        float alpha = ctx->saved_data["alpha"].toDouble();
        float gain = ctx->saved_data["gain"].toDouble();
        float clamp = ctx->saved_data["clamp"].toDouble();
        
        auto dy = saved[0];
        auto x = saved[1];
        auto b = saved[2];
        auto y = saved[3];
        
        auto d_dy = bias_act(dy, b, x, y, _null_tensor, 1, dim - 1L, cuda_idx, alpha, gain, clamp);
        auto d_x = bias_act(d_dx, b, x, y, dy, 2, dim, cuda_idx, 
                            alpha, gain, clamp);
        
        int ndim = d_x.dim() - 1;
        std::vector<int64_t> dims(ndim);
        
        for (int i = 0; i < ndim; i++) {
            if(i != dim) {
                dims.push_back(i);
            }
        }
        
        auto d_b = d_x.sum(dims);
        
        //auto d_y = torch::Tensor();
        
        return {d_dy, d_x, d_b, torch::Tensor(),
                torch::Tensor(),
                torch::Tensor(),
                torch::Tensor(),
                torch::Tensor(),
                torch::Tensor(),
                torch::Tensor()};
        
    }
};

class BiasAct : public Function<BiasAct> {
public:
    
    static torch::Tensor forward(AutogradContext *ctx, torch::Tensor x, torch::Tensor b, int cuda_idx, bool has_2nd, bool yref, int dim, float alpha, float gain, float clamp) {
        
        auto _null_tensor = torch::Tensor();
        c10::MemoryFormat memory_format = x.dim() > 2 && x.stride(1) == 1 ? c10::MemoryFormat::ChannelsLast : c10::MemoryFormat::Contiguous;
        
        x = x.contiguous(memory_format = memory_format);
        b = b.contiguous();
            
        auto y = bias_act(x, b, _null_tensor, _null_tensor, _null_tensor, 0, dim, cuda_idx, alpha, gain, clamp);

        // Save context
        ctx->save_for_backward({!yref && has_2nd ? x : _null_tensor,
                                !yref && has_2nd ? b : _null_tensor,
                                yref ? y : _null_tensor});
        
        ctx->saved_data["memory_format"] = memory_format;
        ctx->saved_data["dim"] = dim;
        ctx->saved_data["alpha"] = alpha;
        ctx->saved_data["gain"] = gain;
        ctx->saved_data["clamp"] = clamp;
        ctx->saved_data["cuda_idx"] = cuda_idx;
        
        return {y};
    }
    
    static tensor_list backward(AutogradContext *ctx, tensor_list grad_outputs) {
        
        auto _null_tensor = torch::Tensor();
        auto saved = ctx->get_saved_variables();
        auto dy = grad_outputs[0];
        c10::MemoryFormat memory_format = ctx->saved_data["memory_format"].toMemoryFormat();
        dy = dy.contiguous(memory_format);
        int dim = ctx->saved_data["dim"].toInt();
        int cuda_idx = ctx->saved_data["cuda_idx"].toInt();
        float alpha = ctx->saved_data["alpha"].toDouble();
        float gain = ctx->saved_data["gain"].toDouble();
        float clamp = ctx->saved_data["clamp"].toDouble();
        
        auto x = saved[0];
        auto b = saved[1];
        auto y = saved[2];
        
        auto dx = bias_act(dy, b, x, y, _null_tensor, 1, dim - 1L, cuda_idx, alpha, gain, clamp);
        
        int ndim = dx.dim() - 1;
        std::vector<int64_t> dims(ndim);
        
        for (int i = 0; i < ndim; i++) {
            if(i != dim) {
                dims.push_back(i);
            }
        }
        
        auto db = dx.sum(dims);
        return {dx, db,
                torch::Tensor(),
                torch::Tensor(),
                torch::Tensor(),
                torch::Tensor(),
                torch::Tensor(),
                torch::Tensor(),
                torch::Tensor()};
        
    }
};


//------------------------------------------------------------------------

STYLEGANR_API void * c_styleganr_bias_act_autograd (void* x, void* b, int cuda_idx, bool has_2nd, bool xref, int dim, float alpha, float gain, float clamp)
{
    //LANTERN_FUNCTION_START
    torch::Tensor res = BiasAct::apply(reinterpret_cast<LanternObject<torch::Tensor>*>(x)->get(), 
                                       reinterpret_cast<LanternObject<torch::Tensor>*>(b)->get(), 
                                       cuda_idx, 
                                       has_2nd, 
                                       xref, 
                                       dim, 
                                       alpha, 
                                       gain, 
                                       clamp);
    return (void*) new LanternObject<torch::Tensor>(res);
    
    //LANTERN_FUNCTION_END
}

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
