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
#include "lantern_ob.h"
#include "styleganr/styleganr.h"
#include "upfirdn2d.h"
#include <torch/torch.h>
//#include "../../utils.hpp"

using namespace torch::autograd;

//------------------------------------------------------------------------

static torch::Tensor upfirdn2d(torch::Tensor x, torch::Tensor f, int upx, int upy, int downx, int downy, int padx0, int padx1, int pady0, int pady1, bool flip, float gain)
{
    // Validate arguments.
    TORCH_CHECK(x.is_cuda(), "x must reside on CUDA device");
    TORCH_CHECK(f.device() == x.device(), "f must reside on the same device as x");
    TORCH_CHECK(f.dtype() == torch::kFloat, "f must be float32");
    TORCH_CHECK(x.numel() <= INT_MAX, "x is too large");
    TORCH_CHECK(f.numel() <= INT_MAX, "f is too large");
    TORCH_CHECK(x.numel() > 0, "x has zero size");
    TORCH_CHECK(f.numel() > 0, "f has zero size");
    TORCH_CHECK(x.dim() == 4, "x must be rank 4");
    TORCH_CHECK(f.dim() == 2, "f must be rank 2");
    TORCH_CHECK((x.size(0)-1)*x.stride(0) + (x.size(1)-1)*x.stride(1) + (x.size(2)-1)*x.stride(2) + (x.size(3)-1)*x.stride(3) <= INT_MAX, "x memory footprint is too large");
    TORCH_CHECK(f.size(0) >= 1 && f.size(1) >= 1, "f must be at least 1x1");
    TORCH_CHECK(upx >= 1 && upy >= 1, "upsampling factor must be at least 1");
    TORCH_CHECK(downx >= 1 && downy >= 1, "downsampling factor must be at least 1");

    // Create output tensor.
    const at::cuda::OptionalCUDAGuard device_guard(device_of(x));
    int outW = ((int)x.size(3) * upx + padx0 + padx1 - (int)f.size(1) + downx) / downx;
    int outH = ((int)x.size(2) * upy + pady0 + pady1 - (int)f.size(0) + downy) / downy;
    TORCH_CHECK(outW >= 1 && outH >= 1, "output must be at least 1x1");
    torch::Tensor y = torch::empty({x.size(0), x.size(1), outH, outW}, x.options(), x.suggest_memory_format());
    TORCH_CHECK(y.numel() <= INT_MAX, "output is too large");
    TORCH_CHECK((y.size(0)-1)*y.stride(0) + (y.size(1)-1)*y.stride(1) + (y.size(2)-1)*y.stride(2) + (y.size(3)-1)*y.stride(3) <= INT_MAX, "output memory footprint is too large");

    // Initialize CUDA kernel parameters.
    upfirdn2d_kernel_params p;
    p.x             = x.data_ptr();
    p.f             = f.data_ptr<float>();
    p.y             = y.data_ptr();
    p.up            = make_int2(upx, upy);
    p.down          = make_int2(downx, downy);
    p.pad0          = make_int2(padx0, pady0);
    p.flip          = (flip) ? 1 : 0;
    p.gain          = gain;
    p.inSize        = make_int4((int)x.size(3), (int)x.size(2), (int)x.size(1), (int)x.size(0));
    p.inStride      = make_int4((int)x.stride(3), (int)x.stride(2), (int)x.stride(1), (int)x.stride(0));
    p.filterSize    = make_int2((int)f.size(1), (int)f.size(0));
    p.filterStride  = make_int2((int)f.stride(1), (int)f.stride(0));
    p.outSize       = make_int4((int)y.size(3), (int)y.size(2), (int)y.size(1), (int)y.size(0));
    p.outStride     = make_int4((int)y.stride(3), (int)y.stride(2), (int)y.stride(1), (int)y.stride(0));
    p.sizeMajor     = (p.inStride.z == 1) ? p.inSize.w : p.inSize.w * p.inSize.z;
    p.sizeMinor     = (p.inStride.z == 1) ? p.inSize.z : 1;

    // Choose CUDA kernel.
    upfirdn2d_kernel_spec spec;
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.scalar_type(), "upfirdn2d_cuda", [&]
    {
        spec = choose_upfirdn2d_kernel<scalar_t>(p);
    });

    // Set looping options.
    p.loopMajor     = (p.sizeMajor - 1) / 16384 + 1;
    p.loopMinor     = spec.loopMinor;
    p.loopX         = spec.loopX;
    p.launchMinor   = (p.sizeMinor - 1) / p.loopMinor + 1;
    p.launchMajor   = (p.sizeMajor - 1) / p.loopMajor + 1;

    // Compute grid size.
    dim3 blockSize, gridSize;
    if (spec.tileOutW < 0) // large
    {
        blockSize = dim3(4, 32, 1);
        gridSize = dim3(
            ((p.outSize.y - 1) / blockSize.x + 1) * p.launchMinor,
            (p.outSize.x - 1) / (blockSize.y * p.loopX) + 1,
            p.launchMajor);
    }
    else // small
    {
        blockSize = dim3(256, 1, 1);
        gridSize = dim3(
            ((p.outSize.y - 1) / spec.tileOutH + 1) * p.launchMinor,
            (p.outSize.x - 1) / (spec.tileOutW * p.loopX) + 1,
            p.launchMajor);
    }

    // Launch CUDA kernel.
    void* args[] = {&p};
    AT_CUDA_CHECK(cudaLaunchKernel(spec.kernel, gridSize, blockSize, args, 0, at::cuda::getCurrentCUDAStream()));
    return y;
};

static torch::Tensor upfirdn2d_forward(torch::Tensor x, torch::Tensor f, int upx, int upy, int downx, int downy, int padx0, int padx1, int pady0, int pady1, bool flip, float gain) 
{
    auto y = x;
    if(f.dim() == 2) {
        y = upfirdn2d(x, f, upx, upy, downx, downy, padx0, padx1, pady0, pady1, flip, gain);
    } else {
        y = upfirdn2d(x, f.unsqueeze(0), upx, 1, downx, 1, padx0, padx1, 0, 0, flip, 1.0);
        y = upfirdn2d(y, f.unsqueeze(1), 1, upy, 1, downy, 0, 0, pady0, pady1, flip, gain);   
    }
    return y;
}

class Upfirdn2d : public Function<Upfirdn2d> {
public:
    
    static torch::Tensor forward(AutogradContext *ctx, torch::Tensor x, torch::Tensor f, int upx, int upy, int downx, int downy, int padx0, int padx1, int pady0, int pady1, bool flip_filter, float gain, int fw, int fh) {
        
        auto y = upfirdn2d_forward(x, f, upx, upy, downx, downy, padx0, padx1, pady0, pady1, flip_filter, gain);
        
        ctx->save_for_backward({f});
        
        ctx->saved_data["ih"] = x.size(0);
        ctx->saved_data["iw"] = x.size(1);
        ctx->saved_data["padx0"] = padx0;
        ctx->saved_data["pady0"] = pady0;
        ctx->saved_data["upx"] = upx;
        ctx->saved_data["upy"] = upy;
        ctx->saved_data["downx"] = downx;
        ctx->saved_data["downy"] = downy;
        ctx->saved_data["flip_filter"] = flip_filter;
        ctx->saved_data["gain"] = gain;
        ctx->saved_data["fw"] = fw;
        ctx->saved_data["fh"] = fh;
        
        return {y};
        
    }
    
    static tensor_list backward(AutogradContext *ctx, tensor_list grad_outputs) {
        
        auto saved = ctx->get_saved_variables();
        auto dy = grad_outputs[0];
        
        auto f = saved[0];
        auto oh = dy.size(0);
        auto ow = dy.size(1);
        
        auto ih = ctx->saved_data["ih"].toInt();
        auto iw = ctx->saved_data["iw"].toInt();
        auto padx0 = ctx->saved_data["padx0"].toInt();
        auto pady0 = ctx->saved_data["pady0"].toInt();
        auto upx = ctx->saved_data["upx"].toInt();
        auto upy = ctx->saved_data["upy"].toInt();
        auto downx = ctx->saved_data["downx"].toInt();
        auto downy = ctx->saved_data["downy"].toInt();
        auto flip_filter = ctx->saved_data["flip_filter"].toBool();
        auto gain = ctx->saved_data["upy"].toDouble();
        auto fw = ctx->saved_data["fw"].toInt();
        auto fh = ctx->saved_data["fh"].toInt();
        
        auto p0 = fw - padx0 - 1;
        auto p1 = iw * upx - ow * downx + padx0 - upx + 1;
        auto p2 = fh - pady0 - 1;
        auto p3 = ih * upy - oh * downy + pady0 - upy + 1;
        
        auto dx = upfirdn2d_forward(dy, f, downx, downy, upx, upy, p0, p1, p2, p3, !flip_filter, gain);
        
        return {dx, torch::Tensor(),
                torch::Tensor(),
                torch::Tensor(),
                torch::Tensor(),
                torch::Tensor(),
                torch::Tensor(),
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

STYLEGANR_API void * c_styleganr_upfirdn2d_autograd (void* x, void* f, int upx, int upy, int downx, int downy, int padx0, int padx1, int pady0, int pady1, bool flip_filter, float gain, int fw, int fh)
{
    //LANTERN_FUNCTION_START
    torch::Tensor res = Upfirdn2d::apply(reinterpret_cast<LanternObject<torch::Tensor>*>(x)->get(), 
                                         reinterpret_cast<LanternObject<torch::Tensor>*>(f)->get(), 
                                         upx, 
                                         upy, 
                                         downx,
                                         downy,
                                         padx0,
                                         padx1,
                                         pady0,
                                         pady1,
                                         flip_filter, 
                                         gain, 
                                         fw,
                                         fh);
    return (void*) new LanternObject<torch::Tensor>(res);
    
    //LANTERN_FUNCTION_END
}

STYLEGANR_API void * c_styleganr_upfirdn2d (void* x, void* f, int upx, int upy, int downx, int downy, int padx0, int padx1, int pady0, int pady1, bool flip, float gain)
{
    //LANTERN_FUNCTION_START
    torch::Tensor result = upfirdn2d(
        reinterpret_cast<LanternObject<torch::Tensor>*>(x)->get(), 
        reinterpret_cast<LanternObject<torch::Tensor>*>(f)->get(), 
        upx, 
        upy, 
        downx, 
        downy, 
        padx0, 
        padx1, 
        pady0, 
        pady1, 
        flip, 
        gain
    );
    return (void*) new LanternObject<torch::Tensor>(result);
    //LANTERN_FUNCTION_END
}