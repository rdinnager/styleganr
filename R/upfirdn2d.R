# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

.parse_scaling <- function(scaling) {
    if(length(scaling) == 1) {
        scaling <- c(scaling, scaling)
    }
    assertthat::assert_that(length(scaling) == 2)
    assertthat::assert_that(all(as.integer(scaling) == scaling))
    sx <- sy <- NULL
    c(sx, sy) %<-% scaling
    assertthat::assert_that(sx >= 1 & sy >= 1)
    return(c(sx, sy))
}

.parse_padding <- function(padding) {
    if(length(padding) == 1 | length(padding) == 2) {
        padding <- c(padding, padding)
    }
    assertthat::assert_that(length(padding) == 2 || length(padding) == 4)
    assertthat::assert_that(all(as.integer(padding) == padding))
    if(length(padding) == 2) {
        padx <- pady <- NULL
        c(padx, pady) %<-% padding
        padding <- c(padx, padx, pady, pady)
    }
    padx0 <- padx1 <- pady0 <- pady0 <- NULL
    c(padx0, padx1, pady0, pady1) %<-% padding
    return(c(padx0, padx1, pady0, pady1))
}

.get_filter_size <- function(f) {
    if(is.null(f)) {
        return(c(1, 1))
    }
    assertthat::assert_that(is_torch_tensor(f) & f$ndim %in% c(1, 2))
    fw <- f$shape[length(f$shape)]
    fh <- f$shape[1]
    fw <- as.interger(fw)
    fh <- as.integer(fh)
    assert_shape(f, c(fh, fw)[1:f$ndim])
    assertthat::assert_that(fw >= 1 & fh >= 1)
    return(c(fw, fh))
}

#----------------------------------------------------------------------------

#' Convenience function to setup 2D FIR filter for [upfirdn2d()].
#' 
#' @param f Torch tensor, or R array of the shape: `c(filter_height, filter_width)` (non-separable),
#' `filter_taps` (separable), `interger()` (impulse), or `NULL` (identity).
#' @param device Result device (default: cpu).
#' @param normalize Normalize the filter so that it retains the magnitude
#' for constant input signal (DC)? (default: TRUE).
#' @param flip_filter Flip the filter? (default: FALSE).
#' @param gain Overall scaling factor for signal magnitude (default: 1).
#' @param separable Return a separable filter? (default: select automatically).
#' 
#' @return Float32 tensor of the shape: `c(filter_height, filter_width)` (non-separable) or
#' `filter_taps` (separable).
setup_filter <- function(f, device = torch_device('cpu'), normalize = TRUE, flip_filter = FALSE, 
                         gain = 1, separable = NULL) {
    # Validate.
    if(is.null(f)) {
        f <- 1
    }
    if(is_torch_tensor(f)) {
        f$to(torch_float32())
    } else {
        f <- torch_tensor(f, dtype = torch_float32)
    }
    assertthat::assert_that(f$ndim %in% c(0, 1, 2))
    assertthat::assert_that(f$numel() > 0)
    if(f$ndim == 0) {
        f =  f$unsqueeze(1)
    }

    # Separable?
    if(is.null(separable)) {
        separable <- (f$ndim == 1 & f$numel() >= 8)
    }
    if(f$ndim == 1 & !separable) {
        f <- f$`ger`(f)
    }
    assertthat::assert_that(f$ndim == (if(separable) 1 else 2))

    # Apply normalize, flip, gain, and device.
    if(normalize) {
        f <- f / f$sum()
    }
    if(flip_filter) {
        f <- f$flip(seq_len(f$ndim))
    }
    f <- f * (gain^(f$ndim / 2))
    f <- f$to(device = device)
    return(f)
}

#----------------------------------------------------------------------------

#' Pad, upsample, filter, and downsample a batch of 2D images.
#' 
#' Performs the following sequence of operations for each channel:
#'     
#' 1. Upsample the image by inserting N-1 zeros after each pixel (`up`).
#' 2. Pad the image with the specified number of zeros on each side (`padding`).
#' Negative padding corresponds to cropping the image.
#' 3. Convolve the image with the specified 2D FIR filter (`f`), shrinking it
#' so that the footprint of all output pixels lies within the input image.
#' 4. Downsample the image by keeping every Nth pixel (`down`).
#' This sequence of operations bears close resemblance to scipy.signal.upfirdn().
#' The fused op is considerably more efficient than performing the same calculation
#' using standard PyTorch ops. It supports gradients of arbitrary order.
#' 
#' @param x Float32/float64/float16 input tensor of the shape
#' `c(batch_size, num_channels, in_height, in_width)`.
#' @param f Float32 FIR filter of the shape
#' `c(filter_height, filter_width)` (non-separable),
#' `filter_taps` (separable), or `NULL` (identity).
#' @param up Integer upsampling factor. Can be a single integer or a vector of integers
#' `c(x, y)` (default: 1).
#' @param down Integer downsampling factor. Can be a single int or a vector
#' `c(x, y)` (default: 1).
#' @param padding Padding with respect to the upsampled image. Can be a single number
#' or a vector `c(x, y)` or `c(x_before, x_after, y_before, y_after)`
#' (default: 0).
#' @param flip_filter `FALSE` = convolution, `TRUE` = correlation (default: `FALSE`).
#' @param gain Overall scaling factor for signal magnitude (default: 1).
#' @param impl Implementation to use. Can be `'ref'` or `'cuda'` 
#' (default: `'cuda'` if `torch::cuda_is_available() == TRUE`, `'ref'` otherwise).
#' @return Tensor of the shape `c(batch_size, num_channels, out_height, out_width)`.
upfirdn2d <- function(x, f, up = 1, down = 1, padding = 0, flip_filter = FALSE, gain = 1, impl = if(cuda_is_available()) 'cuda' else 'ref') {
    
    assertthat::assert_that(is_torch_tensor(x))
    assertthat::assert_that(impl %in% c('ref', 'cuda'))
    if(impl == 'cuda' & x$device$type == 'cuda') {
        return(.upfirdn2d_cuda(up = up, down = down, padding = padding, flip_filter = flip_filter, gain = gain)(x, f))
    }
    return(.upfirdn2d_ref(x, f, up = up, down = down, padding = padding, flip_filter = flip_filter, gain = gain))
}

#----------------------------------------------------------------------------

# Slow reference implementation of `upfirdn2d()` using standard PyTorch ops.
.upfirdn2d_ref <- function(x, f, up = 1, down = 1, padding = 0, flip_filter = FALSE, gain = 1){
    
    # Validate arguments.
    assertthat::assert_that(is_torch_tensor(x), x$ndim == 4)
    if(is.null(f)) {
        f <- torch_ones(1, 1, dtype = torch_float32(), device = x$device)
    }
    assertthat::assert_that(is_torch_tensor(f), f$ndim %in% c(1, 2))
    assertthat::assert_that(f$dtype == torch_float32(), !f$requires_grad)
    
    batch_size <- num_channels <- in_height <- in_width <- 
        upx <- upy <- downx <- downy <- padx0 <- padx1 <- pady0 <- pady1 <- NULL
    
    c(batch_size, num_channels, in_height, in_width) %<-% x$shape
    c(upx, upy) %<-% .parse_scaling(up)
    c(downx, downy) %<-% .parse_scaling(down)
    c(padx0, padx1, pady0, pady1) %<-% .parse_padding(padding)

    # Check that upsampled buffer is not smaller than the filter.
    upW <- in_width * upx + padx0 + padx1
    upH <- in_height * upy + pady0 + pady1
    assertthat::assert_that(upW >= f$shape[length(f$shape)], upH >= f$shape[1])

    # Upsample by inserting zeros.
    x <- x$reshape(c(batch_size, num_channels, in_height, 1, in_width, 1))
    x <- nnf_pad(x, c(0, upx - 1, 0, 0, 0, upy - 1))
    x <- x$reshape(c(batch_size, num_channels, in_height * upy, in_width * upx))

    # Pad or crop.
    x <- nnf_pad(x, c(max(padx0, 0), max(padx1, 0), max(pady0, 0), max(pady1, 0)))
    x <- x[ ,  , max(-(pady0 - 1), 1):(x$shape[3] - max(-(pady1), 0)), 
            max(-(padx0 - 1), 1):(x$shape[4] - max(-(padx1), 0))]

    # Setup filter.
    f <- f * (gain^(f$ndim / 2))
    f <- f$to(x$dtype)
    if(!flip_filter) {
        f <- f$flip(seq_len(f$ndim))
    }

    # Convolve with the filter.
    f <- f$unsqueeze(1)$unsqueeze(2)$`repeat`(c(num_channels, 1, rep(1, f$ndim)))
    #f <- f[np.newaxis, np.newaxis].repeat([num_channels, 1] + [1] * f.ndim)
    if(f$ndim == 4) {
        x <- conv2d_gradfix(input = x, weight = f, groups = num_channels)
    } else {
        x <- conv2d_gradfix(input = x, weight = f$unsqueeze(3), groups = num_channels)
        x <- conv2d_gradfix(input = x, weight = f$unsqueeze(4), groups = num_channels)
    }

    # Downsample by throwing away pixels.
    x <- x[ ,  , slc(1, Inf, downy), slc(1, Inf, downx)]
    return(x)
}

#----------------------------------------------------------------------------

# Fast CUDA implementation of `upfirdn2d()` using custom ops.
.upfirdn2d_cuda <- function(up = 1, down = 1, padding = 0, flip_filter = FALSE, gain = 1) {
    # Parse arguments.
    upx <- upy <- downx <- downy <- padx0 <- padx1 <- pady0 <- pady1 <- NULL
    c(upx, upy) %<-% .parse_scaling(up)
    c(downx, downy) %<-% .parse_scaling(down)
    c(padx0, padx1, pady0, pady1) %<-% .parse_padding(padding)

    # Forward op.
    Upfirdn2dCuda <- autograd_function(
        
        forward = function(ctx, x, f) {
            assertthat::assert_that(is_torch_tensor(x), x$ndim == 4)
            if(is.null(f)) {
                f <- torch_ones(1, 1, dtype = torch_float32(), device = x$device())
            }
            if(f$ndim == 1 & f$shape[1] == 1) {
                f <- f$square()$unsqueeze(1) # Convert separable-1 into full-1x1.
            }
            assertthat::assert_that(is_torch_tensor(f), f$ndim %in% c(1, 2)) 
            y = x
            if(f$ndim == 2) {
                y <- cpp_upfirdn2d(y, f, upx, upy, downx, downy, padx0, padx1, pady0, pady1, flip_filter, gain)
            } else {
                y = cpp_upfirdn2d(y, f$unsqueeze(1), upx, 1, downx, 1, padx0, padx1, 0, 0, flip_filter, 1.0)
                y = cpp_upfirdn2d(y, f$unsqueeze(2), 1, upy, 1, downy, 0, 0, pady0, pady1, flip_filter, gain)   
            }
            ctx$save_for_backward(f, x$shape)
            return(y)
        },

        backward = function(ctx, dy) {
            f <- x_shape <- ih <- iw <- oh <- ow <- fw <- fh <- NULL
            c(f, x_shape) %<-% ctx$saved_variables
            c(., ., ih, iw) %<-% x_shape
            c(., ., oh, ow) %<-% dy$shape
            c(fw, fh) <- .get_filter_size(f)
            p <- c(
                fw - padx0 - 1,
                iw * upx - ow * downx + padx0 - upx + 1,
                fh - pady0 - 1,
                ih * upy - oh * downy + pady0 - upy + 1,
            )
            dx <- NULL
            df <- NULL

            if(ctx$needs_input_grad[1]) {
                dx = .upfirdn2d_cuda(up = down, down = up, padding = p, flip_filter = (!flip_filter), 
                                     gain = gain)(dy, f)
            }

            assertthat::assert_that(!ctx$needs_input_grad[2])
            return(list(x = dx, f = df))
        }
    )

    return(Upfirdn2dCuda)
}

#----------------------------------------------------------------------------

#' Filter a batch of 2D images using the given 2D FIR filter.
#' 
#' By default, the result is padded so that its shape matches the input.
#' User-specified padding is applied on top of that, with negative values
#' indicating cropping. Pixels outside the image are assumed to be zero.
#' 
#' @param x Float32/float64/float16 input tensor of the shape
#' `c(batch_size, num_channels, in_height, in_width)`.
#' @param f Float32 FIR filter of the shape
#' `c(filter_height, filter_width)` (non-separable),
#' `filter_taps` (separable), or `NULL` (identity).
#' @param padding Padding with respect to the output. Can be a single number or a
#' vector `c(x, y)` or `c(x_before, x_after, y_before, y_after)`
#' (default: 0).
#' @param flip_filter `FALSE` = convolution, `TRUE` = correlation (default: `FALSE`).
#' @param gain Overall scaling factor for signal magnitude (default: 1).
#' @param impl Implementation to use. Can be `'ref'` or `'cuda'` 
#' (default: `'cuda'` if `torch::cuda_is_available() == TRUE`, `'ref'` otherwise).
#' @return Tensor of the shape `c(batch_size, num_channels, out_height, out_width)`.
filter2d <- function(x, f, padding = 0, flip_filter = FALSE, gain = 1, impl = if(cuda_is_available()) 'cuda' else 'ref') {
    
    padx0 <- padx1 <- pady0 <- pady1 <- fw <- fh <- NULL
    c(padx0, padx1, pady0, pady1) %<-% .parse_padding(padding)
    c(fw, fh) %<-% .get_filter_size(f)
    p <- c(
        padx0 + fw %/% 2,
        padx1 + (fw - 1) %/% 2,
        pady0 + fh %/% 2,
        pady1 + (fh - 1) %/% 2,
    )
    return(upfirdn2d(x, f, padding = p, flip_filter = flip_filter, gain = gain, impl = impl))
}
    
#----------------------------------------------------------------------------

#' Upsample a batch of 2D images using the given 2D FIR filter.
#' 
#' By default, the result is padded so that its shape is a multiple of the input.
#' User-specified padding is applied on top of that, with negative values
#' indicating cropping. Pixels outside the image are assumed to be zero.
#' 
#' @inheritParams upfirdn2d
#' @return Tensor of the shape `c(batch_size, num_channels, out_height, out_width)`.
upsample2d <- function(x, f, up = 2, padding = 0, flip_filter = FALSE, gain = 1, impl = if(cuda_is_available()) 'cuda' else 'ref') {
    
    upx <- upy <- padx0 <- padx1 <- pady0 <- pady1 <- fw <- fh <- NULL
    c(upx, upy) %<-% .parse_scaling(up)
    c(padx0, padx1, pady0, pady1) %,-% .parse_padding(padding)
    c(fw, fh) %<-% .get_filter_size(f)
    p <- c(
        padx0 + (fw + upx - 1) %/% 2,
        padx1 + (fw - upx) %/% 2,
        pady0 + (fh + upy - 1) %/% 2,
        pady1 + (fh - upy) %/% 2,
    )
    return(upfirdn2d(x, f, up = up, padding = p, flip_filter = flip_filter, gain = gain * upx * upy, impl = impl))
}

#----------------------------------------------------------------------------

#' Downsample a batch of 2D images using the given 2D FIR filter.
#' 
#' By default, the result is padded so that its shape is a fraction of the input.
#' User-specified padding is applied on top of that, with negative values
#' indicating cropping. Pixels outside the image are assumed to be zero.
#'
#' @inheritParams upfirdn2d 
#' 
#' @return Tensor of the shape `c(batch_size, num_channels, out_height, out_width)`.
downsample2d <- function(x, f, down = 2, padding = 0, flip_filter = FALSE, gain = 1, impl = if(cuda_is_available()) 'cuda' else 'ref') {
 
    downx <- downy <- padx0 <- padx1 <- pady0 <- pady1 <- fw <- fh <- NULL
    c(downx, downy) %<-% .parse_scaling(down)
    c(padx0, padx1, pady0, pady1) %<-% .parse_padding(padding)
    c(fw, fh) %<-% .get_filter_size(f)
    p <- c(
        padx0 + (fw - downx + 1) %/% 2,
        padx1 + (fw - downx) %/% 2,
        pady0 + (fh - downy + 1) %/% 2,
        pady1 + (fh - downy) %/% 2,
    )
    return(upfirdn2d(x, f, down = down, padding = p, flip_filter = flip_filter, gain = gain, impl = impl))
    
}

#----------------------------------------------------------------------------
