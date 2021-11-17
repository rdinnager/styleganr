# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

#----------------------------------------------------------------------------

.get_filter_size_lrelu <- function(f) {
    if(is.null(f)) {
        return(c(1, 1))
    }
    assertthat::assert_that(is_torch_tensor(f))
    assertthat::assert_that(1 <= f$ndim, f$ndim <= 2)
    return(c(f$shape[length(f$shape)], f$shape[1])) # width, height
}


#----------------------------------------------------------------------------

#' Filtered leaky ReLU for a batch of 2D images.
#' 
#' Performs the following sequence of operations for each channel:
#' 1. Add channel-specific bias if provided (`b`).
#' 2. Upsample the image by inserting N-1 zeros after each pixel (`up`).
#' 3. Pad the image with the specified number of zeros on each side (`padding`).
#' Negative padding corresponds to cropping the image.
#' 4. Convolve the image with the specified upsampling FIR filter (`fu`), shrinking it
#' so that the footprint of all output pixels lies within the input image.
#' 5. Multiply each value by the provided gain factor (`gain`).
#' 6. Apply leaky ReLU activation function to each value.
#' 7. Clamp each value between -clamp and +clamp, if `clamp` parameter is provided.
#' 8. Convolve the image with the specified downsampling FIR filter (`fd`), shrinking
#' it so that the footprint of all output pixels lies within the input image.
#' 9. Downsample the image by keeping every Nth pixel (`down`).
#' The fused op is considerably more efficient than performing the same calculation
#' using standard PyTorch ops. It supports gradients of arbitrary order.

#' @param x Float32/float16/float64 input tensor of the shape
#' `c(batch_size, num_channels, in_height, in_width)`.
#' @param fu Float32 upsampling FIR filter of the shape
#' `c(filter_height, filter_width)` (non-separable),
#' `filter_taps` (separable), or `NULL` (identity).
#' @param fd Float32 downsampling FIR filter of the shape
#' `c(filter_height, filter_width)` (non-separable),
#' `filter_taps` (separable), or `NULL` (identity).
#' @param b Bias vector, or `NULL` to disable. Must be a 1D tensor of the same type
#' as `x`. The length of vector must must match the channel dimension of `x`.
#' @param slope Slope on the negative side of leaky ReLU (default: 0.2).
#' @param clamp Maximum magnitude for leaky ReLU output (default: NULL).
#' @inheritParams upfirdn2d
#' @return Tensor of the shape `c(batch_size, num_channels, out_height, out_width)`.
filtered_lrelu <- function(x, fu = NULL, fd = NULL, b = NULL, up = 1, down = 1, padding = 0, 
                           gain = sqrt(2), slope = 0.2, clamp = NULL, flip_filter = FALSE, 
                           impl = if(cuda_is_available()) 'cuda' else 'ref'){
    
    assertthat::assert_that(is_torch_tensor(x))
    assertthat::assert_that(impl %in% c('ref', 'cuda'))
    if(impl == 'cuda' & x$device$type == 'cuda') {
        flc <- .filtered_lrelu_cuda(up = up, down = down, padding = padding, gain = gain, slope = slope, clamp = clamp, flip_filter = flip_filter) 
        return(flc(x, fu, fd, b, NULL, 0, 0))
    }
    return(.filtered_lrelu_ref(x, fu = fu, fd = fd, b = b, up = up, down = down, padding = padding, gain = gain, slope = slope, clamp = clamp, flip_filter = flip_filter))

}

#----------------------------------------------------------------------------

# Slow and memory-inefficient reference implementation of `filtered_lrelu()` using
# existing `upfirdn2n()` and `bias_act()` ops.
.filtered_lrelu_ref <- function(x, fu = NULL, fd = NULL, b = NULL, up = 1, down = 1, padding = 0, gain = sqrt(2), slope = 0.2, clamp = NULL, flip_filter = FALSE) {
    
    # if(fu$numel() == 0) {
    #     fu <- NULL
    # }
    # 
    # if(fd$numel() == 0) {
    #     fd <- NULL
    # }
    
    assertthat::assert_that(is_torch_tensor(x), x$ndim == 4)
    fu_w <- fu_h <- fd_w <- fd_h <- px0 <- px1 <- py0 <- py1 <- batch_size <- channels <- in_h <- in_w <- NULL
    c(fu_w, fu_h) %<-% .get_filter_size_lrelu(fu)
    c(fd_w, fd_h) %<-% .get_filter_size_lrelu(fd)
    if(!is.null(b)) {
        assertthat::assert_that(is_torch_tensor(b), b$dtype == x$dtype)
        assert_shape(b, x$shape[2])
    }
    assertthat::assert_that(as.integer(up) == up, up >= 1)
    assertthat::assert_that(as.integer(down) == down, down >= 1)
    c(px0, px1, py0, py1) %<-% .parse_padding(padding)
    assertthat::assert_that(as.double(gain) == gain, gain > 0)
    assertthat::assert_that(as.double(slope) == slope, slope >= 0)
    assertthat::assert_that(is.null(clamp) | (clamp == as.double(clamp) & clamp >= 0))

    # Calculate output size.
    c(batch_size, channels, in_h, in_w) %<-% x$shape
    in_dtype <- x$dtype
    out_w <- (in_w * up + (px0 + px1) - (fu_w - 1) - (fd_w - 1) + (down - 1)) %/% down
    out_h <- (in_h * up + (py0 + py1) - (fu_h - 1) - (fd_h - 1) + (down - 1)) %/% down

    # Compute using existing ops.
    x <- bias_act(x = x, b = b) # Apply bias.
    x <- upfirdn2d(x = x, f = fu, up = up, padding = c(px0, px1, py0, py1), 
                   gain = up^2, flip_filter = flip_filter) # Upsample.
    x <- bias_act(x = x, act = 'lrelu', alpha = slope, gain = gain, clamp = clamp) # Bias, leaky ReLU, clamp.
    x <- upfirdn2d(x = x, f = fd, down = down, flip_filter = flip_filter) # Downsample.

    # Check output shape & dtype.
    assert_shape(x, c(batch_size, channels, out_h, out_w))
    assertthat::assert_that(x$dtype == in_dtype)
    return(x)

}

#----------------------------------------------------------------------------

# Fast CUDA implementation of `filtered_lrelu()` using custom ops.
.filtered_lrelu_cuda <- function(up = 1, down = 1, padding = 0, gain = sqrt(2), slope = 0.2, 
                                 clamp = NULL, flip_filter = FALSE) {

    assertthat::assert_that(as.integer(up) == up, up >= 1)
    assertthat::assert_that(as.integer(down) == down, down >= 1)
    px0 <- px1 <- py0 <- py1 <- NULL
    c(px0, px1, py0, py1) %<-% .parse_padding(padding)
    assertthat::assert_that(as.double(gain) == gain, gain > 0)
    gain <- as.double(gain)
    assertthat::assert_that(slope == as.double(slope), slope >= 0)
    slope <- as.double(slope)
    assertthat::assert_that(is.null(clamp) | (clamp == as.double(clamp) & clamp >= 0))
    clamp <- as.double(if(!is.null(clamp)) clamp else 'Inf')

    # Forward op.
    FilteredLReluCuda <- autograd_function(
        
        forward = function(ctx, x, fu, fd, b, si, sx, sy) {
            assertthat::assert_that(is_torch_tensor(x), x$ndim == 4)
            
            # if(fu$numel() == 0) {
            #     fu <- NULL
            # }
            # 
            # if(fd$numel() == 0) {
            #     fd <- NULL
            # }

            # Replace empty up/downsample kernels with full 1x1 kernels (faster than separable).
            if(is.null(fu)) {
                fu <- torch_ones(1, 1, dtype = torch_float32(), device = x$device)
            }
            if(is.null(fd)) {
                fd <- torch_ones(1, 1, dtype = torch_float32(), device = x$device)
            }
            assertthat::assert_that(1 <= fu$ndim, fu$ndim <= 2)
            assertthat::assert_that(1 <= fd$ndim, fd$ndim <= 2)

            # Replace separable 1x1 kernels with full 1x1 kernels when scale factor is 1.
            if(up == 1 & fu$ndim == 1 & fu$shape[1] == 1) {
                ## fu = fu.square()[None] Holy crap, indexing with None is apparently the same as using np.newaxis!
                ## so we need to unsqueeze here
                fu <- fu$square()$unsqueeze(1)
            }
            if(down == 1 & fd$ndim == 1 & fd$shape[1] == 1) {
                fd <- fd$square()$unsqueeze(1)
            }

            # Missing sign input tensor.
            if(is.null(si)) {
                si <- torch_empty(0)
            }

            # Missing bias tensor.
            if(is.null(b)) {
                b <- torch_zeros(x$shape[2], dtype = x$dtype, device = x$device)
            }

            # Construct internal sign tensor only if gradients are needed.
            write_signs <- (si$numel() == 0) & (x$requires_grad | b$requires_grad)

            # Warn if input storage strides are not in decreasing order due to e.g. channels-last layout.
            strides <- purrr::map_if(seq_len(x$ndim),
                                     ~ x$size(.x) > 1,
                                     ~ x$stride(.x),
                                     .else = ~ NULL) %>%
                purrr::flatten_dbl()
            #strides <- [x.stride(i) for i in range(x.ndim) if x.size(i) > 1]
            if(any(purrr::map2_lgl(strides[-length(strides)], 
                                   strides[2:length(strides)],
                                   ~ .x < .y))) {
                warning("low-performance memory layout detected in filtered_lrelu input")
            }
            
            # Call C++/Cuda plugin if datatype is supported.
            if(as.character(x$dtype) %in% c(as.character(torch_float16()), 
                                            as.character(torch_float32())) &
               Sys.getenv("NO_CUSTOM_OP") == "") {
                # if torch.cuda.current_stream(x.device) != torch.cuda.default_stream(x.device):
                #     warnings.warn("filtered_lrelu called with non-default cuda stream but concurrent execution is not supported", RuntimeWarning)
                y <- so <- return_code <- NULL
                print(list(x = class(x), fu = fu, fd = fd, si = si, up = up, down = down, px0 = px0, px1 = px1, 
                           py0 = py0, py1 = py1, sx = sx, sy = sy, gain = gain, slope = slope, clamp = clamp, 
                           flip_filter = flip_filter, write_signs = write_signs))
                c(y, so, return_code) %<-% cpp_filtered_lrelu(x, fu, fd, b, si, up, down, px0, px1, py0, py1, sx, sy, gain, slope, clamp, flip_filter, write_signs)
            } else {
                return_code = -1
            }
            
            # No Cuda kernel found? Fall back to generic implementation. Still more memory efficient than the reference implementation because
            # only the bit-packed sign tensor is retained for gradient computation.
            if(return_code < 0) {
                
                warning("filtered_lrelu called with parameters that have no optimized CUDA kernel, using generic fallback")

                y <- x$add(b$unsqueeze(-1)$unsqueeze(-1)) # Add bias.
                y <- upfirdn2d(x = y, f = fu, up = up, padding = c(px0, px1, py0, py1), gain = up^2, flip_filter = flip_filter) # Upsample.
                so <- cpp_filtered_lrelu_act(y, si, sx, sy, gain, slope, clamp, write_signs) # Activation function and sign handling. Modifies y in-place.
                y <- upfirdn2d(x = y, f = fd, down = down, flip_filter = flip_filter) # Downsample.
                
            }

            # Prepare for gradient computation.
            ctx$save_for_backward(fu, fd, (if(si$numel() == 0) si else so), x$shape, y$shape, list(sx, sy))
            # ctx.x_shape = x.shape
            # ctx.y_shape = y.shape
            # ctx.s_ofs = sx, sy
            return(y)
        },

        backward = function(ctx, dy) {
            fu <- fd <- si <- xh <- xw <- yh <- yw <- x_shape <- y_shape <- s_ofs <- NULL
            c(fu, fd, si, x_shape, y_shape, s_ofs) %<-% ctx$saved_variables
            c(., ., xh, xw) %<-% x_shape
            c(., ., yh, yw) %<-% y_shape
            c(sx, sy) %<-% s_ofs
            dx  <- NULL # 0
            dfu <- NULL; assertthat::assert_that(!ctx$needs_input_grad[2])
            dfd <- NULL; assertthat::assert_that(!ctx$needs_input_grad[3])
            db  <- NULL # 3
            dsi <- NULL; assertthat::assert_that(!ctx$needs_input_grad[5])
            dsx <- NULL; assertthat::assert_that(!ctx$needs_input_grad[6])
            dsy <- NULL; assertthat::assert_that(!ctx.needs_input_grad[7])

            if(ctx$needs_input_grad[1] | ctx$needs_input_grad[4]) {
                last <- length(fu$shape)
                pp <- c(
                    (fu$shape[last] - 1) + (fd$shape[last] - 1) - px0,
                    xw * up - yw * down + px0 - (up - 1),
                    (fu$shape[1] - 1) + (fd$shape[1] - 1) - py0,
                    xh * up - yh * down + py0 - (up - 1),
                )
                gg <- gain * (up^2) / (down^2)
                ff <- (!flip_filter)
                sx <- sx - (fu$shape[last] - 1) + px0
                sy <- sy - (fu$shape[1]  - 1) + py0
                dx <- .filtered_lrelu_cuda(up = down, down = up, padding = pp, gain = gg, slope = slope, clamp = NULL, flip_filter = ff)(dy, fd, fu, NULL, si, sx, sy)
            }

            if(ctx$needs_input_grad[4]) {
                db <- dx$sum(c(1, 3, 4))
            }

            return(list(x = dx, fu = dfu, fd = dfd, b = db, si = dsi, sx = dsx, sy = dsy))
        }
    )
    
    return(FilteredLReluCuda)
}
#----------------------------------------------------------------------------
