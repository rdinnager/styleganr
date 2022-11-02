#' Fused bias and activation function.
#' 
#' Adds bias `b` to activation tensor `x`, evaluates activation function `act`,
#' and scales the result by `gain`. Each of the steps is optional. In most cases,
#' the fused op is considerably more efficient than performing the same calculation
#' using standard PyTorch ops. It supports first and second order gradients,
#' but not third order gradients.
#' 
#' @param x Input activation tensor. Can be of any shape.
#' @param b Bias vector, or `NULL` to disable. Must be a 1D tensor of the same type
#' as `x`. The shape must be known, and it must match the dimension of `x`
#' corresponding to `dim`.
#' @param dim The dimension in `x` corresponding to the elements of `b`.
#' The value of `dim` is ignored if `b` is not specified.
#' @param act Name of the activation function to evaluate, or `"linear"` to disable.
#' Can be e.g. `"relu"`, `"lrelu"`, `"tanh"`, `"sigmoid"`, `"swish"`, etc.
#' See details for a full list. `NULL` is not allowed.
#' @param alpha Shape parameter for the activation function, or `NULL` to use the default.
#' @param gain Scaling factor for the output tensor, or `NULL` to use default.
#' See details for the default scaling of each activation function.
#' If unsure, consider specifying 1.
#' @param clamp Clamp the output values to `c(-clamp, +clamp)`, or `NULL` to disable
#' the clamping (default).
#' @param impl Name of the implementation to use. Can be `"ref"` or `"cuda"`.
#' 
#' @section Activation Functions:
#' @section Copyright: Note that this function used code from the StyleGAN3 project which 
#' is copyright of Nvidia 2021, and is redistributed in the `torch` package under its
#' original license which can be found here: [https://github.com/NVlabs/stylegan3/blob/main/LICENSE.txt].
#' Note that under the license use is restricted to non-commercial purposes. If you use this function,
#' please make sure your use is acceptable under the license linked above.#' 
#' 
#' @returns `torch_tensor` of the same shape and datatype as `x`.
#' @export
bias_act <- function(x, b = NULL, dim = 2, act = 'linear', alpha = NULL, gain = NULL, clamp = NULL, impl = if(cuda_is_available() & x$device$type == 'cuda') 'cuda' else 'ref') {
  
  stopifnot(is_torch_tensor(x))
  stopifnot(impl %in% c('ref', 'cuda'))
  
  activation_func <- switch(act,
                            linear    = list(name = 'linear',   func = function(x, ...)        x,                        def_alpha = 0,   def_gain = 1,       cuda_idx = 1, ref = '',  has_2nd_grad = FALSE),
                            relu      = list(name = 'relu',     func = function(x, ...)        nnf_relu(x),              def_alpha = 0,   def_gain = sqrt(2), cuda_idx = 2, ref = 'y', has_2nd_grad = FALSE),
                            lrelu     = list(name = 'lrelu',    func = function(x, alpha, ...) nnf_leaky_relu(x, alpha), def_alpha = 0.2, def_gain = sqrt(2), cuda_idx = 3, ref = 'y', has_2nd_grad = FALSE),
                            tanh      = list(name = 'tanh',     func = function(x, ...)        torch_tanh(x),            def_alpha = 0,   def_gain = 1,       cuda_idx = 4, ref = 'y', has_2nd_grad = TRUE),
                            sigmoid   = list(name = 'sigmoid',  func = function(x, ...)        torch_sigmoid(x),         def_alpha = 0,   def_gain = 1,       cuda_idx = 5, ref = 'y', has_2nd_grad = TRUE),
                            elu       = list(name = 'elu',      func = function(x, ...)        nnf_elu(x),               def_alpha = 0,   def_gain = 1,       cuda_idx = 6, ref = 'y', has_2nd_grad = TRUE),
                            selu      = list(name = 'selu',     func = function(x, ...)        nnf_selu(x),              def_alpha = 0,   def_gain = 1,       cuda_idx = 7, ref = 'y', has_2nd_grad = TRUE),
                            softplus  = list(name = 'softplus', func = function(x, ...)        nnf_softplus(x),          def_alpha = 0,   def_gain = 1,       cuda_idx = 8, ref = 'y', has_2nd_grad = TRUE),
                            swish     = list(name = 'swish',    func = function(x, ...)        torch_sigmoid(x) * x,     def_alpha = 0,   def_gain = sqrt(2), cuda_idx = 9, ref = 'x', has_2nd_grad = TRUE)
  )
  
  if(impl == 'cuda' & x$device$type == 'cuda') {
    return(.bias_act_cuda(dim = dim, spec = activation_func, alpha = alpha, gain = gain, clamp = clamp)(x, b))
  } else {
    return(.bias_act_ref(x = x, b = b, dim = dim, spec = activation_func, alpha = alpha, gain = gain, clamp = clamp))
  }
  
}
#----------------------------------------------------------------------------

# Slow reference implementation of `bias_act()` using standard torch ops.
.bias_act_ref <- function(x, b = NULL, dim = 1, spec = list(name = 'linear', func = function(x, ...) x, def_alpha = 0, def_gain = 1, cuda_idx = 1, ref = '', has_2nd_grad = FALSE), alpha = NULL, gain = NULL, clamp = NULL) {
  
  stopifnot(is_torch_tensor(x))
  stopifnot(is.null(clamp) | clamp >= 0)
  
  if(is.null(alpha)) {
    alpha <- as.double(spec$def_alpha)
  } else {
    alpha <- as.double(alpha)
  }
  if(is.null(gain)) {
    gain <- as.double(spec$def_gain)
  } else {
    gain <- as.double(gain)
  }
  if(is.null(clamp)) {
    clamp <-as.double(-1)
  } else {
    clamp <- as.double(clamp)
  }
  
  # Add bias.
  if(!is.null(b)) {
    stopifnot(is_torch_tensor(b) & b$ndim == 1)
    stopifnot(1 <= dim & dim <= x$ndim)
    stopifnot(b$shape[1] == x$shape[dim])
    reshape_vec <- rep(1, x$ndim)
    reshape_vec[dim] <- -1
    x <- x + b$reshape(reshape_vec)
  }
  # Evaluate activation function.
  x <- spec$func(x, alpha = alpha)
  
  # Scale by gain.
  if(gain != 1) {
    x <- x * gain
  }
  # Clamp.
  if(clamp >= 0) {
    x <- x$clamp(-clamp, clamp) 
  }
  return(x)
}

#----------------------------------------------------------------------------

# Fast CUDA implementation of `bias_act()` using custom ops.
.bias_act_cuda <- function(dim = 1, spec = list(name = 'linear', func = function(x, ...) x, def_alpha = 0, def_gain = 1, cuda_idx = 1, ref = '', has_2nd_grad = FALSE), alpha = NULL, gain = NULL, clamp = NULL) {
  # Parse arguments.
  stopifnot(is.null(clamp) | clamp >= 0)
  
  .null_tensor <- torch_empty(0, device = 'cuda')
  
  if(is.null(alpha)) {
    alpha <- as.double(spec$def_alpha)
  } else {
    alpha <- as.double(alpha)
  }
  if(is.null(gain)) {
    gain <- as.double(spec$def_gain)
  } else {
    gain <- as.double(gain)
  }
  if(is.null(clamp)) {
    clamp <-as.double(-1)
  } else {
    clamp <- as.double(clamp)
  }
  
  # Forward op.
  BiasActCuda <- autograd_function(
    
    forward = function(ctx, x, b) {
      if(x$dim() > 2 & x$stride(1) == 1) {
        ctx_memory_format <- torch_channels_last_format()
      } else  {
        ctx_memory_format <- torch_contiguous_format()
      }
      x <- x$contiguous(memory_format = ctx_memory_format)
      if(!is.null(b)) {
        b <- b$contiguous()
      } else {
        b <- .null_tensor
      }
      
      y <- x
      
      if(spec$name != 'linear' | gain != 1 | clamp >= 0 | !torch_equal(b, .null_tensor)) {
        ## dim - 1L to switch to zero-based expected by compiled function 
        y <- cpp_bias_act(x, b, .null_tensor, .null_tensor, .null_tensor, 0, dim - 1L, spec$cuda_idx, alpha, gain, clamp)
      }
      
      ctx$save_for_backward(
        if('x' %in% spec$ref | spec$has_2nd_grad) x else .null_tensor,
        if('x' %in% spec$ref | spec$has_2nd_grad) b else .null_tensor,
        if('y' %in% spec$ref) y else .null_tensor,
        ctx_memory_format)
      
      return(y)
    },
    
    backward = function(ctx, dy) {

      x <- ctx$saved_variables[[1]]
      b <- ctx$saved_variables[[2]]
      y <- ctx$saved_variables[[3]]
      memory_format <- ctx$saved_variables[[4]]
      db <- NULL
      
      dy <- dy$contiguous(memory_format = memory_format)
      
      if(ctx$needs_input_grad[1] | ctx$needs_input_grad[2]) {
        dx <- dy
        if(spec$name != 'linear' | gain != 1 | clamp >= 0) {
          dx <- BiasActCudaGrad(dy, x, b, y)
        }
      }
      
      
      if(ctx$needs_input_grad[2]) {
        sum_dims <- seq_len(dx$ndim)
        sum_dims[dim] <- sum_dims[-dim]
        db <- dx$sum(sum_dims)
      }
      
      return(list(x = dx, b = db))
    }
    
  )
  
  # Backward op.
  BiasActCudaGrad <- autograd_function(
    forward = function(ctx, dy, x, b, y) {
      memory_format <- if(dy$ndim > 2 & dy$stride(1) == 1) torch_channels_last() else torch_contiguous_format()
      dx <- cpp_bias_act(dy, b, x, y, .null_tensor, 1, dim - 1L, spec$cuda_idx, alpha, gain, clamp)
      ctx$save_for_backward(
        if(spec$has_2nd_grad) dy else .null_tensor,
        x, 
        b, 
        y,
        memory_format)
      return(dx)
    },
    
    backward = function(ctx, d_dx) {

      d_dx <- d_dx$contiguous(memory_format = ctx$saved_variables[[5]])
      dy <- ctx$saved_variables[[1]] 
      x <- ctx$saved_variables[[2]] 
      b <- ctx$saved_variables[[3]] 
      y <- ctx$saved_variables[[4]]
      d_dy <- NULL
      d_x <- NULL
      d_b <- NULL
      d_y <- NULL
      
      if(ctx$needs_input_grad[1]) {
        d_dy <- BiasActCudaGrad(d_dx, x, b, y)
      }
      
      if(spec$has_2nd_grad & (ctx$needs_input_grad[2] | ctx$needs_input_grad[3])) {
        d_x <- cpp_bias_act(d_dx, b, x, y, dy, 2, dim - 1L, spec$cuda_idx, alpha, gain, clamp)
      }
      
      if(spec$has_2nd_grad & ctx$needs_input_grad[3]) {
        sum_dims <- seq_len(d_x$ndim)
        sum_dims[dim] <- sum_dims[-dim]
        d_b <- d_x$sum(sum_dims)
      }
      
      return(list(dy = d_dy, x = d_x, b = d_b, y = d_y))
    }
    
  )
  
  return(BiasActCuda)
  
}
