conv2d_gradfix <- function(input, weight, bias=NULL, stride=1, padding=0, dilation=1, groups=1) {
  if(.should_use_custom_op(input)) {
  return(.conv2d_gradfix(transpose = FALSE, weight_shape = weight$shape, stride = stride, padding = padding, output_padding = 0, dilation = dilation, groups = groups)(input, weight, bias))
  }
  return(nnf_conv2d(input = input, weight = weight, bias = bias, stride = stride, padding = padding, dilation = dilation, groups = groups))
}

conv_transpose2d_gradfix <- function(input, weight, bias = NULL, stride = 1, padding = 0, output_padding = 0, groups = 1, dilation = 1) {
  if(.should_use_custom_op(input)) {
    return(.conv2d_gradfix(transpose = TRUE, weight_shape = weight$shape, stride = stride, padding = padding, output_padding = output_padding, groups = groups, dilation = dilation)(input, weight, bias))
  }

  return(nnf_conv_transpose2d(input = input, weight = weight, bias = bias, stride = stride, padding = padding, output_padding = output_padding, groups = groups, dilation = dilation))
}
#----------------------------------------------------------------------------

.should_use_custom_op <- function(input) {
  assertthat::assert_that(is_torch_tensor(input))

  if(!(Sys.getenv("STYLEGAN_GRADFIX_ENABLED") == 1)) {
    return(FALSE)
  }

  if(input$device$type != 'cuda') {
    return(FALSE)
  }

  return(TRUE)
}

.tuple_of_ints <- function(xs, ndim) {
  if(length(xs) == ndim) {
    return(xs)
  }
  xs <- rep(as.integer(xs), ndim)
  assertthat::are_equal(length(xs), ndim)
  assertthat::assert_that(all(purrr::map_lgl(xs, ~is.integer(.x))))
  return(xs)
}

.conv2d_gradfix <- function(transpose, weight_shape, stride, padding, output_padding, dilation, groups) {

  .null_tensor <- torch_empty(0)
  # Parse arguments.

  ndim <- 2
  weight_shape <- as.vector(as.array(weight_shape))
  stride <- .tuple_of_ints(stride, ndim)
  padding <- .tuple_of_ints(padding, ndim)
  output_padding <- .tuple_of_ints(output_padding, ndim)
  dilation <- .tuple_of_ints(dilation, ndim)

  # Lookup from cache.
  ## Realised this is just memoisation, now use the memoise package instead
  # key <- digest::digest(list(transpose, weight_shape, stride, padding, output_padding, dilation, groups))
  # if(.stylegan_env$.conv2d_gradfix_cache$has(key)) {
  #   return(.stylegan_env$.conv2d_gradfix_cache$get(key))
  # }

  # Validate arguments.
  dimr <- seq_len(ndim)
  assertthat::assert_that(groups >= 1)
  assertthat::assert_that(length(weight_shape) == ndim + 2)
  assertthat::assert_that(all(stride[dimr] >= 1))
  assertthat::assert_that(all(padding[dimr] >= 0))
  assertthat::assert_that(all(dilation[dimr] >= 0))
  if(!transpose) {
    assertthat::assert_that(all(output_padding[dimr] == 0))
  } else { # transpose
    assertthat::assert_that(all(0 <= output_padding[dimr] & output_padding[dimr] < pmax(stride[dimr], dilation[dimr])))
  }

  # Helpers.
  common_kwargs <- list(stride=stride, padding=padding, dilation=dilation, groups=groups)
  calc_output_padding <- function(input_shape, output_shape) {
    if(transpose) {
      return(c(0, 0))
    }
    return(
      input_shape[dimr + 2] -
        (output_shape[dimr + 2] - 1) * stride[dimr] -
        (1 - 2 * padding[dimr]) -
        dilation[dimr] * (weight_shape[dimr + 2] - 1)
    )
  }

  # Forward & backward.
  Conv2d <- autograd_function(

    forward = function(ctx, input, weight, bias) {
      
      assertthat::are_equal(weight$shape, weight_shape)
      ctx$save_for_backward(
        if(weight$requires_grad) input else .null_tensor,
        if(input$requires_grad) weight else .null_tensor,
        input_shape = input$shape
      )
      #ctx$save_for_backward()

      # Simple 1x1 convolution => cuBLAS (only on Volta, not on Ampere).
      if(all(weight_shape[3:length(weight_shape)] == c(1, 1)) &
         all(stride == c(1, 1)) &
         all(dilation == c(1, 1)) &
         all(padding == c(0, 0)) &
         Sys.getenv("CUDA_CAPABILITY_MAJOR") < 8) {
        a <- weight$reshape(c(groups, weight_shape[1] %/% groups, weight_shape[2]))
        b <- input$reshape(c(input$shape[1], groups, input$shape[2] %/% groups, -1))
        if(transpose) {
          c <- a$transpose(2, 3) %*% b$permute(c(2, 3, 1, 4))$flatten(3)
        } else {
          c <- a %*% b$permute(c(2, 3, 1, 4))$flatten(3)
        }
        #c <- (a$transpose(2, 3) if transpose else a) @ b.permute(1, 2, 0, 3).flatten(2)
        c <- c$reshape(c(-1, input$shape[1], input$shape[3:length(input$shape)]))$transpose(1, 2)
        if(!is.null(bias)) {
          c <- c + bias$unsqueeze(1)$unsqueeze(3)$unsqueeze(4)
        }
        if(input$stride(1) == 1) {
          mem_format <- torch_channels_last_format()
        } else {
          mem_format <- torch_contiguous_format()
        }
        return(c$contiguous(memory_format = mem_format))
      }
      # General case => cuDNN.
      if(transpose) {
        return(rlang::exec(nnf_conv_transpose2d, input = input, weight = weight,
                           bias = bias, output_padding = output_padding,
                           !!!common_kwargs))
      }

      return(rlang::exec(nnf_conv2d, input = input, weight = weight,
                         bias = bias, !!!common_kwargs))
    },

    backward = function(ctx, grad_output) {
      
      input <- weight <- input_shape <- NULL
      c(input, weight, input_shape) %<-% ctx$saved_variables

      grad_input <- .null_tensor
      grad_weight <- .null_tensor
      grad_bias <- .null_tensor

      if(ctx$needs_input_grad[[1]]) {
        p <- calc_output_padding(input_shape = input_shape, output_shape = grad_output$shape)
        op <- rlang::exec(.conv2d_gradfix, transpose = (!transpose), weight_shape = weight_shape,
                              output_padding = p, !!!common_kwargs)
        grad_input = op(grad_output, weight, NULL)
        assertthat::are_equal(grad_input$shape, input_shape)
      }

      if(ctx$needs_input_grad[[2]] & !(Sys.getenv("STYLEGAN_WEIGHT_GRADIENTS_DISABLED") == 1)) {
        grad_weight <- Conv2dGradWeight(grad_output, input)
        assertthat::are_equal(grad_weight$shape, weight_shape)
      }
      
      if(ctx$needs_input_grad[[3]]) {
        grad_bias <- grad_output$sum(c(1, 3, 4))
      }
      
      return(list(input = grad_input, weight = grad_weight, bias = grad_bias))
    }
  )

  # Gradient with respect to the weights.
  Conv2dGradWeight <- autograd_function(

    forward = function(ctx, grad_output, input) {
     
      ctx$save_for_backward(
        if(input$requires_grad) grad_output else .null_tensor,
        if(grad_output$requires_grad) input else .null_tensor,
        grad_output_shape = grad_output$shape, 
        input_shape = input$shape
      )

      #ctx$save_for_backward(grad_output_shape = grad_output$shape, input_shape = input$shape)
      
      # Simple 1x1 convolution => cuBLAS (on both Volta and Ampere).
      if(all(weight_shape[3:length(weight_shape)] == c(1, 1)) &
         all(stride == c(1, 1)) &
         all(dilation == c(1, 1)) &
         all(padding == c(0, 0))) {

        a <- grad_output$reshape(c(grad_output$shape[1], groups, grad_output$shape[2] %/% groups, -1))$permute(2, 3, 1, 4)$flatten(3)
        b <- input$reshape(input$shape[1], groups, input$shape[1] %/% groups, -1)$permute(2, 3, 1, 4)$flatten(3)
        if(transpose) {
          c <- (b %*% a$transpose(2, 3))$reshape(weight_shape)
        } else {
          c <- (a @ b.transpose(2, 3))$reshape(weight_shape)
        }

        if(input$stride(1) == 1) {
          mem_format <- torch_channels_last_format()
        } else {
          mem_format <- torch_contiguous_format()
        }
        return(c$contiguous(memory_format = mem_format))

      }

      # General case => cuDNN.
      if(transpose) {
        name <- 'torch_cudnn_convolution_transpose_backward_weight'
      } else {
        name <- 'torch_cudnn_convolution_backward_weight'
      }
      flags <- list(benchmark = FALSE, deterministic = FALSE, allow_tf32 = TRUE)
      return(call_torch_function(name, weight_size = weight_shape,
                                 grad_output = grad_output, self = input,
                                 padding = padding, stride = stride,
                                 dilation = dilation, groups = groups,
                                 !!!flags))
    },

    backward = function(ctx, grad2_grad_weight) {
      
      grad_output <- input <- grad_output_shape <- input_shape <- NULL
      c(grad_output, input, grad_output_shape, input_shape) %<-% ctx$saved_variables

      grad2_grad_output <- .null_tensor
      grad2_input = .null_tensor

      if(ctx$needs_input_grad[[1]]) {
        grad2_grad_output <- Conv2d()(input, grad2_grad_weight, NULL)
        assertthat::are_equal(grad2_grad_output$shape, grad_output_shape)
      }

      if(ctx$needs_input_grad[[2]]) {
        p <- calc_output_padding(input_shape = input_shape, output_shape = grad_output_shape)
        op = rlang::exec(.conv2d_gradfix, transpose = (!transpose), weight_shape = weight_shape, output_padding = p, !!!common_kwargs)
        grad2_input <- op(grad_output, grad2_grad_weight, NULL)
        assertthat::are_equal(grad2_input.shape, input_shape)
      }

      return(list(grad_output = grad2_grad_output, input = grad2_input))
    }
  )

  return(Conv2d)

}


