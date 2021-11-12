#' Modulated Convolution
#'
#' @param x
#' @param w
#' @param s
#' @param demodulate
#' @param padding
#' @param input_gain
#'
#' @return
#' @export
#' @importFrom zeallot %<-%
#'
#' @examples
modulated_conv2d <- function(
  x,                  # Input tensor: [batch_size, in_channels, in_height, in_width]
  w,                  # Weight tensor: [out_channels, in_channels, kernel_height, kernel_width]
  s,                  # Style tensor: [batch_size, in_channels]
  demodulate  = TRUE, # Apply weight demodulation?
  padding     = 0,    # Padding: int or [padH, padW]
  input_gain  = NULL # Optional scale factors for the input channels: [], [in_channels], or [batch_size, in_channels]
) {

  batch_size <- x$shape[1]
  c(out_channels, in_channels, kh, kw) %<-% w$shape
  assert_shape(w, c(out_channels, in_channels, kh, kw)) # [OIkk]
  assert_shape(x, c(batch_size, in_channels, NA, NA)) # [NIHW]
  assert_shape(s, c(batch_size, in_channels)) # [NI]

  # Pre-normalize inputs.
  if(demodulate) {
    w <- w * w$square()$mean(c(2, 3, 4), keepdim = TRUE)$rsqrt()
    s <- s * s$square()$mean()$rsqrt()
  }

  # Modulate weights.
  w <- w$unsqueeze(1) # [NOIkk]
  w <- w * s$unsqueeze(2)$unsqueeze(4)$unsqueeze(5) # [NOIkk]

  # Demodulate weights.
  if(demodulate) {
    dcoefs <- (w$square()$sum(dim = c(3,4,5)) + 1e-8)$rsqrt() # [NO]
    w <- w * dcoefs$unsqueeze(3)$unsqueeze(4)$unsqueeze(5) # [NOIkk]
  }

  # Apply input scaling.
  if(!is.null(input_gain)) {
    input_gain <- input_gain$expand(c(batch_size, in_channels)) # [NI]
    w <- w * input_gain$unsqueeze(2)$unsqueeze(4)$unsqueeze(5) # [NOIkk]
  }

  # Execute as one fused op using grouped convolution.
  x <- x$reshape(c(1, -1, x$shape[3:4])) ## *x$shape[2:] * means to 'unpack'
  w <- w$reshape(c(-1, in_channels, kh, kw))
  x <- conv2d_gradfix(input = x, weight = w$to(x$dtype), padding = padding, groups = batch_size)
  x = x$reshape(c(batch_size, -1, x$shape[3:4]))
  return(x)
}

## as far as I can tell, similar to nn_linear but uses a custom op to add an activation to the bias
FullyConnectedLayer <- nn_module(
  initialize = function(
                        in_features,                # Number of input features.
                        out_features,               # Number of output features.
                        activation      = 'linear', # Activation function: 'relu', 'lrelu', etc.
                        bias            = TRUE,     # Apply additive bias before the activation function?
                        lr_multiplier   = 1,        # Learning rate multiplier.
                        weight_init     = 1,        # Initial standard deviation of the weight tensor.
                        bias_init       = 0        # Initial value of the additive bias.
  ) {
    
    self$in_features <- in_features
    self$out_features <- out_features
    self$activation <- activation
    self$weight <- nn_parameter(torch_randn(out_features, in_features) * (weight_init / lr_multiplier))
    if(bias) {
      bias_init <- rep(bias_init, out_features)
      self$bias <- nn_parameter(torch_tensor(bias_init / lr_multiplier))
    }
    self$weight_gain <- lr_multiplier / sqrt(in_features)
    self$bias_gain <- lr_multiplier
    
  },
  
  forward = function(x) {
    w <- self$weight$to(x$dtype) * self$weight_gain
    b <- self$bias
    if(!is.null(b)) {
      b = b$to(x$dtype)
    }
    if(self$bias_gain != 1) {
      b <- b * self$bias_gain
    }
    if(self$activation == 'linear' & !is.null(b)) {
      x <- torch_addmm(b$unsqueeze(1), x, w$t())
    } else {
      x <- x$matmul(w$t())
      x <- contrib_bias_act(x, b, dim = 2, act = self$activation)  ## bias_act: this is custom cuda op, must be imported
    }
    
    return(x)
  },
  
  extra_repr = function() {
    return(glue::glue('in_features={self$in_features}, out_features={self$out_features}, activation={self$activation}'))
  }

)

#----------------------------------------------------------------------------

MappingNetwork <- nn_module(
  initialize = function(
               z_dim,                      # Input latent (Z) dimensionality.
               c_dim,                      # Conditioning label (C) dimensionality, 0 = no labels.
               w_dim,                      # Intermediate latent (W) dimensionality.
               num_ws,                     # Number of intermediate latents to output.
               num_layers      = 2,        # Number of mapping layers.
               lr_multiplier   = 0.01,     # Learning rate multiplier for the mapping layers.
               w_avg_beta      = 0.998     # Decay for tracking the moving average of W during training.
  ) {
  
    self$z_dim <- z_dim
    self$c_dim <- c_dim
    self$w_dim <- w_dim
    self$num_ws <- num_ws
    self$num_layers <- num_layers
    self$w_avg_beta <- w_avg_beta
    
    # Construct layers.
    if(self$c_dim > 0) {
      self$embed <- FullyConnectedLayer(self$c_dim, self$w_dim)
    }
    features <- c(self$z_dim + if(self$c_dim > 0) self$w_dim else 0, rep(self$w_dim, self$num_layers))
    #features = [self.z_dim + (self.w_dim if self.c_dim > 0 else 0)] + [self.w_dim] * self.num_layers
    layers <- purrr::map2(features[1:(length(features) - 1)], features[2:(length(features))],
                          ~FullyConnectedLayer(.x, .y, activation = 'lrelu', lr_multiplier = lr_multiplier))
    
    for(idx in seq_along(layers)) {
      self[[glue::glue("fc{idx}")]] <- layers[[idx]]
    }
    
    # for idx, in_features, out_features in zip(range(num_layers), features[:-1], features[1:]):
    #   layer = FullyConnectedLayer(in_features, out_features, activation='lrelu', lr_multiplier=lr_multiplier)
    #   layer_name <- glue::glue('fc{idx}')
      
    self$w_avg <- nn_buffer(torch_zeros(w_dim))
  },
  
  forward = function(z, c, truncation_psi = 1, truncation_cutoff = NULL, update_emas = FALSE) {
    assert_shape(z, c(NA, self$z_dim))
    if(is.null(truncation_cutoff)) {
      truncation_cutoff <- self$num_ws
    }
     
    # Embed, normalize, and concatenate inputs.
    x <- z$to(torch_float32())
    x <- x * (x$square()$mean(2, keepdim = TRUE) + 1e-8)$rsqrt()
    if(self$c_dim > 0) {
      assert_shape(c, c(NA, self$c_dim))
      y <- self$embed(c$to(torch_float32()))
      y <- y * (y$square()$mean(2, keepdim = TRUE) + 1e-8)$rsqrt()
      if(!is.null(x)) {
        x <- torch$cat(list(x, y), dim = 2)  
      } else {
        x <- y
      }
      
    }
    
    # Execute layers.
    for(idx in seq_along(self$num_layers)) {
      x <- self[[glue::glue('fc{idx}')]](x)
    }
    
    # Update moving average of W.
    if(update_emas) {
      self$w_avg$copy_(x$detach()$mean(dim = 1)$lerp(self$w_avg, self$w_avg_beta))
    }
    
    # Broadcast and apply truncation.
    x <- x$unsqueeze(2)$`repeat`(c(1, self$num_ws, 1))
    if(truncation_psi != 1) {
      x[ , 1:truncation_cutoff] <- self$w_avg$lerp(x[ , 1:truncation_cutoff], truncation_psi)
    }
    
    return(x)
  },
  
  extra_repr = function(){
    return(glue::glue('z_dim={self$z_dim}, c_dim={self$c_dim}, w_dim={self$w_dim}, num_ws={self$num_ws}'))
  }
)

#----------------------------------------------------------------------------


#' library(torch)
#' library(zeallot)
#' 
#' source("R/misc.R")
#' source("R/conv2d_gradfix.R")
#' 
#' z_dim <- 256
#' c_dim <- 0
#' w_dim <- 128
#' num_ws <- 5
#' z <- test_tensor(1, z_dim)$cuda()
#' 
#' mapping <- MappingNetwork(z_dim, c_dim, w_dim, num_ws)
#' mapping <- mapping$cuda()
#' 
#' test <- function() mapping(z, NULL, update_emas = TRUE)
#' tt <- test()
#' 
#' 
#' x <- test_tensor(100, 3, 128, 128)$cuda()$requires_grad_()
#' x$retain_grad()
#' w <- test_tensor(4, 3, 5, 5)$cuda()$requires_grad_()
#' w$retain_grad()
#' s <- test_tensor(100, 3)$cuda()$requires_grad_()
#' s$retain_grad()
#' 
#' 
#' #' I seem to have discovered a bug relating to custom autograd functions. If using a custom autograd function `autograd_grad()` hangs my R session indefinitely, seemingly doing nothing. Here is a reprex:
#' 
#' #' First I can confirm that `autograd_grad()` works with a normal `torch` function:
#' library(torch)
#' 
#' w <- torch_randn(4, 3, 5, 5)$requires_grad_()
#' w$retain_grad()
#' 
#' #' Using a regular torch function works as expected:
#' m <- torch_exp(w)$mean()
#' autograd_grad(m, w, torch_ones_like(w))
#' 
#' #' Now we create a custom autograd function (this is from the example for `autograd_function`), with print statements so we know when the forward and backward are being called.
#' exp2 <- autograd_function(
#'   forward = function(ctx, i) {
#'     print("exp2 forward..")
#'     result <- i$exp()
#'     ctx$save_for_backward(result = result)
#'     result
#'   },
#'   backward = function(ctx, grad_output) {
#'     print("exp2 backward..")
#'     list(i = grad_output * ctx$saved_variables$result)
#'   }
#' )
#' 
#' #' Now we try using it (using a new tensor just to make sure there are no leftover gradients from before).
#' w2 <- torch_randn(4, 3, 5, 5)$requires_grad_()
#' w2$retain_grad()
#' m2 <- exp2(w2)$mean()
#' 
#' #' Now if we just try to run `autograd_backward` it seems to work fine, which is good.
#' w2$grad
#' autograd_backward(m2)
#' autograd_backward(m2, torch_ones_like(w2))
#' w2$grad
#' 
#' #' But if we try and calculate a gradient using `autograd_grad` instead, the function hangs forever, with no error message(s). The print statement I placed in forward and backward never get printed, implying they simply are not being run.
#' w3 <- torch_randn(4, 3, 5, 5)$requires_grad_()
#' w3$retain_grad()
#' m3 <- exp2(w3)$mean()
#' #' If I run this then my session hangs:
#' 
#' 
#' #' Note that this happens also if I do this all on a GPU, so it is not specific to the CPU calculations.
#' 
#' #' My session info:
#' sessionInfo()
#' 
#' autograd_grad(m3, w3, torch_ones_like(w3), create_graph = TRUE)
#' 
#' w4 <- torch_randn(4, 3, 5, 5)$requires_grad_()
#' w4$retain_grad()
#' m4 <- exp2(w4)$mean()
#' #' If I run this then my session hangs:
#' autograd_grad(m4, w4, torch_ones_like(w4))
#' 
#' 
#' Sys.setenv(STYLEGAN_GRADFIX_ENABLED = 1)
#' Sys.setenv(CUDA_CAPABILITY_MAJOR = 7)
#' Sys.setenv(STYLEGAN_WEIGHT_GRADIENTS_DISABLED = 0)
#' 
#' test <- modulated_conv2d(x, w, s)
#' 
#' m <- torch_mean(test)
#' #m$requires_grad_()
#' 
#' autograd_backward(m)
#' autograd_grad(m, w, torch_ones_like(w))
#' 
#' 
#' 
#' SynthesisInput <- nn_module(
#'   initialize = function(self,
#'                         w_dim,          # Intermediate latent (W) dimensionality.
#'                         channels,       # Number of output channels.
#'                         size,           # Output spatial size: int or [width, height].
#'                         sampling_rate,  # Output sampling rate.
#'                         bandwidth      # Output bandwidth.
#'   ) {
#'   
#'     self.w_dim = w_dim
#'     self.channels = channels
#'     self.size = np.broadcast_to(np.asarray(size), [2])
#'     self.sampling_rate = sampling_rate
#'     self.bandwidth = bandwidth
#'     
#'     # Draw random frequencies from uniform 2D disc.
#'     freqs = torch.randn([self.channels, 2])
#'     radii = freqs.square().sum(dim=1, keepdim=True).sqrt()
#'     freqs /= radii * radii.square().exp().pow(0.25)
#'     freqs *= bandwidth
#'     phases = torch.rand([self.channels]) - 0.5
#'     
#'     # Setup parameters and buffers.
#'     self.weight = torch.nn.Parameter(torch.randn([self.channels, self.channels]))
#'     self.affine = FullyConnectedLayer(w_dim, 4, weight_init=0, bias_init=[1,0,0,0])
#'     self.register_buffer('transform', torch.eye(3, 3)) # User-specified inverse transform wrt. resulting image.
#'     self.register_buffer('freqs', freqs)
#'     self.register_buffer('phases', phases)
#' 
#'   },
#'   
#'   def forward(self, w):
#'     # Introduce batch dimension.
#'     transforms = self.transform.unsqueeze(0) # [batch, row, col]
#'   freqs = self.freqs.unsqueeze(0) # [batch, channel, xy]
#'   phases = self.phases.unsqueeze(0) # [batch, channel]
#'   
#'   # Apply learned transformation.
#'   t = self.affine(w) # t = (r_c, r_s, t_x, t_y)
#'   t = t / t[:, :2].norm(dim=1, keepdim=True) # t' = (r'_c, r'_s, t'_x, t'_y)
#'   m_r = torch.eye(3, device=w.device).unsqueeze(0).repeat([w.shape[0], 1, 1]) # Inverse rotation wrt. resulting image.
#'   m_r[:, 0, 0] = t[:, 0]  # r'_c
#'   m_r[:, 0, 1] = -t[:, 1] # r'_s
#'   m_r[:, 1, 0] = t[:, 1]  # r'_s
#'   m_r[:, 1, 1] = t[:, 0]  # r'_c
#'   m_t = torch.eye(3, device=w.device).unsqueeze(0).repeat([w.shape[0], 1, 1]) # Inverse translation wrt. resulting image.
#'   m_t[:, 0, 2] = -t[:, 2] # t'_x
#'   m_t[:, 1, 2] = -t[:, 3] # t'_y
#'   transforms = m_r @ m_t @ transforms # First rotate resulting image, then translate, and finally apply user-specified transform.
#'   
#'   # Transform frequencies.
#'   phases = phases + (freqs @ transforms[:, :2, 2:]).squeeze(2)
#'   freqs = freqs @ transforms[:, :2, :2]
#'   
#'   # Dampen out-of-band frequencies that may occur due to the user-specified transform.
#'   amplitudes = (1 - (freqs.norm(dim=2) - self.bandwidth) / (self.sampling_rate / 2 - self.bandwidth)).clamp(0, 1)
#'   
#'   # Construct sampling grid.
#'   theta = torch.eye(2, 3, device=w.device)
#'   theta[0, 0] = 0.5 * self.size[0] / self.sampling_rate
#'   theta[1, 1] = 0.5 * self.size[1] / self.sampling_rate
#'   grids = torch.nn.functional.affine_grid(theta.unsqueeze(0), [1, 1, self.size[1], self.size[0]], align_corners=False)
#'   
#'   # Compute Fourier features.
#'   x = (grids.unsqueeze(3) @ freqs.permute(0, 2, 1).unsqueeze(1).unsqueeze(2)).squeeze(3) # [batch, height, width, channel]
#'   x = x + phases.unsqueeze(1).unsqueeze(2)
#'   x = torch.sin(x * (np.pi * 2))
#'   x = x * amplitudes.unsqueeze(1).unsqueeze(2)
#'   
#'   # Apply trainable mapping.
#'   weight = self.weight / np.sqrt(self.channels)
#'   x = x @ weight.t()
#'   
#'   # Ensure correct shape.
#'   x = x.permute(0, 3, 1, 2) # [batch, channel, height, width]
#'   misc.assert_shape(x, [w.shape[0], self.channels, int(self.size[1]), int(self.size[0])])
#'   return x
#'   
#'   def extra_repr(self):
#'     return '\n'.join([
#'       f'w_dim={self.w_dim:d}, channels={self.channels:d}, size={list(self.size)},',
#'       f'sampling_rate={self.sampling_rate:g}, bandwidth={self.bandwidth:g}'])
#'   
#' )
