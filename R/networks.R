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
#' @importFrom zeallot `%<-%`
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
  out_channels <- in_channels <- kh <- kw <- NULL
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
                        bias_init       = 0         # Initial value of the additive bias.
  ) {
    
    self$in_features <- in_features
    self$out_features <- out_features
    self$activation <- activation
    self$weight <- nn_parameter(torch_randn(out_features, in_features) * (weight_init / lr_multiplier))
    if(bias) {
      bias_init <- array(bias_init, dim = out_features)
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
      x <- bias_act(x, b, dim = 2, act = self$activation)  ## bias_act: this is custom cuda op, must be imported
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
      self[[glue::glue("fc{idx - 1}")]] <- layers[[idx]]
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
        x <- torch_cat(list(x, y), dim = 2)  
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

SynthesisInput <- nn_module(
  initialize = function(w_dim,          # Intermediate latent (W) dimensionality.
                        channels,       # Number of output channels.
                        size,           # Output spatial size: int or c(width, height)
                        sampling_rate,  # Output sampling rate.
                        bandwidth       # Output bandwidth.
  ) {

    self$w_dim <- w_dim
    self$channels <- channels
    self$size <- array(size, dim = 2)
    self$sampling_rate <- sampling_rate
    self$bandwidth <- bandwidth

    # Draw random frequencies from uniform 2D disc.
    freqs <- torch_randn(c(self$channels, 2))
    radii <- freqs$square()$sum(dim = 1, keepdim = TRUE)$sqrt()
    freqs <- freqs / (radii * radii$square()$exp()$pow(0.25))
    freqs <- freqs * bandwidth
    phases <- torch_rand(self$channels) - 0.5

    # Setup parameters and buffers.
    self$weight <- nn_parameter(torch_randn(c(self$channels, self$channels)))
    self$affine <- FullyConnectedLayer(w_dim, 4, weight_init = 0, bias_init = c(1, 0, 0, 0))
    self$transform <- nn_buffer(torch_eye(3, 3)) # User-specified inverse transform wrt. resulting image.
    self$freqs <- nn_buffer(freqs)
    self$phases <- nn_buffer(phases)

  },

  forward = function(w) {
    # Introduce batch dimension.
    transforms <- self$transform$unsqueeze(1) # [batch, row, col]
    freqs <- self$freqs$unsqueeze(1) # [batch, channel, xy]
    phases <- self$phases$unsqueeze(1) # [batch, channel]
  
    # Apply learned transformation.
    t <- self$affine(w) # t = (r_c, r_s, t_x, t_y)
    t <- t / t[ , 1:2]$norm(dim = 2, keepdim = TRUE) # t' = (r'_c, r'_s, t'_x, t'_y)
    m_r <- torch_eye(3, device = w$device)$unsqueeze(1)$`repeat`(c(w$shape[1], 1, 1)) # Inverse rotation wrt. resulting image.
    m_r[ , 1, 1] <- t[ , 1]  # r'_c
    m_r[ , 1, 2] <- -t[ , 2] # r'_s
    m_r[ , 2, 1] <- t[ , 2]  # r'_s
    m_r[ , 2, 2] <- t[ , 1]  # r'_c
    m_t <- torch_eye(3, device = w$device)$unsqueeze(1)$`repeat`(c(w$shape[1], 1, 1)) # Inverse translation wrt. resulting image.
    m_t[ , 1, 3] = -t[ , 3] # t'_x
    m_t[ , 2, 3] = -t[ , 4] # t'_y
    transforms <- m_r %*% m_t %*% transforms # First rotate resulting image, then translate, and finally apply user-specified transform.
  
    # Transform frequencies.
    phases <- phases + (freqs %*% transforms[ , 1:2, 3:Inf])$squeeze(3)
    freqs <- freqs %*% transforms[ , 1:2, 1:2]
  
    # Dampen out-of-band frequencies that may occur due to the user-specified transform.
    amplitudes <- (1 - (freqs$norm(dim = 3) - self$bandwidth) / (self$sampling_rate / 2 - self$bandwidth))$clamp(0, 1)
  
    # Construct sampling grid.
    theta <- torch_eye(2, 3, device = w$device)
    theta[1, 1] <- 0.5 * self$size[1] / self$sampling_rate
    theta[2, 2] <- 0.5 * self$size[2] / self$sampling_rate
    grids <- nnf_affine_grid(theta$unsqueeze(1), c(1, 1, self$size[2], self$size[1]), align_corners = FALSE)
  
    # Compute Fourier features.
    x <- (grids$unsqueeze(4) %*% freqs$permute(c(1, 3, 2))$unsqueeze(2)$unsqueeze(3))$squeeze(4) # [batch, height, width, channel]
    x <- x + phases$unsqueeze(2)$unsqueeze(3)
    x <- torch_sin(x * (pi * 2))
    x = x * amplitudes$unsqueeze(2)$unsqueeze(3)
  
    # Apply trainable mapping.
    weight <- self$weight / sqrt(self$channels)
    x <- x %*% weight$t()
  
    # Ensure correct shape.
    x <- x$permute(c(1, 4, 2, 3)) # [batch, channel, height, width]
    assert_shape(x, c(w$shape[1], self$channels, as.integer(self$size[2]), as.integer(self$size[1])))
    return(x)
  },

  extra_repr = function() {
    return(glue::glue('w_dim={self$w_dim}, channels={self$channels}, size={self$size},\nsampling_rate={self$sampling_rate}, bandwidth={self$bandwidth}'))
  }

)

SynthesisLayer <- nn_module(
  initialize = function(
               w_dim,                          # Intermediate latent (W) dimensionality.
               is_torgb,                       # Is this the final ToRGB layer?
               is_critically_sampled,          # Does this layer use critical sampling?
               use_fp16,                       # Does this layer use FP16?
               
               # Input & output specifications.
               in_channels,                    # Number of input channels.
               out_channels,                   # Number of output channels.
               in_size,                        # Input spatial size: int or [width, height].
               out_size,                       # Output spatial size: int or [width, height].
               in_sampling_rate,               # Input sampling rate (s).
               out_sampling_rate,              # Output sampling rate (s).
               in_cutoff,                      # Input cutoff frequency (f_c).
               out_cutoff,                     # Output cutoff frequency (f_c).
               in_half_width,                  # Input transition band half-width (f_h).
               out_half_width,                 # Output Transition band half-width (f_h).
               
               # Hyperparameters.
               conv_kernel         = 3,        # Convolution kernel size. Ignored for final the ToRGB layer.
               filter_size         = 6,        # Low-pass filter size relative to the lower resolution when up/downsampling.
               lrelu_upsampling    = 2,        # Relative sampling rate for leaky ReLU. Ignored for final the ToRGB layer.
               use_radial_filters  = FALSE,    # Use radially symmetric downsampling filter? Ignored for critically sampled layers.
               conv_clamp          = 256,      # Clamp the output to [-X, +X], None = disable clamping.
               magnitude_ema_beta  = 0.999     # Decay rate for the moving average of input magnitudes.
  ) {
    
    self$w_dim <- w_dim
    self$is_torgb <- is_torgb
    self$is_critically_sampled <- is_critically_sampled
    self$use_fp16 <- use_fp16
    self$in_channels <- in_channels
    self$out_channels <- out_channels
    self$in_size <- array(in_size, dim = 2) #np.broadcast_to(np.asarray(in_size), [2])
    self$out_size <- array(out_size, dim = 2) #np.broadcast_to(np.asarray(out_size), [2])
    self$in_sampling_rate <- in_sampling_rate
    self$out_sampling_rate <- out_sampling_rate
    self$tmp_sampling_rate <- max(in_sampling_rate, out_sampling_rate) * (if(is_torgb) 1 else lrelu_upsampling)
    self$in_cutoff <- in_cutoff
    self$out_cutoff <- out_cutoff
    self$in_half_width <- in_half_width
    self$out_half_width <- out_half_width
    self$conv_kernel <- if(is_torgb) 1 else conv_kernel
    self$conv_clamp <- conv_clamp
    self$magnitude_ema_beta <- magnitude_ema_beta
    
    # Setup parameters and buffers.
    self$affine <- FullyConnectedLayer(self$w_dim, self$in_channels, bias_init = 1)
    self$weight <- nn_parameter(torch_randn(c(self$out_channels, self$in_channels, self$conv_kernel, self$conv_kernel)))
    self$bias <- nn_parameter(torch_zeros(self$out_channels))
    self$magnitude_ema <- nn_buffer(torch_scalar_tensor(1.0))
    
    # Design upsampling filter.
    self$up_factor <- as.integer(round(self$tmp_sampling_rate / self$in_sampling_rate))
    assertthat::are_equal(self$in_sampling_rate * self$up_factor, self$tmp_sampling_rate)
    self$up_taps <- if(self$up_factor > 1 & !self$is_torgb) filter_size * self$up_factor else 1
    up_filter <- self$design_lowpass_filter(
      numtaps = self$up_taps, cutoff = self$in_cutoff, 
      width = self$in_half_width * 2, fs = self$tmp_sampling_rate)
    if(up_filter$numel() > 0) {
      self$up_filter <- nn_buffer(up_filter)
    } else {
      self$up_filter <- NULL
    }
    
    # Design downsampling filter.
    self$down_factor = as.integer(round(self$tmp_sampling_rate / self$out_sampling_rate))
    assertthat::are_equal(self$out_sampling_rate * self$down_factor, self$tmp_sampling_rate)
    self$down_taps <- if(self$down_factor > 1 & !self$is_torgb) filter_size * self$down_factor else 1
    self$down_radial <- use_radial_filters & !self$is_critically_sampled
    down_filter <- self$design_lowpass_filter(
      numtaps = self$down_taps, cutoff = self$out_cutoff, width = self$out_half_width * 2, fs = self$tmp_sampling_rate, radial = self$down_radial)
    if(down_filter$numel() > 0) {
      self$down_filter <- nn_buffer(down_filter)
    } else {
      self$down_filter <- NULL
    }
    
    # Compute padding.
    pad_total <- (self$out_size - 1) * self$down_factor + 1 # Desired output size before downsampling.
    pad_total <- pad_total - (self$in_size + self$conv_kernel - 1) * self$up_factor # Input size after upsampling.
    pad_total <- pad_total + self$up_taps + self$down_taps - 2 # Size reduction caused by the filters.
    pad_lo <- (pad_total + self$up_factor) %/% 2 # Shift sample locations according to the symmetric interpretation (Appendix C.3).
    pad_hi <- pad_total - pad_lo
    self$padding <- c(as.integer(pad_lo[1]), as.integer(pad_hi[1]), as.integer(pad_lo[2]), as.integer(pad_hi[2]))
    
  },
    
  forward = function(x, w, noise_mode = 'random', force_fp32 = FALSE, update_emas = FALSE) {
    assertthat::assert_that(noise_mode %in% c('random', 'const', 'none')) # unused
    assert_shape(x, c(NA, self$in_channels, as.integer(self$in_size[2]), as.integer(self$in_size[1])))
    assert_shape(w, c(x$shape[1], self$w_dim))
    
    # Track input magnitude.
    if(update_emas) {
      magnitude_cur <- x$detach()$to(torch_float32())$square()$mean()
      self$magnitude_ema$copy_(magnitude_cur$lerp(self$magnitude_ema, self$magnitude_ema_beta))
    }
    input_gain <- self$magnitude_ema$rsqrt()
    
    # Execute affine layer.
    styles <- self$affine(w)
    if(self$is_torgb) {
      weight_gain <- 1 / sqrt(self$in_channels * (self$conv_kernel^2))
      styles <- styles * weight_gain
    }
    
    # Execute modulated conv2d.
    dtype = if(self$use_fp16 & !force_fp32 & x$device$type == 'cuda') torch_float16() else torch_float32()
    x = modulated_conv2d(x = x$to(dtype), w = self$weight, s = styles,
                         padding = self$conv_kernel - 1, demodulate = (!self$is_torgb), input_gain = input_gain)
    
    # Execute bias, filtered leaky ReLU, and clamping.
    gain <- if(self$is_torgb) 1 else sqrt(2)
    slope <- if(self$is_torgb) 1 else 0.2
    x <- filtered_lrelu(x = x, fu = self$up_filter, fd = self$down_filter, b = self$bias$to(x$dtype),
                        up = self$up_factor, down = self$down_factor, padding = self$padding, 
                        gain = gain, slope = slope, clamp = self$conv_clamp)
    
    # Ensure correct shape and dtype.
    assert_shape(x, c(NA, self$out_channels, as.integer(self$out_size[2]), as.integer(self$out_size[1])))
    assertthat::assert_that(x$dtype == dtype)
    return(x)
  },
  
  design_lowpass_filter = function(numtaps, cutoff, width, fs, radial = FALSE) {
    
    assertthat::assert_that(numtaps >= 1)
  
    # Identity filter.
    if(numtaps == 1) {
      return(torch_empty(0))
    }
    
    # Separable Kaiser low-pass filter.
    if(!radial) {
      f <- scipy_signal_firwin(numtaps = numtaps, cutoff = cutoff, width = width, fs = fs)
      return(torch_tensor(f, dtype = torch_float32()))
    }
    
    # Radially symmetric jinc-based filter.
    x <- (seq(0, numtaps - 1) - (numtaps - 1) / 2) / fs
    r <- rlang::exec(hypot, !!!(meshgrid(x, x) %>% setNames(c("x1", "x2"))))
    #r = np.hypot(*np.meshgrid(x, x))
    f <- besselJ(2 * cutoff * (pi * r), 1) / (pi * r)
    #f <- scipy.special.j1()
    beta <- kaiser_beta(kaiser_atten(numtaps, width / (fs / 2)))
    w <- signal::kaiser(numtaps, beta)
    f <- f * outer(w, w)
    f <- f / sum(f)
    return(torch_tensor(f, dtype = torch_float32()))
  },
  
  extra_repr = function() {
    return(glue::glue(paste(
      'w_dim={self.w_dim:d}, is_torgb={self.is_torgb},',
      'is_critically_sampled={self.is_critically_sampled}, use_fp16={self.use_fp16},',
      'in_sampling_rate={self.in_sampling_rate:g}, out_sampling_rate={self.out_sampling_rate:g},',
      'in_cutoff={self.in_cutoff:g}, out_cutoff={self.out_cutoff:g},',
      'in_half_width={self.in_half_width:g}, out_half_width={self.out_half_width:g},',
      'in_size={list(self.in_size)}, out_size={list(self.out_size)},',
      'in_channels={self.in_channels:d}, out_channels={self.out_channels:d}',
      sep = "\n")))
  }
)

SynthesisNetwork <- nn_module(
  initialize = function(
               w_dim,                          # Intermediate latent (W) dimensionality.
               img_resolution,                 # Output image resolution.
               img_channels,                   # Number of color channels.
               channel_base        = 32768,    # Overall multiplier for the number of channels.
               channel_max         = 512,      # Maximum number of channels in any layer.
               num_layers          = 14,       # Total number of layers, excluding Fourier features and ToRGB.
               num_critical        = 2,        # Number of critically sampled layers at the end.
               first_cutoff        = 2,        # Cutoff frequency of the first layer (f_{c,0}).
               first_stopband      = 2**2.1,   # Minimum stopband of the first layer (f_{t,0}).
               last_stopband_rel   = 2**0.3,   # Minimum stopband of the last layer, expressed relative to the cutoff.
               margin_size         = 10,       # Number of additional pixels outside the image.
               output_scale        = 0.25,     # Scale factor for the output image.
               num_fp16_res        = 4,        # Use FP16 for the N highest resolutions.
               ...                             # Arguments for SynthesisLayer. (as a list)
  ) {
  
    self$w_dim <- w_dim
    self$num_ws <- num_layers + 2
    self$img_resolution <- img_resolution
    self$img_channels <- img_channels
    self$num_layers <- num_layers
    self$num_critical <- num_critical
    self$margin_size <- margin_size
    self$output_scale <- output_scale
    self$num_fp16_res <- num_fp16_res
    
    # Geometric progression of layer cutoffs and min. stopbands.
    last_cutoff <- self$img_resolution / 2 # f_{c,N}
    last_stopband <- last_cutoff * last_stopband_rel # f_{t,N}
    exponents <- pmin((seq_len(self$num_layers + 1) - 1) / (self$num_layers - self$num_critical), 1)
    cutoffs <- first_cutoff * (last_cutoff / first_cutoff)^exponents # f_c[i]
    stopbands <- first_stopband * (last_stopband / first_stopband)^exponents # f_t[i]
    
    # Compute remaining layer parameters.
    sampling_rates <- exp2(ceiling(log2(pmin(stopbands * 2, self$img_resolution)))) # s[i]
    half_widths <- pmax(stopbands, sampling_rates / 2) - cutoffs # f_h[i]
    sizes <- sampling_rates + self$margin_size * 2
    sizes[(length(sizes)-1):length(sizes)] = self$img_resolution
    channels <- as.integer(round(pmin((channel_base / 2) / cutoffs, channel_max)))
    channels[length(channels)] <- self$img_channels
    
    # Construct layers.
    self$input <- SynthesisInput(
      w_dim = self$w_dim, channels = as.integer(channels[1]), size = as.integer(sizes[1]),
      sampling_rate = sampling_rates[1], bandwidth = cutoffs[1])
    self$layer_names = character()
    for(idx in seq_len(self$num_layers + 1)) {
      prev <- max(idx - 1, 1)
      is_torgb <- (idx == (self$num_layers + 1))
      is_critically_sampled <- (idx >= self$num_layers + 1 - self$num_critical)
      use_fp16 <- (sampling_rates[idx] * (2^self$num_fp16_res) > self$img_resolution)
      layer <- SynthesisLayer( 
        w_dim = self$w_dim, is_torgb = is_torgb, is_critically_sampled = is_critically_sampled, use_fp16 = use_fp16,
        in_channels = as.integer(channels[prev]), out_channels = as.integer(channels[idx]),
        in_size = as.integer(sizes[prev]), out_size = as.integer(sizes[idx]),
        in_sampling_rate = as.integer(sampling_rates[prev]), out_sampling_rate = as.integer(sampling_rates[idx]),
        in_cutoff = cutoffs[prev], out_cutoff = cutoffs[idx],
        in_half_width = half_widths[prev], out_half_width = half_widths[idx],
        ...)
      name <- glue::glue('L{idx - 1}_{layer$out_size[1]}_{layer$out_channels}')
      self[[name]] <- layer
      self$layer_names <- c(self$layer_names, name)
    }
    
  },
  
  forward = function(ws, ...) {
    assert_shape(ws, c(NA, self$num_ws, self$w_dim))
    ws <- ws$to(torch_float32())$unbind(dim = 2)
    
    # Execute layers.
    x <- self$input(ws[[1]])
    # for name, w in zip(self.layer_names, ws[1:]):
    #   x = getattr(self, name)(x, w, **layer_kwargs)
    names_w <- purrr::transpose(list(name = self$layer_names, w = ws[2:length(ws)]))
    for(i in names_w) {
      x <- self[[i$name]](x, i$w, ...)
    }
    
    if(self$output_scale != 1) {
      x <- x * self$output_scale
    }
    
    # Ensure correct shape and dtype.
    assert_shape(x, c(NA, self$img_channels, self$img_resolution, self$img_resolution))
    x <- x$to(torch_float32())
    return(x)
  },
  
  extra_repr = function() {
    return(glue::glue(paste(
      'w_dim={self$w_dim}, num_ws={self$num_ws},',
      'img_resolution={self$img_resolution}, img_channels={self$img_channels},',
      'num_layers={self$num_layers}, num_critical={self$num_critical},',
      'margin_size={self$margin_size}, num_fp16_res={self$num_fp16_res}',
      sep = "\n")))
  }

)

#----------------------------------------------------------------------------

#' @export
Generator <- nn_module(
  initialize = function(
               z_dim,                      # Input latent (Z) dimensionality.
               c_dim,                      # Conditioning label (C) dimensionality.
               w_dim,                      # Intermediate latent (W) dimensionality.
               img_resolution,             # Output resolution.
               img_channels,               # Number of output color channels.
               mapping_kwargs     = NULL,  # Arguments for MappingNetwork (as a named list).
               ...                         # Arguments for SynthesisNetwork.
  ) {
    
    self$z_dim <- z_dim
    self$c_dim <- c_dim
    self$w_dim <- w_dim
    self$img_resolution <- img_resolution
    self$img_channels <- img_channels
    self$synthesis <- SynthesisNetwork(w_dim = w_dim, img_resolution = img_resolution, img_channels = img_channels, ...)
    self$num_ws = self$synthesis$num_ws
    if(!is.null(mapping_kwargs)) {
      self$mapping <- rlang::exec(MappingNetwork, 
                                  z_dim = z_dim, c_dim = c_dim, w_dim = w_dim, num_ws = self$num_ws, 
                                  !!!mapping_kwargs)
    } else {
      self$mapping <- MappingNetwork(z_dim = z_dim, c_dim = c_dim, w_dim = w_dim, 
                                     num_ws = self$num_ws)
    }
  },

  forward = function(z, c, truncation_psi = 1, truncation_cutoff = NULL, update_emas = FALSE, ...) {
    ws <- self$mapping(z, c, truncation_psi = truncation_psi, truncation_cutoff = truncation_cutoff, 
                      update_emas = update_emas)
    img <- self$synthesis(ws, update_emas = update_emas, ...)
    return(img)
  }
)