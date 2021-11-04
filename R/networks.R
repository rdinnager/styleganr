
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

x <- test_tensor(100, 3, 128, 128)$cuda()
w <- test_tensor(4, 3, 5, 5)$cuda()
s <- test_tensor(100, 3)$cuda()

Sys.setenv(STYLEGAN_GRADFIX_ENABLED = 1)

test <- modulated_conv2d(x, w, s)
