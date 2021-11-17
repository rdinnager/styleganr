assert_shape <- function(tensor, ref_shape) {
  if(tensor$ndim != length(ref_shape)) {
    stop(glue::glue('Wrong number of dimensions: got {tensor$ndim}, expected {length(ref_shape)}'))
  }

  shape_match <- purrr::map2_lgl(tensor$shape, ref_shape,
                                 ~.x == .y)
  shape_match <- shape_match | is.na(ref_shape)

  if(any(!shape_match)) {
    stop(glue::glue("Shapes do not match: Wrong size for dimensions {which(!shape_match)}"))
  }

  return(invisible(TRUE))

}

test_tensor <- function(...) {
  dims <- list(...)
  dat <- rnorm(do.call(prod, dims))
  torch_tensor(array(dat, dim = unlist(dims)))
}

is_torch_tensor <- function(x) {
  inherits(x, "torch_tensor")
}

`%*%.default` <-.Primitive("%*%") # assign default as current definition
`%*%` = function(x, ...){ #make S3
  UseMethod("%*%", x)
}

#' @export
`%*%.torch_tensor` <- function(e1, e2) {
  if(!is_torch_tensor(e1)) {
    e1 <- torch_tensor(e1, device = e2$device)
  }

  torch_matmul(e1, e2)
}

call_torch_function <- function(name, ...) {
  args <- rlang::list2(...)
  f <- getNamespace("torch")[[name]]
  do.call(f, args)
}

## signal processing helpers

kaiser_atten <- function(numtaps, width) {
  a <- 2.285 * (numtaps - 1) * pi * width + 7.95
  return(a)
}

kaiser_beta <- function(a) {
  if(a > 50) {
    beta <- 0.1102 * (a - 8.7)
  } else {
    if(a > 21) {
      beta <- 0.5842 * (a - 21)^0.4 + 0.07886 * (a - 21)
    } else {
      beta <- 0
    }
  }
  return(beta)
}


scipy_signal_firwin <- function(numtaps, cutoff, width = NULL,
                                scale = TRUE, fs = 2) {
  
  nyq <- 0.5 * fs
  cutoff <- cutoff / nyq
  
  atten <- kaiser_atten(numtaps, width / nyq)
  beta <- kaiser_beta(atten)
  window <- signal::kaiser(numtaps, beta)
  
  res <- signal::fir1(numtaps - 1, cutoff, type = "DC-1", window = window, scale = scale)
  
  if(scale) {
    scale_frequency <- 0.0
    alpha <- 0.5 * (numtaps - 1)
    m <- seq(0, numtaps - 1) - alpha 
    c <- cos(pi * m * scale_frequency)
    
    s <- sum(res * c)
    res <- res / s
  }
  
  return(res)
  
}


# code taken from pracma::meshgrid()
meshgrid <- function (x, y = x) {
  x <- c(x)
  y <- c(y)
  n <- length(x)
  m <- length(y)
  X <- matrix(rep(x, each = m), nrow = m, ncol = n)
  Y <- matrix(rep(y, times = n), nrow = m, ncol = n)
  return(list(X = X, Y = Y))
}

hypot <- function(x1, x2) {
  sqrt(x1^2 + x2^2)
}

exp2 <- function(x) {
  2^x
}