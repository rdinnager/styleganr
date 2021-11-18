library(styleganr)
Sys.setenv(NO_CUSTOM_OP = 1)

G <- Generator(z_dim = 512, c_dim = 0, w_dim = 512, img_resolution = 512, img_channels = 3)$cuda()
G_state <- load_state_dict("F:/misc/yuruchara-512x512-final.pt")

G$load_state_dict(G_state)
G$eval()

z <- torch_randn(1, 512)#$cuda()
c <- torch_zeros(1)

#debug(G$synthesis)

as_cimg <- function(x) {
  im <- as_array(x$clamp(-1, 1)$cpu()) 
  im <- aperm(im, c(4, 3, 1, 2))
  im <- (im + 1) / 2
  im <- imager::as.cimg(im)
  im
}
plot(as_cimg(test))

sample_image <- function(G, z) {
  
  with_no_grad({
    dev <- G$mapping$fc0$weight$device
    z <- torch_tensor(z)$unsqueeze(1)$to(device = dev)
    im <- G(z, NULL)$cpu()
  })
  
  as_cimg(im)
  
}

plot(sample_image(G, rnorm(512)))

G <- Generator(128, 0, 16, 48, 3)$cuda()
z <- torch_randn(1, 128)$cuda()
c <- torch_zeros(1)$cuda()

x <- G(z, c)



source("R/misc.R")
x <- test_tensor(c(4, 20))
x$numel()


#library(torch)
#library(zeallot)

source("R/misc.R")
# source("R/conv2d_gradfix.R")
# source("R/networks.R")

z_dim <- 256
c_dim <- 0
w_dim <- 128
num_ws <- 5
z <- test_tensor(1, z_dim)$cuda()

mapping <- MappingNetwork(z_dim, c_dim, w_dim, num_ws)
mapping <- mapping$cuda()

test <- function() mapping(z, NULL, update_emas = TRUE)
tt <- test()
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

