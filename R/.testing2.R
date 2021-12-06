library(styleganr)
Sys.setenv(NO_CUSTOM_OP = 1)

G <- sgr_get_model(device = "cuda")

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

sample_image <- function(G, z) {
  
  with_no_grad({
    dev <- G$mapping$fc0$weight$device
    z <- torch_tensor(z)$unsqueeze(1)$to(device = dev)
    im <- G(z, NULL)$cpu()
  })
  
  as_cimg(im)
  
}

plot(sample_image(G, rnorm(512)))