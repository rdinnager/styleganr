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

is_tensor <- function(x) {
  inherits(x, "torch_tensor")
}

`%*%.default` <-.Primitive("%*%") # assign default as current definition
`%*%` = function(x, ...){ #make S3
  UseMethod("%*%", x)
}

#' @export
`%*%.torch_tensor` <- function(e1, e2) {
  if(!is_tensor(e1)) {
    e1 <- torch_tensor(e1, device = e2$device)
  }

  torch_mm(e1, e2)
}
