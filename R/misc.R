assert_shape <- function(tensor, ref_shape) {
  if(tensor$ndim != length(ref_shape)) {
    stop(glue::glue('Wrong number of dimensions: got {tensor$ndim}, expected {length(ref_shape)}'))
  }
# for idx, (size, ref_size) in enumerate(zip(tensor.shape, ref_shape)):
#   if ref_size is None:
#   pass
# elif isinstance(ref_size, torch.Tensor):
#   with suppress_tracer_warnings(): # as_tensor results are registered as constants
#   symbolic_assert(torch.equal(torch.as_tensor(size), ref_size), f'Wrong size for dimension {idx}')
# elif isinstance(size, torch.Tensor):
#   with suppress_tracer_warnings(): # as_tensor results are registered as constants
#   symbolic_assert(torch.equal(size, torch.as_tensor(ref_size)), f'Wrong size for dimension {idx}: expected {ref_size}')
# elif size != ref_size:
#   raise AssertionError(f'Wrong size for dimension {idx}: got {size}, expected {ref_size}')

}
