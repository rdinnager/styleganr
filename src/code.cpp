#include <Rcpp.h>
#include <iostream>
#define STYLEGANR_HEADERS_ONLY
#include "styleganr/styleganr.h"
#define TORCH_IMPL
#define IMPORT_TORCH
#include <torch.h>


// [[Rcpp::export]]
XPtrTorchTensor cpp_bias_act (XPtrTorchTensor x, XPtrTorchTensor b, XPtrTorchTensor xref, XPtrTorchTensor yref, XPtrTorchTensor dy, int grad, int dim, int act, float alpha, float gain, float clamp)
{
  return XPtrTorchTensor(_styleganr_bias_act(
      x.get(), 
      b.get(), 
      xref.get(), 
      yref.get(), 
      dy.get(), 
      grad, 
      dim,
      act, 
      alpha, 
      gain, 
      clamp)
  );
}