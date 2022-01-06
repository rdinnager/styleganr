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
  return XPtrTorchTensor(c_styleganr_bias_act(
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

// [[Rcpp::export]]
XPtrTorchTensor cpp_bias_act_autograd (XPtrTorchTensor x, XPtrTorchTensor b, int cuda_idx, bool has_2nd, bool yref_bool, int dim, float alpha, float gain, float clamp)
{
  return XPtrTorchTensor(c_styleganr_bias_act_autograd(
      x.get(), 
      b.get(), 
      cuda_idx, 
      has_2nd, 
      yref_bool,
      dim, 
      alpha,
      gain, 
      clamp)
  );
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_upfirdn2d (XPtrTorchTensor x, XPtrTorchTensor f, int upx, int upy, int downx, int downy, int padx0, int padx1, int pady0, int pady1, bool flip, float gain)
{
  return XPtrTorchTensor(c_styleganr_upfirdn2d(
      x.get(), 
      f.get(), 
      upx, 
      upy, 
      downx, 
      downy, 
      padx0, 
      padx1, 
      pady0, 
      pady1, 
      flip, 
      gain)
  );
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_upfirdn2d_autograd (XPtrTorchTensor x, XPtrTorchTensor f, int upx, int upy, int downx, int downy, int padx0, int padx1, int pady0, int pady1, bool flip_filter, float gain, int fw, int fh)
{
  return XPtrTorchTensor(c_styleganr_upfirdn2d_autograd(
      x.get(), 
      f.get(), 
      upx, 
      upy, 
      downx,
      downy,
      padx0,
      padx1,
      pady0,
      pady1,
      flip_filter, 
      gain, 
      fw,
      fh)
  );
}

// [[Rcpp::export]]
XPtrTorchTensor cpp_filtered_lrelu_act (XPtrTorchTensor x, XPtrTorchTensor si, int sx, int sy, float gain, float slope, float clamp, bool writeSigns)
{
  return XPtrTorchTensor(c_styleganr_filtered_lrelu_act(
      x.get(), 
      si.get(), 
      sx, 
      sy, 
      gain, 
      slope, 
      clamp, 
      writeSigns)
  );
}

// [[Rcpp::export]]
XPtrTorchTuple cpp_filtered_lrelu (XPtrTorchTensor x, XPtrTorchTensor fu, XPtrTorchTensor fd, XPtrTorchTensor b, XPtrTorchTensor si, int up, int down, int px0, int px1, int py0, int py1, int sx, int sy, float gain, float slope, float clamp, bool flip_filters, bool writeSigns)
{
  return XPtrTorchTuple(c_styleganr_filtered_lrelu(
      x.get(), 
      fu.get(), 
      fd.get(), 
      b.get(), 
      si.get(), 
      up, 
      down, 
      px0, 
      px1, 
      py0, 
      py1, 
      sx, 
      sy, 
      gain, 
      slope,
      clamp, 
      flip_filters, 
      writeSigns)
  );
}