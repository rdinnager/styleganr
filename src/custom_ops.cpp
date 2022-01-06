#include <Rcpp.h>
#include <iostream>
#define STYLEGANR_HEADERS_ONLY
#include "styleganr/styleganr.h"
#define TORCH_IMPL
#define IMPORT_TORCH
#include <torch.h>
#include "styleganr_types.h"


// [[Rcpp::export]]
torch::Tensor cpp_bias_act (torch::Tensor x, torch::Tensor b, torch::Tensor xref, torch::Tensor yref, torch::Tensor dy, int grad, int dim, int act, float alpha, float gain, float clamp)
{
  return torch::Tensor(c_styleganr_bias_act(
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
torch::Tensor cpp_upfirdn2d (torch::Tensor x, torch::Tensor f, int upx, int upy, int downx, int downy, int padx0, int padx1, int pady0, int pady1, bool flip, float gain)
{
  return torch::Tensor(c_styleganr_upfirdn2d(
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
torch::Tensor cpp_filtered_lrelu_act (torch::Tensor x, torch::Tensor si, int sx, int sy, float gain, float slope, float clamp, bool writeSigns)
{
  return torch::Tensor(c_styleganr_filtered_lrelu_act(
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
styleganr::TensorTensorInt cpp_filtered_lrelu (torch::Tensor x, torch::Tensor fu, torch::Tensor fd, torch::Tensor b, torch::Tensor si, int up, int down, int px0, int px1, int py0, int py1, int sx, int sy, float gain, float slope, float clamp, bool flip_filters, bool writeSigns)
{
  return styleganr::TensorTensorInt(c_styleganr_filtered_lrelu(
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