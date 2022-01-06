#define STYLEGANR_HEADERS_ONLY
#include "styleganr/styleganr.h"
#include "styleganr_types.h"
#include <torch.h>

namespace styleganr {
  TensorTensorInt::operator SEXP () const 
  {
    Rcpp::List output;
    output.push_back(torch::Tensor(TensorTensorInt_get_0(this->x_.get())));
    output.push_back(torch::Tensor(TensorTensorInt_get_1(this->x_.get())));
    output.push_back(TensorTensorInt_get_2(this->x_.get()));
    return output;
  }
  TensorTensorInt::TensorTensorInt (void* x)
  {
    x_ = std::shared_ptr<void>(x, TensorTensorInt_delete);
  }
}