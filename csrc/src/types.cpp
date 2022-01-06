#include <torch/torch.h>
#define LANTERN_TYPES_IMPL
#include <lantern/types.h>
#include "styleganr/styleganr.h"
#include "types.h"


namespace make_raw {
void* TensorTensorInt (const alias::TensorTensorInt& x)
{
  return std::make_unique<alias::TensorTensorInt>(x).release();
}
}

namespace from_raw {
alias::TensorTensorInt& TensorTensorInt (void* x)
{
  return *reinterpret_cast<alias::TensorTensorInt*>(x);
}
}


STYLEGANR_API void* TensorTensorInt_get_0 (void* self)
{
  auto self_ = from_raw::TensorTensorInt(self);
  return make_raw::Tensor(std::get<0>(self_));
}

STYLEGANR_API void* TensorTensorInt_get_1 (void* self)
{
  auto self_ = from_raw::TensorTensorInt(self);
  return make_raw::Tensor(std::get<1>(self_));
}

STYLEGANR_API int TensorTensorInt_get_2 (void* self)
{
  auto self_ = from_raw::TensorTensorInt(self);
  return std::get<2>(self_);
}

STYLEGANR_API void TensorTensorInt_delete (void* self)
{
  delete reinterpret_cast<alias::TensorTensorInt*>(self);
}
