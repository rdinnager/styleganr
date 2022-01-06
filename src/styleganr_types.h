#include <torch.h>

namespace styleganr {
  class TensorTensorInt {
  public:
    std::shared_ptr<void> x_;
    operator SEXP () const;
    TensorTensorInt (void* x);
  };
}
