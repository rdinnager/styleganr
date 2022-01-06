namespace alias {
using TensorTensorInt = std::tuple<torch::Tensor,torch::Tensor,int>;
}

namespace make_raw {
void* TensorTensorInt (const alias::TensorTensorInt& x);
}

namespace from_raw {
alias::TensorTensorInt& TensorTensorInt (void* x);
}