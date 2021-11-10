#define LANTERN_BUILD
#include "lantern/lantern.h"
#include <torch/torch.h>
#include "../../utils.hpp"


void* _lantern_contrib_bias_act (void* x, void* b, void* xref, void* yref, void* dy, int grad, int dim, int act, float alpha, float gain, float clamp)
{
    LANTERN_FUNCTION_START
    throw std::runtime_error("lantern's `bias_act` is only supported on CUDA runtimes.");
    LANTERN_FUNCTION_END
}