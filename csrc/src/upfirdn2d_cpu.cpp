//#define LANTERN_BUILD
//#include "lantern/lantern.h"
#include <torch/torch.h>
//#include "../../utils.hpp"


void* _styleganr_upfirdn2d (void* x, void* f, int upx, int upy, int downx, int downy, int padx0, int padx1, int pady0, int pady1, bool flip, float gain)
{
  //LANTERN_FUNCTION_START
  throw std::runtime_error("C++ `upfirdn2d` is only supported on CUDA runtimes.");
  //LANTERN_FUNCTION_END
}