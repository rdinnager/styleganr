//#define LANTERN_BUILD
//#include "lantern/lantern.h"
#include <torch/torch.h>
#include "styleganr/styleganr.h"
//#include "../../utils.hpp"


STYLEGANR_API void * c_styleganr_filtered_lrelu_act (void* x, void* si, int sx, int sy, float gain, float slope, float clamp, bool writeSigns)
{
  //LANTERN_FUNCTION_START
  throw std::runtime_error("C++ `filtered_lrelu_act` is only supported on CUDA runtimes.");
  //LANTERN_FUNCTION_END
}