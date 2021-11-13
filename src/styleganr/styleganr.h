#ifdef _WIN32
#ifndef STYLGANR_HEADERS_ONLY
#define STYLEGANR_API extern "C" __declspec(dllexport)
#else
#define STYLEGANR_API extern "C" __declspec(dllimport)
#endif
#else
#define STYLEGANR_API extern "C"
#endif

STYLEGANR_API void* c_styleganr_bias_act (void* x, void* b, void* xref, void* yref, void* dy, int grad, int dim, int act, float alpha, float gain, float clamp);
STYLEGANR_API void* _styleganr_upfirdn2d (void* x, void* f, int upx, int upy, int downx, int downy, int padx0, int padx1, int pady0, int pady1, bool flip, float gain);
STYLEGANR_API void* _styleganr_filtered_lrelu_act (void* x, void* si, int sx, int sy, float gain, float slope, float clamp, bool writeSigns);




