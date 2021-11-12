#ifdef _WIN32
#ifndef STYLGANR_HEADERS_ONLY
#define STYLEGANR_API extern "C" __declspec(dllexport)
#else
#define STYLEGANR_API extern "C" __declspec(dllimport)
#endif
#else
#define STYLEGANR_API extern "C"
#endif

STYLEGANR_API void* _stylganr_act_bias (void* x, void* b, void* xref, void* yref, void* dy, int grad, int dim, int act, float alpha, float gain, float clamp);




