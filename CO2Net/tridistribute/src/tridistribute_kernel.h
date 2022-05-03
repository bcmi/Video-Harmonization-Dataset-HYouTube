#ifndef _TRIDISTRIBUTE_KERNEL
#define _TRIDISTRIBUTE_KERNEL

#include <THC/THC.h>

__global__ void TriDistributeForward(const int nthreads, const float* mask, float* lut, float* lut_count, const float* input_image, const float* output_image, const int dim, const int shift, const float binsize, const int width, const int height, const int batch);

int TriDistributeForwardLaucher(const float* mask, float* lut, float* lut_count, const float* input_image, const float* output_image, const int dim, const int shift, const float binsize, const int width, const int height, const int batch, cudaStream_t stream);

__global__ void TriDistributeBackward(const int nthreads, const float* mask, const float* input_image, float* output_image_grad, const float* lut_grad, const float* lut_count, const int dim, const int shift, const float binsize, const int width, const int height, const int batch);

int TriDistributeBackwardLaucher(const float* mask, const float* input_image, float* output_image_grad, const float* lut_grad, const float* lut_count, const int dim, const int shift, const float binsize, const int width, const int height, const int batch, cudaStream_t stream);

__global__ void TriDistributeCount(const int nthreads, float* lut_count, const float* input_image, const float* output_image, const int dim, const int shift, const float binsize, const int width, const int height, const int batch);

int TriDistributeCountLaucher(float* lut_count, const float* input_image, const float* output_image, const int dim, const int shift, const float binsize, const int width, const int height, const int batch, cudaStream_t stream);

__global__ void TriDistributeDivide(const int nthreads, float* lut, const float* lut_count, const int dim, const int shift, const int batch);

int TriDistributeDivideLaucher(float* lut, const float* lut_count, const int dim, const int shift, const int batch, cudaStream_t stream);


#endif