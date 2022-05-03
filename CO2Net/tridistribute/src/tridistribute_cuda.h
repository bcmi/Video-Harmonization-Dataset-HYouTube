#ifndef TRIDISTRIBUTE_CUDA_H
#define TRIDISTRIBUTE_CUDA_H

#import <torch/extension.h>

int trildistribute_forward_cuda(torch::Tensor mask, torch::Tensor lut, torch::Tensor lut_count, torch::Tensor input_image, torch::Tensor output_image,
                      int lut_dim, int shift, float binsize, int width, int height, int batch);

int tridistribute_backward_cuda(torch::Tensor mask, torch::Tensor input_image, torch::Tensor output_image_grad, torch::Tensor lut_count, torch::Tensor lut_grad,
                       int lut_dim, int shift, float binsize, int width, int height, int batch);

int tridistribute_count_cuda(torch::Tensor lut_count, torch::Tensor input_image, torch::Tensor output_image,
                      int lut_dim, int shift, float binsize, int width, int height, int batch);


int tridistribute_divide_cuda(torch::Tensor lut, torch::Tensor lut_count, int lut_dim, int shift, int batch);

#endif