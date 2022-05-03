#ifndef TRIDISTRIBUTE_H
#define TRIDISTRIBUTE_H

#include<torch/extension.h>

int trildistribute_forward(torch::Tensor lut, torch::Tensor lut_count, torch::Tensor input_image, torch::Tensor output_image,
                      int lut_dim, int shift, float binsize, int width, int height, int batch);

int tridistribute_backward(torch::Tensor input_image, torch::Tensor output_image_grad, torch::Tensor lut_count, torch::Tensor lut_grad,
                       int lut_dim, int shift, float binsize, int width, int height, int batch);

int tridistribute_count(torch::Tensor lut_count, torch::Tensor input_image, torch::Tensor output_image,
                      int lut_dim, int shift, float binsize, int width, int height, int batch);


int tridistribute_divide_cuda(torch::Tensor lut, torch::Tensor lut_count, int lut_dim, int shift);

#endif
