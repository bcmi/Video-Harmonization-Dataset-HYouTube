#include "tridistribute_kernel.h"
#include <torch/extension.h>
#include <THC/THC.h>

int tridistribute_forward_cuda(torch::Tensor mask, torch::Tensor lut, torch::Tensor lut_count, torch::Tensor input_image, torch::Tensor output_image,
                      int lut_dim, int shift, float binsize, int width, int height, int batch)
{
    // Grab the input tensor
    float * lut_flat = lut.data<float>();
    float * input_image_flat = input_image.data<float>();
    float * output_image_flat = output_image.data<float>();
    float * lut_count_flat = lut_count.data<float>();
    float * mask_flat = mask.data<float>();

    TriDistributeForwardLaucher(mask_flat, lut_flat, lut_count_flat, input_image_flat, output_image_flat, lut_dim, shift, binsize, width, height, batch, at::cuda::getCurrentCUDAStream());

    return 1;
}

int tridistribute_backward_cuda(torch::Tensor mask, torch::Tensor input_image, torch::Tensor output_image_grad, torch::Tensor lut_count, torch::Tensor lut_grad,
                       int lut_dim, int shift, float binsize, int width, int height, int batch)
{
    // Grab the input tensor
    float * lut_count_flat = lut_count.data<float>();
    float * output_image_grad_flat = output_image_grad.data<float>();
    float * input_image_flat = input_image.data<float>();
    float * lut_grad_flat = lut_grad.data<float>();
    float * mask_flat = mask.data<float>();


    TriDistributeBackwardLaucher(mask_flat, input_image_flat, output_image_grad_flat, lut_count_flat, lut_grad_flat, lut_dim, shift, binsize, width, height, batch, at::cuda::getCurrentCUDAStream());

    return 1;
}

int tridistribute_count_cuda(torch::Tensor lut_count, torch::Tensor input_image, torch::Tensor output_image,
                      int lut_dim, int shift, float binsize, int width, int height, int batch)
{

    float * lut_count_flat = lut_count.data<float>();
    float * output_image_flat = output_image.data<float>();
    float * input_image_flat = input_image.data<float>();

    TriDistributeCountLaucher(lut_count_flat, input_image_flat, output_image_flat, lut_dim, shift, binsize, width, height, batch, at::cuda::getCurrentCUDAStream());

    return 1;

}

int tridistribute_divide_cuda(torch::Tensor lut, torch::Tensor lut_count, int lut_dim, int shift, int batch)
{
    float * lut_count_flat = lut_count.data<float>();
    float * lut_flat = lut.data<float>();

    TriDistributeDivideLaucher(lut_flat, lut_count_flat, lut_dim, shift, batch, at::cuda::getCurrentCUDAStream());

    return 1;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &tridistribute_forward_cuda, "TriDistribute forward");
  m.def("backward", &tridistribute_backward_cuda, "TriDistribute backward");
  m.def("count", &tridistribute_count_cuda, "TriDistribute count");
  m.def("divide", &tridistribute_divide_cuda, "TriDistribute divide");
}

