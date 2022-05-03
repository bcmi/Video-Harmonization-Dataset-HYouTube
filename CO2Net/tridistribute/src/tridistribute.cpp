#include "trilinear.h"

void TriDistributeCountCpu(float* lut_count, const float* input_image, const float* output_image, const int dim, const int shift, const float binsize, const int width, const int height, const int channels);

void TriDistributeForwardCpu(float* lut, const float* lut_count, const float* input_image, const float* output_image, const int dim, const int shift, const float binsize, const int width, const int height, const int channels);

void TriDistributeBackwardCpu(const float* input_image, float* output_image_grad, const float* lut_grad, const float* lut_count, const int dim, const int shift, const float binsize, const int width, const int height, const int channels);

void TriDistributeDivideCpu(float *lut, const float* lut_count, const int dim, const int shift);

int tridistribute_divide(torch::Tensor lut, torch::Tensor lut_count, int lut_dim, int shift)
{
    float * lut_flat = lut.data<float>();
    float * lut_count_flat = lut_count.data<float>();

    TriDistributeDivideCpu(lut_count, lut_count_flat, dim, shift);

    return 1;

}

int tridistribute_count(torch::Tensor lut_count, torch::Tensor input_image, torch::Tensor output_image,
                      int lut_dim, int shift, float binsize, int width, int height, int batch)
{
        // Grab the input tensor
    float * input_image_flat = input_image.data<float>();
    float * output_image_flat = output_image.data<float>();
    float * lut_count_flat = lut_count_flat.data<float>();

    auto image_size = image.sizes();
    int channels = image_size[1];
    if (channels != 3)
    {
        return 0;
    }

    TriDistributeCountCpu(lut_count_flat, input_image_flat, output_image_flat, lut_dim, shift, binsize, width, height, channels);

    return 1;


}


int trildistribute_forward(torch::Tensor lut, torch::Tensor lut_count, torch::Tensor input_image, torch::Tensor output_image,
                      int lut_dim, int shift, float binsize, int width, int height, int batch)
{
    // Grab the input tensor
    float * lut_flat = lut.data<float>();
    float * input_image_flat = input_image.data<float>();
    float * output_image_flat = output_image.data<float>();
    float * lut_count_flat = lut_count_flat.data<float>();

    // whether color image
    auto image_size = image.sizes();
    int channels = image_size[1];
    if (channels != 3)
    {
        return 0;
    }

    TriDistributeForwardCpu(lut_flat, lut_count_flat, image_flat, output_flat, lut_dim, shift, binsize, width, height, channels);

    return 1;
}

int trilinear_backward(torch::Tensor input_image, torch::Tensor output_image_grad, torch::Tensor lut_count, torch::Tensor lut_grad,
                       int lut_dim, int shift, float binsize, int width, int height, int batch)
{
    // Grab the input tensor
    float * lut_count_flat = lut_count.data<float>();
    float * output_image_grad_flat = output_image_grad.data<float>();
    float * input_image_flat = input_image.data<float>();
    float * lut_grad_flat = lut_grad.data<float>();

    // whether color image
    auto image_size = image.sizes();
    int channels = image_size[1];
    if (channels != 3)
    {
        return 0;
    }

    TriLinearBackwardCpu(input_image_flat, output_image_grad_flat, lut_grad_flat, lut_count_flat, lut_dim, shift, binsize, width, height, channels);
    return 1;
}


void TriDistributeDivideCpu(float *lut, const float* lut_count, const int dim, const int shift)
{
    const int output_size = shift;
    int index = 0;
    for (index = 0;index < output_size; ++index)
    {
        lut[index] = lut[index] / lut_count[index];
        lut[index + shift] = lut[index + shift] / lut_count[index + shift];
        lut[index + shift * 2] = lut[index + shift*2] / lut_count[index + shift * 2];

    }
}

void TriDistributeCountCpu(float* lut_count, , const float* input_image, const float* output_image, const int dim, const int shift, const float binsize, const int width, const int height, const int channels))
{
    const int output_size = height * width;;

    int index = 0;
    for (index = 0; index < output_size; ++index)
    {
        float r = input_image[index];
	float g = input_image[index + width * height];
	float b = input_image[index + width * height * 2];

	int r_id = floor(r / binsize);
	int g_id = floor(g / binsize);
	int b_id = floor(b / binsize);

	float r_d = fmod(r,binsize) / binsize;
	float g_d = fmod(g,binsize) / binsize;
	float b_d = fmod(b,binsize) / binsize;

	int id000 = r_id + g_id * dim + b_id * dim * dim;
        int id100 = r_id + 1 + g_id * dim + b_id * dim * dim;
        int id010 = r_id + (g_id + 1) * dim + b_id * dim * dim;
        int id110 = r_id + 1 + (g_id + 1) * dim + b_id * dim * dim;
        int id001 = r_id + g_id * dim + (b_id + 1) * dim * dim;
        int id101 = r_id + 1 + g_id * dim + (b_id + 1) * dim * dim;
        int id011 = r_id + (g_id + 1) * dim + (b_id + 1) * dim * dim;
        int id111 = r_id + 1 + (g_id + 1) * dim + (b_id + 1) * dim * dim;


    lut_count[id000] += 1;
    lut_count[id100] += 1;
    lut_count[id010] += 1;
    lut_count[id110] += 1;
    lut_count[id001] += 1;
    lut_count[id101] += 1;
    lut_count[id011] += 1;
    lut_count[id111] += 1;

    lut_count[id000 + shift] += 1;
    lut_count[id100 + shift] += 1;
    lut_count[id010 + shift] += 1;
    lut_count[id110 + shift] += 1;
    lut_count[id001 + shift] += 1;
    lut_count[id101 + shift] += 1;
    lut_count[id011 + shift] += 1;
    lut_count[id111 + shift] += 1;

    lut_count[id000 + shift * 2] += 1;
    lut_count[id100 + shift * 2] += 1;
    lut_count[id010] + shift * 2 += 1;
    lut_count[id110 + shift * 2] += 1;
    lut_count[id001 + shift * 2] += 1;
    lut_count[id101 + shift * 2] += 1;
    lut_count[id011 + shift * 2] += 1;
    lut_count[id111 + shift * 2] += 1;


}


void TriDistributeForwardCpu(float* lut, consta float* lut_count, const float* input_image, const float* output_image, const int dim, const int shift, const float binsize, const int width, const int height, const int channels)
{
    const int output_size = height * width;;

    int index = 0;
    for (index = 0; index < output_size; ++index)
    {
        float r = input_image[index];
	float g = input_image[index + width * height];
	float b = input_image[index + width * height * 2];

	int r_id = floor(r / binsize);
	int g_id = floor(g / binsize);
	int b_id = floor(b / binsize);

	float r_d = fmod(r,binsize) / binsize;
	float g_d = fmod(g,binsize) / binsize;
	float b_d = fmod(b,binsize) / binsize;

	int id000 = r_id + g_id * dim + b_id * dim * dim;
        int id100 = r_id + 1 + g_id * dim + b_id * dim * dim;
        int id010 = r_id + (g_id + 1) * dim + b_id * dim * dim;
        int id110 = r_id + 1 + (g_id + 1) * dim + b_id * dim * dim;
        int id001 = r_id + g_id * dim + (b_id + 1) * dim * dim;
        int id101 = r_id + 1 + g_id * dim + (b_id + 1) * dim * dim;
        int id011 = r_id + (g_id + 1) * dim + (b_id + 1) * dim * dim;
        int id111 = r_id + 1 + (g_id + 1) * dim + (b_id + 1) * dim * dim;

	float w000 = (1-r_d)*(1-g_d)*(1-b_d);
        float w100 = r_d*(1-g_d)*(1-b_d);
        float w010 = (1-r_d)*g_d*(1-b_d);
        float w110 = r_d*g_d*(1-b_d);
        float w001 = (1-r_d)*(1-g_d)*b_d;
        float w101 = r_d*(1-g_d)*b_d;
        float w011 = (1-r_d)*g_d*b_d;
        float w111 = r_d*g_d*b_d;


    lut[id000] += w000 * output_image[index] / lut_count[id000];
        lut[id100] += w100 * output_image[index] / lut_count[id000];
        lut[id010] += w010 * output_image[index] / lut_count[id010];
        lut[id110] += w110 * output_image[index] / lut_count[id110];
        lut[id001] += w001 * output_image[index] / lut_count[id001];
        lut[id101] += w101 * output_image[index] / lut_count[id101];
        lut[id011] += w011 * output_image[index] / lut_count[id011];
        lut[id111] += w111 * output_image[index] / lut_count[id111];

    lut[id000 + shift] += w000 * output_image[index + width * height] / lut_count[id000 + shift];
        lut[id100 + shift] += w100 * output_image[index + width * height] / lut_count[id000 + shift];
        lut[id010 + shift] += w010 * output_image[index + width * height] / lut_count[id010 + shift];
        lut[id110 + shift] += w110 * output_image[index + width * height] / lut_count[id110 + shift];
        lut[id001 + shift] += w001 * output_image[index + width * height] / lut_count[id001 + shift];
        lut[id101 + shift] += w101 * output_image[index + width * height] / lut_count[id101 + shift];
        lut[id011 + shift] += w011 * output_image[index + width * height] / lut_count[id011 + shift];
        lut[id111 + shift] += w111 * output_image[index + width * height] / lut_count[id111 + shift];

    lut[id000 + shift * 2] += w000 * output_image[index + width * height * 2] / lut_count[id000 + shift * 2];
        lut[id100 + shift * 2] += w100 * output_image[index + width * height * 2] / lut_count[id000 + shift * 2];
        lut[id010 + shift * 2] += w010 * output_image[index + width * height * 2] / lut_count[id010 + shift * 2];
        lut[id110 + shift * 2] += w110 * output_image[index + width * height * 2] / lut_count[id110 + shift * 2];
        lut[id001 + shift * 2] += w001 * output_image[index + width * height * 2] / lut_count[id001 + shift * 2];
        lut[id101 + shift * 2] += w101 * output_image[index + width * height * 2] / lut_count[id101 + shift * 2];
        lut[id011 + shift * 2] += w011 * output_image[index + width * height * 2] / lut_count[id011 + shift * 2];
        lut[id111 + shift * 2] += w111 * output_image[index + width * height * 2] / lut_count[id111 + shift] * 2;






    /*
	output[index] = w000 * lut[id000] + w100 * lut[id100] +
                        w010 * lut[id010] + w110 * lut[id110] +
                        w001 * lut[id001] + w101 * lut[id101] +
                        w011 * lut[id011] + w111 * lut[id111];

	output[index + width * height] = w000 * lut[id000 + shift] + w100 * lut[id100 + shift] +
                      		         w010 * lut[id010 + shift] + w110 * lut[id110 + shift] +
                       		         w001 * lut[id001 + shift] + w101 * lut[id101 + shift] +
                               	         w011 * lut[id011 + shift] + w111 * lut[id111 + shift];

	output[index + width * height * 2] = w000 * lut[id000 + shift * 2] + w100 * lut[id100 + shift * 2] +
                           		     w010 * lut[id010 + shift * 2] + w110 * lut[id110 + shift * 2] +
                           		     w001 * lut[id001 + shift * 2] + w101 * lut[id101 + shift * 2] +
                                	     w011 * lut[id011 + shift * 2] + w111 * lut[id111 + shift * 2];
    */
    }
}

void TriLinearBackwardCpu(const float* input_image, float* output_image_grad, const float* lut_grad, const float* lut_count, const int dim, const int shift, const float binsize, const int width, const int height, const int channels)
{
    const int output_size = height * width;

    int index = 0;
    for (index = 0; index < output_size; ++index)
    {
        float r = input_image[index];
	float g = input_image[index + width * height];
	float b = input_image[index + width * height * 2];

	int r_id = floor(r / binsize);
	int g_id = floor(g / binsize);
	int b_id = floor(b / binsize);

	float r_d = fmod(r,binsize) / binsize;
	float g_d = fmod(g,binsize) / binsize;
	float b_d = fmod(b,binsize) / binsize;

	int id000 = r_id + g_id * dim + b_id * dim * dim;
        int id100 = r_id + 1 + g_id * dim + b_id * dim * dim;
        int id010 = r_id + (g_id + 1) * dim + b_id * dim * dim;
        int id110 = r_id + 1 + (g_id + 1) * dim + b_id * dim * dim;
        int id001 = r_id + g_id * dim + (b_id + 1) * dim * dim;
        int id101 = r_id + 1 + g_id * dim + (b_id + 1) * dim * dim;
        int id011 = r_id + (g_id + 1) * dim + (b_id + 1) * dim * dim;
        int id111 = r_id + 1 + (g_id + 1) * dim + (b_id + 1) * dim * dim;

	float w000 = (1-r_d)*(1-g_d)*(1-b_d);
        float w100 = r_d*(1-g_d)*(1-b_d);
        float w010 = (1-r_d)*g_d*(1-b_d);
        float w110 = r_d*g_d*(1-b_d);
        float w001 = (1-r_d)*(1-g_d)*b_d;
        float w101 = r_d*(1-g_d)*b_d;
        float w011 = (1-r_d)*g_d*b_d;
        float w111 = r_d*g_d*b_d;

    output_image_grad[index] = w000 * lut_grad[id000] / lut_count[id000]  + w100 * lut_grad[id100] / lut_count[id100]+
                        w010 * lut_grad[id010] / lut_count[id010] + w110 * lut_grad[id110] / lut_count[id110] +
                        w001 * lut_grad[id001] / lut_count[id001] + w101 * lut_grad[id101] / lut_count[id101] +
                        w011 * lut_grad[id011] / lut_count[id011] + w111 * lut_grad[id111] / lut_count[id111];

    output_image_grad[index + width * height] = w000 * lut_grad[id000 + shift] / lut_count[id000 + shift]  + w100 * lut_grad[id100 + shift] / lut_count[id100 + shift]+
                        w010 * lut_grad[id010 + shift] / lut_count[id010 + shift] + w110 * lut_grad[id110 + shift] / lut_count[id110 + shift] +
                        w001 * lut_grad[id001 + shift] / lut_count[id001 + shift] + w101 * lut_grad[id101 + shift] / lut_count[id101 + shift] +
                        w011 * lut_grad[id011 + shift] / lut_count[id011 + shift] + w111 * lut_grad[id111 + shift] / lut_count[id111 + shift];

    output_image_grad[index + width * height * 2] = w000 * lut_grad[id000 + shift * 2] / lut_count[id000 + shift * 2]  + w100 * lut_grad[id100 + shift * 2] / lut_count[id100 + shift * 2]+
                        w010 * lut_grad[id010 + shift * 2] / lut_count[id010 + shift * 2] + w110 * lut_grad[id110 + shift * 2] / lut_count[id110 + shift * 2] +
                        w001 * lut_grad[id001 + shift * 2] / lut_count[id001 + shift * 2] + w101 * lut_grad[id101 + shift * 2] / lut_count[id101 + shift * 2] +
                        w011 * lut_grad[id011 + shift * 2] / lut_count[id011 + shift * 2] + w111 * lut_grad[id111 + shift * 2] / lut_count[id111 + shift * 2];


    /*
	lut_grad[id000] += w000 * image_grad[index];
        lut_grad[id100] += w100 * image_grad[index];
        lut_grad[id010] += w010 * image_grad[index];
        lut_grad[id110] += w110 * image_grad[index];
        lut_grad[id001] += w001 * image_grad[index];
        lut_grad[id101] += w101 * image_grad[index];
        lut_grad[id011] += w011 * image_grad[index];
        lut_grad[id111] += w111 * image_grad[index];

        lut_grad[id000 + shift] += w000 * image_grad[index + width * height];
        lut_grad[id100 + shift] += w100 * image_grad[index + width * height];
        lut_grad[id010 + shift] += w010 * image_grad[index + width * height];
        lut_grad[id110 + shift] += w110 * image_grad[index + width * height];
        lut_grad[id001 + shift] += w001 * image_grad[index + width * height];
        lut_grad[id101 + shift] += w101 * image_grad[index + width * height];
        lut_grad[id011 + shift] += w011 * image_grad[index + width * height];
        lut_grad[id111 + shift] += w111 * image_grad[index + width * height];

        lut_grad[id000 + shift* 2] += w000 * image_grad[index + width * height * 2];
        lut_grad[id100 + shift* 2] += w100 * image_grad[index + width * height * 2];
        lut_grad[id010 + shift* 2] += w010 * image_grad[index + width * height * 2];
        lut_grad[id110 + shift* 2] += w110 * image_grad[index + width * height * 2];
        lut_grad[id001 + shift* 2] += w001 * image_grad[index + width * height * 2];
        lut_grad[id101 + shift* 2] += w101 * image_grad[index + width * height * 2];
        lut_grad[id011 + shift* 2] += w011 * image_grad[index + width * height * 2];
        lut_grad[id111 + shift* 2] += w111 * image_grad[index + width * height * 2];
    */
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &trilinear_forward, "Trilinear forward");
  m.def("backward", &trilinear_backward, "Trilinear backward");
}
