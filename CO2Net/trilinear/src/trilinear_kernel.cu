#include <math.h>
#include <float.h>
#include "trilinear_kernel.h"

#define CUDA_1D_KERNEL_LOOP(i, n)                            \
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
            i += blockDim.x * gridDim.x)


__global__ void TriLinearForward(const int nthreads,  const float* lut_count, const float* lut, const float* image, float* output, const int dim, const int shift, const float binsize, const int width, const int height, const int batch) {
        CUDA_1D_KERNEL_LOOP(index, nthreads) {

        int current_batch_num = index / (height * width);
	int remain = index - current_batch_num * height * width;
	int new_index = current_batch_num * height * width * 3 + remain;

        float r = image[new_index];
	float g = image[new_index + width * height];
	float b = image[new_index + width * height * 2];

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

        if (lut_count[id000 + current_batch_num * shift * 3] == 0) {w000 = 0;}
        if (lut_count[id001 + current_batch_num * shift * 3] == 0) {w001 = 0;}
        if (lut_count[id010 + current_batch_num * shift * 3] == 0) {w010 = 0;}
        if (lut_count[id011 + current_batch_num * shift * 3] == 0) {w011 = 0;}
        if (lut_count[id100 + current_batch_num * shift * 3] == 0) {w100 = 0;}
        if (lut_count[id101 + current_batch_num * shift * 3] == 0) {w101 = 0;}
        if (lut_count[id110 + current_batch_num * shift * 3] == 0) {w110 = 0;}
        if (lut_count[id111 + current_batch_num * shift * 3] == 0) {w111 = 0;}
        float w_sum = w000 + w001 + w010 + w011 + w100 + w101 + w110 + w111;
        if (w_sum > 0){
        w000 = w000 / w_sum;
        w001 = w001 / w_sum;
        w010 = w010 / w_sum;
        w011 = w011 / w_sum;
        w100 = w100 / w_sum;
        w101 = w101 / w_sum;
        w110 = w110 / w_sum;
        w111 = w111 / w_sum;
        }

        output[new_index] = w000 * lut[id000 + current_batch_num * shift * 3] + w100 * lut[id100 + current_batch_num * shift * 3] +
                        w010 * lut[id010 + current_batch_num * shift * 3] + w110 * lut[id110 + current_batch_num * shift * 3] +
                        w001 * lut[id001 + current_batch_num * shift * 3] + w101 * lut[id101 + current_batch_num * shift * 3] +
                        w011 * lut[id011 + current_batch_num * shift * 3] + w111 * lut[id111 + current_batch_num * shift * 3];

        output[new_index + width * height] = w000 * lut[id000 + shift + current_batch_num * shift * 3] + w100 * lut[id100 + shift + current_batch_num * shift * 3] +
                                                 w010 * lut[id010 + shift + current_batch_num * shift * 3] + w110 * lut[id110 + shift + current_batch_num * shift * 3] +
                                                 w001 * lut[id001 + shift + current_batch_num * shift * 3] + w101 * lut[id101 + shift + current_batch_num * shift * 3] +
                                                 w011 * lut[id011 + shift + current_batch_num * shift * 3] + w111 * lut[id111 + shift + current_batch_num * shift * 3];

        output[new_index + width * height * 2] = w000 * lut[id000 + shift * 2 + current_batch_num * shift * 3] + w100 * lut[id100 + shift * 2 + current_batch_num * shift * 3] +
                                                     w010 * lut[id010 + shift * 2 + current_batch_num * shift * 3] + w110 * lut[id110 + shift * 2 + current_batch_num * shift * 3] +
                                                     w001 * lut[id001 + shift * 2 + current_batch_num * shift * 3] + w101 * lut[id101 + shift * 2 + current_batch_num * shift * 3] +
                                                     w011 * lut[id011 + shift * 2 + current_batch_num * shift * 3] + w111 * lut[id111 + shift * 2 + current_batch_num * shift * 3];




    }
}


int TriLinearForwardLaucher(const float* lut_count, const float* lut, const float* image, float* output, const int lut_dim, const int shift, const float binsize, const int width, const int height, const int batch, cudaStream_t stream) {
    const int kThreadsPerBlock = 1024;
    const int output_size = height * width * batch;
    cudaError_t err;


    TriLinearForward<<<(output_size + kThreadsPerBlock - 1) / kThreadsPerBlock, kThreadsPerBlock, 0, stream>>>(output_size, lut_count, lut, image, output, lut_dim, shift, binsize, width, height, batch);

    err = cudaGetLastError();
    if(cudaSuccess != err) {
        fprintf( stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString( err ) );
        exit( -1 );
    }

    return 1;
}


__global__ void TriLinearBackward(const int nthreads, const float* image, const float* image_grad, float* lut_grad, const int dim, const int shift, const float binsize, const int width, const int height, const int batch) {
        CUDA_1D_KERNEL_LOOP(index, nthreads) {


       int current_batch_num = index / (height * width);
	int remain = index - current_batch_num * height * width;
	int new_index = current_batch_num * height * width * 3 + remain;



    float r = image[new_index];
    float g = image[new_index + width * height ];
    float b = image[new_index + width * height * 2];

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

    atomicAdd(lut_grad + id000 + current_batch_num * shift * 3, image_grad[new_index] * w000);
    atomicAdd(lut_grad + id100 + current_batch_num * shift * 3, image_grad[new_index] * w100);
    atomicAdd(lut_grad + id010 + current_batch_num * shift * 3, image_grad[new_index] * w010);
    atomicAdd(lut_grad + id110 + current_batch_num * shift * 3, image_grad[new_index] * w110);
    atomicAdd(lut_grad + id001 + current_batch_num * shift * 3, image_grad[new_index] * w001);
    atomicAdd(lut_grad + id101 + current_batch_num * shift * 3, image_grad[new_index] * w101);
    atomicAdd(lut_grad + id011 + current_batch_num * shift * 3, image_grad[new_index] * w011);
    atomicAdd(lut_grad + id111 + current_batch_num * shift * 3, image_grad[new_index] * w111);

    atomicAdd(lut_grad + id000 + shift + current_batch_num * shift * 3, image_grad[new_index + width * height] * w000);
    atomicAdd(lut_grad + id100 + shift + current_batch_num * shift * 3, image_grad[new_index + width * height] * w100);
    atomicAdd(lut_grad + id010 + shift + current_batch_num * shift * 3, image_grad[new_index + width * height] * w010);
    atomicAdd(lut_grad + id110 + shift + current_batch_num * shift * 3, image_grad[new_index + width * height] * w110);
    atomicAdd(lut_grad + id001 + shift + current_batch_num * shift * 3, image_grad[new_index + width * height] * w001);
    atomicAdd(lut_grad + id101 + shift + current_batch_num * shift * 3, image_grad[new_index + width * height] * w101);
    atomicAdd(lut_grad + id011 + shift + current_batch_num * shift * 3, image_grad[new_index + width * height] * w011);
    atomicAdd(lut_grad + id111 + shift + current_batch_num * shift * 3, image_grad[new_index + width * height] * w111);

    atomicAdd(lut_grad + id000 + shift * 2 + current_batch_num * shift * 3, image_grad[new_index + width * height * 2] * w000);
    atomicAdd(lut_grad + id100 + shift * 2 + current_batch_num * shift * 3, image_grad[new_index + width * height * 2] * w100);
    atomicAdd(lut_grad + id010 + shift * 2 + current_batch_num * shift * 3, image_grad[new_index + width * height * 2] * w010);
    atomicAdd(lut_grad + id110 + shift * 2 + current_batch_num * shift * 3, image_grad[new_index + width * height * 2] * w110);
    atomicAdd(lut_grad + id001 + shift * 2 + current_batch_num * shift * 3, image_grad[new_index + width * height * 2] * w001);
    atomicAdd(lut_grad + id101 + shift * 2 + current_batch_num * shift * 3, image_grad[new_index + width * height * 2] * w101);
    atomicAdd(lut_grad + id011 + shift * 2 + current_batch_num * shift * 3, image_grad[new_index + width * height * 2] * w011);
    atomicAdd(lut_grad + id111 + shift * 2 + current_batch_num * shift * 3, image_grad[new_index + width * height * 2] * w111);
}
    }

int TriLinearBackwardLaucher(const float* image, const float* image_grad, float* lut_grad, const int lut_dim, const int shift, const float binsize, const int width, const int height, const int batch, cudaStream_t stream) {
    const int kThreadsPerBlock = 1024;
    const int output_size = height * width * batch;
    cudaError_t err;

    TriLinearBackward<<<(output_size + kThreadsPerBlock - 1) / kThreadsPerBlock, kThreadsPerBlock, 0, stream>>>(output_size, image, image_grad, lut_grad, lut_dim, shift, binsize, width, height, batch);

    err = cudaGetLastError();
    if(cudaSuccess != err) {
        fprintf( stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString( err ) );
        exit( -1 );
    }

    return 1;
}


__global__ void TriLinearMap(const int nthreads, const float* lut_count,  const float* image, float* output_map, const int dim, const int shift, const float binsize, const int width, const int height, const int batch) {
        CUDA_1D_KERNEL_LOOP(index, nthreads) {

        int current_batch_num = index / (height * width);
	int remain = index - current_batch_num * height * width;
	int new_index = current_batch_num * height * width * 3 + remain;

        float r = image[new_index];
	float g = image[new_index + width * height];
	float b = image[new_index + width * height * 2];

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

        if (lut_count[id000 + current_batch_num * shift * 3] == 0 && lut_count[id100 + current_batch_num * shift * 3] == 0.0
            && lut_count[id010 + current_batch_num * shift * 3]==0 && lut_count[id110 + current_batch_num * shift * 3]==0 &&
               lut_count[id001 + current_batch_num * shift * 3]==0 && lut_count[id101 + current_batch_num * shift * 3]==0 &&
               lut_count[id011 + current_batch_num * shift * 3]==0 && lut_count[id111 + current_batch_num * shift * 3]==0 )
        {
            output_map[new_index] = 1;
            output_map[new_index + width * height] = 1;
            output_map[new_index + width * height * 2] = 1;
        }
        else
        {
            output_map[new_index] = 0;
            output_map[new_index + width * height] = 0;
            output_map[new_index + width * height * 2] = 0;
        }




    }
}


int TriLinearMapLaucher(const float* lut_count, const float* image, float* output_map, const int lut_dim, const int shift, const float binsize, const int width, const int height, const int batch, cudaStream_t stream) {
    const int kThreadsPerBlock = 1024;
    const int output_size = height * width * batch;
    cudaError_t err;


    TriLinearMap<<<(output_size + kThreadsPerBlock - 1) / kThreadsPerBlock, kThreadsPerBlock, 0, stream>>>(output_size, lut_count, image, output_map, lut_dim, shift, binsize, width, height, batch);

    err = cudaGetLastError();
    if(cudaSuccess != err) {
        fprintf( stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString( err ) );
        exit( -1 );
    }

    return 1;
}