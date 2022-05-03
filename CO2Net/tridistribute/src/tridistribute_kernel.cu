#include <math.h>
#include <float.h>
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include "tridistribute_kernel.h"

#define CUDA_1D_KERNEL_LOOP(i, n)                            \
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
            i += blockDim.x * gridDim.x)



__global__ void TriDistributeForward(const int nthreads, const float* mask, float* lut, float* lut_count, const float* input_image, const float* output_image, const int dim, const int shift, const float binsize, const int width, const int height, const int batch){
        CUDA_1D_KERNEL_LOOP(index, nthreads) {
    if (mask[index] > 0.5) {




	int current_batch_num = index / (height * width);
	int remain = index - current_batch_num * height * width;
	int new_index = current_batch_num * height * width * 3 + remain;

	float r = input_image[new_index];
	float g = input_image[new_index + width * height];
	float b = input_image[new_index + width * height * 2];



	int r_id = floor(r / binsize);
	int g_id = floor(g / binsize);
	int b_id = floor(b / binsize);

	//printf("current index: %d, batch: %d, r g b : %f, %f, %f\n", index, batch, r, g, b);

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




        atomicAdd(lut_count + id000 + current_batch_num * shift * 3, w000);
        atomicAdd(lut_count + id100 + current_batch_num * shift * 3, w100);
        atomicAdd(lut_count + id010 + current_batch_num * shift * 3, w010);
        atomicAdd(lut_count + id110 + current_batch_num * shift * 3, w110);
        atomicAdd(lut_count + id001 + current_batch_num * shift * 3, w001);
        atomicAdd(lut_count + id101 + current_batch_num * shift * 3, w101);
        atomicAdd(lut_count + id011 + current_batch_num * shift * 3, w011);
        atomicAdd(lut_count + id111 + current_batch_num * shift * 3, w111);

        atomicAdd(lut_count + id000 + shift + current_batch_num * shift * 3, w000);
        atomicAdd(lut_count + id100 + shift + current_batch_num * shift * 3, w100);
        atomicAdd(lut_count + id010 + shift + current_batch_num * shift * 3, w010);
        atomicAdd(lut_count + id110 + shift + current_batch_num * shift * 3, w110);
        atomicAdd(lut_count + id001 + shift + current_batch_num * shift * 3, w001);
        atomicAdd(lut_count + id101 + shift + current_batch_num * shift * 3, w101);
        atomicAdd(lut_count + id011 + shift + current_batch_num * shift * 3, w011);
        atomicAdd(lut_count + id111 + shift + current_batch_num * shift * 3, w111);

        atomicAdd(lut_count + id000 + shift * 2 + current_batch_num * shift * 3, w000);
        atomicAdd(lut_count + id100 + shift * 2 + current_batch_num * shift * 3, w100);
        atomicAdd(lut_count + id010 + shift * 2 + current_batch_num * shift * 3, w010);
        atomicAdd(lut_count + id110 + shift * 2 + current_batch_num * shift * 3, w110);
        atomicAdd(lut_count + id001 + shift * 2 + current_batch_num * shift * 3, w001);
        atomicAdd(lut_count + id101 + shift * 2 + current_batch_num * shift * 3, w101);
        atomicAdd(lut_count + id011 + shift * 2 + current_batch_num * shift * 3, w011);
        atomicAdd(lut_count + id111 + shift * 2 + current_batch_num * shift * 3, w111);


        atomicAdd(lut + id000 + current_batch_num * shift * 3, output_image[new_index] * w000 );
        atomicAdd(lut + id100 + current_batch_num * shift * 3, output_image[new_index] * w100);
        atomicAdd(lut + id010 + current_batch_num * shift * 3, output_image[new_index] * w010 );
        atomicAdd(lut + id110 + current_batch_num * shift * 3, output_image[new_index] * w110 );
        atomicAdd(lut + id001 + current_batch_num * shift * 3, output_image[new_index] * w001 );
        atomicAdd(lut + id101 + current_batch_num * shift * 3, output_image[new_index] * w101 );
        atomicAdd(lut + id011 + current_batch_num * shift * 3, output_image[new_index] * w011 );
        atomicAdd(lut + id111 + current_batch_num * shift * 3, output_image[new_index] * w111 );

        atomicAdd(lut + id000 + shift + current_batch_num * shift * 3, output_image[new_index + width * height] * w000);
        atomicAdd(lut + id100 + shift + current_batch_num * shift * 3, output_image[new_index + width * height] * w100);
        atomicAdd(lut + id010 + shift + current_batch_num * shift * 3, output_image[new_index + width * height] * w010);
        atomicAdd(lut + id110 + shift + current_batch_num * shift * 3, output_image[new_index + width * height] * w110);
        atomicAdd(lut + id001 + shift + current_batch_num * shift * 3, output_image[new_index + width * height] * w001);
        atomicAdd(lut + id101 + shift + current_batch_num * shift * 3, output_image[new_index + width * height] * w101);
        atomicAdd(lut + id011 + shift + current_batch_num * shift * 3, output_image[new_index + width * height] * w011);
        atomicAdd(lut + id111 + shift + current_batch_num * shift * 3, output_image[new_index + width * height] * w111);

        atomicAdd(lut + id000 + shift * 2 + current_batch_num * shift * 3, output_image[new_index + width * height * 2] * w000);
        atomicAdd(lut + id100 + shift * 2 + current_batch_num * shift * 3, output_image[new_index + width * height * 2] * w100);
        atomicAdd(lut + id010 + shift * 2 + current_batch_num * shift * 3, output_image[new_index + width * height * 2] * w010);
        atomicAdd(lut + id110 + shift * 2 + current_batch_num * shift * 3, output_image[new_index + width * height * 2] * w110);
        atomicAdd(lut + id001 + shift * 2 + current_batch_num * shift * 3, output_image[new_index + width * height * 2] * w001);
        atomicAdd(lut + id101 + shift * 2 + current_batch_num * shift * 3, output_image[new_index + width * height * 2] * w101);
        atomicAdd(lut + id011 + shift * 2 + current_batch_num * shift * 3, output_image[new_index + width * height * 2] * w011);
        atomicAdd(lut + id111 + shift * 2 + current_batch_num * shift * 3, output_image[new_index + width * height * 2] * w111);





        }}
}


int TriDistributeForwardLaucher(const float* mask, float* lut, float* lut_count, const float* input_image, const float* output_image, const int dim, const int shift, const float binsize, const int width, const int height, const int batch, cudaStream_t stream) {
    const int kThreadsPerBlock = 1024;
    const int output_size = height * width * batch;
    cudaError_t err;


    TriDistributeForward<<<(output_size + kThreadsPerBlock - 1) / kThreadsPerBlock, kThreadsPerBlock, 0, stream>>>(output_size, mask, lut, lut_count, input_image, output_image, dim, shift, binsize, width, height, batch);

    err = cudaGetLastError();
    if(cudaSuccess != err) {
        fprintf( stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString( err ) );
        exit( -1 );
    }

    return 1;
}

__global__ void TriDistributeBackward(const int nthreads, const float* mask, const float* input_image, float* output_image_grad, const float* lut_grad, const float* lut_count, const int dim, const int shift, const float binsize, const int width, const int height, const int batch){
        CUDA_1D_KERNEL_LOOP(index, nthreads) {
        if (mask[index] > 0.5) {

        int current_batch_num = index / (height * width);
	    int remain = index - current_batch_num * height * width;
	    int new_index = current_batch_num * height * width * 3 + remain;

        float r = input_image[new_index];
        float g = input_image[new_index + width * height];
        float b = input_image[new_index + width * height * 2];

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


        output_image_grad[new_index] = w000 * lut_grad[id000  + current_batch_num * shift * 3] / lut_count[id000  + current_batch_num * shift * 3]   + w100 * lut_grad[id100  + current_batch_num * shift * 3] / lut_count[id100  + current_batch_num * shift * 3]+
                        w010 * lut_grad[id010 + current_batch_num * shift * 3] / lut_count[id010 + current_batch_num * shift * 3] + w110 * lut_grad[id110 + current_batch_num * shift * 3] / lut_count[id110 + current_batch_num * shift * 3] +
                        w001 * lut_grad[id001 + current_batch_num * shift * 3] / lut_count[id001 + current_batch_num * shift * 3] + w101 * lut_grad[id101 + current_batch_num * shift * 3] / lut_count[id101 + current_batch_num * shift * 3] +
                        w011 * lut_grad[id011 + current_batch_num * shift * 3] / lut_count[id011 + current_batch_num * shift * 3] + w111 * lut_grad[id111 + current_batch_num * shift * 3] / lut_count[id111 + current_batch_num * shift * 3];

        output_image_grad[new_index + width * height] = w000 * lut_grad[id000 + shift + current_batch_num * shift * 3] / lut_count[id000 + shift + current_batch_num * shift * 3] + w100 * lut_grad[id100 + shift + current_batch_num * shift * 3] / lut_count[id100 + shift + current_batch_num * shift * 3] +
                            w010 * lut_grad[id010 + shift + current_batch_num * shift * 3] / lut_count[id010 + shift + current_batch_num * shift * 3] + w110 * lut_grad[id110 + shift + current_batch_num * shift * 3] / lut_count[id110 + shift + current_batch_num * shift * 3] +
                            w001 * lut_grad[id001 + shift + current_batch_num * shift * 3] / lut_count[id001 + shift + current_batch_num * shift * 3] + w101 * lut_grad[id101 + shift + current_batch_num * shift * 3] / lut_count[id101 + shift + current_batch_num * shift * 3] +
                            w011 * lut_grad[id011 + shift + current_batch_num * shift * 3] / lut_count[id011 + shift + current_batch_num * shift * 3] + w111 * lut_grad[id111 + shift + current_batch_num * shift * 3] / lut_count[id111 + shift + current_batch_num * shift * 3];

        output_image_grad[new_index + width * height * 2] = w000 * lut_grad[id000 + shift * 2 + current_batch_num * shift * 3] / lut_count[id000 + shift * 2 + current_batch_num * shift * 3]  + w100 * lut_grad[id100 + shift * 2 + current_batch_num * shift * 3] / lut_count[id100 + shift * 2 + current_batch_num * shift * 3]+
                            w010 * lut_grad[id010 + shift * 2 + current_batch_num * shift * 3] / lut_count[id010 + shift * 2 + current_batch_num * shift * 3] + w110 * lut_grad[id110 + shift * 2 + current_batch_num * shift * 3] / lut_count[id110 + shift * 2 + current_batch_num * shift * 3] +
                            w001 * lut_grad[id001 + shift * 2 + current_batch_num * shift * 3] / lut_count[id001 + shift * 2 + current_batch_num * shift * 3] + w101 * lut_grad[id101 + shift * 2 + current_batch_num * shift * 3] / lut_count[id101 + shift * 2 + current_batch_num * shift * 3] +
                            w011 * lut_grad[id011 + shift * 2 + current_batch_num * shift * 3] / lut_count[id011 + shift * 2 + current_batch_num * shift * 3] + w111 * lut_grad[id111 + shift * 2 + current_batch_num * shift * 3] / lut_count[id111 + shift * 2 + current_batch_num * shift * 3];


        }
        }
}
int TriDistributeBackwardLaucher(const float* mask, const float* input_image, float* output_image_grad, const float* lut_grad, const float* lut_count, const int dim, const int shift, const float binsize, const int width, const int height, const int batch, cudaStream_t stream) {
    const int kThreadsPerBlock = 1024;
    const int output_size = height * width * batch;
    cudaError_t err;

    TriDistributeBackward<<<(output_size + kThreadsPerBlock - 1) / kThreadsPerBlock, kThreadsPerBlock, 0, stream>>>(output_size, mask, input_image, output_image_grad, lut_grad, lut_count, dim, shift, binsize, width, height, batch);

    err = cudaGetLastError();
    if(cudaSuccess != err) {
        fprintf( stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString( err ) );
        exit( -1 );
    }

    return 1;
}

__global__ void TriDistributeCount(const int nthreads, float* lut_count, const float* input_image, const float* output_image, const int dim, const int shift, const float binsize, const int width, const int height, const int batch) {
    CUDA_1D_KERNEL_LOOP(index, nthreads) {
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



    atomicAdd(lut_count + id000, w000);
        atomicAdd(lut_count + id100, w100);
        atomicAdd(lut_count + id010, w010);
        atomicAdd(lut_count + id110, w110);
        atomicAdd(lut_count + id001, w001);
        atomicAdd(lut_count + id101, w101);
        atomicAdd(lut_count + id011, w011);
        atomicAdd(lut_count + id111, w111);

        atomicAdd(lut_count + id000 + shift, w000);
        atomicAdd(lut_count + id100 + shift, w100);
        atomicAdd(lut_count + id010 + shift, w010);
        atomicAdd(lut_count + id110 + shift, w110);
        atomicAdd(lut_count + id001 + shift, w001);
        atomicAdd(lut_count + id101 + shift, w101);
        atomicAdd(lut_count + id011 + shift, w011);
        atomicAdd(lut_count + id111 + shift, w111);

        atomicAdd(lut_count + id000 + shift * 2, w000);
        atomicAdd(lut_count + id100 + shift * 2, w100);
        atomicAdd(lut_count + id010 + shift * 2, w010);
        atomicAdd(lut_count + id110 + shift * 2, w110);
        atomicAdd(lut_count + id001 + shift * 2, w001);
        atomicAdd(lut_count + id101 + shift * 2, w101);
        atomicAdd(lut_count + id011 + shift * 2, w011);
        atomicAdd(lut_count + id111 + shift * 2, w111);



    }

}

int TriDistributeCountLaucher(float* lut_count, const float* input_image, const float* output_image, const int dim, const int shift, const float binsize, const int width, const int height, const int batch, cudaStream_t stream){
    const int kThreadsPerBlock = 1024;
    const int output_size = height * width * batch;
    cudaError_t err;

    TriDistributeCount<<<(output_size + kThreadsPerBlock - 1) / kThreadsPerBlock, kThreadsPerBlock, 0, stream>>>(output_size, lut_count, input_image, output_image, dim, shift, binsize, width, height, batch);

    err = cudaGetLastError();
    if(cudaSuccess != err) {
        fprintf( stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString( err ) );
        exit( -1 );
    }

    return 1;
}

__global__ void TriDistributeDivide(const int nthreads, float* lut, const float* lut_count, const int dim, const int shift, const int batch){
    CUDA_1D_KERNEL_LOOP(index, nthreads){
        int current_batch_num = index / (shift);
	int remain = index - current_batch_num * shift;
	int new_index = current_batch_num * shift * 3 + remain;

        if (lut_count[new_index] > 0 ) {lut[new_index] = lut[new_index] / lut_count[new_index];}
        if (lut_count[new_index + shift] > 0 ) {lut[new_index + shift] = lut[new_index + shift ] / lut_count[new_index + shift];}
        if (lut_count[new_index + shift*2] > 0 ) {lut[new_index + shift*2] = lut[new_index + shift*2] / lut_count[new_index + shift*2];}
    }

}

int TriDistributeDivideLaucher(float* lut, const float* lut_count, const int dim, const int shift, const int batch, cudaStream_t stream)
{
    const int kThreadsPerBlock = 1024;
    const int output_size = shift * batch;
    cudaError_t err;

    TriDistributeDivide<<<(output_size + kThreadsPerBlock - 1) / kThreadsPerBlock, kThreadsPerBlock, 0, stream>>>(output_size, lut, lut_count, dim, shift, batch);

    err = cudaGetLastError();
    if(cudaSuccess != err) {
        fprintf( stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString( err ) );
        exit( -1 );
    }

    return 1;


}