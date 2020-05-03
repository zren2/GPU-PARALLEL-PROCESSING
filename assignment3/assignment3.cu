#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

__device__ int *sm;

__global__ void reduce1(int *a, int *b) {
	int tid = threadIdx.x;
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	sm[tid] = a[i];
	for (int j = 1; j < blockDim.x; j *= 2) {
		if (tid % (2 * j) == 0) {
			sm[tid] = sm[tid] >= sm[tid + j] ? sm[tid] : sm[tid + j];
		}
		__syncthreads();
	}
	if (tid == 0) {
		b[blockIdx.x] = sm[0];
	}
}

__global__ void reduce2(int *a, int *b) {
	extern __shared__ int sdata[];
	int tid = threadIdx.x;
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	sdata[tid] = a[i];
	__syncthreads();
	for (int j = 1; j < blockDim.x; j *= 2) {
		int idx = 2 * j * tid;
		if (idx < blockDim.x) {
			sdata[tid] = sdata[tid] >= sdata[tid + j] ? sdata[tid] : sdata[tid + j];
		}
		__syncthreads();
	}
	if (tid == 0) {
		b[blockIdx.x] = sdata[0];
	}
}

__global__ void reduce3(int *a, int *b) {
	extern __shared__ int sdata[];
	int tid = threadIdx.x;
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	sdata[tid] = a[i];
	__syncthreads();
	for (int j = blockDim.x / 2; j > 0; j >>= 1) {
		if (tid < j) {
			sdata[tid] = sdata[tid] >= sdata[tid + j] ? sdata[tid] : sdata[tid + j];
		}
		__syncthreads();
	}
	if (tid == 0) {
		b[blockIdx.x] = sdata[0];
	}
}

int main() {
	int n = 1000;
	int i;
	int *a1;
	int *a2;
	int *a3;
	int *dev_a1;
	int *dev_a2;
	int *dev_a3;
	int *b1;
	int *b2;
	int *b3;
	int *dev_b1;
	int *dev_b2;
	int *dev_b3;

	cudaMalloc((void**)&dev_a1, sizeof(int) * n);
	cudaMalloc((void**)&dev_a2, sizeof(int) * n);
	cudaMalloc((void**)&dev_a3, sizeof(int) * n);
	cudaMalloc((void**)&dev_b1, sizeof(int) * n);
	cudaMalloc((void**)&dev_b2, sizeof(int) * n);
	cudaMalloc((void**)&dev_b3, sizeof(int) * n);

	cudaMallocHost((void**)&a1, sizeof(int) * n);
	cudaMallocHost((void**)&a2, sizeof(int) * n);
	cudaMallocHost((void**)&a3, sizeof(int) * n);
	cudaMallocHost((void**)&b1, sizeof(int) * n);
	cudaMallocHost((void**)&b2, sizeof(int) * n);
	cudaMallocHost((void**)&b3, sizeof(int) * n);

	for (i = 0; i < n; i++) {
		a1[i] = rand();
		a2[i] = rand();
		a3[i] = rand();
	}

	clock_t start_time1 = clock();
	cudaMalloc((void**)&sm, sizeof(int) * n);
	cudaMemcpy(dev_a1, a1, sizeof(int) * n, cudaMemcpyHostToDevice);
	reduce1<<<100, 1>>>(dev_a1, dev_b1);
	cudaMemcpy(b1, dev_b1, sizeof(int) * n, cudaMemcpyDeviceToHost);
	clock_t end_time1 = clock();
	printf("Time consuming using GLOBAL MEMORY is %f ms. \n", static_cast<double>(end_time1 - start_time1)/CLOCKS_PER_SEC*1000);

	clock_t start_time2 = clock();
	cudaMemcpy(dev_a2, a2, sizeof(int) * n, cudaMemcpyHostToDevice);
	reduce2<<<100, 1>>>(dev_a2, dev_b2);
	cudaMemcpy(b2, dev_b2, sizeof(int) * n, cudaMemcpyDeviceToHost);
	clock_t end_time2 = clock();
	printf("Time consuming using INTERLEAVING ADDRESSING SHARED MEMORY is %f ms. \n", static_cast<double>(end_time2 - start_time1)/CLOCKS_PER_SEC*1000);

	clock_t start_time3 = clock();
	cudaMemcpy(dev_a3, a3, sizeof(int) * n, cudaMemcpyHostToDevice);
	reduce3<<<100, 1>>>(dev_a3, dev_b3);
	cudaMemcpy(b3, dev_b3, sizeof(int) * n, cudaMemcpyDeviceToHost);
	clock_t end_time3 = clock();
	printf("Time consuming using SEQUENTIAL ADDRESSING SHARED MEMORY is %f ms. \n", static_cast<double>(end_time3 - start_time3)/CLOCKS_PER_SEC*1000);

	cudaFree(dev_a1);
	cudaFree(dev_a2);
	cudaFree(dev_a3);
	cudaFree(dev_b1);
	cudaFree(dev_b2);
	cudaFree(dev_b3);

	cudaFreeHost(a1);
	cudaFreeHost(a2);
	cudaFreeHost(a3);
	cudaFreeHost(b1);
	cudaFreeHost(b2);
	cudaFreeHost(b3);

	return 0;
}#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

__device__ int *sm;

__global__ void reduce1(int *a, int *b) {
	int tid = threadIdx.x;
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	sm[tid] = a[i];
	for (int j = 1; j < blockDim.x; j *= 2) {
		if (tid % (2 * j) == 0) {
			sm[tid] = sm[tid] >= sm[tid + j] ? sm[tid] : sm[tid + j];
		}
		__syncthreads();
	}
	if (tid == 0) {
		b[blockIdx.x] = sm[0];
	}
}

__global__ void reduce2(int *a, int *b) {
	extern __shared__ int sdata[];
	int tid = threadIdx.x;
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	sdata[tid] = a[i];
	__syncthreads();
	for (int j = 1; j < blockDim.x; j *= 2) {
		int idx = 2 * j * tid;
		if (idx < blockDim.x) {
			sdata[tid] = sdata[tid] >= sdata[tid + j] ? sdata[tid] : sdata[tid + j];
		}
		__syncthreads();
	}
	if (tid == 0) {
		b[blockIdx.x] = sdata[0];
	}
}

__global__ void reduce3(int *a, int *b) {
	extern __shared__ int sdata[];
	int tid = threadIdx.x;
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	sdata[tid] = a[i];
	__syncthreads();
	for (int j = blockDim.x / 2; j > 0; j >>= 1) {
		if (tid < j) {
			sdata[tid] = sdata[tid] >= sdata[tid + j] ? sdata[tid] : sdata[tid + j];
		}
		__syncthreads();
	}
	if (tid == 0) {
		b[blockIdx.x] = sdata[0];
	}
}

int main() {
	int n = 1000;
	int i;
	int *a1;
	int *a2;
	int *a3;
	int *dev_a1;
	int *dev_a2;
	int *dev_a3;
	int *b1;
	int *b2;
	int *b3;
	int *dev_b1;
	int *dev_b2;
	int *dev_b3;

	cudaMalloc((void**)&dev_a1, sizeof(int) * n);
	cudaMalloc((void**)&dev_a2, sizeof(int) * n);
	cudaMalloc((void**)&dev_a3, sizeof(int) * n);
	cudaMalloc((void**)&dev_b1, sizeof(int) * n);
	cudaMalloc((void**)&dev_b2, sizeof(int) * n);
	cudaMalloc((void**)&dev_b3, sizeof(int) * n);

	cudaMallocHost((void**)&a1, sizeof(int) * n);
	cudaMallocHost((void**)&a2, sizeof(int) * n);
	cudaMallocHost((void**)&a3, sizeof(int) * n);
	cudaMallocHost((void**)&b1, sizeof(int) * n);
	cudaMallocHost((void**)&b2, sizeof(int) * n);
	cudaMallocHost((void**)&b3, sizeof(int) * n);

	for (i = 0; i < n; i++) {
		a1[i] = rand();
		a2[i] = rand();
		a3[i] = rand();
	}

	clock_t start_time1 = clock();
	cudaMalloc((void**)&sm, sizeof(int) * n);
	cudaMemcpy(dev_a1, a1, sizeof(int) * n, cudaMemcpyHostToDevice);
	reduce1<<<100, 1>>>(dev_a1, dev_b1);
	cudaMemcpy(b1, dev_b1, sizeof(int) * n, cudaMemcpyDeviceToHost);
	clock_t end_time1 = clock();
	printf("Time consuming using GLOBAL MEMORY is %f ms. \n", static_cast<double>(end_time1 - start_time1)/CLOCKS_PER_SEC*1000);

	clock_t start_time2 = clock();
	cudaMemcpy(dev_a2, a2, sizeof(int) * n, cudaMemcpyHostToDevice);
	reduce2<<<100, 1>>>(dev_a2, dev_b2);
	cudaMemcpy(b2, dev_b2, sizeof(int) * n, cudaMemcpyDeviceToHost);
	clock_t end_time2 = clock();
	printf("Time consuming using INTERLEAVING ADDRESSING SHARED MEMORY is %f ms. \n", static_cast<double>(end_time2 - start_time1)/CLOCKS_PER_SEC*1000);

	clock_t start_time3 = clock();
	cudaMemcpy(dev_a3, a3, sizeof(int) * n, cudaMemcpyHostToDevice);
	reduce3<<<100, 1>>>(dev_a3, dev_b3);
	cudaMemcpy(b3, dev_b3, sizeof(int) * n, cudaMemcpyDeviceToHost);
	clock_t end_time3 = clock();
	printf("Time consuming using SEQUENTIAL ADDRESSING SHARED MEMORY is %f ms. \n", static_cast<double>(end_time3 - start_time3)/CLOCKS_PER_SEC*1000);

	cudaFree(dev_a1);
	cudaFree(dev_a2);
	cudaFree(dev_a3);
	cudaFree(dev_b1);
	cudaFree(dev_b2);
	cudaFree(dev_b3);

	cudaFreeHost(a1);
	cudaFreeHost(a2);
	cudaFreeHost(a3);
	cudaFreeHost(b1);
	cudaFreeHost(b2);
	cudaFreeHost(b3);

	return 0;
}
