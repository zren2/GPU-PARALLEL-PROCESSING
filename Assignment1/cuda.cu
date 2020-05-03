#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

__global__ void mult(int *a, int *b, int *c, int n) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int sum = 0;
	if (col < n && row < n) {
		for (int i = 0; i < n; i++)
			sum += a[row * n + i] * b[i * n + col];
		c[row * n + col] = sum;
	}
}

int main() {
	int n;
	int i, j, k;
	int *a, *b, *c;
	int *dev_a, *dev_b, *dev_c;

	printf("Please enter the size of matrix: \n");
	scanf("%d", &n);

	cudaMalloc((void**)&dev_a, sizeof(int) * n * n);
	cudaMalloc((void**)&dev_b, sizeof(int) * n * n);
	cudaMalloc((void**)&dev_c, sizeof(int) * n * n);

	cudaMallocHost((void**)&a, sizeof(int) * n * n);
	cudaMallocHost((void**)&b, sizeof(int) * n * n);
	cudaMallocHost((void**)&c, sizeof(int) * n * n);

	for (i = 0; i < n; i++){
		for (j = 0; j < n; j++){
			a[i * n + j] = round(rand() % 2);
			b[i * n + j] = round(rand() % 2);
		}
	}

	printf("Start calculating...\n");
	clock_t start_time = clock();
	cudaMemcpy(dev_a, a, sizeof(int) * n * n, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, b, sizeof(int) * n * n, cudaMemcpyHostToDevice);

	mult<<<n, n>>>(dev_a, dev_b, dev_c, n);
	clock_t end_time = clock();
	printf("Time consuming of calculating %dx%d matrix using GPU is %f ms.\n", n, n, static_cast<double>(end_time - start_time)/CLOCKS_PER_SEC*1000);
	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);
	cudaFreeHost(a);
	cudaFreeHost(b);
	cudaFreeHost(c);
	return 0;
}
