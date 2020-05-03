#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

__global__ void n_avg(int *a, int *b, int i, int n) {
	for (int j = i; j < i + n; j++) {
		atomicAdd(&b[i], a[j]);
	}
	b[i] /= n;
}

int main() {
	int m = 10000;
	int n = 32;
	int i;
	int block = 256;
	int grid = 256;
	int *a;
	int *dev_a;
	int *b;
	int *dev_b;

	printf("N is %d\n", n);
	printf("DimBlock is %d\n", block);
	printf("DimGrid is %d\n", grid);

	cudaMalloc((void**)&dev_a, sizeof(int) * m);
	cudaMalloc((void**)&dev_b, sizeof(int) * (m - n + 1));
	cudaMallocHost((void**)&a, sizeof(int) * m);
	cudaMallocHost((void**)&b, sizeof(int) * (m - n + 1));

	for (i = 0; i < m; i++) {
		a[i] = rand();
	}

	for (i = 0; i < m - n + 1; i++) {
		b[i] = 0;
	}

	clock_t start_time = clock();
	cudaMemcpy(dev_a, a, sizeof(int) * m, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, b, sizeof(int) * (m - n + 1), cudaMemcpyHostToDevice);

	for (i = 0; i < m - n + 1; i++) {
		n_avg<<<grid, block>>>(dev_a, dev_b, i, n);
	}

	clock_t end_time = clock();
	printf("Time consuming is %f ms. \n", static_cast<double>(end_time - start_time)/CLOCKS_PER_SEC*1000);

	cudaFree(dev_a);
	cudaFreeHost(a);
	return 0;
}
