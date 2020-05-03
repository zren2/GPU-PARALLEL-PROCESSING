
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

int main() {
	int n;
	int i, j, k;
	printf("Please enter the size of matrix: \n");
	scanf("%d", &n);

	int *a, *b, *c;
	cudaMallocHost((void**)&a, sizeof(int) * n * n);
	cudaMallocHost((void**)&b, sizeof(int) * n * n);
	cudaMallocHost((void**)&c, sizeof(int) * n * n);

	for (i = 0; i < n; i++){
		for (j = 0; j < n; j++){
			a[i * n + j] = round(rand() % 2);
			b[i * n + j] = round(rand() % 2);
		}
	}

	printf("Start...\n");
	clock_t start_time = clock();
	for (i = 0; i < n; i++){
		for (j = 0; j < n; j++){
			int tmp = 0;
			for (k = 0; k < n; k++)
				tmp += a[i * n + k] * b[k * n + j];
			c[i * n + j] = tmp;
		}
	}
	clock_t end_time = clock();

	printf("Time of calculating %dx%d matrix using CPU is %f ms.\n", n, n, static_cast<double>(end_time - start_time)/CLOCKS_PER_SEC*1000);
	cudaFreeHost(a);
	cudaFreeHost(b);
	cudaFreeHost(c);
	return 0;
}

    
