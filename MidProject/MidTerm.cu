#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#define MAX_TREE_HT 100

struct MinHeapNode {
	char data;
	unsigned freq;
	struct MinHeapNode *left, *right;
};

struct MinHeap {
	unsigned size;
	unsigned capacity;
	struct MinHeapNode** array;
};

//__global__ struct MinHeapNode* newNode(char data, unsigned freq, struct MinHeapNode* temp) {
__global__ void newNode(char data, unsigned freq, struct MinHeapNode *temp) {
	//struct MinHeapNode* temp = (struct MinHeapNode*)malloc(sizeof(struct MinHeapNode));
	temp = (struct MinHeapNode*)malloc(sizeof(struct MinHeapNode));

	temp->left = temp->right = NULL;
	temp->data = data;
	temp->freq = freq;

	//return temp;
}

//__global__ struct MinHeap* createMinHeap(unsigned capacity, struct MinHeap *minHeap) {
__global__ void createMinHeap(unsigned capacity, struct MinHeap *minHeap) {
	//struct MinHeap* minHeap = (struct MinHeap*)malloc(sizeof(struct MinHeap));
	minHeap = (struct MinHeap*)malloc(sizeof(struct MinHeap));

	minHeap->size = 0;

	minHeap->capacity = capacity;

	minHeap->array = (struct MinHeapNode**)malloc(minHeap-> capacity * sizeof(struct MinHeapNode*));
	//return minHeap;
}

__global__ void swapMinHeapNode(struct MinHeapNode** a, struct MinHeapNode** b) {
	struct MinHeapNode* t = *a;
	*a = *b;
	*b = t;
}

__global__ void minHeapify(struct MinHeap* minHeap, int idx) {
	int smallest = idx;
	int left = 2 * idx + 1;
	int right = 2 * idx + 2;

	if (left < minHeap->size && minHeap->array[left]-> freq < minHeap->array[smallest]->freq) smallest = left;

	if (right < minHeap->size && minHeap->array[right]-> freq < minHeap->array[smallest]->freq) smallest = right;

	if (smallest != idx) {
		swapMinHeapNode<<<1,1>>>(&minHeap->array[smallest], &minHeap->array[idx]);
		minHeapify<<<1,1>>>(minHeap, smallest);
	}
}

__device__ int isSizeOne(struct MinHeap* minHeap) {
	return (minHeap->size == 1);
}

//__global__ struct MinHeapNode* extractMin(struct MinHeap* minHeap) {
__global__ void extractMin(struct MinHeap *minHeap, struct MinHeapNode *temp) {
	//struct MinHeapNode* temp = minHeap->array[0];
	temp = minHeap->array[0];
	minHeap->array[0] = minHeap->array[minHeap->size - 1];

	--minHeap->size;
	minHeapify<<<1,1>>>(minHeap, 0);

	//return temp;
}

__global__ void insertMinHeap(struct MinHeap* minHeap, struct MinHeapNode* minHeapNode) {
	++minHeap->size;
	int i = minHeap->size - 1;

	while (i && minHeapNode->freq < minHeap->array[(i - 1) / 2]->freq) {
		minHeap->array[i] = minHeap->array[(i - 1) / 2];
		i = (i - 1) / 2;
	}

	minHeap->array[i] = minHeapNode;
}

__global__ void buildMinHeap(struct MinHeap* minHeap) {
	int n = minHeap->size - 1;
	int i;

	for (i = (n - 1) / 2; i >= 0; --i) minHeapify<<<1,1>>>(minHeap, i);
}

__global__ void printArr(int *arr, int n) {
	int i;
	for (i = 0; i < n; i++) printf("%d", arr[i]);

	printf("\n");
}

__device__ int isLeaf(struct MinHeapNode* root) {
	return !(root->left) && !(root->right);
}

//__global__ struct MinHeap* createAndBuildMinHeap(char *data, int *freq int size) {
__global__ void createAndBuildMinHeap(char *data, int *freq, int size, struct MinHeap *minHeap) {
	//struct MinHeap* minHeap = createMinHeap(size);
	createMinHeap<<<1,1>>>(size, minHeap);

	//for (int i = 0; i < size; ++i) minHeap->array[i] = newNode(data[i], freq[i]);
	for (int i  = 0; i < size; i++) newNode<<<1,1>>>(data[i], freq[i], minHeap->array[i]);

	minHeap->size = size;
	buildMinHeap<<<1,1>>>(minHeap);

	//return minHeap;
}

//__global__ struct MinHeapNode* buildHuffmanTree(char *data, int *freq, int size) {
__global__ void buildHuffmanTree(char *data, int *freq, int size, struct MinHeapNode *temp) {
	struct MinHeapNode *left, *right, *top;
	//struct MinHeap* minHeap = createAndBuildMinHeap(data, freq, size);
	struct MinHeap *minHeap;
	createAndBuildMinHeap<<<1,1>>>(data, freq, size, minHeap);
	while (!isSizeOne(minHeap)) {
		//left = extractMin(minHeap);
		extractMin<<<1,1>>>(minHeap, left);
		//right = extractMin(minHeap);
		extractMin<<<1,1>>>(minHeap, right);
		//top = newNode('$', left->freq + right->freq);
		newNode<<<1,1>>>('$', left->freq + right->freq, top);

		top->left = left;
		top->right = right;

		insertMinHeap<<<1,1>>>(minHeap, top);
	}
	//return extractMin(minHeap);
	extractMin<<<1,1>>>(minHeap, temp);
}

__global__ void printCodes(struct MinHeapNode *root, int *arr, int top) {
	if (root->left) {
		arr[top] = 0;
		printCodes<<<1,1>>>(root->left, arr, top + 1);
	}

	if (root->right) {
		arr[top] = 1;
		printCodes<<<1,1>>>(root->right, arr, top + 1);
	}

	if (isLeaf(root)) {
		printf("%c: ", root->data);
		printArr<<<1,1>>>(arr, top);
	}
}

__global__ void HuffmanCodes(char *data, int *freq, int size) {
	//struct MinHeapNode* root = buildHuffmanTree(data, freq, size);
	struct MinHeapNode *root;
	buildHuffmanTree<<<1,1>>>(data, freq, size, root);

	int *dev_arr;
	int top = 0;

	cudaMalloc((void**)&dev_arr, sizeof(int) * MAX_TREE_HT);

	printCodes<<<1,1>>>(root, dev_arr, top);

	cudaFree(dev_arr);
}

int main() {

	int i;
	int size = 6;
	char *arr;
	char *dev_arr;
	int *freq;
	int *dev_freq;

	cudaMalloc((void**)&dev_arr, sizeof(char) * size);
	cudaMalloc((void**)&dev_freq, sizeof(int) * size);

	cudaMallocHost((void**)&arr, sizeof(char) * size);
	cudaMallocHost((void**)&freq, sizeof(int) * size);

	int f[] = { 5, 9, 12, 13, 16, 45 };

	for (i = 0; i < 6; i++) {
		arr[i] = (char)(97 + i);
		freq[i] = f[i];
	}

	cudaMemcpy(dev_arr, arr, sizeof(char) * size, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_freq, freq, sizeof(int) * size, cudaMemcpyHostToDevice);



	HuffmanCodes<<<1,1>>>(arr, freq, size);

	cudaFree(dev_arr);
	cudaFree(dev_freq);
	cudaFreeHost(arr);
	cudaFreeHost(freq);

	return 0;
}
