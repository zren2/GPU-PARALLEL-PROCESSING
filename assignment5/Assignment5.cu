#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <string>
#include <vector>
#include <iostream>

using namespace std;

int main() {
  string str = "\002banana\003";
  vector<string> table;
  for (int i = 0; i < str.length(); i++) {
    string temp = str.substr(i, str.length()) + str.substr(0, i);
    table.push_back(temp);
  }
  thrust::device_vector<char*> device_table;
  for (int i = 0; i < table.size(); i++) {
    char* temp;
    cudaMalloc((void**)&temp, sizeof(char) * (str.length() + 1));
    cudaMemcpy(temp, table[i].c_str(), sizeof(char) * (str.length() + 1), cudaMemcpyHostToDevice);
    device_table.push_back(temp);
  }
  thrust::sort(device_table.begin(), device_table.end());
  char* result;
  cudaMallocHost((void**)&result, sizeof(char) * (device_table.size() + 1));
  for (int i = 0; i < device_table.size(); i++) {
    char* temp;
    cudaMallocHost((void**)&temp, sizeof(char) * (str.length() + 1));
    cudaMemcpy(temp, device_table[i], sizeof(char) * (str.length() + 1), cudaMemcpyDeviceToHost);
    result[i] = temp[str.length() - 1];
    cudaFreeHost(temp);
  }
  cout << result << endl;
  for (int i = 0; i < device_table.size(); i++) {
    cudaFree(device_table[i]);
  }
  cudaFreeHost(result);
  return 0;
}
