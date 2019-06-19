#pragma once
#include <math.h>
#include "CudaLibs.h"
class ReluGPU {
public:
	float* inputRef = NULL;
	float* output = NULL;
	float* gradient = NULL;

	int inputWidth = 0;

	ReluGPU(int inputWidth);
	void Forward(float* input);
	void Backward(float* upstreamGradient);

	void CheckCuda(cudaError_t ce) {
		//printf("%s\n", cudaGetErrorString(ce));
	}
	int BLOCK_SIZE = 32;
	int GRID_SIZE = 32;
};