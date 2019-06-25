#pragma once
#include <stdlib.h>
#include "CudaLibs.h"
class MSELossGPU {
public:
	float* output = NULL;
	float* gradient = NULL;
	float* error = NULL;
	int inputWidth = 0;

	MSELossGPU(int inputWidth);
	void CalculateLoss(float* input, float* expectedOutput);
	float GetError();

	void CheckCuda(cudaError_t ce) {
		//printf("%s\n", cudaGetErrorString(ce));
	}
	int BLOCK_SIZE = 32;
	int GRID_SIZE = 32;
};