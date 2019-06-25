#pragma once
#include <math.h>
#include "CudaLibs.h"
class DropoutGPU {
public:
	float* inputRef = NULL;
	float* output = NULL;
	float* dropoutMask = NULL;
	float dropoutRate = 0.2;
	float* gradient = NULL;
	curandState* randomStates;

	int inputWidth = 0;

	DropoutGPU(int inputWidth, float dropoutRate);
	void Forward(float* input);
	void Backward(float* upstreamGradient);

	void CheckCuda(cudaError_t ce) {
		//printf("%s\n", cudaGetErrorString(ce));
	}
	int BLOCK_SIZE = 32;
	int GRID_SIZE = 32;
};