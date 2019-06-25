#pragma once
#include <stdlib.h>
#include "CudaLibs.h"

class DenseGPU {
public:
	float* inputRef = NULL;
	float* output = NULL;
	float* gradient = NULL;

	int inputWidth = 0;
	int outputWidth = 0;

	float* weights = NULL;
	float* weightErrors = NULL;

	float* biases = NULL;
	float* biasErrors = NULL;

	DenseGPU(int inputWidth, int outputWidth);
	void Initialize(float multiplier, float offset);
	void Forward(float* input);	
	void Backward(float* upstreamGradient);
	void UpdateWeights(float learningRate);
	void CheckCuda(cudaError_t ce) {
		 //printf("%s\n", cudaGetErrorString(ce));
	}
	int BLOCK_SIZE = 32;
	int GRID_SIZE = 32;
};