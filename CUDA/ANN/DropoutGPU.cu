#pragma once
#pragma once
#include "DropoutGPU.h"

__global__
void InitializeKernel(curandState* state, int inputWidth) {
	int threadId = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadId >= inputWidth) return;
	curand_init(0, threadId, 0, &state[threadId]);
}

__global__
void ForwardKernelDropout(curandState* randomStates, float dropoutRate, float* input, float* output, float* dropoutMask, int inputWidth) {
	int threadId = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadId >= inputWidth) return;

	curandState_t state = randomStates[threadId];
	float rand = curand_uniform(&state);
	randomStates[threadId] = state;

	float inputR = input[threadId];

	float mask = 0;
	if (rand > dropoutRate) mask = 1;

	dropoutMask[threadId] = mask;
	output[threadId] = inputR * (1.0 / dropoutRate) * mask;
}

__global__
void BackwardKernelDropout(float* inputRef, float* upstreamGradient, float* gradient, float* dropoutMask, float dropoutRate, int inputWidth) {
	int threadId = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadId >= inputWidth) return;

	gradient[threadId] = dropoutMask[threadId] * inputRef[threadId] * upstreamGradient[threadId] * (1.0 / dropoutRate);
}

DropoutGPU::DropoutGPU(int inputWidth, float dropoutRate) {
	this->inputWidth = inputWidth;
	this->dropoutRate = dropoutRate;
	CheckCuda(cudaMalloc(&inputRef, sizeof(float*)));
	CheckCuda(cudaMalloc(&output, inputWidth * sizeof(float)));
	CheckCuda(cudaMalloc(&gradient, inputWidth * sizeof(float)));
	CheckCuda(cudaMalloc(&dropoutMask, inputWidth * sizeof(float)));

	cudaMalloc(&randomStates, inputWidth * sizeof(curandState));
	dim3 blockSize(BLOCK_SIZE, 1, 1);
	dim3 gridSize(GRID_SIZE, 1, 1);
	InitializeKernel <<<blockSize, gridSize >>> (randomStates, inputWidth);
}

void DropoutGPU::Forward(float* input) {
	dim3 blockSize(BLOCK_SIZE, 1, 1);
	dim3 gridSize(GRID_SIZE, 1, 1);
	ForwardKernelDropout <<<blockSize, gridSize >>> (randomStates, dropoutRate, input, output, dropoutMask, inputWidth);

	inputRef = input;
}

void DropoutGPU::Backward(float* upstreamGradient) {
	dim3 blockSize(BLOCK_SIZE, 1);
	dim3 gridSize(GRID_SIZE, 1, 1);
	BackwardKernelDropout <<<blockSize, gridSize >>> (inputRef, upstreamGradient, gradient, dropoutMask, dropoutRate, inputWidth);
}