#pragma once
#include "SigmoidGPU.h"

__device__
float sigmoid(float input) {
	return 1.0 / (1.0 + exp(-1 * input));
}

__device__
float d_sigmoid(float z) {
	return sigmoid(z)*(1.0 - sigmoid(z));
}

__global__
void ForwardKernel(float* input, float* output, int inputWidth) {	
	int threadId = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadId >= inputWidth) return;

	output[threadId] = sigmoid(input[threadId]);
}

__global__
void BackwardKernel(float* inputRef, float* upstreamGradient, float* gradient, int inputWidth) {
	int threadId = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadId >= inputWidth) return;

	gradient[threadId] = d_sigmoid(inputRef[threadId]) * upstreamGradient[threadId];
}

SigmoidGPU::SigmoidGPU(int inputWidth) {
	this->inputWidth = inputWidth;
	CheckCuda(cudaMalloc(&inputRef, sizeof(float*)));
	CheckCuda(cudaMalloc(&output, inputWidth * sizeof(float)));
	CheckCuda(cudaMalloc(&gradient, inputWidth * sizeof(float)));
}

void SigmoidGPU::Forward(float* input) {
	dim3 blockSize(BLOCK_SIZE, 1, 1);
	dim3 gridSize(GRID_SIZE, 1, 1);
	ForwardKernel <<<blockSize, gridSize >>> (input, output, inputWidth);

	inputRef = input;
}

void SigmoidGPU::Backward(float* upstreamGradient) {
	dim3 blockSize(BLOCK_SIZE, 1);
	dim3 gridSize(GRID_SIZE, 1, 1);
	BackwardKernel <<<blockSize, gridSize >>> (inputRef, upstreamGradient, gradient, inputWidth);
}