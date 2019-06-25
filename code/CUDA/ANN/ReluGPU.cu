#pragma once
#include "ReluGPU.h"

__device__
float relu(float input) {
	if (input >= 0) {
		return input;
	}
	return 0.1*input;
}

__device__
float d_relu(float input) {
	if (input >= 0) {
		return 1;
	}
	return 0.1;
}

__global__
void ForwardKernelRelu(float* input, float* output, int inputWidth) {
	int threadId = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadId >= inputWidth) return;

	output[threadId] = relu(input[threadId]);
}

__global__
void BackwardKernelRelu(float* inputRef, float* upstreamGradient, float* gradient, int inputWidth) {
	int threadId = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadId >= inputWidth) return;

	gradient[threadId] = d_relu(inputRef[threadId]) * upstreamGradient[threadId];
}

ReluGPU::ReluGPU(int inputWidth) {
	this->inputWidth = inputWidth;
	CheckCuda(cudaMalloc(&inputRef, sizeof(float*)));
	CheckCuda(cudaMalloc(&output, inputWidth * sizeof(float)));
	CheckCuda(cudaMalloc(&gradient, inputWidth * sizeof(float)));
}

void ReluGPU::Forward(float* input) {
	dim3 blockSize(BLOCK_SIZE, 1, 1);
	dim3 gridSize(GRID_SIZE, 1, 1);
	ForwardKernelRelu <<<blockSize, gridSize >>> (input, output, inputWidth);

	inputRef = input;
}

void ReluGPU::Backward(float* upstreamGradient) {
	dim3 blockSize(BLOCK_SIZE, 1);
	dim3 gridSize(GRID_SIZE, 1, 1);
	BackwardKernelRelu <<<blockSize, gridSize >>> (inputRef, upstreamGradient, gradient, inputWidth);
}