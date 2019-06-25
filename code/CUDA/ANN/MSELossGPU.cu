#pragma once
#include "MSELossGPU.h"
__device__
float SquareError(float output, float expectedOutput) {
	return (((output - expectedOutput) * (output - expectedOutput))) * 0.5;
}

__device__
float d_SquareError(float output, float expectedOutput) {
	return (output - expectedOutput);
}

__global__
void LossKernel(float* input, float* expectedOutput, float* gradient, float* output, float* error, int inputWidth) {
	int threadId = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadId >= inputWidth) return;
	float se = SquareError(input[threadId], expectedOutput[threadId]);
	output[threadId] = se;
	error[threadId] += se;

	gradient[threadId] = d_SquareError(input[threadId], expectedOutput[threadId]);
}

MSELossGPU::MSELossGPU(int inputWidth) {
	this->inputWidth = inputWidth;
	CheckCuda(cudaMalloc(&output, inputWidth * sizeof(float)));
	CheckCuda(cudaMalloc(&gradient, inputWidth * sizeof(float)));
	CheckCuda(cudaMalloc(&error, inputWidth * sizeof(float)));
}

void MSELossGPU::CalculateLoss(float* input, float* expectedOutput) {
	dim3 blockSize(BLOCK_SIZE, 1, 1);
	dim3 gridSize(GRID_SIZE, 1, 1);
	LossKernel <<<blockSize, gridSize >>> (input, expectedOutput, gradient, output, error, inputWidth);
}

float MSELossGPU::GetError() {
	float* errorArr = new float[inputWidth];

	CheckCuda(cudaMemcpy(errorArr, error, inputWidth * sizeof(float), cudaMemcpyDeviceToHost));
	cudaDeviceSynchronize();

	float sum = 0;
	for (int i = 0; i < inputWidth; i++) {
		sum += errorArr[i];
		errorArr[i] = 0;
	}

	CheckCuda(cudaMemcpy(error, errorArr, inputWidth * sizeof(float), cudaMemcpyHostToDevice));
	cudaDeviceSynchronize();

	return sum;
}