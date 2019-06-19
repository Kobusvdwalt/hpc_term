#pragma once
#include "DenseGPU.h"
__global__
void InitializeKernel(float multiplier, float offset, float* weights, float* weightErrors, float* biases, float* biasErrors, int inputWidth, int outputWidth) {
	int threadId = blockIdx.x * blockDim.x + threadIdx.x;
	curandState_t state;
	curand_init(0, outputWidth * threadId + 1, 0, &state);

	if (threadId < inputWidth) {
		for (int o = 0; o < outputWidth; o++) {
			weights[inputWidth * o + threadId] = curand_uniform(&state) * + offset;
			weightErrors[inputWidth * o + threadId] = 0;
		}
	}

	if (threadId < outputWidth) {
		biases[threadId] = curand_uniform(&state) *  0.01;
		biasErrors[threadId] = 0;
	}
}

__global__
void ForwardKernel(float* input, float* weights, float* biases, float* output, int inputWidth, int outputWidth) {

	int threadId = blockIdx.x * blockDim.x + threadIdx.x;

	if (threadId < outputWidth) {
		float sum = 0;
		for (int i = 0; i < inputWidth; i++) {
			sum += input[i] * weights[inputWidth * threadId + i];
		}
		output[threadId] = sum +biases[threadId];
	}
}

__global__
void BackwardKernel(float* inputRef, float* upstreamGradient, float* gradient, float* weights, float* weightErrors, float* biasErrors, int inputWidth, int outputWidth) {
	int threadId = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadId < inputWidth) {
		float sum = 0;
		for (int o = 0; o < outputWidth; o++) {
			weightErrors[inputWidth * o + threadId] += inputRef[threadId] * upstreamGradient[o];
			sum += weights[inputWidth * o + threadId] * upstreamGradient[o];
		}
		gradient[threadId] = sum;
	}

	if (threadId < outputWidth) {
		biasErrors[threadId] += upstreamGradient[threadId];
	}	
}

__global__
void UpdateWeightsKernel(float learningRate, float* weights, float* weightErrors, float* biases, float* biasErrors, int inputWidth, int outputWidth) {
	int threadId = blockIdx.x * blockDim.x + threadIdx.x;

	if (threadId < inputWidth * outputWidth) {
		weights[threadId] -= weightErrors[threadId] * learningRate;
		weightErrors[threadId] = 0;
	}
	
	if (threadId < outputWidth) {
		biases[threadId] -= biasErrors[threadId] * learningRate;
		biasErrors[threadId] = 0;
	}
}

DenseGPU::DenseGPU(int inputWidth, int outputWidth) {
	this->inputWidth = inputWidth;
	this->outputWidth = outputWidth;
	
	CheckCuda(cudaMalloc(&output, sizeof(float) * outputWidth));
	CheckCuda(cudaMalloc(&gradient, sizeof(float) * inputWidth));
	CheckCuda(cudaMalloc(&biasErrors, sizeof(float) * inputWidth));

	CheckCuda(cudaMalloc(&weights, inputWidth * outputWidth * sizeof(float)));
	CheckCuda(cudaMalloc(&weightErrors, inputWidth * outputWidth * sizeof(float)));

	CheckCuda(cudaMalloc(&biases, inputWidth * sizeof(float)));
	CheckCuda(cudaMalloc(&biasErrors, inputWidth * sizeof(float)));
}

void DenseGPU::Initialize(float multiplier, float offset) {
	dim3 blockSize(BLOCK_SIZE, 1, 1);
	dim3 gridSize(GRID_SIZE, 1, 1);
	InitializeKernel <<<blockSize, gridSize >>> (multiplier, offset, weights, weightErrors, biases, biasErrors, inputWidth, outputWidth);
}

void DenseGPU::Forward(float* input) {
	dim3 blockSize(BLOCK_SIZE, 1, 1);
	dim3 gridSize(GRID_SIZE, 1, 1);
	ForwardKernel <<<blockSize, gridSize >>> (input, weights, biases, output, inputWidth, outputWidth);
	inputRef = input;
}

void DenseGPU::Backward(float* upstreamGradient) {
	dim3 blockSize(BLOCK_SIZE, 1, 1);
	dim3 gridSize(GRID_SIZE, 1, 1);
	BackwardKernel <<<blockSize, gridSize >>> (inputRef, upstreamGradient, gradient, weights, weightErrors, biasErrors, inputWidth, outputWidth);
}

void DenseGPU::UpdateWeights(float learningRate) {
	dim3 blockSize(outputWidth, 1, 1);
	dim3 gridSize(inputWidth, 1, 1);

	UpdateWeightsKernel <<<blockSize, gridSize >>> (learningRate, weights, weightErrors, biases, biasErrors, inputWidth, outputWidth);
}