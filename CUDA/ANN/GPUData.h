#pragma once
#include <stdio.h>
#include <cuda.h>
#include "DenseLayer.h"
#include "MSELayer.h"
#include "ReluLayer.h"
#include "SigmoidLayer.h"

typedef struct Network {
	// Manual initialization	
	float* inputs = NULL;
	float* outputs = NULL;
	float* weights = NULL;
	float* biases = NULL;

	float* inputError = NULL;
	float* outputError = NULL;
	float* weightError = NULL;
	float* biasError = NULL;
	
	
	// Automatic initialization
	int* layerCount;
	int* layerWidth;
	int* neuronCount = NULL;
	int* weightCount = NULL;
	Network() {
	}
	__host__ __device__
	int GetNeuronIndex(int layer, int neuron) {
		int start = 0;
		for (int l = 0; l < layer; l++) {
			start += layerWidth[l];
		}
		return start + neuron;
	}
	__host__ __device__
	int GetWeightIndex(int layer, int neuron, int weight) {
		int start = 0;
		for (int l = 0; l < layer; l++) {
			start += layerWidth[l]*(layerWidth[l+1]);
		}

		return start + neuron*layerWidth[layer+1] + weight;
	}

	// Helper functions
	__host__ __device__
	float sigmoid(float input) {
		return 1.0 / (1.0 + exp(-1 * input));
	}
	__host__ __device__
	float d_sigmoid(float z) {
		return sigmoid(z)*(1.0 - sigmoid(z));
	}
	__host__ __device__
	float SquareError(float output, float expectedOutput) {
		return (((output - expectedOutput) * (output - expectedOutput))) * 0.5;
	}
	__host__ __device__
	float d_SquareError(float output, float expectedOutput) {
		return (output - expectedOutput);
	}

	void Initialize() {
		// Initialize weights and weight errors
		for (int w = 0; w < *weightCount; w++) {
			weights[w] = (float)rand() / RAND_MAX - 0.5;
			weightError[w] = 0;
		}
		// Initialize biases and bias errors
		for (int b = 0; b < *neuronCount; b++) {
			biases[b] = ((float)rand() / RAND_MAX) * 0.01;
			biasError[b] = 0;
		}
	}
	__host__ __device__
	void Forward(float* input, float* output) {
		// Copy input
		float* outputsPtr = &outputs[GetNeuronIndex(0, 0)];
		for (int n = 0; n < layerWidth[0]; n++) {
			outputsPtr[n] = input[n];
		}
		// Run forward
		float sum = 0;
		float* prevOutPtr = NULL;
		for (int l = 1; l < *layerCount; l++) {			
			for (int n = 0; n < layerWidth[l]; n++) {
				sum = 0;
				prevOutPtr = &outputs[GetNeuronIndex(l - 1, 0)];
				for (int pn = 0; pn < layerWidth[l-1]; pn++) {
					sum += prevOutPtr[pn] * weights[GetWeightIndex(l-1, pn, n)];
				}

				inputs[GetNeuronIndex(l, n)] = sum + biases[GetNeuronIndex(l, n)];
				outputs[GetNeuronIndex(l, n)] = sigmoid(sum);
			}
		}
		// Copy output
		for (int n = 0; n < layerWidth[*layerCount - 1]; n++) {
			output[n] = outputs[GetNeuronIndex(*layerCount - 1, n)];
		}
	}
	__host__ __device__
	void Backward(float* expectedOutput) {
		// Copy expected output
		int neuronIndex = GetNeuronIndex(*layerCount - 1, 0);
		float* outputsPtr = &outputs[neuronIndex];
		float* outputErrPtr = &outputError[neuronIndex];
		float* inputsPtr = &inputs[neuronIndex];
		float* inputErrPtr = &inputError[neuronIndex];
		float* biasErrPtr = &biasError[neuronIndex];
		for (int n = 0; n < layerWidth[*layerCount - 1]; n++) {
			outputErrPtr[n] = d_SquareError(outputsPtr[n], expectedOutput[n]);
			inputErrPtr[n] = d_sigmoid(inputsPtr[n]) * outputErrPtr[n];
			biasErrPtr[n] = inputErrPtr[n];
		}
		for (int l = *layerCount - 2; l >= 0; l--) {
			for (int n = 0; n < layerWidth[l]; n++) {				
				for (int nn = 0; nn < layerWidth[l + 1]; nn++) {
					weightError[GetWeightIndex(l, n, nn)] =
						outputs[GetNeuronIndex(l, n)] * inputError[GetNeuronIndex(l + 1, nn)];
					
				}

				float sum = 0;
				for (int nn = 0; nn < layerWidth[l + 1]; nn++) {
					sum += weights[GetWeightIndex(l, n, nn)] * inputError[GetNeuronIndex(l + 1, nn)];
				}

				// Upstream error
				outputError[GetNeuronIndex(l, n)] = sum;
				inputError[GetNeuronIndex(l, n)] = d_sigmoid(inputs[GetNeuronIndex(l, n)]) * sum;
				biasError[n] = inputError[GetNeuronIndex(l, n)];
			}
		}
	}
	float learningRate = 0.001;
	__host__ __device__
	void UpdateWeights() {
		for (int w = 0; w < *weightCount; w++) {
			weights[w] -= weightError[w] * learningRate;
			weightError[w] = 0;
		}
		for (int b = 0; b < *neuronCount; b++) {
			biases[b] -= biasError[b] * learningRate;
			biasError[b] = 0;
		}
	}
};

typedef struct GPUData {
	float* trainImages = NULL;
	float* trainLabels = NULL;
	int trainCount = 0;

	GPUData() {
	}

	__host__ __device__
		float GetPixel(int sample, int x, int y) {
		return trainImages[(sample * (28 * 28)) + (y * 28 + x)];
	}
};