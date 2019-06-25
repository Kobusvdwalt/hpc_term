#pragma once
#include <stdlib.h>
class Dense {
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

	Dense(int inputWidth, int outputWidth) {
		this->inputWidth = inputWidth;
		this->outputWidth = outputWidth;

		output = new float[inputWidth];
		gradient = new float[inputWidth];
	}

	void Initialize(float multiplier, float offset) {
		weights = new float[inputWidth * outputWidth];
		weightErrors = new float[inputWidth * outputWidth];
		for (int i = 0; i < inputWidth * outputWidth; i++) {
			weights[i] = ((float)rand() / RAND_MAX) * multiplier + offset;
			weightErrors[i] = 0;
		}

		biases = new float[outputWidth];
		biasErrors = new float[outputWidth];
		for (int o = 0; o < outputWidth; o++) {
			biases[o] = ((float)rand() / RAND_MAX) * 0.01;
			biasErrors[o] = 0;
		}
	}

	void Forward(float* input) {
		for (int o = 0; o < outputWidth; o++) {
			float sum = 0;
			for (int i = 0; i < inputWidth; i++) {
				sum += input[i] * weights[inputWidth * o + i];
			}

			output[o] = sum + biases[o];
		}
		inputRef = input;
	}
	void Backward(float* upstreamGradient) {
		for (int i = 0; i < inputWidth; i++) {
			float sum = 0;
			for (int o = 0; o < outputWidth; o++) {
				weightErrors[inputWidth * o + i] += inputRef[i] * upstreamGradient[o];
				sum += weights[inputWidth * o + i] * upstreamGradient[o];
			}
			gradient[i] = sum;
		}

		for (int o = 0; o < outputWidth; o++) {
			biasErrors[o] += upstreamGradient[o];
		}
	}

	float learningRate = 0.01;
	void UpdateWeights() {
		for (int i = 0; i < inputWidth * outputWidth; i ++) {
			weights[i] -= weightErrors[i] * learningRate;
			weightErrors[i] = 0;
		}
		for (int o = 0; o < outputWidth; o++) {
			biases[o] -= biasErrors[o] * learningRate;
			biasErrors[o] = 0; 
		}
	}
};