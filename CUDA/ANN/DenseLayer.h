#pragma once
#include <stdlib.h>
class DenseLayer {
public:
	float* inputRef = NULL;
	float* output = NULL;
	float* gradient = NULL;
	
	int inputWidth = 0;
	int outputWidth = 0;

	float** weights = NULL;
	float** weightErrors = NULL;

	float* biases = NULL;
	float* biasErrors = NULL;

	DenseLayer(int inputWidth, int outputWidth) {
		this->inputWidth = inputWidth;
		this->outputWidth = outputWidth;

		output = new float[inputWidth];
		gradient = new float[inputWidth];
	}

	void Initialize(float multiplier, float offset) {
		weights = new float*[inputWidth];
		weightErrors = new float*[inputWidth];
		for (int i = 0; i < inputWidth; i++) {
			weights[i] = new float[outputWidth];
			weightErrors[i] = new float[inputWidth];
			for (int o = 0; o < outputWidth; o++) {
				weights[i][o] = ((float)rand() / RAND_MAX) * multiplier + offset;
				weightErrors[i][o] = 0;
			}
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
				sum += input[i] * weights[i][o];
			}

			output[o] = sum + biases[o];
		}
		inputRef = input;
	}
	void Backward(float* upstreamGradient) {
		for (int i = 0; i < inputWidth; i++) {
			float sum = 0;
			for (int o = 0; o < outputWidth; o++) {
				weightErrors[i][o] += inputRef[i] * upstreamGradient[o];
				sum += weights[i][o] * upstreamGradient[o];
			}
			gradient[i] = sum;
		}

		for (int o = 0; o < outputWidth; o++) {
			biasErrors[o] += upstreamGradient[o];
		}
	}

	float learningRate = 0.01;
	void UpdateWeights() {
		for (int i = 0; i < inputWidth; i ++) {
			for (int o = 0; o < outputWidth; o++) {
				weights[i][o] -= weightErrors[i][o] * learningRate;
				weightErrors[i][o] = 0;
			}
		}
		for (int o = 0; o < outputWidth; o++) {
			biases[o] -= biasErrors[o] * learningRate;
			biasErrors[o] = 0; 
		}
	}
};