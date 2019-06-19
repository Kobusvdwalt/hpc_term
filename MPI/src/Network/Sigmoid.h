#pragma once
#include <math.h>
class Sigmoid {
public:
	float* inputRef = NULL;
	float* output = NULL;
	float* gradient = NULL;
	
	int inputWidth = 0;

	Sigmoid(int inputWidth) {
		this->inputWidth = inputWidth;
		gradient = new float[inputWidth];
		output = new float[inputWidth];
	}

	float sigmoid(float input) {
		return 1.0 / (1.0 + exp(-1 * input));
	}

	float d_sigmoid(float z) {
		return sigmoid(z)*(1.0 - sigmoid(z));
	}

	void Forward(float* input) {
		for (int i = 0; i < inputWidth; i++) {
			output[i] = sigmoid(input[i]);
		}
		inputRef = input;
	}

	void Backward(float* upstreamGradient) {
		for (int i = 0; i < inputWidth; i++) {
			gradient[i] = d_sigmoid(inputRef[i]) * upstreamGradient[i];
		}
	}
};