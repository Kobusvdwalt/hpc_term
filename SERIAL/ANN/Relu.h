#pragma once
class Relu {
public:
	float* inputRef = NULL;
	float* output = NULL;
	float* gradient = NULL;

	int inputWidth = 0;

	Relu(int inputWidth) {
		this->inputWidth = inputWidth;
		gradient = new float[inputWidth];
		output = new float[inputWidth];
	}

	float relu(float input) {
		if (input >= 0) {
			return input;
		}
		return 0.1*input;
	}

	float d_relu(double input) {
		if (input >= 0) {
			return 1;
		}
		return 0.1;
	}

	void Forward(float* input) {
		for (int i = 0; i < inputWidth; i++) {
			output[i] = relu(input[i]);
		}
		inputRef = input;
	}

	void Backward(float* upstreamGradient) {
		for (int i = 0; i < inputWidth; i++) {
			gradient[i] = d_relu(inputRef[i]) * upstreamGradient[i];
		}
	}
};