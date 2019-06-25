#pragma once
#include <stdlib.h>
class MSELoss {
public:
	float* output = NULL;
	float* gradient = NULL;
	
	float errorSum = 0;

	int inputWidth = 0;

	MSELoss(int inputWidth) {
		this->inputWidth = inputWidth;
		gradient = new float[inputWidth];
		output = new float[inputWidth];
	}

	float SquareError(float output, float expectedOutput) {
		return (((output - expectedOutput) * (output - expectedOutput))) * 0.5;
	}

	float d_SquareError(float output, float expectedOutput) {
		return (output - expectedOutput);
	}

	void CalculateLoss(float* input, float* expectedOutput) {
		this->errorSum = 0;
		for (int i = 0; i < inputWidth; i++) {
			output[i] = SquareError(input[i], expectedOutput[i]);
			gradient[i] = d_SquareError(input[i], expectedOutput[i]);
			this->errorSum += output[i];
		}
	}
};