#pragma once
#include <stdio.h>
#include "CudaLibs.h"
#include "GPUData.h"



void PrepareGPU(
	float** trainImages, float** trainLabels, int trainCount, 
	float** testImages, float** testLabels, int testCount, int width, int height);
void LaunchKernel(dim3 block, dim3 grid, GPUData gpuData, Network** networkList);

void CheckCuda(cudaError_t ce) {
	printf("%s\n", cudaGetErrorString(ce));
}