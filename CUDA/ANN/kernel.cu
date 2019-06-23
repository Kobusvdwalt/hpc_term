#include "kernel.h"

__host__ __device__
bool Match(float* set1, float* set2, int count) {
	int max1 = 0;
	int max2 = 0;
	for (int i = 0; i < count; i++) {
		if (set1[max1] < set1[i]) {
			max1 = i;
		}
		if (set2[max2] < set2[i]) {
			max2 = i;
		}
	}


	if (max1 == max2) {
		return true;
	}
	return false;
}

__global__ void TrainKernel(GPUData gpuData, Network** networkList) {
	int networkId = blockIdx.x * blockDim.x + threadIdx.x;

	float output[10];

	float* sampleImage = gpuData.trainImages + 784 * networkId;
	float* sampleLabel = gpuData.trainLabels + 10 * networkId;

	for (int j = 0; j < 5; j++) {
		networkList[networkId]->Forward(sampleImage, output);
		networkList[networkId]->Backward(sampleLabel);
	}
	networkList[networkId]->UpdateWeights();
}

void LaunchKernel(dim3 block, dim3 grid, GPUData gpuData, Network** networkList) {
	TrainKernel <<<block, grid >>> (gpuData, networkList);
}

void PrepareGPU(float** trainImages, float** trainLabels, int trainCount, float** testImages, float** testLabels, int testCount, int width, int height) {
	
	// Transfer training data to GPU
	GPUData gpuData = GPUData();
	int trainImagesByteCount = width * height * trainCount * sizeof(float);
	int trainLabelsByteCount = 10 * trainCount * sizeof(float);

	CheckCuda(cudaMalloc(&gpuData.trainImages, size_t(trainImagesByteCount)));
	CheckCuda(cudaMalloc(&gpuData.trainLabels, size_t(trainLabelsByteCount)));
	gpuData.trainCount = trainCount;

	for (int i = 0; i < trainCount; i++) {
		cudaMemcpy(&gpuData.trainImages[i*(width*height)], trainImages[i], width*height * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(&gpuData.trainLabels[i*(10)], trainLabels[i], 10 * sizeof(float), cudaMemcpyHostToDevice);
	}
	cudaDeviceSynchronize();

	// Create collection of networks
	
	Network* networkCPU = new Network();
	int lc = 3;
	networkCPU->layerCount = new int(3);
	networkCPU->layerWidth = new int[3]{ 784, 64, 10 };
	networkCPU->neuronCount = new int(784 + 64 + 10);
	networkCPU->weightCount = new int(784 * 64 + 64 * 10);

	networkCPU->inputs = new float[*networkCPU->neuronCount * sizeof(float)];
	networkCPU->outputs = new float[*networkCPU->neuronCount * sizeof(float)];
	networkCPU->weights = new float[*networkCPU->weightCount * sizeof(float)];
	networkCPU->biases = new float[*networkCPU->neuronCount * sizeof(float)];

	networkCPU->inputError = new float[*networkCPU->neuronCount * sizeof(float)];
	networkCPU->outputError = new float[*networkCPU->neuronCount * sizeof(float)];
	networkCPU->weightError = new float[*networkCPU->weightCount * sizeof(float)];
	networkCPU->biasError = new float[*networkCPU->neuronCount * sizeof(float)];

	networkCPU->Initialize();
	
	// Data
	int numberOfNetworks = 8192;
	Network** networkList = new Network*[numberOfNetworks];
	
	for (int i = 0; i < numberOfNetworks; i++) {
		Network* network = new Network();
		// Aloocate GPU		
		CheckCuda(cudaMalloc(&network->layerCount, sizeof(int)));
		CheckCuda(cudaMalloc(&network->neuronCount, sizeof(int)));
		CheckCuda(cudaMalloc(&network->weightCount, sizeof(int)));
		CheckCuda(cudaMalloc(&network->layerWidth, *networkCPU->layerCount * sizeof(int)));

		CheckCuda(cudaMalloc(&network->inputs, *networkCPU->neuronCount * sizeof(float)));
		CheckCuda(cudaMalloc(&network->outputs, *networkCPU->neuronCount * sizeof(float)));
		CheckCuda(cudaMalloc(&network->weights, *networkCPU->weightCount * sizeof(float)));
		CheckCuda(cudaMalloc(&network->biases, *networkCPU->neuronCount * sizeof(float)));

		CheckCuda(cudaMalloc(&network->inputError, *networkCPU->neuronCount * sizeof(float)));
		CheckCuda(cudaMalloc(&network->outputError, *networkCPU->neuronCount * sizeof(float)));
		CheckCuda(cudaMalloc(&network->weightError, *networkCPU->weightCount * sizeof(float)));
		CheckCuda(cudaMalloc(&network->biasError, *networkCPU->neuronCount * sizeof(float)));

		// Copy CPU to GPU
		CheckCuda(cudaMemcpy(network->layerCount, networkCPU->layerCount, sizeof(int), cudaMemcpyHostToDevice));
		CheckCuda(cudaMemcpy(network->neuronCount, networkCPU->neuronCount, sizeof(int), cudaMemcpyHostToDevice));
		CheckCuda(cudaMemcpy(network->weightCount, networkCPU->weightCount, sizeof(int), cudaMemcpyHostToDevice));
		CheckCuda(cudaMemcpy(network->layerWidth, networkCPU->layerWidth, *networkCPU->layerCount * sizeof(int), cudaMemcpyHostToDevice));

		CheckCuda(cudaMemcpy(network->inputs, networkCPU->inputs, *networkCPU->neuronCount * sizeof(float), cudaMemcpyHostToDevice));
		CheckCuda(cudaMemcpy(network->outputs, networkCPU->outputs, *networkCPU->neuronCount * sizeof(float), cudaMemcpyHostToDevice));
		CheckCuda(cudaMemcpy(network->weights, networkCPU->weights, *networkCPU->weightCount * sizeof(float), cudaMemcpyHostToDevice));
		CheckCuda(cudaMemcpy(network->biases, networkCPU->biases, *networkCPU->neuronCount * sizeof(float), cudaMemcpyHostToDevice));

		CheckCuda(cudaMemcpy(network->inputError, networkCPU->inputError, *networkCPU->neuronCount * sizeof(float), cudaMemcpyHostToDevice));
		CheckCuda(cudaMemcpy(network->outputError, networkCPU->outputError, *networkCPU->neuronCount * sizeof(float), cudaMemcpyHostToDevice));
		CheckCuda(cudaMemcpy(network->weightError, networkCPU->weightError, *networkCPU->weightCount * sizeof(float), cudaMemcpyHostToDevice));
		CheckCuda(cudaMemcpy(network->biasError, networkCPU->biasError, *networkCPU->neuronCount * sizeof(float), cudaMemcpyHostToDevice));

		// Pointer
		Network* networkPtr;
		CheckCuda(cudaMalloc(&networkPtr, sizeof(Network)));
		CheckCuda(cudaMemcpy(networkPtr, network, sizeof(Network), cudaMemcpyHostToDevice));
		networkList[i] = networkPtr;
	}

	Network** networkListPtr;
	CheckCuda(cudaMalloc(&networkListPtr, numberOfNetworks * sizeof(Network*)));
	CheckCuda(cudaMemcpy(networkListPtr, networkList, numberOfNetworks * sizeof(Network*), cudaMemcpyHostToDevice));

	dim3 blockSize(256, 1);
	dim3 gridSize(numberOfNetworks / 256, 1, 1);

	for (int e = 0; e < 50; e++) {
		double error = 0;
		for (int i = 0; i < trainCount; i++) {
			LaunchKernel(blockSize, gridSize, gpuData, networkListPtr);
			cudaDeviceSynchronize();
			printf("Epoch : %d\n", e);
		}
	}

	cudaFree(gpuData.trainImages);
	cudaFree(gpuData.trainLabels);
	
}

// Serial test code :

/*

		printf("Epoch : %d\n", e);
		printf("Train Error : %lf\n", error / trainCount);

		error = 0;
		for (int i = 0; i < testCount; i++) {
			networkHost.Forward(testImages[i], output);
			if (Match(testLabels[i], output, 10) == false) {
				error += 1.0;
			}
		}
		printf("Test Error : %lf\n", error / testCount);
*/