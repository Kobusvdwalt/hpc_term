#include "Main.h"

void LoadImageData(string path, int width, int height, int count, float** data) {
	printf("Load image data.\n");

	for (int i = 0; i < count; i++) {
		bitmap_image image = bitmap_image(path + to_string(i) + ".bmp");
		data[i] = new float[width * height];
		for (int x = 0; x < image.width(); x++) {
			for (int y = 0; y < image.height(); y++) {
				data[i][y*width + x] = ((float)image.get_pixel(x, y).red)/255.0;
			}
		}
	}
}

void LoadLabelData(string path, int count, float** label) {
	printf("Load label data.\n");
	ifstream inFile;
	inFile.open(path + "labels.txt");
	if (!inFile) {
		cout << "Unable to open file";
		exit(1); // terminate with error
	}

	int x = 0;
	int i = 0;
	while (inFile >> x && i < count) {
		label[i] = new float[10];
		for (int o = 0; o < 10; o++) {
			label[i][o] = 0;
		}
		label[i][x] = 1.0;
		i++;
	}

	inFile.close();
}

int FindMax(float* set, int count) {
	int max = 0;
	for (int i = 0; i < count; i++) {
		if (set[max] < set[i]) {
			max = i;
		}
	}
	return max;
}

void CheckCudaM(cudaError_t ce) {
	//printf("%s\n", cudaGetErrorString(ce));
}

void TrainGPU(int width, int height, int trainCount, int testCount) {
	//load training data
	float** trainImages = new float*[trainCount];
	float** trainLabels = new float*[trainCount];
	LoadImageData("../../DATASET/digits28/train/", width, height, trainCount, trainImages);
	LoadLabelData("../../DATASET/digits28/train/", trainCount, trainLabels);

	//load testing data
	float** testImages = new float*[testCount];
	float** testLabels = new float*[testCount];
	LoadImageData("../../DATASET/digits28/test/", width, height, testCount, testImages);
	LoadLabelData("../../DATASET/digits28/test/", testCount, testLabels);

	float** trainImagesGPU = new float*[trainCount];
	float** trainLabelsGPU = new float*[trainCount];

	float** testImagesGPU = new float*[testCount];
	float** testLabelsGPU = new float*[testCount];

	for (int i = 0; i < trainCount; i++) {
		CheckCudaM(cudaMalloc(&trainImagesGPU[i], width * height * sizeof(float)));
		CheckCudaM(cudaMalloc(&trainLabelsGPU[i], 10 * sizeof(float)));

		CheckCudaM(cudaMemcpy(trainImagesGPU[i], trainImages[i], width * height * sizeof(float), cudaMemcpyHostToDevice));
		CheckCudaM(cudaMemcpy(trainLabelsGPU[i], trainLabels[i], 10 * sizeof(float), cudaMemcpyHostToDevice));
	}

	for (int i = 0; i < testCount; i++) {
		CheckCudaM(cudaMalloc(&testImagesGPU[i], width * height * sizeof(float)));
		CheckCudaM(cudaMalloc(&testLabelsGPU[i], 10 * sizeof(float)));

		CheckCudaM(cudaMemcpy(testImagesGPU[i], testImages[i], width * height * sizeof(float), cudaMemcpyHostToDevice));
		CheckCudaM(cudaMemcpy(testLabelsGPU[i], testLabels[i], 10 * sizeof(float), cudaMemcpyHostToDevice));
	}

	// Layer 1 w*h -> 64
	DropoutGPU* dropout = new DropoutGPU(width*height, 0.2);
	DenseGPU* d1 = new DenseGPU(width*height, 256);
	ReluGPU* r1 = new ReluGPU(256);
	d1->Initialize(0.01, 0);

	// Layer 2
	DenseGPU* d2 = new DenseGPU(256, 64);
	ReluGPU* r2 = new ReluGPU(64);
	d2->Initialize(0.01, 0);

	// Layer 3
	DenseGPU* d3 = new DenseGPU(64, 10);
	SigmoidGPU* s = new SigmoidGPU(10);
	d3->Initialize(2, -1);
	
	MSELossGPU* mse = new MSELossGPU(10);
	
	int imageSample = 0;
	double mseSum = 0;
	float* out = new float[10];
	float learningRate = 0.01;
	for (int e = 0; e < 20; e++) {
		mseSum = 0;
		cudaEvent_t start, stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start);
		for (int i = 0; i < trainCount; i++) {
			imageSample = rand() % trainCount;
			dropout->Forward(trainImagesGPU[imageSample]);
			d1->Forward(dropout->output);
			r1->Forward(d1->output);

			d2->Forward(r1->output);
			r2->Forward(d2->output);

			d3->Forward(r2->output);
			s->Forward(d3->output);

			mse->CalculateLoss(s->output, trainLabelsGPU[imageSample]);

			s->Backward(mse->gradient);
			d3->Backward(s->gradient);

			r2->Backward(d3->gradient);
			d2->Backward(r2->gradient);

			r1->Backward(d2->gradient);
			d1->Backward(r1->gradient);

			d1->UpdateWeights(learningRate);
			d2->UpdateWeights(learningRate);
			d3->UpdateWeights(learningRate);
		}
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		float milliseconds = 0;
		cudaEventElapsedTime(&milliseconds, start, stop);

		mseSum += mse->GetError();
		printf("Epoch : %d\n", e);
		printf("Train Error : %lf\n", mseSum / trainCount);
		printf("Time : %lf\n", milliseconds);
		printf("Learning rate : %lf\n", learningRate);

		learningRate *= 0.9;

		mseSum = 0;
		

		for (int i = 0; i < testCount; i++) {
			imageSample = i;
			d1->Forward(testImagesGPU[imageSample]);
			r1->Forward(d1->output);

			d2->Forward(r1->output);
			r2->Forward(d2->output);

			d3->Forward(r2->output);
			s->Forward(d3->output);

			CheckCudaM(cudaMemcpy(out, s->output, 10 * sizeof(float), cudaMemcpyDeviceToHost));
			cudaDeviceSynchronize();

			if (FindMax(testLabels[imageSample], 10) != FindMax(out, 10)) {
				mseSum += 1.0;
			}
		}
		printf("Test Error : %.2lf\n\n", (mseSum / testCount) * 100);
	}

	for (int i = 0; i < trainCount; i++) {
		cudaFree(trainImagesGPU[i]);
		cudaFree(trainLabelsGPU[i]);
	}

	for (int i = 0; i < testCount; i++) {
		cudaFree(testImagesGPU[i]);
		cudaFree(testLabelsGPU[i]);
	}
}

int main(int argc, char **argv) {
	TrainGPU(28, 28, 6000, 1000);
	
	printf("End\n");
	return 0;
}