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

void TrainCPU(int width, int height, int trainCount, int testCount) {	
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

	// Layer 1
	Dense* d1 = new Dense(width*height, 256);
	Relu* r1 = new Relu(256);
	d1->Initialize(0.01, 0);

	// Layer 2
	Dense* d2 = new Dense(256, 64);
	Relu* r2 = new Relu(64);
	d2->Initialize(0.01, 0);

	// Layer 3
	Dense* d3 = new Dense(64, 10);	
	Sigmoid* s = new Sigmoid(10);
	d3->Initialize(2, -1);

	MSELoss* mse = new MSELoss(10);

	int imageSample = 0;
	double mseSum = 0;

	for (int e = 0; e < 20; e++) {
		mseSum = 0;
		std::clock_t c_start = std::clock();
		for (int i = 0; i < trainCount; i++) {
			imageSample = rand() % trainCount;
			d1->Forward(trainImages[imageSample]);
			r1->Forward(d1->output);

			d2->Forward(r1->output);
			r2->Forward(d2->output);

			d3->Forward(r2->output);
			s->Forward(d3->output);

			mse->CalculateLoss(s->output, trainLabels[imageSample]);

			s->Backward(mse->gradient);
			d3->Backward(s->gradient);

			r2->Backward(d3->gradient);
			d2->Backward(r2->gradient);

			r1->Backward(d2->gradient);
			d1->Backward(r1->gradient);

			d1->UpdateWeights();
			d2->UpdateWeights();
			d3->UpdateWeights();

			mseSum += mse->errorSum;
		}
		std::clock_t c_end = std::clock();
		double milliseconds = 1000.0 * (c_end - c_start) / CLOCKS_PER_SEC;

		printf("Epoch : %d\n", e);
		printf("Train Error : %lf\n", mseSum/trainCount);
		printf("Time : %lf\n", milliseconds);

		mseSum = 0;
		for (int i = 0; i < testCount; i++) {
			imageSample = i;
			d1->Forward(testImages[imageSample]);
			r1->Forward(d1->output);

			d2->Forward(r1->output);
			r2->Forward(d2->output);

			d3->Forward(r2->output);
			s->Forward(d3->output);

			if (FindMax(testLabels[imageSample], 10) != FindMax(s->output, 10)) {
				mseSum += 1.0;
			}

		}
		printf("Test Error : %.2lf\n\n", (mseSum / testCount)*100);
	}
}


int main(int argc, char **argv) {
	TrainCPU(28, 28, 6000, 1000);
	
	printf("End\n");
	return 0;
}