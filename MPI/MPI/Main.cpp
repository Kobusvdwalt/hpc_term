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

int main(int argc, char *argv[]) {
    int proccessCount, processId;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &proccessCount);
    MPI_Comm_rank(MPI_COMM_WORLD, &processId);
    printf("P count : %d\n", proccessCount);
    printf("This rank : %d\n", processId);
    
	int trainCount = 60000;
	int testCount = 10000;
	int width = 28;
	int height = 28;
	
	float** trainImages = new float*[trainCount];
	float** trainLabels = new float*[trainCount];
	float** testImages = new float*[testCount];
	float** testLabels = new float*[testCount];

	if (processId == 0) {		
		// Load data from disk
		LoadImageData("../DATASET/digits28/train/", width, height, trainCount, trainImages);
		LoadLabelData("../DATASET/digits28/train/", trainCount, trainLabels);
		LoadImageData("../DATASET/digits28/test/", width, height, testCount, testImages);
		LoadLabelData("../DATASET/digits28/test/", testCount, testLabels);
	} else {
		// Allocate memory
		for (int i=0; i < trainCount; i ++) {
			trainImages[i] = new float[width*height];
			trainLabels[i] = new float[10];			
		}
		for (int i=0; i < testCount; i ++) {
			testImages[i] = new float[width*height];
			testLabels[i] = new float[10];			
		}
	}
	
	// Distribute training data
	MPI_Barrier(MPI_COMM_WORLD);
	for (int i = 0; i < trainCount; i ++) {			
		MPI_Bcast(trainImages[i], width*height, MPI_FLOAT, 0, MPI_COMM_WORLD);
		MPI_Bcast(trainLabels[i], 10, MPI_FLOAT, 0, MPI_COMM_WORLD);
	}

	srand (processId);
	
	// Create network
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
	
	// Train loop	
	for (int me = 0; me < 50; me ++) {
		std::clock_t c_start = std::clock();

		// Train
		mseSum = 0;
		for (int i = 0; i < trainCount/proccessCount; i++) {
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

		MPI_Barrier(MPI_COMM_WORLD);

		// Info
		std::clock_t c_end = std::clock();
		double milliseconds = 1000.0 * (c_end - c_start) / CLOCKS_PER_SEC;
		if (processId == 0) {
			printf("Epoch : %d\n", me);
			printf("Train Error : %lf\n", mseSum/trainCount);
			printf("Time : %lf\n", milliseconds);
		}		

		// Test
		if (processId == 0) {
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

		MPI_Barrier(MPI_COMM_WORLD);

		// Sum network paramaters
		float* tempWeightsD1 = new float[d1->inputWidth * d1->outputWidth];
		float* tempBiasesD1 = new float[d1->outputWidth];

		float* tempWeightsD2 = new float[d2->inputWidth * d2->outputWidth];
		float* tempBiasesD2 = new float[d2->outputWidth];

		float* tempWeightsD3 = new float[d3->inputWidth * d3->outputWidth];
		float* tempBiasesD3 = new float[d3->outputWidth];

		MPI_Allreduce(d1->weights, tempWeightsD1, d1->inputWidth * d1->outputWidth, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
		MPI_Allreduce(d1->biases, tempBiasesD1, d1->outputWidth, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
		
		MPI_Allreduce(d2->weights, tempWeightsD2, d2->inputWidth * d2->outputWidth, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
		MPI_Allreduce(d2->biases, tempBiasesD2, d2->outputWidth, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);

		MPI_Allreduce(d3->weights, tempWeightsD3, d3->inputWidth * d3->outputWidth, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
		MPI_Allreduce(d3->biases, tempBiasesD3, d3->outputWidth, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);

		// Average paramaters
		for (int i=0; i < d1->inputWidth * d1->outputWidth; i ++) d1->weights[i] = tempWeightsD1[i] / (float)proccessCount;
		for (int i=0; i < d1->outputWidth; i ++) d1->biases[i] = tempBiasesD1[i] / (float)proccessCount;

		for (int i=0; i < d2->inputWidth * d2->outputWidth; i ++) d2->weights[i] = tempWeightsD2[i] / (float)proccessCount;
		for (int i=0; i < d2->outputWidth; i ++) d2->biases[i] = tempBiasesD2[i] / (float)proccessCount;

		for (int i=0; i < d3->inputWidth * d3->outputWidth; i ++) d3->weights[i] = tempWeightsD3[i] / (float)proccessCount;
		for (int i=0; i < d3->outputWidth; i ++) d3->biases[i] = tempBiasesD3[i] / (float)proccessCount;
	}

   MPI_Finalize();
   printf("Test1\n");
}