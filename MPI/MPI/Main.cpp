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
    
	int trainCount = 10;
	int testCount = 1000;
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
	
	// Train loop
	
	for (int me = 0; me < 10; me ++) {
		// Train
		TrainSwarm(width, height, trainImages, trainLabels, trainCount, testImages, testLabels, testCount);

		float* weightTemp_d1 = new float[d1->inputWidth * d1->outputWidth];
		float* weightTemp_d2 = new float[d2->inputWidth * d2->outputWidth];
		float* weightTemp_d3 = new float[d3->inputWidth * d3->outputWidth];

		float* weightNew_d1 = new float[d1->inputWidth * d1->outputWidth];
		float* weightNew_d2 = new float[d2->inputWidth * d2->outputWidth];
		float* weightNew_d3 = new float[d3->inputWidth * d3->outputWidth];

		// Collect network paramaters
		if (processId == 0) {
			
			for (int i=1; i < proccessCount; i ++) {
				// Receive weight from node
				MPI_Recv(weightTemp_d1, d1->inputWidth * d1->outputWidth, MPI_FLOAT, i, 0, MPI_COMM_WORLD,  MPI_STATUS_IGNORE);
				MPI_Recv(weightTemp_d2, d2->inputWidth * d2->outputWidth, MPI_FLOAT, i, 0, MPI_COMM_WORLD,  MPI_STATUS_IGNORE);
				MPI_Recv(weightTemp_d3, d3->inputWidth * d3->outputWidth, MPI_FLOAT, i, 0, MPI_COMM_WORLD,  MPI_STATUS_IGNORE);

				// 
			}
			

		} else {
			MPI_Send(d1->weights, d1->inputWidth * d1->outputWidth, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
			MPI_Send(d2->weights, d2->inputWidth * d2->outputWidth, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
			MPI_Send(d3->weights, d3->inputWidth * d3->outputWidth, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
		}
		

		// Distribute network update
	}

/*
	if (processId == 1) {
		for (int i = 0; i < 784; i ++) {
			float val = trainImages[9][i];
			if (val < 0.1) {
				printf("     ");
			} else {
				printf("%.2lf ", val);
			}
			if (i % 28 == 0) {
				printf("\n");
			}
		}
	}
 */

   MPI_Finalize();
   printf("Test1\n");
}