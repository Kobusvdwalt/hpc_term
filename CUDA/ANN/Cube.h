#pragma once
// Useful way to store multiple features (image layers)
class Cube {
public:
	double*** values;
	int width = 0;
	int height = 0;
	int depth = 0;

	Cube(int width, int height, int depth) {
		this->width = width;
		this->height = height;
		this->depth = depth;

		values = new double**[width];
		for (int x = 0; x < width; x++) {
			values[x] = new double*[height];
			for (int y = 0; y < height; y++) {
				values[x][y] = new double[depth];
				for (int z = 0; z < depth; z++) {
					values[x][y][z] = 0;
				}
			}
		}
	}

	void Print() {
		for (int z = 0; z < this->depth; z++) {
			for (int y = 0; y < this->height; y++) {
				for (int x = 0; x < this->width; x++) {
					printf("%.6lf ", this->values[x][y][z]);
				}
				printf("\n");
			}
		}
		printf("\n");
	}

	void PrintStats() {
		printf("Width: %d\n", width);
		printf("Height: %d\n", height);
		printf("Depth: %d\n", depth);
	}
};