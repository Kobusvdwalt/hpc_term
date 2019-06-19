mpicxx -std=c++0x ./src/Main.cpp -o ./dist/main
mpirun -np 4 ./dist/main