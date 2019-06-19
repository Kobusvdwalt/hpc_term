
This project contains 3 implementations of a Fully Connected Neural Network.

SERIAL
- The serial implementation was built in Visual Studio and is structed as a Visual Studio 
  solution. To build and run this code you need to open the solution and compile it with
  visual studio.

CUDA
- The CUDA implementation was also built in Visual Studio and is also structed as a Visual Studio 
  solution. To build and run this code you need to open the solution and compile it with
  visual studio. You also need to have the CUDA sdk installed.

MPI
- The MPI implementation was build with mpich on linux. A make file is included to compile the   code. A run script is incuded to run the code.

Note : 
- We used Visual Studio for the SERIAL and CUDA implementations but when we tried to use 
  the Open MPI solutions for windows we experienced difficulties. When we tried to port the CUDA
  code to run on linux we also experienced problems where linux did not want to regognize the GPU
  and thus could not execute the CUDA commands. Thus we settled on this project structure, with
  the SERIAL and CUDA implementations built in Visual Studio on Windows. And the MPI   implementation being built with a simple text editor and the mpicc compiler on Linux.


The first implementation is a serial implementation which was used as the benchmark
for our tests. The serial implementation consists of a network containing
multiple layers. We only have one type of layer in this project which is 
refered to as a dense layer. A dense layer receives input to each of it's input 
neurons. This input signal is then fed forward to the output neurons. For every 
output neuron, a sum each input neurons times a weight is computed. This sum is
then passed through a non linear activation function and set as the output of a 
particular neuron.

We use 2 different activation functions in this network. Namely Sigmoid and ReLu
The Sigmoid activation function is the classical activation function. The Sigmoid
has the desirable characteristic of having an output with range between 0 and 1.
Sigmoid however has a problem in that the gradients calculated by backpropegation
get smaller and smaller with each layer that is added to the network. 

To address the vanishing gradient problem we make use of a ReLu function in the 
hidden layers of the network. The ReLu function has a gradient of 1 which means
that gradients don't reduce in magnatude as you move away from the output layer.
The relu layer is also very simple and thus provides a significant performance
increase compared to Sigmoid.

Since we desire output values between 0 and 1 the sigmiod function is still valueable
to us for the last layer.

The second implementation is a parallel implementation built in CUDA which runs on 
a GPU.

The third implementation is another parallel implementation built in MPI. MPI
allows us to execute.