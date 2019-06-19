
This project contains 3 implementations of a Fully Connected Neural Network.

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