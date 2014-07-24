CC=g++
CFLAGS=-c -Wall -O2
DEBUG=
all: nn_test

nn_test: test.o neural_network.o activation_function.o back_propagation.o normalization.o
	$(CC) $(DEBUG) test.o neural_network.o activation_function.o back_propagation.o normalization.o -o nn_test

test.o: test.cpp NeuralNetwork.h BackPropagation.h
	$(CC) $(CFLAGS) $(DEBUG) test.cpp -o test.o

neural_network.o: NeuralNetwork.cpp ActivationFunction.h
	$(CC) $(CFLAGS) $(DEBUG) NeuralNetwork.cpp -o neural_network.o

activation_function.o: ActivationFunction.cpp
	$(CC) $(CFLAGS) $(DEBUG) ActivationFunction.cpp -o activation_function.o

back_propagation.o: BackPropagation.cpp
	$(CC) $(CFLAGS) $(DEBUG) BackPropagation.cpp -o back_propagation.o

normalization.o: Normalization.cpp
	$(CC) $(CFLAGS) $(DEBUG) Normalization.cpp -o normalization.o

clean:
	rm nn_test *.o
