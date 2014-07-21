all:
	g++ test.cpp NeuralNetwork.cpp -o nn_test
clean:
	rm nn_test
