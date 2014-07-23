#include "NeuralNetwork.h"
#include "ActivationFunction.h"
#include <vector>
#include <iostream>
using namespace std;

NeuralNetwork::NeuralNetwork(int numInput, int numHidden, int numOutput,
        ActivationFunction &hidden, ActivationFunction &output)
:hiddenActi(hidden), outputActi(output)
{

    this->numInput = numInput;
    this->numHidden = numHidden;
    this->numOutput = numOutput;

    inputs.resize(numInput);
    ihWeights.resize(numInput);
    for(unsigned int i = 0; i != numInput; ++i){
        ihWeights[i].resize(numHidden);
    }
    ihSums.resize(numHidden);
    ihBiases.resize(numHidden);
    ihOutputs.resize(numHidden);

    hoWeights.resize(numHidden);
    for(unsigned int i = 0; i != numHidden; ++i){
        hoWeights[i].resize(numOutput);
    }
    hoSums.resize(numOutput);
    hoBiases.resize(numOutput);
    outputs.resize(numOutput);
}

int NeuralNetwork::getNumInput(){
    return numInput;
}

int NeuralNetwork::getNumHidden(){
    return numHidden;
}

int NeuralNetwork::getNumOutput(){
    return numOutput;
}

void NeuralNetwork::setWeights(vector<double> &weights){
    int numWeights = (numInput + numOutput) * numHidden + numHidden + numOutput;
    if(weights.size() != numWeights){
        cout << "The number of weights does not match network structure" << endl;
        return;
    }
    int k = 0;

    // weights in input to hidden connections
    for(int i = 0; i != numInput; ++i){
        for(int j = 0; j != numHidden; ++j){
            ihWeights[i][j] = weights[k++];
        }
    }
    // weights in biases of hidden layer
    for(int i = 0; i != numHidden; ++i){
        ihBiases[i] = weights[k++];
    }
    // weights in hidden to output connections
    for(int i = 0; i != numHidden; ++i){
        for(int j = 0; j != numOutput; ++j){
            hoWeights[i][j] = weights[k++];
        }
    }
    // weights in biases of output layer
    for(int i = 0; i != numOutput; ++i){
        hoBiases[i] = weights[k++];
    }
}

void NeuralNetwork::getWeights(vector<double> &weights){
    int numWeights = (numInput + numOutput) * numHidden + numHidden + numOutput;
    weights.resize(numWeights);
    int k = 0;

    // weights in input to hidden connections
    for(int i = 0; i != numInput; ++i){
        for(int j = 0; j != numHidden; ++j){
            weights[k++] = ihWeights[i][j];
        }
    }
    // weights in biases of hidden layer
    for(int i = 0; i != numHidden; ++i){
        weights[k++] = ihBiases[i];
    }
    // weights in hidden to output connections
    for(int i = 0; i != numHidden; ++i){
        for(int j = 0; j != numOutput; ++j){
            weights[k++] = hoWeights[i][j];
        }
    }
    // weights in biases of output layer
    for(int i = 0; i != numOutput; ++i){
        weights[k++] = hoBiases[i];
    }
}

vector<double> &NeuralNetwork::computeOutputs(vector<double> &xValues){
    if(xValues.size() != numInput){
        cout << "The input values does not match the network input layers" << endl;
        return outputs;
    }
    // initialize input to hidden sums 
    for(int i = 0; i != numHidden; ++i){
        ihSums[i] = 0;
    }
    // initialize hidden to output sums
    for(int i = 0; i != numOutput; ++i){
        hoSums[i] = 0;
    }
    // copy x values to input 
    for(int i = 0; i != numInput; ++i){
        inputs[i] = xValues[i];
    }
    // compute input to hidden sums
    for(int i = 0; i != numInput; ++i){
        for(int j = 0; j != numHidden; ++j){
            ihSums[j] = ihWeights[i][j] * inputs[i];
        }
    }
    // add biases to hidden sums
    for(int i = 0; i != numHidden; ++i){
        ihSums[i] += ihBiases[i];
    }
    // determine input to hidden output
    for(int i = 0; i != numHidden; ++i){
        ihOutputs[i] = hiddenActi(ihSums[i]);
    }
    // compute hidden to output sums
    for(int i = 0; i != numHidden; ++i){
        for(int j = 0; j != numOutput; ++j){
            hoSums[j] = hoWeights[i][j] * ihOutputs[j];
        }
    }
    // add biases to output sums
    for(int i = 0; i != numOutput; ++i){
        hoSums[i] = hoSums[i] + hoBiases[i];
    }
    // determine hidden to output output
    for(int i = 0; i != numOutput; ++i){
        outputs[i] = outputActi(hoSums[i]);
    }
    return outputs;
}

