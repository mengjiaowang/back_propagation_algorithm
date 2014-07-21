#include "NeuralNetwork.h"
#include <vector>
using namespace std;

NeuralNetwork::NeuralNetwork(int numInput, int numHidden, int numOutput){

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

    oGrads.resize(numOutput);
    hGrads.resize(numHidden);

    ihPrevWeightsDelta.resize(numInput);
    for(unsigned int i = 0; i != numInput; ++i){
        ihPrevWeightsDelta[i].resize(numHidden);
    }
    ihPrevBiasesDelta.resize(numHidden);
    hoPrevWeightsDelta.resize(numHidden);
    for(unsigned int i = 0; i != numHidden; ++i){
        hoPrevWeightsDelta[i].resize(numOutput);
    }
    hoPrevBiasesDelta.resize(numOutput);
}

void NeuralNetwork::updateWeights(vector<double> &tValues, double eta, double alpha){

}

void NeuralNetwork::setWeights(vector<double> &weights){

}

vector<double> & NeuralNetwork::getWeights(){
    return inputs;
}

void NeuralNetwork::computeOutputs(vector<double> &xValues){

}
