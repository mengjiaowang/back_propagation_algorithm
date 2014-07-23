#include "BackPropagation.h"
#include "NeuralNetwork.h"
#include <iostream>
#include <ctime>

BackPropagation::BackPropagation(NeuralNetwork &nn): nn(nn){

    this->numInput = this->nn.getNumInput();
    this->numHidden = this->nn.getNumHidden();
    this->numOutput = this->nn.getNumOutput();

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

void BackPropagation::setLearningRate(double eta){
    this->eta = eta;
}

void BackPropagation::setMomentum(double alpha){
    this->alpha = alpha;
}

void BackPropagation::setMaxIteration(int iteration){
    this->maxIteration = iteration;
}

void BackPropagation::initializeWeights(){
    srand(time(0));
    int numWeights = (numInput + numOutput) * numHidden + numHidden + numOutput;
    vector<double> weight;
    weight.resize(numWeights);
    for(int i = 0; i != numWeights; ++i){
        int r = rand() % 100 + 1;
        double w = r/100.0;
        weight[i] = w;
    }
    nn.setWeights(weight);
}

void BackPropagation::train(vector<vector<double> > &dataset){
    initializeWeights();
    int iteration = maxIteration;
    while(iteration--) {
        double error = 0;
        for(unsigned int i = 0; i != dataset.size(); ++i){
            vector<double> xValues(dataset[i].begin(), dataset[i].begin() + numInput);
            vector<double> yValues(dataset[i].begin() + numInput, dataset[i].end());
            nn.computeOutputs(xValues);
            error += getError(nn.outputs, yValues);
            updateWeights(yValues);
        }
        cout << "Iteration #" << maxIteration - iteration << "\tError:"
            << error/dataset.size() << endl;
    }
}

double BackPropagation::getError(vector<double> &output, vector<double> &yValues){
    double error = 0;
    for(int i = 0; i != output.size(); ++i){
        error += (output[i] - yValues[i]) * (output[i] - yValues[i]);
    }
    return error;
}

void BackPropagation::updateWeights(vector<double> &yValues){
    if(nn.outputs.size() != yValues.size()){
        cout << "The yValues does not match network structure" << endl;
        return;
    }
    // compute the output gradients
    for(int i = 0; i != oGrads.size(); ++i){
        oGrads[i] = nn.outputActi.derivative(nn.outputs[i]) * (yValues[i] - nn.outputs[i]);
    }
    // compute the hidden gradients
    for(int i = 0; i != hGrads.size(); ++i){
        double sum = 0.0;
        for(int j = 0; j != numOutput; ++j){
            sum += oGrads[j] * nn.hoWeights[i][j];
        }
        hGrads[i] = nn.hiddenActi.derivative(nn.ihOutputs[i]) * sum;
    }
    // update hidden to output weights
    for(int i = 0; i != nn.hoWeights.size(); ++i){
        for(int j = 0; j != nn.hoWeights[i].size(); ++j){
            double delta = eta * oGrads[j] * nn.ihOutputs[i];
            nn.hoWeights[i][j] += delta;
            nn.hoWeights[i][j] += alpha * hoPrevWeightsDelta[i][j];
            hoPrevWeightsDelta[i][j] = delta;
        }
    }
    // update hidden to output biases
    for(int i = 0; i != nn.hoBiases.size(); ++i){
        double delta = eta * oGrads[i] * 1.0;
        nn.hoBiases[i] += delta;
        nn.hoBiases[i] += alpha * hoPrevBiasesDelta[i];
        hoPrevBiasesDelta[i] = delta;
    }
    // update input to hidden layer weights
    for(int i = 0; i != nn.ihWeights.size(); ++i){
        for(int j = 0; j != nn.ihWeights.size(); ++j){
            double delta = eta * hGrads[j] * nn.inputs[i];
            nn.ihWeights[i][j] += delta;
            nn.ihWeights[i][j] += ihPrevWeightsDelta[i][j];
        }
    }
    // update input to hidden layer biases
    for(int i = 0; i != nn.ihBiases.size(); ++i){
        double delta = eta * hGrads[i] * 1.0;
        nn.ihBiases[i] += delta;
        nn.ihBiases[i] += alpha * ihPrevBiasesDelta[i];
    }
}
