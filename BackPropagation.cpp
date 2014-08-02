#include "BackPropagation.h"
#include "NeuralNetwork.h"
#include <iostream>
#include <ctime>
#include <cmath>
#include <cstdlib>

BackPropagation::BackPropagation(NeuralNetwork &nn): nn(nn){

    this->numInput = this->nn.getNumInput();
    this->numHidden = this->nn.getNumHidden();
    this->numOutput = this->nn.getNumOutput();

    oGrads.resize(numOutput);
    hGrads.resize(numHidden);

    ihPrevWeightsDelta.resize(numInput);
    for(unsigned int i = 0; i != numInput; ++i){
         ihPrevWeightsDelta[i].resize(numHidden);
         for(unsigned int j = 0; j != numHidden; ++j){
            ihPrevWeightsDelta[i][j] = 0;
         }
    }

    ihPrevBiasesDelta.resize(numHidden);

    for(unsigned int i = 0; i != numHidden; ++i){
        ihPrevBiasesDelta[i] = 0;
    }

    hoPrevWeightsDelta.resize(numHidden);
    for(unsigned int i = 0; i != numHidden; ++i){
        hoPrevWeightsDelta[i].resize(numOutput);
        for(unsigned int j = 0; j != numOutput; ++j){
            hoPrevWeightsDelta[i][j] = 0;
        }
    }

    hoPrevBiasesDelta.resize(numOutput);
    for(unsigned int i = 0; i != numOutput; ++i){
        hoPrevBiasesDelta[i] = 0;
    }
}

void BackPropagation::setLearningRate(double eta){
    this->eta = eta;
}

void BackPropagation::setMomentum(double alpha){
    this->alpha = alpha;
}

void BackPropagation::setMaxIteration(unsigned int iteration){
    this->maxIteration = iteration;
}

void BackPropagation::setMinChangeRate(double rate){
    this->minChangeRate = rate;
}

void BackPropagation::initializeWeights(){
    srand(time(0));
    unsigned int numWeights = (numInput + numOutput) * numHidden + numHidden + numOutput;
    vector<double> weight;
    weight.resize(numWeights);
    for(unsigned int i = 0; i != numWeights; ++i){
        unsigned int r = rand() % 100000 + 1;
        double w = r/100000.0;
        weight[i] = w;
    }
    nn.setWeights(weight);
}

void BackPropagation::initializeGradients(){
    for(unsigned int i = 0; i != oGrads.size(); ++i){
        oGrads[i] = 0;
    }
    for(unsigned int i = 0; i != hGrads.size(); ++i){
        hGrads[i] = 0;
    }
}

void BackPropagation::trainStochastic(vector<vector<double> > &dataset){
    initializeWeights();
    unsigned int iteration = maxIteration;
    double preError = 0;
    while(iteration--) {
        double error = 0;
        for(unsigned int i = 0; i != dataset.size(); ++i){
            initializeGradients();
            vector<double> xValues(dataset[i].begin(), dataset[i].begin() + numInput);
            vector<double> yValues(dataset[i].begin() + numInput, dataset[i].end());
            nn.computeOutputs(xValues);
            error += getErrorSquare(nn.outputs, yValues);
            calculateGradients(yValues);
            if(iteration != 0 || i != dataset.size() - 1) updateWeights();
        }
        cout << "Iteration #" << maxIteration - iteration << "\tError:"
            << error/dataset.size() << endl;
        if(fabs(error - preError)/preError < minChangeRate) break;
        preError = error;
    }
}

void BackPropagation::trainBatch(vector<vector<double> > &dataset){
    initializeWeights();
    unsigned int iteration = maxIteration;
    double preError = 0;
    while(iteration--) {
        double error = 0;
        initializeGradients();
        for(unsigned int i = 0; i != dataset.size(); ++i){
            vector<double> xValues(dataset[i].begin(), dataset[i].begin() + numInput);
            vector<double> yValues(dataset[i].begin() + numInput, dataset[i].end());
            nn.computeOutputs(xValues);
            error += getErrorSquare(nn.outputs, yValues);
            calculateGradients(yValues);
        }
        if(iteration != 0) updateWeights();
        cout << "Iteration #" << maxIteration - iteration << "\tCost:"
            << error/(2*dataset.size()) << endl;
        if(fabs(error - preError)/preError < minChangeRate) break;
        preError = error;
    }
}

double BackPropagation::getErrorSquare(vector<double> &output, vector<double> &yValues){
    double error = 0;
    for(unsigned int i = 0; i != output.size(); ++i){
        error += (output[i] - yValues[i]) * (output[i] - yValues[i]);
    }
    return error;
}

void BackPropagation::calculateGradients(vector<double> &yValues){
    if(nn.outputs.size() != yValues.size()){
        cout << "The yValues does not match network structure" << endl;
        return;
    }
    // compute the output gradients
    for(unsigned int i = 0; i != oGrads.size(); ++i){
        oGrads[i] += nn.outputActi.derivative(nn.outputs[i]) * (yValues[i] - nn.outputs[i]);
    }
    // compute the hidden gradients
    for(unsigned int i = 0; i != hGrads.size(); ++i){
        double sum = 0.0;
        for(unsigned int j = 0; j != numOutput; ++j){
            sum += oGrads[j] * nn.hoWeights[i][j];
        }
        hGrads[i] += nn.hiddenActi.derivative(nn.ihOutputs[i]) * sum;
    }
}

void BackPropagation::updateWeights(){
    // update hidden to output weights
    for(unsigned int i = 0; i != nn.hoWeights.size(); ++i){
        for(unsigned int j = 0; j != nn.hoWeights[i].size(); ++j){
            double delta = eta * oGrads[j] * nn.ihOutputs[i];
            nn.hoWeights[i][j] += delta;
            nn.hoWeights[i][j] += alpha * hoPrevWeightsDelta[i][j];
            hoPrevWeightsDelta[i][j] = delta;
        }
    }
    // update hidden to output biases
    for(unsigned int i = 0; i != nn.hoBiases.size(); ++i){
        double delta = eta * oGrads[i] * 1.0;
        nn.hoBiases[i] += delta;
        nn.hoBiases[i] += alpha * hoPrevBiasesDelta[i];
        hoPrevBiasesDelta[i] = delta;
    }
    // update input to hidden layer weights
    for(unsigned int i = 0; i != nn.ihWeights.size(); ++i){
        for(unsigned int j = 0; j != nn.ihWeights.size(); ++j){
            double delta = eta * hGrads[j] * nn.inputs[i];
            nn.ihWeights[i][j] += delta;
            nn.ihWeights[i][j] += alpha * ihPrevWeightsDelta[i][j];
            ihPrevWeightsDelta[i][j] = delta;
        }
    }
    // update input to hidden layer biases
    for(unsigned int i = 0; i != nn.ihBiases.size(); ++i){
        double delta = eta * hGrads[i] * 1.0;
        nn.ihBiases[i] += delta;
        nn.ihBiases[i] += alpha * ihPrevBiasesDelta[i];
        ihPrevBiasesDelta[i] = delta;
    }
}
