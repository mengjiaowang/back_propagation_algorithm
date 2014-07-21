#include "BackPropagation.h"
#include "NeuralNetwork.h"

BackPropagation::BackPropagation(NeuralNetwork &nn): nn(nn){
}

void BackPropagation::setLearningRate(double eta){
    this->eta = eta;
}

void BackPropagation::setMomentum(double alpha){
    this->alpha = alpha;
}
