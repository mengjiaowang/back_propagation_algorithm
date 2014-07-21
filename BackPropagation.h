#ifndef BACKPROPAGATION_H_
#define BACKPROPAGATION_H_

#include "NeuralNetwork.h"

class BackPropagation{
    public:

        BackPropagation(NeuralNetwork &nn);
        void setLearningRate(double eta);
        void setMomentum(double alpha);

    private:

        NeuralNetwork &nn;
        double eta; // learning rate
        double alpha; // momentum
};

#endif

