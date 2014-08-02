#ifndef BACKPROPAGATION_H_
#define BACKPROPAGATION_H_

#include "NeuralNetwork.h"

class BackPropagation{
    public:

        BackPropagation(NeuralNetwork &nn);
        void setLearningRate(double eta);
        void setMomentum(double alpha);
        void setMaxIteration(unsigned int iteration);
        void setMinChangeRate(double rate);
        void trainBatch(vector<vector<double> > &dataset);
        void trainStochastic(vector<vector<double> > &dataset);

    private:

        NeuralNetwork &nn;
        unsigned int numInput;
        unsigned int numHidden;
        unsigned int numOutput;

        double eta; // learning rate
        double alpha; // momentum
        unsigned int maxIteration;
        double minChangeRate;

        double getErrorSquare(vector<double> &output, vector<double> &yValues);
        void updateWeights(); // in-place update
        void calculateGradients(vector<double> &yValues);
        void initializeWeights();
        void initializeGradients();

        // Back Propagation Extra Data //
        // output gradients for back propagation
        vector<double> oGrads;
        // hidden gradients for back propagation
        vector<double> hGrads;
        // for momentum with back propagation
        vector<vector<double> > ihPrevWeightsDelta;
        vector<double> ihPrevBiasesDelta;
        vector<vector<double> > hoPrevWeightsDelta;
        vector<double> hoPrevBiasesDelta;
};

#endif

