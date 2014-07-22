#ifndef BACKPROPAGATION_H_
#define BACKPROPAGATION_H_

#include "NeuralNetwork.h"

class BackPropagation{
    public:

        BackPropagation(NeuralNetwork &nn);
        void setLearningRate(double eta);
        void setMomentum(double alpha);
        void setMaxIteration(int iteration);
        void train(vector<vector<double> > &dataset);

    private:

        NeuralNetwork &nn;
        int numInput;
        int numHidden;
        int numOutput;
        double eta; // learning rate
        double alpha; // momentum
        int maxIteration;

        double getError(vector<double> &output, vector<double> &yValues);
        void updateWeights(vector<double> &yValues); // in-place update
        void initializeWeights();

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

