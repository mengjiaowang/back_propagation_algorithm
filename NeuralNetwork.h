#ifndef NEURALNETWORK_H_
#define NEURALNETWORK_H_

#include <vector>
#include "ActivationFunction.h"

using namespace std;

class NeuralNetwork{

    public:
        
        NeuralNetwork(int numInput, int numHidden, int numOutput, 
                ActivationFunction &hidden, ActivationFunction &output);
        void updateWeights(vector<double> &tValues, double eta, double alpha);
        void setWeights(vector<double> &weights);
        void getWeights(vector<double> &weights);
        void computeOutputs(vector<double> &xValues);

        // Neural Network Parameters //
        // input
        vector<double> inputs;
        // input to hidden
        vector<vector<double> > ihWeights;
        vector<double> ihSums;
        vector<double> ihBiases;
        vector<double> ihOutputs;
        // hidden to output
        vector<vector<double> > hoWeights;
        vector<double> hoSums;
        vector<double> hoBiases;
        // onput
        vector<double> outputs;

    private:

        int numInput;
        int numHidden;
        int numOutput;

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

        ActivationFunction &hiddenActi;
        ActivationFunction &outputActi;

};

#endif
