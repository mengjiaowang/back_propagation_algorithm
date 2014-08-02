#ifndef NEURALNETWORK_H_
#define NEURALNETWORK_H_

#include <vector>
#include "ActivationFunction.h"

using namespace std;

class NeuralNetwork{
    friend class BackPropagation;
    public:
        
        NeuralNetwork(unsigned int numInput, unsigned int numHidden, unsigned int numOutput, 
                ActivationFunction &hidden, ActivationFunction &output);
        void setWeights(vector<double> &weights);
        void getWeights(vector<double> &weights);
        vector<double> &computeOutputs(vector<double> &xValues);
        unsigned int getNumInput();
        unsigned int getNumHidden();
        unsigned int getNumOutput();

    protected: // Neural Network Parameters //

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

        ActivationFunction &hiddenActi;
        ActivationFunction &outputActi;

    private:

        unsigned int numInput;
        unsigned int numHidden;
        unsigned int numOutput;
};

#endif
