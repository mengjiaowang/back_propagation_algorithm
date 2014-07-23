#ifndef NEURALNETWORK_H_
#define NEURALNETWORK_H_

#include <vector>
#include "ActivationFunction.h"

using namespace std;

class NeuralNetwork{
    friend class BackPropagation;
    public:
        
        NeuralNetwork(int numInput, int numHidden, int numOutput, 
                ActivationFunction &hidden, ActivationFunction &output);
        void setWeights(vector<double> &weights);
        void getWeights(vector<double> &weights);
        vector<double> &computeOutputs(vector<double> &xValues);
        int getNumInput();
        int getNumHidden();
        int getNumOutput();

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

        int numInput;
        int numHidden;
        int numOutput;
};

#endif
