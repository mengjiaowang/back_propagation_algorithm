#include <iostream>
#include <fstream>
#include "NeuralNetwork.h"
#include "BackPropagation.h"
#include "Normalization.h"

using namespace std;

int main(){

    // define network structure
    Sigmoid sigmoid;
    Tanh tanh;
    NeuralNetwork nn(13,20,3, sigmoid, tanh);
    BackPropagation bp(nn);

    // prepare training data
    fstream datafile;
    datafile.open ("data/wine.dat", ios::in);
    vector<vector<double> > dataset;
    while(!datafile.eof()){
        vector<double> row;
        dataset.push_back(row);
        double attribute = 0, label = 0;
        datafile >> label;
        for(int i = 0; i != 13; i ++){
            datafile >> attribute;
            dataset.back().push_back(attribute);
        }
        dataset.back().push_back(0);
        dataset.back().push_back(0);
        dataset.back().push_back(0);
        dataset.back()[12+label] = 1;
    }
    datafile.close();

    // normalization
    Normalization norm;
    norm.normalize(dataset, 0, 13);
    vector<vector<double> > train(dataset.begin(), dataset.begin()+150);
    vector<vector<double> > test(dataset.begin()+150, dataset.end());
    
    // start the training process
    bp.setLearningRate(0.1);
    bp.setMomentum(0.04);
    bp.setMaxIteration(30);
    bp.train(train);

    // start the test process
    for(int i = 0; i != test.size(); ++i){
        vector<double> xValues(test[i].begin(), test[i].begin()+13);
        vector<double> yValues(test[i].begin()+13, test[i].end());
        vector<double> &outputs = nn.computeOutputs(xValues);
        cout << "Test: output:(" << outputs[0] << "," <<
            outputs[1] << "," << outputs[2] << ")" <<
            " tValues:(" << yValues[0] << "," << yValues[1] << "," <<
            yValues[2] << ")" << endl;
    }

    return 0;
}
