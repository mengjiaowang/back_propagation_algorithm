#include <iostream>
#include <fstream>
#include "NeuralNetwork.h"
#include "BackPropagation.h"

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
    vector<vector<double> > train(dataset.begin(), dataset.begin()+100);
    vector<vector<double> > test(dataset.begin()+100, dataset.end());
    
    // start the training process
    bp.setLearningRate(0.1);
    bp.setMomentum(0.04);
    bp.setMaxIteration(30);
    bp.train(train);

    return 0;
}
