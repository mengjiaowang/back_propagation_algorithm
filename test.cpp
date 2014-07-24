#include <iostream>
#include <fstream>
#include "NeuralNetwork.h"
#include "BackPropagation.h"
#include "Normalization.h"

using namespace std;

void loadDataset(const char *path, vector<vector<double> > &dataset);
void evaluation(NeuralNetwork &nn, vector<vector<double> > &testdata);

int main(int argc, char* argv[]){

    // commend line parameters
    if(argc != 3){
        cout << "Wrong Parameters!" << endl;
        return 0;
    }
    double learningRate = std::atof(argv[1]);
    double maxIteration = std::atoi(argv[2]);

    // define network structure
    Sigmoid sigmoid;
    Tanh tanh;
    NeuralNetwork nn(13,200,3, sigmoid, tanh);
    BackPropagation bp(nn);

    // prepare training data
    vector<vector<double> > dataset;
    string file = "datasets/wine.dat";
    loadDataset(file.c_str(), dataset);

    // normalization
    Normalization norm;
    norm.normalize(dataset, 0, 12);
    vector<vector<double> > train(dataset.begin(), dataset.end());
    vector<vector<double> > test(dataset.begin(), dataset.end());
    
    // start the training process
    bp.setLearningRate(learningRate);
    bp.setMomentum(0.0);
    bp.setMaxIteration(maxIteration);
    bp.trainBatch(train);
    //bp.trainStochastic(train);

    // start the evaluation process
    evaluation(nn, test);

    return 0;
}

void evaluation(NeuralNetwork &nn, vector<vector<double> > &test){

    int correct = 0;
    for(int i = 0; i != test.size(); ++i){
        vector<double> xValues(test[i].begin(), test[i].begin()+13);
        vector<double> yValues(test[i].begin()+13, test[i].end());
        vector<double> &outputs = nn.computeOutputs(xValues);
        if(outputs[0] > outputs[1] && outputs[0] > outputs[2] && yValues[0] == 1) correct ++;
        if(outputs[1] > outputs[0] && outputs[1] > outputs[2] && yValues[1] == 1) correct ++;
        if(outputs[2] > outputs[0] && outputs[2] > outputs[1] && yValues[2] == 1) correct ++;
    }
    cout << "Accuracy:" << (double)correct/test.size() << endl;
}

void loadDataset(const char *path, vector<vector<double> > &dataset){
    fstream datafile;
    datafile.open (path, ios::in);
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

}
