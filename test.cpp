#include <iostream>
#include <fstream>
#include "NeuralNetwork.h"
#include "BackPropagation.h"
#include "Normalization.h"

using namespace std;

void loadDataset(const char *path, vector<vector<double> > &dataset);
void evaluation(NeuralNetwork &nn, vector<vector<double> > &testdata);
int findMaximum(vector<double> & arr);

int numRow = 0;
int numLabels = 0;
int numAttr = 0;

int main(int argc, char* argv[]){

    // commend line parameters
    if(argc != 3){
        cout << "Wrong Parameters!" << endl;
        return 0;
    }
    double learningRate = std::atof(argv[1]);
    double maxIteration = std::atoi(argv[2]);

    // prepare training data
    vector<vector<double> > dataset;
//    string file = "datasets/wine.dat";
    string file = "datasets/linear.dat";
    loadDataset(file.c_str(), dataset);

    // define network structure
    Sigmoid sigmoid;
    Tanh tanh;
    NeuralNetwork nn(numAttr, 100, numLabels, sigmoid, tanh);
    BackPropagation bp(nn);

    // normalization
    Normalization norm;
    norm.normalize(dataset, 0, numAttr-1);
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
        vector<double> xValues(test[i].begin(), test[i].begin()+numAttr);
        vector<double> yValues(test[i].begin()+numAttr, test[i].end());
        vector<double> &outputs = nn.computeOutputs(xValues);
        int index = findMaximum(outputs);
        if(yValues[index] == 1) correct ++;
    }
    cout << "Accuracy:" << (double)correct/test.size() << endl;
}

void loadDataset(const char *path, vector<vector<double> > &dataset){
    fstream datafile;
    datafile.open (path, ios::in);
    datafile >> numRow >> numLabels >> numAttr;

    cout << "The number of instances: " << numRow << endl 
         << "The number of lables: " << numLabels << endl
         << "The number of attributes: " << numAttr << endl;

    while(!datafile.eof()){
        vector<double> row;
        dataset.push_back(row);
        double attribute = 0, label = 0;
        datafile >> label;
        for(int i = 0; i != numAttr; i ++){
            datafile >> attribute;
            dataset.back().push_back(attribute);
        }
        for(int i = 0; i != numLabels; ++i){
            dataset.back().push_back(0);
        }
        dataset.back()[numAttr+label-1] = 1;
    }
    datafile.close();

}

int findMaximum(vector<double> & arr){
    if(arr.size() == 0) return -1;
    double max = arr[0];
    int index = 0;
    for(int i = 1; i != arr.size(); ++i){
        if(max < arr[i]){
            max = arr[i];
            index = i;
        }
    }
    return index; 
}
