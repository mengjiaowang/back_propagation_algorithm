#include <iostream>
#include <fstream>
#include <cstdlib>
#include "NeuralNetwork.h"
#include "BackPropagation.h"
#include "Normalization.h"

using namespace std;

void loadDataset(const char *path, vector<vector<double> > &dataset);
void evaluation(NeuralNetwork &nn, vector<vector<double> > &testdata);
int findMaximum(vector<double> & arr);

unsigned int numRow = 0;
unsigned int numLabels = 0;
unsigned int numAttr = 0;

int main(int argc, char* argv[]){

    // commend line parameters
    if(argc != 5){
        cout << "Wrong Parameters!" << endl;
        cout << "Usage: ./nn_test learningRate monentum maxIteration minChangeRage" << endl;
        return 0;
    }
    double learningRate = atof(argv[1]);
    double momentum = atof(argv[2]);
    double maxIteration = atoi(argv[3]);
    double minChangeRate = atof(argv[4]);

    // prepare training data
    vector<vector<double> > dataset;
    // string file = "datasets/wine.dat";
    string file = "datasets/linear.dat";
    loadDataset(file.c_str(), dataset);

    // define network structure
    Sigmoid sigmoid;
    Tanh tanh;
    NeuralNetwork nn(numAttr, 10, numLabels, sigmoid, tanh);
    BackPropagation bp(nn);

    // normalization
    StudentTNormalization norm;
    cout << dataset.size() << endl;
    norm.normalize(dataset, 0u, numAttr-1);
    cout << dataset.size() << endl;
    vector<vector<double> > train(dataset.begin(), dataset.end());
    vector<vector<double> > test(dataset.begin(), dataset.end());

    // start the training process
    bp.setLearningRate(learningRate);
    bp.setMomentum(momentum);
    bp.setMaxIteration(maxIteration);
    bp.setMinChangeRate(minChangeRate);
    bp.trainBatch(train);
    //bp.trainStochastic(train);

    // start the evaluation process
    evaluation(nn, test);

    return 0;
}

void evaluation(NeuralNetwork &nn, vector<vector<double> > &test){

    int correct = 0;
    for(unsigned int i = 0; i != test.size(); ++i){
        vector<double> xValues(test[i].begin(), test[i].begin()+numAttr);
        vector<double> yValues(test[i].begin()+numAttr, test[i].end());
        vector<double> &outputs = nn.computeOutputs(xValues);
        int index = findMaximum(outputs);
        if(yValues[index] == 1) correct ++;
    }
    cout << "Accuracy:" << ((double)correct)/test.size() 
        << "\tTest Size:" << test.size() << endl;
}

void loadDataset(const char *path, vector<vector<double> > &dataset){
    fstream datafile;
    datafile.open (path, ios::in);
    datafile >> numRow >> numLabels >> numAttr;

    cout << "The number of instances: " << numRow << endl 
        << "The number of lables: " << numLabels << endl
        << "The number of attributes: " << numAttr << endl;

    for(unsigned int j = 0; j != numRow; ++j){
        vector<double> row;
        dataset.push_back(row);
        double attribute = 0, label = 0;
        datafile >> label;
        for(unsigned int i = 0; i != numAttr; i ++){
            datafile >> attribute;
            dataset.back().push_back(attribute);
        }
        for(unsigned int i = 0; i != numLabels; ++i){
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
    for(unsigned int i = 1; i != arr.size(); ++i){
        if(max < arr[i]){
            max = arr[i];
            index = i;
        }
    }
    return index; 
}
