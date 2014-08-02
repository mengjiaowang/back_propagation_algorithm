#include "Normalization.h"
#include <vector>
#include <iostream>
#include <cstdlib>
#include <limits>
#include <cmath>
using namespace std;

NormalizationBase::NormalizationBase(){
}

StudentTNormalization::StudentTNormalization(){
}

void StudentTNormalization::normalize(vector<vector<double> > &dataset){
    if(dataset.size() == 0){
        cout << "empty dataset!" << endl;
        return;
    }
    unsigned int numAttr = dataset[0].size();
    normalize(dataset, 0, numAttr-1);
}

void StudentTNormalization::normalize(vector<vector<double> > &dataset, unsigned int start, unsigned int end){
    if(dataset.size() == 0){
        cout << "empty dataset!" << endl;
        return;
    }
    if(start < 0 || end >= dataset.size()){
        cout << "index out of range" << endl;
        return;
    }
    unsigned int numInst = dataset.size();

    for(unsigned int i = start; i <= end; ++i){
        double mean = getMean(dataset, i);
        double std = getStandardDeviation(dataset, mean, i);

        if(std - 0 > 1E-6){
            for(unsigned int j = 0; j != numInst; ++j){
                dataset[j][i] = (dataset[j][i]-mean)/std;
            }
        }
    }
}

double StudentTNormalization::getMean(vector<vector<double> > &data, unsigned int colIndex){
    double sum = 0.0f;
    for(unsigned int i = 0; i != data.size(); ++i){
        sum += data[i][colIndex];
    }
    return sum/data.size();
}

double StudentTNormalization::getStandardDeviation(vector<vector<double> > &data, double mean, unsigned int colIndex){
    double var = 0.0f;
    for(unsigned int i = 0; i != data.size(); ++i){
        var += (data[i][colIndex] - mean) * (data[i][colIndex] - mean);
    }
    return sqrt(var);
}

ScalingNormalization::ScalingNormalization(){
    this->lower = 0.0f;
    this->upper = 1.0f;
}

void ScalingNormalization::setScale(double lower, double upper){
    if(lower > upper){
        cout << "lower bound must smaller than upper bound" << endl;
        return;
    }
    this->lower = lower;
    this->upper = upper;
}

void ScalingNormalization::normalize(vector<vector<double> > &dataset){
    if(dataset.size() == 0){
        cout << "empty dataset!" << endl;
        return;
    }
    unsigned int numAttr = dataset[0].size();
    normalize(dataset, 0, numAttr-1);

}

void ScalingNormalization::normalize(vector<vector<double> > &dataset, unsigned int start, unsigned int end){
    if(dataset.size() == 0){
        cout << "empty dataset!" << endl;
        return;
    }
    if(start < 0 || end >= dataset.size()){
        cout << "index out of range" << endl;
        return;
    }
    unsigned int numInst = dataset.size();

    for(unsigned int i = start; i <= end; ++i){
        double max = std::numeric_limits<double>::min();
        double min = std::numeric_limits<double>::max();

        for(unsigned int j = 0; j != numInst; ++j){
            if(max < dataset[j][i]) max = dataset[j][i];
            if(min > dataset[j][i]) min = dataset[j][i];
        }

        double scale = max - min;
        if(scale - 0 < 1E-6){
            for(unsigned int j = 0; j != numInst; ++j){
                dataset[j][i] = scale;
            }
        }
        else{
            for(unsigned int j = 0; j != numInst; ++j){
                dataset[j][i] = ((upper - lower)*(dataset[j][i]-min))/scale;
            }
        }
    }
}
