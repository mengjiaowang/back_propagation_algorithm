#include "Normalization.h"
#include <vector>
#include <iostream>
using namespace std;

Normalization::Normalization(){
}

void Normalization::normalize(vector<vector<double> > &dataset){
    if(dataset.size() == 0){
        cout << "empty dataset!" << endl;
        return;
    }
    unsigned int numAttr = dataset[0].size();
    normalize(dataset, 0, numAttr-1);

}

void Normalization::normalize(vector<vector<double> > &dataset, int start, int end){
    if(dataset.size() == 0){
        cout << "empty dataset!" << endl;
        return;
    }
    if(start < 0 || end >= dataset.size()){
        cout << "index out of range" << endl;
        return;
    }
    unsigned int numInst = dataset.size();

    for(int i = start; i <= end; ++i){
        double max = std::numeric_limits<double>::min();
        double min = std::numeric_limits<double>::max();

        for(int j = 0; j != numInst; ++j){
            if(max < dataset[j][i]) max = dataset[j][i];
            if(min > dataset[j][i]) min = dataset[j][i];
        }

        double scale = max - min;
        if(scale - 0 < 1E-6){
            for(int j = 0; j != numInst; ++j){
                dataset[j][i] = scale;
            }    
        }
        else{
            for(int j = 0; j != numInst; ++j){
                dataset[j][i] = (dataset[j][i]-min)/scale;
            }
        }
    }
}
