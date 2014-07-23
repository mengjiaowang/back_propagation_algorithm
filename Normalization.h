#ifndef NORMALIZATION_H_
#define NORMALIZATION_H_

#include <vector>
using namespace std;

class Normalization{
    public:
        Normalization();
        void normalize(vector<vector<double> > &dataset);
        void normalize(vector<vector<double> > &dataset, int start, int end);
};

#endif
