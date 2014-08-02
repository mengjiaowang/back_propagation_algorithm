#ifndef NORMALIZATION_H_
#define NORMALIZATION_H_

#include <vector>
using namespace std;

class NormalizationBase{
    public:
        NormalizationBase();
        virtual void normalize(vector<vector<double> > &dataset)=0;
        virtual void normalize(vector<vector<double> > &dataset, unsigned int start, unsigned int end)=0;
};

class StudentTNormalization: public NormalizationBase{
    public:
        StudentTNormalization();
        void normalize(vector<vector<double> > &dataset);
        void normalize(vector<vector<double> > &dataset, unsigned int start, unsigned int end);
    private:
        double getMean(vector<vector<double> > &data, unsigned int columnIndex);
        double getStandardDeviation(vector<vector<double> > &data, double mean, unsigned int columnIndex);
};

class ScalingNormalization: public NormalizationBase{
    public:
        ScalingNormalization();
        void normalize(vector<vector<double> > &dataset);
        void normalize(vector<vector<double> > &dataset, unsigned int start, unsigned int end);
        void setScale(double lower, double upper);
    private:
        double lower;
        double upper;

};
#endif
