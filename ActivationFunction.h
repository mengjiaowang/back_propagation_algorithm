#ifndef ACTIVIATION_FUNCTION_H_
#define ACTIVIATION_FUNCTION_H_

class ActivationFunction{
    public:
        ActivationFunction();
        virtual double operator()(double x) = 0;
        virtual double derivative(double x) = 0;
    private:
};

class Sigmoid:public ActivationFunction{
    public:
        Sigmoid();
        double operator()(double x);
        double derivative(double x);
};

class Tanh:public ActivationFunction{
    public:
        Tanh();
        double operator()(double x);
        double derivative(double x);
};

#endif
