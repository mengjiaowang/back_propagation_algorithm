#ifndef ACTIVIATION_FUNCTION_H_
#define ACTIVIATION_FUNCTION_H_

class ActivationFunction{
    public:
        ActivationFunction();
        virtual double operator()(int x) = 0;
        //virtual double derivative(int x) = 0;
    private:
};

class Sigmoid:public ActivationFunction{
    public:
        Sigmoid();
        double operator()(int x);
        //double derivative(int x);
};

class Tanh:public ActivationFunction{
    public:
        Tanh();
        double operator()(int x);
        //double derivative(int x);
};

#endif
