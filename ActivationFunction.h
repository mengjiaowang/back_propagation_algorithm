#ifndef ACTIVIATION_FUNCTION_H_
#define ACTIVIATION_FUNCTION_H_

class ActivationFunction{
    public:
        ActivationFunction();
};

class Sigmoid: ActivationFunction{
    public:
        Sigmoid();
        double operator()(int x, int c);
};

class Tanh: ActivationFunction{
    public:
        Tanh();
        double operator()(int x, int c);
};

#endif
