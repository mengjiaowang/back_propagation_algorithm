#include "ActivationFunction.h"
#include <cmath>

ActivationFunction::ActivationFunction(){
}

Sigmoid::Sigmoid(){
}

double Sigmoid::operator()(int x){
    return 1/(1+exp(-1*x));
}

double Sigmoid::derivative(int x){
    return x*(1-x);
}

Tanh::Tanh(){
}

double Tanh::operator()(int x){
    return (exp(x)-exp(-1*x))/(exp(x)+exp(-1*x));
}

double Tanh::derivative(int x){
    return 1-(pow(exp(x)-exp(-1*x),2)/pow(exp(x)+exp(-1*x),2));
}
