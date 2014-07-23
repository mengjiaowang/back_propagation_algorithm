#include "ActivationFunction.h"
#include <cmath>

ActivationFunction::ActivationFunction(){
}

Sigmoid::Sigmoid(){
}

double Sigmoid::operator()(double x){
    return 1/(1+exp(-1*x));
}

double Sigmoid::derivative(double x){
    return x*(1-x);
}

Tanh::Tanh(){
}

double Tanh::operator()(double x){
    return (exp(x)-exp(-1*x))/(exp(x)+exp(-1*x));
}

double Tanh::derivative(double x){
    return 1-(pow(exp(x)-exp(-1*x),2)/pow(exp(x)+exp(-1*x),2));
}
