#include "ActivationFunction.h"
#include <cmath>

ActivationFunction::ActivationFunction(){
}

Sigmoid::Sigmoid(){
}

double Sigmoid::operator()(int x){
    return 1/(1+exp(-1*x));
}

Tanh::Tanh(){
}

double Tanh::operator()(int x){
    return (exp(x)-exp(-1*x))/(exp(x)+exp(-1*x));
}
