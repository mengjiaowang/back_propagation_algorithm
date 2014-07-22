#include <iostream>
#include "NeuralNetwork.h"
#include "BackPropagation.h"

using namespace std;

int main(){
    Sigmoid sigmoid;
    Tanh tanh;
    NeuralNetwork nn(3,4,2, sigmoid, tanh);
    BackPropagation bp(nn);
    return 0;
}
