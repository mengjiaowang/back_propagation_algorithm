#include <iostream>
#include "NeuralNetwork.h"
#include "BackPropagation.h"

using namespace std;

int main(){
    NeuralNetwork nn(3,4,2);
    BackPropagation bp(nn);
    return 0;
}
