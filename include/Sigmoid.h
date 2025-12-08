#ifndef SIGMOID_H
#define SIGMOID_H

#include "ActivationFunction.h"
#include <cmath>

class Sigmoid : public ActivationFunction {
public:
    float activate(float x) override;

    float derivative(float x) override;
};

#endif