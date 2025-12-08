#ifndef ACTIVATION_FUNCTION_H
#define ACTIVATION_FUNCTION_H

class ActivationFunction {
public:
    virtual ~ActivationFunction() {}

    // Função de ativação
    virtual float activate(float x) = 0;

    // Derivada da função de ativação (necessária para backpropagation)
    virtual float derivative(float x) = 0;
};

#endif
