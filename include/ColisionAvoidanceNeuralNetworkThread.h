#ifndef COLISIONAVOIDANCENEURALNETWORKTHREAD_H
#define COLISIONAVOIDANCENEURALNETWORKTHREAD_H
#include "Aria.h"
#include "ClassRobo.h"
#include "NeuralNetwork.h"
#include "ExpectedMovement.h"

class ColisionAvoidanceNeuralNetworkThread : public ArASyncTask{
public:
    PioneerRobot *robo;
    ArCondition myCondition;
    ArMutex myMutex;
    int sonar[8];
    NeuralNetwork *neuralNetwork;

public:
ColisionAvoidanceNeuralNetworkThread(PioneerRobot *_robo, NeuralNetwork *_neuralNetwork);
    void* runThread(void*);
    void waitOnCondition();
    void lockMutex();
    void unlockMutex();
    void tratamentoRna();
};

#endif // COLISIONAVOIDANCENEURALNETWORKTHREAD_H
