#ifndef WALLFOLLOWERTHREAD_H
#define WALLFOLLOWERTHREAD_H
#include "Aria.h"
#include "ClassRobo.h"

class WallFollowerThread : public ArASyncTask
{
public:
    PioneerRobot *robo;
    ArCondition myCondition;
    ArMutex myMutex;
    int sonar[8];
    int parede_direita = 0;

public:
    WallFollowerThread(PioneerRobot *_robo);
    void *runThread(void *);
    void waitOnCondition();
    void lockMutex();
    void unlockMutex();
    float Proporcional(float erro, float pGain);
    void seguirParedeDSImples();
    void seguirParedeDComP();
};

#endif // WALLFOLLOWERTHREAD_H
