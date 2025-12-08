#ifndef COLISIONAVOIDANCETHREAD_H
#define COLISIONAVOIDANCETHREAD_H
#include "Aria.h"
#include "ClassRobo.h"

class ColisionAvoidanceThread : public ArASyncTask
{
public:
    PioneerRobot *robo;
    ArCondition myCondition;
    ArMutex myMutex;
    int sonar[8];

public:
    ColisionAvoidanceThread(PioneerRobot *_robo);
    void *runThread(void *);
    void waitOnCondition();
    void lockMutex();
    void unlockMutex();
    void tratamentoSimples();
};

#endif // COLISIONAVOIDANCETHREAD_H
