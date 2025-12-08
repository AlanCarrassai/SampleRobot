#ifndef LASERTHREAD_H
#define LASERTHREAD_H
#include "Aria.h"
#include "ClassRobo.h"

class LaserThread : public ArASyncTask
{
public:
    PioneerRobot *robo;
    ArCondition myCondition;
    ArMutex myMutex;

public:
    LaserThread(PioneerRobot *_robo);
    void *runThread(void *);
    void waitOnCondition();
    void lockMutex();
    void unlockMutex();
};

#endif // LASERTHREAD_H
