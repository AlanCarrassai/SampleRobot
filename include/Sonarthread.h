#ifndef SONARTHREAD_H
#define SONARTHREAD_H
#include "Aria.h"
#include "ClassRobo.h"

class SonarThread : public ArASyncTask
{
public:
    PioneerRobot *robo;
    ArCondition myCondition;
    ArMutex myMutex;
    int sonar[8];

public:
    SonarThread(PioneerRobot *_robo);
    void *runThread(void *);
    void waitOnCondition();
    void lockMutex();
    void unlockMutex();
    int *getCurrentSonarReadings();
    void printSonarReadings();
};

#endif // SONARTHREAD_H
