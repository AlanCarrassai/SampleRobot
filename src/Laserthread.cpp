#include "Laserthread.h"
#include <iostream>

LaserThread::LaserThread(PioneerRobot *_robo)
{
    this->robo = _robo;
}

void *LaserThread::runThread(void *)
{
    while (this->getRunningWithLock())
    {
        myMutex.lock();
        robo->getLaser();
        myMutex.unlock();
        ArUtil::sleep(2000);
    }

    ArLog::log(ArLog::Normal, "Example thread: requested stop running, ending thread.");
    return NULL;
}

void LaserThread::waitOnCondition() { myCondition.wait(); }

void LaserThread::lockMutex() { myMutex.lock(); }

void LaserThread::unlockMutex() { myMutex.unlock(); }
