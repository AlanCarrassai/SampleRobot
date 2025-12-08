#include "Colisionavoidancethread.h"
#include "Config.h"
#include <iostream>

ColisionAvoidanceThread::ColisionAvoidanceThread(PioneerRobot *_robo)
{
      this->robo = _robo;
}

void *ColisionAvoidanceThread::runThread(void *)
{
      while (this->getRunningWithLock())
      {
            myMutex.lock();
            robo->getAllSonar(sonar);
            tratamentoSimples();
            // ArUtil::sleep(1000);
            myMutex.unlock();
      }

      ArLog::log(ArLog::Normal, "Colision Avoidance.");
      return NULL;
}

void ColisionAvoidanceThread::waitOnCondition() { myCondition.wait(); }

void ColisionAvoidanceThread::lockMutex() { myMutex.lock(); }

void ColisionAvoidanceThread::unlockMutex() { myMutex.unlock(); }

void ColisionAvoidanceThread::tratamentoSimples()
{
    robo->Move(200, 200);
}
