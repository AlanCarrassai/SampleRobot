#include "Sonarthread.h"
#include <iostream>

SonarThread::SonarThread(PioneerRobot *_robo)
{
  this->robo = _robo;
}

void *SonarThread::runThread(void *)
{
  while (this->getRunningWithLock())
  {
    myMutex.lock();

    robo->getAllSonar(sonar);
    printSonarReadings();

    myMutex.unlock();
    ArUtil::sleep(1000);
  }

  ArLog::log(ArLog::Normal, "Example thread: requested stop running, ending thread.");
  return NULL;
}

void SonarThread::waitOnCondition() { myCondition.wait(); }

void SonarThread::lockMutex() { myMutex.lock(); }

void SonarThread::unlockMutex() { myMutex.unlock(); }

int *SonarThread::getCurrentSonarReadings() { return sonar; }

void SonarThread::printSonarReadings()
{
  for (int i = 0; i < 8; i++)
    std::cout << "Sonar " << i << " " << sonar[i] << "\t";
  std::cout << "\n";
}
