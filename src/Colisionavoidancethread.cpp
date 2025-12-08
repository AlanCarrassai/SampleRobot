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
      int sumD = (sonar[3] * LIMIARFRENTE) + ((sonar[2] + sonar[1]) * LIMIARDIAGONAIS) + (sonar[0] * LIMIARLATERAIS); // 2
      int sumE = (sonar[4] * LIMIARFRENTE) + ((sonar[5] + sonar[6]) * LIMIARDIAGONAIS) + (sonar[7] * LIMIARLATERAIS); // 1
      int dirMov = 1;

      if (robo->robot.isHeadingDone())
      {
            std::cout << "A ultima rotacao foi concluida \n";
            if (sumD > sumE)
                  dirMov = 2;
            if (sonar[3] <= LIMIARFRENTE / 5 || sonar[4] <= LIMIARFRENTE / 5)
            {
                  robo->Move(-VELOCIDADEDESLOCAMENTO, -VELOCIDADEDESLOCAMENTO);
                  std::cout << "Frente perto \n";
            }
            else if (sonar[0] <= LIMIARLATERAIS)
            {
                  robo->Rotaciona(5, dirMov, VELOCIDADEROTACAO);
                  std::cout << "Esquerda perto \n";
            }
            else if (sonar[1] <= LIMIARDIAGONAIS || sonar[2] <= LIMIARDIAGONAIS)
            {
                  robo->Rotaciona(15, dirMov, VELOCIDADEROTACAO);
                  std::cout << "DDE perto \n";
            }
            else if (sonar[3] <= LIMIARFRENTE || sonar[4] <= LIMIARFRENTE)
            {
                  robo->Rotaciona(45, dirMov, VELOCIDADEROTACAO);
                  std::cout << "Frente afastado \n";
            }
            else if (sonar[5] <= LIMIARDIAGONAIS || sonar[6] <= LIMIARDIAGONAIS)
            {
                  robo->Rotaciona(15, dirMov, VELOCIDADEROTACAO);
                  std::cout << "DDD perto \n";
            }
            else if (sonar[7] <= LIMIARLATERAIS)
            {
                  robo->Rotaciona(5, dirMov, VELOCIDADEROTACAO);
                  std::cout << "Direita perto \n";
            }
            else
            {
                  robo->Move(VELOCIDADEDESLOCAMENTO, VELOCIDADEDESLOCAMENTO);
                  std::cout << "Seguir em frente \n";
            }
      }
      else
            std::cout << "Executando rotacao previa \n";
}
