#include "Wallfollowerthread.h"
#include "Config.h"
#include <iostream>

WallFollowerThread::WallFollowerThread(PioneerRobot *_robo)
{
    this->robo = _robo;
}

void *WallFollowerThread::runThread(void *)
{
    while (this->getRunningWithLock())
    {
        myMutex.lock();
        robo->getAllSonar(sonar);
        seguirParedeDSImples();
        myMutex.unlock();
    }

    ArLog::log(ArLog::Normal, "Colision Avoidance.");
    return NULL;
}

void WallFollowerThread::waitOnCondition() { myCondition.wait(); }

void WallFollowerThread::lockMutex() { myMutex.lock(); }

void WallFollowerThread::unlockMutex() { myMutex.unlock(); }

// So um teste para ver o quanto melhorara
float WallFollowerThread::Proporcional(float erro, float pGain)
{
    float pTerm = 0;
    pTerm = erro * pGain;
    // cout<<"pTerm: "<<pTerm<<" Erro: "<<erro<< "\n";
    return pTerm;
}

void WallFollowerThread::seguirParedeDSImples()
{
    float angulo;
    angulo = Proporcional(200 - sonar[7], 0.05);

    if (robo->robot.isHeadingDone() && robo->robot.isMoveDone())
    {
        // Uma re para nao beijar a parede
        if (sonar[3] <= LIMIARFRENTE / 5 || sonar[4] <= LIMIARFRENTE / 5)
        {
            robo->Move(-VELOCIDADEDESLOCAMENTO, -VELOCIDADEDESLOCAMENTO);
            std::cout << "Frente perto \n";
        }
        // Uma re para nao beijar a parede
        else if (sonar[2] <= LIMIARFRENTE / 4 || sonar[5] <= LIMIARFRENTE / 4)
        {
            robo->Move(-VELOCIDADEDESLOCAMENTO, -VELOCIDADEDESLOCAMENTO);
            std::cout << "DDD ou DDE perto \n";
        }
        // Parede em frente
        else if (sonar[3] < LIMITELEITURA / 4 && sonar[4] < LIMITELEITURA / 4)
        {
            std::cout << "Parede em frente \n";
            robo->Rotaciona(10, 1, VELOCIDADEROTACAO); // virar a esquerda
            parede_direita = 0;
        }
        // Procurar parede
        else if (parede_direita && sonar[7] == LIMITELEITURA)
        {
            std::cout << "Procurar parede \n";
            robo->Rotaciona(-15, 1, VELOCIDADEROTACAO); // virar a direita para procurar a parede
        }
        // Parede a direita
        else if (sonar[7] < LIMITELEITURA || sonar[6] < LIMITELEITURA || sonar[5] < LIMITELEITURA)
        {
            std::cout << "Direita: " << sonar[7] << " Parede: " << parede_direita << "\n";
            parede_direita = 1;
            std::cout << "Parede a direita \n";

            if (sonar[7] <= 200)
            {
                std::cout << "Correcao a esquerda 7\n";
                robo->Rotaciona(5, 1, VELOCIDADEROTACAO); // virar a esquerda
            }
            else if (sonar[6] <= 500)
            {
                std::cout << "Correcao a esquerda 6\n";
                robo->Rotaciona(5, 1, VELOCIDADEROTACAO); // virar a esquerda
            }
            else if (sonar[5] <= 700)
            {
                std::cout << "Correcao a esquerda 5\n";
                robo->Rotaciona(5, 1, VELOCIDADEROTACAO); // virar a esquerda
            }
            else if (sonar[3] <= 1000 || sonar[4] <= 1000)
            {
                std::cout << "Correcao a esquerda 3 4\n";
                robo->Rotaciona(5, 1, VELOCIDADEROTACAO); // virar a esquerda
            }
            else if (sonar[7] > 500)
            {
                std::cout << "Correcao a direita \n";
                robo->Rotaciona(-5, 1, VELOCIDADEROTACAO); // virar a direita
            }
            else
            {
                std::cout << "Parede a direita, mas seguir em frente \n";
                robo->Move(VELOCIDADEDESLOCAMENTO, VELOCIDADEDESLOCAMENTO); // seguir em frente
            }
        }
        // Quina
        else if ((sonar[3] <= LIMITELEITURA / 5 && sonar[4] <= LIMITELEITURA / 5) && sonar[7] < LIMITELEITURA / 5)
        {
            parede_direita = 1;
            std::cout << "Quina a direita \n";
            robo->Rotaciona(90, 1, VELOCIDADEROTACAO); // virar a esquerda
        }
        else
        {
            parede_direita = 0;
            std::cout << "Nenhuma parede detectada \n";
            robo->Move(VELOCIDADEDESLOCAMENTO, VELOCIDADEDESLOCAMENTO); // seguir em frente
        }
        //}
    }
}

void WallFollowerThread::seguirParedeDComP()
{
    float angulo = 0, velF = 0;

    angulo = Proporcional(200 - sonar[7], 0.05);
    velF = Proporcional(1000 - (sonar[3] + sonar[4]) / 2, 0.5);
    
    // distDDD_DDE = Proporcional(1250 - (sonar[2]+sonar[5])/2, 0.05);
    std::cout << "VelF: " << velF;

    // Procurar parede
    if (parede_direita && sonar[7] == LIMITELEITURA)
    {
        std::cout << "Procurar parede \n";
        robo->Rotaciona(-angulo, 1, VELOCIDADEROTACAO); // virar a direita para procurar a parede -15
    }
    // Parede a direita
    else if (sonar[7] < LIMITELEITURA)
    {
        std::cout << "Direita: " << sonar[7] << " Parede: " << parede_direita << "\n";
        parede_direita = 1;
        std::cout << "Parede a direita \n";
        robo->Rotaciona(angulo, 1, VELOCIDADEROTACAO);
    }
    // Quina
    else if ((sonar[3] <= LIMITELEITURA && sonar[4] <= LIMITELEITURA) && sonar[7] < LIMITELEITURA)
    {
        parede_direita = 1;
        std::cout << "Quina a direita \n";
        robo->Rotaciona(angulo, 1, VELOCIDADEROTACAO);
    }

    // if(robo->robot.isHeadingDone())
    //{
    if (((sonar[3] + sonar[4]) / 2) < LIMITELEITURA)
        robo->Move(velF * -1, velF * -1);
    else
        robo->Move(VELOCIDADEDESLOCAMENTO, VELOCIDADEDESLOCAMENTO);
    //}
}
