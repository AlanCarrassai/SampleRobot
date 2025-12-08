#include "ColisionAvoidanceNeuralNetworkThread.h"
#include "Config.h"
#include <iostream>

ColisionAvoidanceNeuralNetworkThread::ColisionAvoidanceNeuralNetworkThread(PioneerRobot *_robo, NeuralNetwork *_neuralNetwork)
{
    this->robo = _robo;
    this->neuralNetwork = _neuralNetwork;
}

void* ColisionAvoidanceNeuralNetworkThread::runThread(void*)
{
    while(this->getRunningWithLock())
    {
        myMutex.lock();
        robo->getAllSonar(sonar);
        tratamentoRna();
        ArUtil::sleep(500); //Esse delay pode ser importante, principalmente dependendo da velocidade do computador em que o codigo estiver rodando, pois pode acabar empilhando muitas acoes para o robo realizar.
        myMutex.unlock();
    }

    ArLog::log(ArLog::Normal, "Colision Avoidance.");
    return NULL;
}

void ColisionAvoidanceNeuralNetworkThread::waitOnCondition() { myCondition.wait(); }

void ColisionAvoidanceNeuralNetworkThread::lockMutex() { myMutex.lock(); }

void ColisionAvoidanceNeuralNetworkThread::unlockMutex() { myMutex.unlock(); }

//Essa funcao pode ser alterada completamente, essa e apenas uma sugestao de como os dados de saida da rede poderiam ser convertidos em acoes do robo
void ColisionAvoidanceNeuralNetworkThread::tratamentoRna()
{
    if(robo->robot.isHeadingDone() && robo->robot.isMoveDone())//Esperar o robo terminar de executar a acao anterior antes de enviar um novo comando (voces podem remover esse teste ou so utilizar o teste pela rotacao, caso queiram que as interacoes sejam mais rapidas)
    {
        ExpectedMovement movement =  neuralNetwork->definirAcao(sonar[0], sonar[1], sonar[2], sonar[3], sonar[4], sonar[5], sonar[6], sonar[7]);
        printf("\nDirecaoRotacaoProcessada %f DirecaoMovimento %f AnguloRotacao %f", movement.DirecaoRotacaoProcessada, movement.DirecaoMovimento, movement.AnguloRotacao);
        movement.ProcessarMovimento();
        printf("\nDirecaoRotacao %f DirecaoMovimentoProcessada %f AnguloRotacaoProcessado %f", movement.DirecaoRotacao, movement.DirecaoMovimentoProcessada, movement.AnguloRotacaoProcessado);

        if(movement.DirecaoRotacaoProcessada == 999 || movement.AnguloRotacaoProcessado == 999 || movement.DirecaoMovimentoProcessada == 999)
        {
            robo->pararMovimento();
        }
        else
        {
            if(movement.DirecaoRotacaoProcessada == 0)//Não rotacionar
            {
                robo->Move(movement.DirecaoMovimentoProcessada, movement.DirecaoMovimentoProcessada);
            }
            else//rotacionar
            {
                 robo->Rotaciona(movement.AnguloRotacaoProcessado, movement.DirecaoMovimentoProcessada, VELOCIDADEROTACAO);
            }
        }
    }
}

