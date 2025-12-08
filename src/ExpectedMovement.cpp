#include "ExpectedMovement.h"
#include "Config.h"

ExpectedMovement::ExpectedMovement(float _direcaoRotacao, float _direcaoMovimento, float _anguloRotacao)
{
    this->DirecaoRotacao = _direcaoRotacao;
    this->DirecaoMovimento = _direcaoMovimento;
    this->AnguloRotacao = _anguloRotacao;
}

void ExpectedMovement::ProcessarMovimento()
{
    if (this->DirecaoRotacao >= 0.125 && this->DirecaoRotacao < 0.375)//Direita
       this->DirecaoRotacaoProcessada = 2; //Girar para a direita
    else if (this->DirecaoRotacao >= 0.375 && this->DirecaoRotacao < 0.625)//Esquerda
       this->DirecaoRotacaoProcessada = 1; //Girar para a esquerda
    else if (this->DirecaoRotacao >= 0.625 && this->DirecaoRotacao < 0.875)//Frente
        this->DirecaoRotacaoProcessada = 0; //Não girar e seguir em frente
    else
        this->DirecaoRotacaoProcessada = -999;//Parar movimento

    if (this->AnguloRotacao  >= 0.1 && this->AnguloRotacao  < 0.3)//0 Seguir em frente
        this->AnguloRotacaoProcessado = 0;
    else if (this->AnguloRotacao >= 0.3 && this->AnguloRotacao < 0.5)//5 Lateral
        this->AnguloRotacaoProcessado = 5;
    else if (this->AnguloRotacao >= 0.5 && this->AnguloRotacao < 0.7)//15 Diagonal
        this->AnguloRotacaoProcessado = 15;
    else if (this->AnguloRotacao >= 0.7 && this->AnguloRotacao < 0.9)//45 Frontal
        this->AnguloRotacaoProcessado = 45;
    else
        this->AnguloRotacaoProcessado = -999;//Parar movimento 

    if (this->DirecaoMovimento >= 0.1 && this->DirecaoMovimento < 0.5)//Frente
        this->DirecaoMovimentoProcessada = VELOCIDADEDESLOCAMENTO;
    else if (this->DirecaoMovimento >= 0.5 && this->DirecaoMovimento < 0.9)//Re
        this->DirecaoMovimentoProcessada = -VELOCIDADEDESLOCAMENTO;
    else
        this->DirecaoMovimentoProcessada = -999;
        
}
