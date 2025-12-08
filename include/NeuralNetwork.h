#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "Sigmoid.h"
#include "ExpectedMovement.h"


#define PadroesValidacao 56
#define PadroesTreinamento 56 
#define Sucesso 0.04		    // 0.0004
#define NumeroCiclos 100000     // Exibir o progresso do treinamento a cada NumeroCiclos ciclos

//Sigmoide
#define TaxaAprendizado 0.3     //0.3 converge super rápido e com uma boa precisão (sigmoide na oculta).
#define Momentum 0.9            // Dificulta a convergencia da rede em minimos locais, fazendo com que convirja apenas quando realmente se tratar de um valor realmente significante.
#define MaximoPesoInicial 0.5


//Saidas da rede neural (Exemplo): Vocês precisam definir os intervalos entre 0 e 1 para cada uma das saídas de um mesmo neuronio.
//Alem disso, esses sao exemplos, voces podem ter mais tipos de saidas, por exemplo defni que o robo ira rotacionar para a direita, esquerda ou nao rotacionar, no geral isso nao teria como alterar, entao deixei como exemplo.

//Direcao de rotacao (Neuronio da camada de saida 1)
//   Direita              Reto            Esquerda
//0.125 - 0.375      0.375 - 0.625      0.625 - 0.875
//    0,25				  0,5                0,75
#define OUT_DR_DIREITA    0.25    
#define OUT_DR_ESQUERDA   0.5   
#define OUT_DR_FRENTE     0.75

//Para a direcao de movimento nao ha muita diferenca, entao acredito que voces possam adotar esses valores
//Direcao de movimento (Neuronio da camada de saida 2)
//	  Frente		    Re
//   0.1 - 0.5      0.5 - 0.9
#define OUT_DM_FRENTE     0.3      
#define OUT_DM_RE         0.7

//O angulo nao possui receita de bolo, voces podem altera-lo em diferentes niveis, ou ate lidar com valores continuos
//Angulo de rotacao  (Neuronio da camada de saida 3)
#define OUT_AR_SEM_ROTACAO  0.2    //0
#define OUT_AR_LATERAL      0.4    //5
#define OUT_AR_DIAGONAL     0.6    //15
#define OUT_AR_FRONTAL      0.8    //45
//...

//Essa e uma sugestao, voces tambem podem trabalhar com a velocidade de movbvimento tambem sendo retornada pela rede neural, pois quanto mais proximo dos obstaculos, mais lento deveria ser o movimento
//Velocidade de movimento (Neuronio da camada de saida 4)

#define ALCANCE_MAX_SENSOR 5000

//Sobre o numero de neuronio das camadas, a camada de entrada ira refletir o numero de sensores, entao seriam esses 8. Se voces possuissem mais variaveis relevantes para essa operacao, poderiam utiliza-las. 
//Pensem que ate mesmo a velocidade de movimento atual do robo poderia ser utilizada como entrada para decidir no momento t+1
// Camada de entrada
#define NodosEntrada 8

//A quantidade de neuronios nessa camada esta fortemente vinculada a complexidade do problema, sendo uma boa pratica iniciar os esperimentos com pelo menos um neuronio a mais do que na camada de entrada.
// Camada oculta
#define NodosOcultos 9

//Essa camada ira definir a quantidade de diferentes variaveis de saida, nesse meu exemplo sao elas  direcao de rotacao (DR), direcao de movimento (DM) e angulo de rotacao (AR).
//Mas como eu disse no comentario acima, a rede poderia ter um quarto neuronio na camada de saida, para definir a velocidade de mopvimento do robo, ou ate outras saidas que voces condiderem importanes para a resolucao do problema.
// Camada de saída
#define NodosSaida 3

//Estrutura da rede neural, sintam-se livres para adicionar novas camadas intermediarias, alterar a funcao de ativacao, bias e etc.
class NeuralNetwork {
public:
    int i, j, p, q, r;
    int IntervaloTreinamentosPrintTela;
    int IndiceRandom[PadroesTreinamento];
    long CiclosDeTreinamento;
    float Rando;
    float Error;
    float AcumulaPeso;

    int esquerda = 0;
    int diagonal_esquerda_lateral = 0;
    int diagonal_esquerda_frontal = 0;
    int frente_esquerda = 0;
    int direita = 0;
    int diagonal_direita_lateral = 0;
    int diagonal_direita_frontal = 0;
    int frente_direita = 0;

    // Camada oculta
    float Oculto[NodosOcultos];
    float PesosCamadaOculta[NodosEntrada + 1][NodosOcultos];
    float OcultoDelta[NodosOcultos];
    float AlteracaoPesosOcultos[NodosEntrada + 1][NodosOcultos];
    ActivationFunction* activationFunctionCamadasOcultas;

    // Camada de saída
    float Saida[NodosSaida];
    float SaidaDelta[NodosSaida];
    float PesosSaida[NodosOcultos + 1][NodosSaida];
    float AlterarPesosSaida[NodosOcultos + 1][NodosSaida];
    ActivationFunction* activationFunctionCamadaSaida;

    float ValoresSensores[1][NodosEntrada] = {{0, 0, 0, 0, 0, 0, 0, 0}};

    //Exemplo de dadod de treinamento, cada um representando a distancia lida por um sensor
    const float Input[PadroesTreinamento][NodosEntrada] = {
    //ESQUERDA 							  FRETENE								  DIREITA
    // {0, 		1, 		2, 		3, 		4, 		5, 		6, 		7}                   {0,      0,       0,       0,       0,       0,       0,       0,},
        {5000,      5000,       5000,     5000,     5000,      5000,       5000,       5000},
        {650,      750,       1000,     4000,    5000,      5000,       5000,       5000},
        {700,      1000,       1500,     3700,     4600,      4800,       4750,       4900},
        {500,      700,       1300,     3400,     4000,      4400,       4650,       4700},
        {440,      600,       900,     3000,     3800,      4800,       4600,       4400},
        {580,      800,       1700,     3250,     3700,      5000,       4700,       5000},
        {600,      900,       1400,     5000,     4500,      4800,       5000,       5000},
        {750,      500,       900,     2800,     3000,      3700,       4400,       5000},
        {600,      750,       1400,     3500,     4300,      4400,       4500,       4600},
        {5000,      5000,       5000,     5000,     4700,      2000,       700,       600},
        {4500,      4300,       5000,     4600,     4000,      2200,       800,       500},
        {4800,      4900,       4600,     4000,     4100,      1900,       1000,       400},
        {4650,      4700,       4400,     3800,     3500,      1650,       650,       550},
        {5000,      4900,       4700,     4200,     3700,      1800,       750,       650},
        {3400,      3600,       3600,     2700,     2600,      1400,       500,       450},
        {4700,      4400,       4200,     3700,     3300,      1500,       900,       550},
        {4900,      4600,       4500,     3800,     4200,      4350,       600,       700},
        {3500,      3000,       2700,     1000,     900,      400,       500,       600},
        {4900,      4600,       4500,     900,     850,      2000,       1700,       1670},    
        {4000,      3800,       3000,     1300,     1000,      450,       600,       700},
        {3000,      2600,       2000,     450,     600,      1700,       1400,       1500},
        {1000,      1350,       1400,     700,     800,      3300,       4000,       5000},
        {500,      700,       1200,     1000,     900,      2700,       4000,       4200},
        {1500,      1350,       1200,     400,     500,      2800,       4700,       4850},
        {800,      900,       600,     650,     700,      1600,       3400,       3900},
        {700,      1700,       2600,     3600,     4100,      2800,       1500,       700},
        {1200,      1700,       2200,     4200,     3800,      2400,       1500,       1000},
        {2000,      2400,       2900,     3600,     3900,      3000,       2200,       1900},
        {1000,      1800,       2600,     5000,     4700,      3500,       2600,       2000},
        {600,      1000,       1400,     3800,     4000,      2700,       1900,       1500},
        {2000,      2500,       3000,     5000,     4900,      2200,       1300,       800},
        {1700,      2300,       2900,     4000,     4300,      1600,       1000,       600},
    };
    float InputNormalizado[PadroesTreinamento][NodosEntrada];

    //Exemplo de output esperado para os dados de treinamento acima
    const float Objetivo[PadroesTreinamento][NodosSaida] = {
    //   DR,  AR,  DM
        {OUT_DR_FRENTE, OUT_AR_SEM_ROTACAO, OUT_DM_FRENTE}, 
        {OUT_DR_DIREITA, OUT_AR_FRONTAL, OUT_DM_FRENTE},
        {OUT_DR_DIREITA, OUT_AR_FRONTAL, OUT_DM_FRENTE},
        {OUT_DR_DIREITA, OUT_AR_FRONTAL, OUT_DM_FRENTE},
        {OUT_DR_DIREITA, OUT_AR_FRONTAL, OUT_DM_FRENTE},
        {OUT_DR_DIREITA, OUT_AR_FRONTAL, OUT_DM_FRENTE},
        {OUT_DR_DIREITA, OUT_AR_FRONTAL, OUT_DM_FRENTE},
        {OUT_DR_DIREITA, OUT_AR_FRONTAL, OUT_DM_FRENTE},
        {OUT_DR_DIREITA, OUT_AR_FRONTAL, OUT_DM_FRENTE},
        {OUT_DR_ESQUERDA, OUT_AR_FRONTAL, OUT_DM_FRENTE},
        {OUT_DR_ESQUERDA, OUT_AR_FRONTAL, OUT_DM_FRENTE},
        {OUT_DR_ESQUERDA, OUT_AR_FRONTAL, OUT_DM_FRENTE},
        {OUT_DR_ESQUERDA, OUT_AR_FRONTAL, OUT_DM_FRENTE},
        {OUT_DR_ESQUERDA, OUT_AR_FRONTAL, OUT_DM_FRENTE},
        {OUT_DR_ESQUERDA, OUT_AR_FRONTAL, OUT_DM_FRENTE},
        {OUT_DR_ESQUERDA, OUT_AR_FRONTAL, OUT_DM_FRENTE},
        {OUT_DR_ESQUERDA, OUT_AR_FRONTAL, OUT_DM_FRENTE},
        {OUT_DR_ESQUERDA, OUT_AR_FRONTAL, OUT_DM_FRENTE},
        {OUT_DR_ESQUERDA, OUT_AR_FRONTAL, OUT_DM_FRENTE},
        {OUT_DR_ESQUERDA, OUT_AR_FRONTAL, OUT_DM_FRENTE},
        {OUT_DR_ESQUERDA, OUT_AR_FRONTAL, OUT_DM_FRENTE},
        {OUT_DR_DIREITA, OUT_AR_FRONTAL, OUT_DM_FRENTE},
        {OUT_DR_DIREITA, OUT_AR_FRONTAL, OUT_DM_FRENTE},
        {OUT_DR_DIREITA, OUT_AR_FRONTAL, OUT_DM_FRENTE},
        {OUT_DR_DIREITA, OUT_AR_FRONTAL, OUT_DM_FRENTE},
        {OUT_DR_FRENTE, OUT_AR_SEM_ROTACAO, OUT_DM_FRENTE},
        {OUT_DR_FRENTE, OUT_AR_SEM_ROTACAO, OUT_DM_FRENTE},
        {OUT_DR_FRENTE, OUT_AR_SEM_ROTACAO, OUT_DM_FRENTE},
        {OUT_DR_FRENTE, OUT_AR_SEM_ROTACAO, OUT_DM_FRENTE},
        {OUT_DR_FRENTE, OUT_AR_SEM_ROTACAO, OUT_DM_FRENTE},
        {OUT_DR_FRENTE, OUT_AR_SEM_ROTACAO, OUT_DM_FRENTE},
        {OUT_DR_FRENTE, OUT_AR_SEM_ROTACAO, OUT_DM_FRENTE},
    };
    
    //Aqui eu utilizei os mesmos valores, mas o correto sera definir dados de validacao diferentes daqueles apresentados a rede em seu treinamento, para garantir que ela nao tenha apenas "decorado" as respostas.
    //Dados de validação
    const float InputValidacao[PadroesValidacao][NodosEntrada] = {
        {4670,      4700,       4300,     4800,     4700,     4567,       4580,       4890},
        {500,      600,       800,     3800,    4700,      4650,       4800,       4650},
        {600,      800,       1200,     3400,     4400,      4500,       4650,       5000},
        {700,      750,       1400,     3200,     4200,      4500,       5000,       5000},
        {400,      550,       780,     2900,     3500,      4600,       4400,       4000}, 
        {450,      850,       1500,     3000,     3650,      4800,       4450,       4800},
        {650,      1000,       1550,     4700,     4400,      4600,       4800,       4800},
        {600,      450,       800,     2400,     2800,      3300,       4000,       4800},
        {550,      700,       1300,     3300,     4500,      4600,       4700,       4900},
        {4800,      4700,       4650,     4800,     4400,      2200,       800,       500},
        {4600,      4000,       4900,     4400,     3700,      1900,       700,       650},
        {4700,      5000,       4800,     3900,     4000,      1700,       900,       450},
        {4500,      4600,       4000,     3600,     3300,      1500,       600,       700},
        {4750,      4800,       4500,     4000,     3500,      1600,       650,       550},
        {3600,      3800,       3800,     2800,     2700,      1550,       560,       480},
        {4800,      4700,       4350,     3900,     3450,      1400,       750,       600},
        {4650,      4550,       4200,     3500,     4000,      4100,       550,       650},
        {3900,      3300,       3000,     1200,     1100,      500,       700,       900},
        {4650,      4550,       4200,     750,     600,      1650,       1300,       1200},
        {3700,      3500,       2400,     1000,     800,      400,       550,       650},
        {3200,      2800,       2200,     650,     900,      1900,       1600,       1700},
        {600,      1200,       1000,     500,     700,      3500,       4000,       4500},
        {400,      600,       1000,     800,     700,      2300,       3700,       4000},
        {1300,      1250,       1000,     350,     450,      2500,       4550,       4700},
        {600,      800,       500,     550,     600,      1500,       3300,       3700},
        {600,      1600,       2500,     3500,     4000,      2700,       1400,       650},
        {1100,      1600,       2100,     4000,     3700,      2300,       1400,       900},
        {1850,      2300,       3300,     4200,     4400,      3000,       2100,       1760},
        {900,      1700,       2500,     4800,     4600,      3300,       2400,       1800},
        {700,      1100,       1500,     3900,     4200,      2900,       2100,       1700},
        {2100,      2600,       3400,     4900,     4700,      2000,       1200,       850},
        {1800,      2400,       3000,     4100,     4400,      1700,       1100,       550},
    };
    float InputValidacaoNormalizado[PadroesValidacao][NodosEntrada];
    
    const float ObjetivoValidacao[PadroesValidacao][NodosSaida] = {
        {OUT_DR_FRENTE, OUT_AR_SEM_ROTACAO, OUT_DM_FRENTE}, 
        {OUT_DR_DIREITA, OUT_AR_FRONTAL, OUT_DM_FRENTE},
        {OUT_DR_DIREITA, OUT_AR_FRONTAL, OUT_DM_FRENTE},    
        {OUT_DR_DIREITA, OUT_AR_FRONTAL, OUT_DM_FRENTE},
        {OUT_DR_DIREITA, OUT_AR_FRONTAL, OUT_DM_FRENTE},
        {OUT_DR_DIREITA, OUT_AR_FRONTAL, OUT_DM_FRENTE},
        {OUT_DR_DIREITA, OUT_AR_FRONTAL, OUT_DM_FRENTE},
        {OUT_DR_DIREITA, OUT_AR_FRONTAL, OUT_DM_FRENTE},
        {OUT_DR_DIREITA, OUT_AR_FRONTAL, OUT_DM_FRENTE},
        {OUT_DR_ESQUERDA, OUT_AR_FRONTAL, OUT_DM_FRENTE},
        {OUT_DR_ESQUERDA, OUT_AR_FRONTAL, OUT_DM_FRENTE},
        {OUT_DR_ESQUERDA, OUT_AR_FRONTAL, OUT_DM_FRENTE},
        {OUT_DR_ESQUERDA, OUT_AR_FRONTAL, OUT_DM_FRENTE},
        {OUT_DR_ESQUERDA, OUT_AR_FRONTAL, OUT_DM_FRENTE},
        {OUT_DR_ESQUERDA, OUT_AR_FRONTAL, OUT_DM_FRENTE},
        {OUT_DR_ESQUERDA, OUT_AR_FRONTAL, OUT_DM_FRENTE},
        {OUT_DR_ESQUERDA, OUT_AR_FRONTAL, OUT_DM_FRENTE},
        {OUT_DR_ESQUERDA, OUT_AR_FRONTAL, OUT_DM_FRENTE},
        {OUT_DR_ESQUERDA, OUT_AR_FRONTAL, OUT_DM_FRENTE},
        {OUT_DR_ESQUERDA, OUT_AR_FRONTAL, OUT_DM_FRENTE},
        {OUT_DR_ESQUERDA, OUT_AR_FRONTAL, OUT_DM_FRENTE},
        {OUT_DR_DIREITA, OUT_AR_FRONTAL, OUT_DM_FRENTE},
        {OUT_DR_DIREITA, OUT_AR_FRONTAL, OUT_DM_FRENTE},
        {OUT_DR_DIREITA, OUT_AR_FRONTAL, OUT_DM_FRENTE},
        {OUT_DR_DIREITA, OUT_AR_FRONTAL, OUT_DM_FRENTE},
        {OUT_DR_FRENTE, OUT_AR_SEM_ROTACAO, OUT_DM_FRENTE},
        {OUT_DR_FRENTE, OUT_AR_SEM_ROTACAO, OUT_DM_FRENTE},
        {OUT_DR_FRENTE, OUT_AR_SEM_ROTACAO, OUT_DM_FRENTE},
        {OUT_DR_FRENTE, OUT_AR_SEM_ROTACAO, OUT_DM_FRENTE},
        {OUT_DR_FRENTE, OUT_AR_SEM_ROTACAO, OUT_DM_FRENTE},
        {OUT_DR_FRENTE, OUT_AR_SEM_ROTACAO, OUT_DM_FRENTE},
        {OUT_DR_FRENTE, OUT_AR_SEM_ROTACAO, OUT_DM_FRENTE},
    };
    
    //--

public:
    NeuralNetwork();
    void treinarRedeNeural();
    void inicializacaoPesos();
    int treinoInicialRede();
    void PrintarValores();
    ExpectedMovement testarValor();
    ExpectedMovement definirAcao(int sensor0, int sensor1, int sensor2, int sensor3, int sensor4, int sensor5, int sensor6, int sensor7);
    void validarRedeNeural();
    void treinarValidar();
    void normalizarEntradas();
    void setupCamadas() ;
};

#endif // NEURALNETWORK_H
