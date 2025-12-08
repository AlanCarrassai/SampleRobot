#include "NeuralNetwork.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <typeinfo>

NeuralNetwork::NeuralNetwork()
{
    srand(time(NULL));
    setupCamadas();
    normalizarEntradas();
    treinarValidar();
}

void NeuralNetwork::treinarValidar()
{
    treinarRedeNeural();
	validarRedeNeural();
}

void NeuralNetwork::setupCamadas() 
{
    activationFunctionCamadasOcultas = new Sigmoid();
    activationFunctionCamadaSaida = new Sigmoid();
}

void NeuralNetwork::validarRedeNeural() 
{
    float ErrorValidacao = 0.0;
    printf("\n--- Validação ---\n");

    for (int p = 0; p < PadroesValidacao; p++) 
    {
        for (i = 0; i < NodosOcultos; i++) 
        {
            AcumulaPeso = PesosCamadaOculta[NodosEntrada][i];

            for (j = 0; j < NodosEntrada; j++) 
            {
                AcumulaPeso += InputValidacaoNormalizado[p][j] * PesosCamadaOculta[j][i];
            }

            Oculto[i] = activationFunctionCamadasOcultas->activate(AcumulaPeso);
        }

        for (i = 0; i < NodosSaida; i++) 
        {
            AcumulaPeso = PesosSaida[NodosOcultos][i];

            for (j = 0; j < NodosOcultos; j++) 
            {
                AcumulaPeso += Oculto[j] * PesosSaida[j][i];
            }

            Saida[i] = activationFunctionCamadaSaida->activate(AcumulaPeso);

            ErrorValidacao += 0.5 * (ObjetivoValidacao[p][i] - Saida[i]) * (ObjetivoValidacao[p][i] - Saida[i]);
        }
        printf("Padrao %d | Objetivo: %.2f  %.2f  %.2f | Saida: %.2f  %.2f  %.2f\n", p, ObjetivoValidacao[p][0], ObjetivoValidacao[p][1], ObjetivoValidacao[p][2], Saida[0], Saida[1], Saida[2]);
    }

    printf("Erro de validacao: %f\n", ErrorValidacao);
}

void NeuralNetwork::normalizarEntradas()
{
    // normaliza para [0, 1]
    for (int p = 0; p < PadroesTreinamento; p++) 
    {
        for (int j = 0; j < NodosEntrada; j++) 
        {
            InputNormalizado[p][j] = Input[p][j] / (float)ALCANCE_MAX_SENSOR;
            printf("  Entrada[%d][%d]: original = %.2f  ->  normalizado = %.4f\n", 
                p, j, Input[p][j], InputNormalizado[p][j]);
        }
    }

    for (int p = 0; p < PadroesValidacao; p++) 
    {
        for (int j = 0; j < NodosEntrada; j++) 
        {
            InputValidacaoNormalizado[p][j] = InputValidacao[p][j] / (float)ALCANCE_MAX_SENSOR;
            printf("  Validação[%d][%d]: original = %.2f  ->  normalizado = %.4f\n", 
                p, j, InputValidacao[p][j], InputValidacaoNormalizado[p][j]);
        }
    }

   printf("encerrou");
}


void NeuralNetwork::treinarRedeNeural()
{
	int numeroTreinamentosExecutados = 0;
	int p = 0;

    printf("\nTreinar Rede Neural: \n\n");
	for (p = 0; p < PadroesTreinamento; p++)
		IndiceRandom[p] = p;

	numeroTreinamentosExecutados = treinoInicialRede(); // Efetua o treinamento
	if (numeroTreinamentosExecutados > 0)					 // Se "numeroTreinamentosExecutados" for maior do que zero, indica que a rede executou "numeroTreinamentosExecutados" treinamentos, para aprender o padrão.
		printf("\n\n\nA rede foi treinada com sucesso, apos %d trinamentos", numeroTreinamentosExecutados);
}

void NeuralNetwork::inicializacaoPesos() {
    srand(time(NULL));

    for (i = 0; i < NodosEntrada + 1; i++) {
        for (j = 0; j < NodosOcultos; j++) {
            float scale;

            scale = MaximoPesoInicial;
            
            float Rando = (float)rand() / (float)RAND_MAX;
            PesosCamadaOculta[i][j] = (Rando * 2.0f - 1.0f) * scale;
            AlteracaoPesosOcultos[i][j] = 0.0f;
        }
    }

    for (i = 0; i < NodosOcultos + 1; i++) {
        for (j = 0; j < NodosSaida; j++) {
            float scale;

            scale = MaximoPesoInicial;
            
            float Rando = (float)rand() / (float)RAND_MAX;
            PesosSaida[i][j] = (Rando * 2.0f - 1.0f) * scale;
            AlterarPesosSaida[i][j] = 0.0f;
        }
    }
}

int NeuralNetwork::treinoInicialRede()
{
	IntervaloTreinamentosPrintTela = NumeroCiclos;
	inicializacaoPesos();

	// Inicio do treinamento
	for (CiclosDeTreinamento = 1; CiclosDeTreinamento < 2147483647; CiclosDeTreinamento++)
	{
		// Randomizar a ordem dos treinamentos, evitando que caia nas regioes de minimos comuns
		for (p = 0; p < PadroesTreinamento; p++)
		{
			q = rand() % PadroesTreinamento;
			r = IndiceRandom[p];
			IndiceRandom[p] = IndiceRandom[q];
			IndiceRandom[q] = r;
		}
		Error = 0.0;
		// Navega entre todos os padroes de treinamento
		for (q = 0; q < PadroesTreinamento; q++)
		{
			p = IndiceRandom[q];

			// Processa a ativacoes da(s) camada(s) oculta(s) - Por hora ha apenas uma camada oculta
            float AcumulaPesoLinear[NodosOcultos];
			for (i = 0; i < NodosOcultos; i++)
			{
				AcumulaPesoLinear[i] = PesosCamadaOculta[NodosEntrada][i]; //bias
				for (j = 0; j < NodosEntrada; j++)
				{
                    AcumulaPesoLinear[i] += InputNormalizado[p][j] * PesosCamadaOculta[j][i];
				}
                Oculto[i] = activationFunctionCamadasOcultas->activate(AcumulaPesoLinear[i]); //Função de ativação
			}

			// Processa a ativacoes dos neuronios da camada de saida e calcula o erro
			for (i = 0; i < NodosSaida; i++)
			{
				AcumulaPeso = PesosSaida[NodosOcultos][i];
				for (j = 0; j < NodosOcultos; j++)
				{
					AcumulaPeso += Oculto[j] * PesosSaida[j][i];
				}
                Saida[i] = activationFunctionCamadaSaida->activate(AcumulaPeso);  //Função de ativação
                SaidaDelta[i] = (Objetivo[p][i] - Saida[i]) * activationFunctionCamadaSaida->derivative(AcumulaPeso); //Calculo do erro
				Error += 0.5 * (Objetivo[p][i] - Saida[i]) * (Objetivo[p][i] - Saida[i]);
			}

			// Backpropagation - Propaga o erro para as camadas anteriores
			for (i = 0; i < NodosOcultos; i++)
			{
                float erroPropagado = 0.0f;
				for (j = 0; j < NodosSaida; j++)
				{
                    erroPropagado += PesosSaida[i][j] * SaidaDelta[j];
				}
                OcultoDelta[i] = erroPropagado * activationFunctionCamadasOcultas->derivative(AcumulaPesoLinear[i]);// Funcao Delta/Regra Delta 
			}

			// Atualiza os pesos, indo da entrada para a camada oculta
			for (i = 0; i < NodosOcultos; i++)
			{
				AlteracaoPesosOcultos[NodosEntrada][i] = TaxaAprendizado * OcultoDelta[i] + Momentum * AlteracaoPesosOcultos[NodosEntrada][i];
				PesosCamadaOculta[NodosEntrada][i] += AlteracaoPesosOcultos[NodosEntrada][i];
				for (j = 0; j < NodosEntrada; j++)
				{
					AlteracaoPesosOcultos[j][i] = TaxaAprendizado * InputNormalizado[p][j] * OcultoDelta[i] + Momentum * AlteracaoPesosOcultos[j][i];
					PesosCamadaOculta[j][i] += AlteracaoPesosOcultos[j][i];
				}
			}

			// Atualiza os pesos, indo da camada oculta para a saida
			for (i = 0; i < NodosSaida; i++)
			{
				AlterarPesosSaida[NodosOcultos][i] = TaxaAprendizado * SaidaDelta[i] + Momentum * AlterarPesosSaida[NodosOcultos][i];
				PesosSaida[NodosOcultos][i] += AlterarPesosSaida[NodosOcultos][i];
				for (j = 0; j < NodosOcultos; j++)
				{
					AlterarPesosSaida[j][i] = TaxaAprendizado * Oculto[j] * SaidaDelta[i] + Momentum * AlterarPesosSaida[j][i];
					PesosSaida[j][i] += AlterarPesosSaida[j][i];
				}
			}
		}

		// Exibir na tela o estatus do treinamento
		IntervaloTreinamentosPrintTela = IntervaloTreinamentosPrintTela - 1;
		if (IntervaloTreinamentosPrintTela == 0)
		{
			printf("\nCiclosDeTreinamento: %ld  Error: %f \n\n", CiclosDeTreinamento, Error);
			PrintarValores();

			if (CiclosDeTreinamento == 1)
				IntervaloTreinamentosPrintTela = NumeroCiclos - 1;
			else
				IntervaloTreinamentosPrintTela = NumeroCiclos;
		}

		// Encerra o treinamento caso o erro seja menor do que o erro considerado aceitavel (configurado)
        printf("\nCiclosDeTreinamento: %ld  Error: %f  Sucesso: %f\n\n", CiclosDeTreinamento, Error, Sucesso);
		if (Error < Sucesso)
        {
            printf("Treinamento encerrado com sucesso, segue o resultado prelimira:");
            PrintarValores();
			return CiclosDeTreinamento;
        }
	}

	printf("\nCiclosDeTreinamento: %ld  Error: %f \n\n", CiclosDeTreinamento, Error);
	PrintarValores();
	printf("Treinamento concluido! \n\n");
	IntervaloTreinamentosPrintTela = 1;
}

void NeuralNetwork::PrintarValores()
{
	for (p = 0; p < PadroesTreinamento; p++)
	{
		printf("  Padrao de treinamento: %d \n", p);
		printf("  Entrada: ");
		for (i = 0; i < NodosEntrada; i++)
			printf(" %f  ", InputNormalizado[p][i]);

		printf("\n  Objetivo: ");
		for (i = 0; i < NodosSaida; i++)
			printf(" %f ", Objetivo[p][i]);

		// Computar as ativacoes da camada oculta
		for (i = 0; i < NodosOcultos; i++)
		{
			AcumulaPeso = PesosCamadaOculta[NodosEntrada][i];
			for (j = 0; j < NodosEntrada; j++)
			{
				AcumulaPeso += InputNormalizado[p][j] * PesosCamadaOculta[j][i];
			}

            Oculto[i] = activationFunctionCamadasOcultas->activate(AcumulaPeso); //Funcao de ativacao
		}

		// Computar as ativacoes da camada de saida e calcular os seus erros
		for (i = 0; i < NodosSaida; i++)
		{
			AcumulaPeso = PesosSaida[NodosOcultos][i];
			for (j = 0; j < NodosOcultos; j++)
			{
				AcumulaPeso += Oculto[j] * PesosSaida[j][i];
			}
			
            Saida[i] = activationFunctionCamadaSaida->activate(AcumulaPeso); //Funcao de ativacao
		}

		printf("\n  Saida: ");
		for (i = 0; i < NodosSaida; i++)
			printf(" %f ", Saida[i]);
	}
}

ExpectedMovement NeuralNetwork::testarValor()
{
    float direcaoRotacao, direcaoMovimento, anguloRotacao;

	printf("  Testando valores: \n");
	printf("  Training Pattern: \n");
	printf("  Input \n");
	for (i = 0; i < NodosEntrada; i++)
		printf(" %f ", ValoresSensores[0][i]);

	for (p = 0; p < 1; p++)
	{
		//Computar as ativacoes da camada oculta
		for (i = 0; i < NodosOcultos; i++)
		{
			AcumulaPeso = PesosCamadaOculta[NodosEntrada][i];
			for (j = 0; j < NodosEntrada; j++)
			{
				AcumulaPeso += ValoresSensores[p][j] * PesosCamadaOculta[j][i];
			}

            Oculto[i] = activationFunctionCamadasOcultas->activate(AcumulaPeso); //Funcao de ativacao
		}

		//Computar as ativacoes da camada de saida e calcular os seus erros
		for (i = 0; i < NodosSaida; i++)
		{
			AcumulaPeso = PesosSaida[NodosOcultos][i];
			for (j = 0; j < NodosOcultos; j++)
			{
				AcumulaPeso += Oculto[j] * PesosSaida[j][i];
			}

            Saida[i] = activationFunctionCamadaSaida->activate(AcumulaPeso); //Funcao de ativacao
		}

		printf("  Saida \n");
		for (i = 0; i < NodosSaida; i++)
			printf(" %f ", Saida[i]);

		printf("\n\n");

		printf("Direcao da Rotacao: \n");
		if (Saida[0] >= 0.125 && Saida[0] < 0.375)
			printf("DIREITA\n\n");

		else if (Saida[0] >= 0.375 && Saida[0] < 0.625)
			printf("ESQUERDA\n\n");
		else if (Saida[0] >= 0.625 && Saida[0] < 0.875)
			printf("FRENTE\n\n");
		else
			printf("VALOR INDEFINIDO\n\n");

		printf("Angulo de Rotacao: \n");
        if (Saida[1] >= 0.1 && Saida[1] < 0.3)
			printf("SEM ROTACAO 0 GRAUS\n\n");
		else if (Saida[1] >= 0.3 && Saida[1] < 0.5)
			printf("LATERAL 5 GRAUS\n\n");
		else if (Saida[1] >= 0.5 && Saida[1] < 0.7)
			printf("DIAGONAL 15 GRAUS\n\n");
		else if (Saida[1] >= 0.7 && Saida[1] < 0.9)
			printf("FRONTAL 45 GRAUS\n\n");
		else
			printf("VALOR INDEFINIDO\n\n");

		printf("Direcao de movimento: \n");
		if (Saida[2] >= 0.1 && Saida[2] < 0.5)
			printf("FRENTE\n\n");
		else if (Saida[2] >= 0.5 && Saida[2] < 0.9)
			printf("RE\n\n");
		else
			printf("VALOR INDEFINIDO\n\n");

		printf(" %f ", Saida[0]);
	}

    return ExpectedMovement(Saida[0], Saida[2], Saida[1]);
}

ExpectedMovement NeuralNetwork::definirAcao(int sensor0, int sensor1, int sensor2, int sensor3, int sensor4, int sensor5, int sensor6, int sensor7)
{
    ValoresSensores[0][0] = (float)sensor0 / ALCANCE_MAX_SENSOR;
    ValoresSensores[0][1] = (float)sensor1 / ALCANCE_MAX_SENSOR;
    ValoresSensores[0][2] = (float)sensor2 / ALCANCE_MAX_SENSOR;
    ValoresSensores[0][3] = (float)sensor3 / ALCANCE_MAX_SENSOR;
    ValoresSensores[0][4] = (float)sensor4 / ALCANCE_MAX_SENSOR;
    ValoresSensores[0][5] = (float)sensor5 / ALCANCE_MAX_SENSOR;
    ValoresSensores[0][6] = (float)sensor6 / ALCANCE_MAX_SENSOR;
    ValoresSensores[0][7] = (float)sensor7 / ALCANCE_MAX_SENSOR;

	printf(" sensor0: %d  ", sensor0);
	printf(" sensor1: %d  ", sensor1);
	printf(" sensor2: %d  ", sensor2);
	printf(" sensor3: %d  ", sensor3);
	printf(" sensor4: %d  ", sensor4);
	printf(" sensor5: %d  ", sensor5);
	printf(" sensor6: %d  ", sensor6);
	printf(" sensor7: %d  ", sensor7);

    printf("Valores normalizados dos sensores:\n");
    for (int i = 0; i < 8; i++)
        printf(" Sensor%d: %.4f ", i, ValoresSensores[0][i]);

	return testarValor();
}