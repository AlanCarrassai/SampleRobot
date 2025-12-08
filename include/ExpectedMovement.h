#ifndef EXPECTEDMOVEMENT_H
#define EXPECTEDMOVEMENT_H


class ExpectedMovement {
  public:
  float DirecaoRotacao;
  float DirecaoMovimento;
  float AnguloRotacao;

  float DirecaoRotacaoProcessada;
  float DirecaoMovimentoProcessada;
  float AnguloRotacaoProcessado;

  ExpectedMovement(float _direcaoRotacao, float _direcaoMovimento, float _anguloRotacao);
  void ProcessarMovimento();
};


#endif
