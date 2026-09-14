# T1 - Comparacao de classificadores

Projeto da disciplina CIC407 - Inteligencia Artificial e Aplicacoes.

O trabalho compara tres algoritmos de classificacao para prever a situacao academica de estudantes (`Dropout`, `Enrolled` ou `Graduate`) usando a base [Students Dropout and Academic Success](https://www.kaggle.com/datasets/missionjee/students-dropout-and-academic-success-dataset).

## Arquivos

- `T1_aprendizado_maquina.ipynb`: relatorio executavel com analise, graficos, metricas e conclusao.
- `T1_aprendizado_maquina.py`: versao Python equivalente ao notebook.
- `data/data.csv`: base usada na analise.
- `requirements.txt`: bibliotecas necessarias para reproduzir o projeto.

## Como executar

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook T1_aprendizado_maquina.ipynb
```

Tambem e possivel executar a versao em script:

```bash
python T1_aprendizado_maquina.py
```

## Resultados da execucao atual

Com divisao estratificada de 80% para treino e 20% para teste, usando `random_state=42`, a SVM com kernel RBF obteve o melhor resultado: acuracia de 76,4% e F1-macro de 68,5%.

## Integrantes

Preencher os nomes completos e RAs do grupo antes da entrega.
