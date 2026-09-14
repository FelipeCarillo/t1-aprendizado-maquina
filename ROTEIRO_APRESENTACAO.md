# Roteiro de apresentacao - T1

Tempo sugerido: 5 a 7 minutos. O professor conversa com cada grupo, portanto todos devem saber explicar o fluxo inteiro, mesmo com a divisao de falas abaixo.

## 1. Abertura - Edgar

"Nosso projeto compara tres algoritmos de classificacao para prever a situacao academica final de estudantes: evasao, permanencia no curso ou graduacao. Usamos a base Students Dropout and Academic Success, disponivel no Kaggle. Ela possui 4.424 registros, 36 variaveis preditivas e tres classes no alvo: Dropout, Enrolled e Graduate."

"Escolhemos essa base porque ela representa um problema real de classificacao multiclasse e tem atributos demograficos, financeiros e academicos."

## 2. Analise e preparo - Felipe

"Primeiro verificamos a qualidade dos dados. A versao usada nao possui valores ausentes nem linhas duplicadas. Mantivemos as 36 colunas como preditoras e a coluna Target como alvo."

"Algumas colunas possuem codigos numericos que representam categorias, como curso, nacionalidade e modalidade de candidatura. Para elas aplicamos one-hot encoding. As variaveis quantitativas foram mantidas como numericas."

"Fizemos uma divisao unica de 80% para treino e 20% para teste, com random_state igual a 42 e stratify no alvo. Assim a proporcao das tres classes e preservada e todos os modelos sao avaliados no mesmo teste."

## 3. Modelos comparados - Gabriel

"Comparamos Arvore de Decisao, KNN com k igual a 5 e SVM com kernel RBF. A Arvore de Decisao aprende regras e nao precisa de normalizacao. KNN usa a proximidade entre casos e SVM encontra fronteiras entre as classes; por isso normalizamos as variaveis quantitativas para esses dois modelos."

"Todo o preparo esta dentro de pipelines do scikit-learn. Isso significa que codificacao e normalizacao sao ajustadas apenas com o treino, evitando vazamento de informacao do teste."

## 4. Avaliacao - Isaías

"Para cada algoritmo calculamos matriz de confusao e classification report, que mostra precision, recall e F1-score por classe. Tambem comparamos acuracia e F1-macro."

"Usamos F1-macro como a metrica principal porque as classes nao possuem o mesmo tamanho. Ela calcula o F1 de cada classe e faz uma media dando o mesmo peso para todas, inclusive Enrolled, que e a menor classe."

"A Arvore de Decisao obteve acuracia de 66,8% e F1-macro de 60,2%. O KNN obteve 69,6% de acuracia e 60,6% de F1-macro."

## 5. Melhor resultado - Marcelo

"A SVM foi o melhor modelo: acuracia de 76,4% e F1-macro de 68,5%. Portanto, ela apresentou o melhor desempenho equilibrado entre as tres classes."

"Mesmo no melhor modelo, a classe Enrolled teve o menor recall, de aproximadamente 37%. Isso indica que o modelo ainda tem mais dificuldade para identificar estudantes que continuam matriculados, e a matriz de confusao permite enxergar essas trocas de classe."

"Em custo-beneficio, a Arvore de Decisao e mais simples de interpretar. Ja a SVM teve o melhor resultado, mas e menos direta de explicar. A escolha depende de priorizar desempenho ou interpretabilidade."

## 6. Novos exemplos e encerramento - Yuri

"Tambem criamos tres perfis hipoteticos completos, que nao existem na base: um aluno com desempenho forte, um perfil com risco de evasao e um perfil intermediario. Todos os modelos previram Graduate para o perfil forte e Dropout para o perfil de risco. Para o perfil intermediario, Arvore e SVM previram Enrolled, enquanto o KNN previu Graduate."

"Como limitacao, usamos informacoes do primeiro e do segundo semestre. Portanto, o modelo nao representa uma previsao feita apenas na matricula. Alem disso, ele encontra padroes estatisticos e nao deve ser usado sozinho para decidir sobre um estudante."

"Em resumo, cumprimos o fluxo completo: analise da base, preparacao, divisao estratificada, treinamento de tres classificadores, avaliacao e previsoes para novos casos."

## Perguntas provaveis

### Por que usaram estratificacao?

Porque a distribuicao das classes e desigual. Com `stratify=y`, treino e teste mantem proporcoes parecidas de Dropout, Enrolled e Graduate, tornando a avaliacao mais justa.

### Por que normalizar somente KNN e SVM?

KNN calcula distancias e SVM depende de escalas para construir fronteiras adequadas. Na Arvore, os cortes em atributos nao dependem da escala, entao normalizacao nao e necessaria.

### O que e vazamento de dados?

E usar, durante o treino ou no preparo, informacao que pertence ao teste. Para evitar isso, os transformadores foram colocados nos pipelines e ajustados somente com o conjunto de treino.

### Por que a SVM foi melhor?

Os atributos preparados permitem que a SVM encontre fronteiras nao lineares entre as classes. Neste teste, ela separou melhor os casos de Dropout e Graduate. Isso e uma conclusao para esta base e esta divisao, nao uma garantia para qualquer conjunto de dados.

### O que a matriz de confusao mostra?

As celulas da diagonal sao os acertos de cada classe. As demais celulas mostram quais classes foram confundidas pelo modelo. Ela complementa a acuracia, pois revela onde os erros acontecem.
