# Projeto de Segmentacao de Clientes com Aprendizado Nao Supervisionado

## 1. Visao geral

Este projeto analisa uma base de clientes de uma campanha de marketing para identificar grupos com comportamentos semelhantes. O objetivo principal foi transformar dados demograficos, familiares, financeiros e de consumo em segmentos de clientes que possam apoiar decisoes de marketing mais direcionadas.

A analise foi desenvolvida no notebook `notebook/Unsupervised_Learning_Project.ipynb`, utilizando o arquivo de dados `data/marketing_campaign (2).csv`. Os dados brutos e os documentos da pasta `docs` sao ignorados pelo Git por conterem arquivos de dados e materiais de apoio do projeto.

O trabalho segue quatro perspectivas analiticas:

1. Analise descritiva: o que existe na base e como os clientes se distribuem.
2. Analise diagnostica: quais padroes e diferencas aparecem entre os grupos.
3. Analise preditiva: quais possibilidades de previsao podem ser construidas a partir dos segmentos.
4. Analise prescritiva: quais decisoes de negocio podem ser recomendadas com base nos resultados.

## 2. Problema de negocio

A empresa possui clientes com diferentes niveis de renda, gastos, composicao familiar, idade e resposta a promocoes. Tratar todos os clientes da mesma maneira pode reduzir a efetividade das campanhas.

A pergunta central do projeto foi:

> Quais perfis de clientes podem ser identificados a partir de suas caracteristicas e comportamentos de consumo, e como esses perfis podem orientar campanhas de marketing mais eficientes?

## 3. Estrutura do projeto

```text
.
├── data/
│   └── marketing_campaign (2).csv
├── docs/
│   └── documento de apoio da analise
├── notebook/
│   └── Unsupervised_Learning_Project.ipynb
├── .gitignore
└── README.md
```

Os arquivos CSV, a pasta `docs` e os arquivos Word nao sao versionados, conforme definido no `.gitignore`.

## 4. Tecnologias utilizadas

- Python
- pandas e NumPy para manipulacao dos dados
- Matplotlib e Seaborn para visualizacao
- scikit-learn para pre-processamento, PCA e clusterizacao
- Yellowbrick para avaliacao do metodo do cotovelo
- Jupyter Notebook para desenvolvimento e registro da analise

## 5. Tratamento e preparacao dos dados

O notebook executa as seguintes etapas:

### 5.1 Carregamento

A base e carregada como um arquivo separado por tabulacao:

```python
data = pd.read_csv("../data/marketing_campaign (2).csv", sep="\t")
```

### 5.2 Tratamento de valores ausentes

As linhas com valores ausentes sao removidas para evitar problemas nas etapas de transformacao e modelagem:

```python
data = data.dropna()
```

### 5.3 Conversao de datas

A coluna `Dt_Customer` e convertida para o formato de data. A partir dela, foi criada a variavel `Customer_For`, que representa o tempo desde o cadastro do cliente em relacao a data mais recente da base.

### 5.4 Engenharia de atributos

Foram criadas novas variaveis para tornar os perfis mais interpretaveis:

- `Age`: idade estimada a partir do ano de nascimento.
- `Spent`: soma dos gastos com vinhos, frutas, carnes, peixes, doces e produtos de ouro.
- `Living_Width`: classificacao da situacao familiar em `Alone` ou `Partner`.
- `Children`: soma de criancas e adolescentes no domicilio.
- `Family_Size`: tamanho estimado da familia.
- `Is_Parent`: indicador de existencia de filhos no domicilio.
- `Total_Promos`: quantidade de campanhas aceitas pelo cliente.

As variaveis de gastos tambem foram renomeadas para nomes mais curtos e claros, como `Wines`, `Fruits`, `Meat`, `Fish`, `Sweets` e `Gold`.

### 5.5 Remocao de colunas redundantes

Foram removidas colunas que nao seriam utilizadas diretamente na modelagem, incluindo identificadores, datas originais, variaveis constantes e campos redundantes.

### 5.6 Tratamento de valores extremos

Foram aplicados limites para reduzir a influencia de registros extremos de idade e renda:

- idade inferior a 90 anos;
- renda inferior a 600.000.

Esses limites devem ser revisados em uma aplicacao produtiva com base em regras de negocio e validacao estatistica.

### 5.7 Transformacao das variaveis

As variaveis categoricas foram convertidas para valores numericos com `LabelEncoder`. Depois, as variaveis utilizadas na segmentacao foram padronizadas com `StandardScaler`, garantindo que atributos em escalas diferentes nao dominassem o calculo das distancias.

As variaveis diretamente relacionadas a resposta de campanhas, reclamacoes e resposta final foram retiradas da base usada para criar os clusters:

- `AcceptedCmp1` a `AcceptedCmp5`;
- `Response`;
- `Complain`.

Essa separacao evita que o agrupamento seja definido diretamente pelo resultado das campanhas que posteriormente serao analisadas.

## 6. Reducao de dimensionalidade

Foi aplicado o PCA, ou Analise de Componentes Principais, para reduzir as variaveis padronizadas a tres componentes:

```python
pca = PCA(n_components=3)
pca_data = pca.fit_transform(scaled_ds)
```

Os componentes `PC1`, `PC2` e `PC3` foram utilizados para visualizar os clientes em um espaco tridimensional. Clientes proximos nesse espaco possuem perfis gerais mais semelhantes considerando as variaveis utilizadas.

O PCA facilita a visualizacao, mas os componentes nao devem ser interpretados como variaveis originais isoladas. Cada componente e uma combinacao de varias caracteristicas.

## 7. Definicao do numero de clusters

O metodo do cotovelo foi utilizado como apoio para avaliar diferentes quantidades de grupos. A analise registrada no notebook indicou quatro clusters como uma escolha pratica, pois a reducao da inercia se torna menos significativa depois desse ponto.

O metodo do cotovelo e uma referencia, nao uma prova definitiva de que quatro grupos sejam a unica solucao correta. A escolha tambem deve considerar estabilidade, interpretabilidade e utilidade para o negocio.

## 8. Segmentacao dos clientes

A segmentacao final foi realizada com clusterizacao hierarquica aglomerativa:

```python
AC = AgglomerativeClustering(n_clusters=4)
yhat_AC = AC.fit_predict(PCA_ds)
```

Os grupos foram adicionados ao conjunto de dados original e ao DataFrame do PCA:

```python
PCA_ds["Clusters"] = yhat_AC
data["Clusters"] = yhat_AC
```

Os numeros dos clusters sao identificadores tecnicos. O cluster 3 nao e necessariamente melhor que o cluster 0; a interpretacao deve ser feita a partir das caracteristicas medias e da distribuicao de cada grupo.

## 9. Analise descritiva: o que foi observado

A analise descritiva mostrou que:

- os clientes possuem diferentes niveis de renda e consumo;
- a maior parte dos clientes se concentra em faixas mais baixas ou intermediarias de gasto;
- existe uma relacao positiva entre renda e gasto total: clientes com maior renda tendem a apresentar maior consumo;
- a maioria dos clientes possui zero ou uma crianca pequena e zero ou um adolescente;
- a maior parte das familias possui entre um e tres membros;
- os clientes estao principalmente em faixas adultas e maduras de idade;
- a maior parte dos clientes nao aceitou nenhuma das cinco promocoes analisadas;
- a quantidade de compras realizadas em ofertas varia entre os grupos.

Os graficos de dispersao, boxenplot, countplot e KDE foram utilizados para examinar essas distribuicoes.

## 10. Analise diagnostica: diferencas entre os clusters

Com base nos graficos e nas distribuicoes observadas, os grupos apresentaram os seguintes perfis gerais:

### Cluster 0

Apresenta menor nivel de gasto e, em geral, menor renda. A maior parte dos clientes realiza poucas compras em ofertas. E um grupo de menor consumo relativo.

### Cluster 1

Apresenta comportamento intermediario em gasto, mas se destaca pela maior dispersao no numero de compras promocionais. Ha clientes com participacao mais frequente em ofertas.

### Cluster 2

Concentra os maiores niveis de gasto e uma faixa mais ampla de consumo. E o grupo com maior potencial de valor financeiro, embora tambem apresente variabilidade interna.

### Cluster 3

Apresenta baixo gasto na distribuicao observada. Apesar de aparecer associado a clientes de maior renda em alguns graficos de renda e gasto, os valores de consumo nao sao uniformemente altos. Por isso, esse grupo deve ser analisado com cuidado e nao deve ser classificado automaticamente como grupo de maior rentabilidade.

### Promocoes

A distribuicao de `Total_Promos` mostra predominancia do valor zero. Portanto, a maioria dos clientes nao aceitou nenhuma promocao. Como o grafico apresenta contagens absolutas, a comparacao entre clusters deve ser complementada por proporcoes dentro de cada cluster.

### Compras em ofertas

O cluster 1 apresenta maior dispersao em `NumDealsPurchases`, com alguns clientes realizando muitas compras promocionais. O cluster 2 apresenta menor numero tipico de compras em ofertas, apesar de possuir maior gasto total. Isso indica que gasto total e dependencia de promocoes nao sao a mesma coisa.

### Variaveis pessoais

As variaveis `Kidhome`, `Teenhome`, `Children`, `Family_Size`, `Is_Parent`, `Age`, `Education` e `Living_Width` ajudam a descrever os grupos, mas nao explicam isoladamente o gasto total. A renda, o historico de consumo e o comportamento de compras parecem ter maior poder de diferenciacao.

## 11. Analise preditiva: o que o projeto permite prever

O notebook atual nao treina um modelo supervisionado para prever uma nova resposta de campanha, valor de gasto ou probabilidade de compra. Portanto, o resultado preditivo deve ser entendido como uma oportunidade de extensao, e nao como uma entrega ja concluida.

A segmentacao produz variaveis que podem ser usadas em modelos futuros, por exemplo:

- probabilidade de aceitar uma nova campanha;
- probabilidade de realizar uma compra;
- valor esperado de gasto;
- probabilidade de responder a uma oferta promocional;
- risco de baixa participacao ou abandono.

Para implementar essa etapa, seria necessario separar treino e teste, definir uma variavel-alvo, treinar modelos supervisionados e avaliar metricas como acuracia, precisao, recall, F1, ROC-AUC ou erro medio, conforme o tipo de previsao.

Um cuidado importante e evitar vazamento de dados. Variaveis de resposta de campanhas so devem ser usadas como alvo ou como atributos historicos validos, respeitando a ordem temporal do problema.

## 12. Analise prescritiva: recomendacoes de negocio

A recomendacao final e substituir campanhas generalizadas por uma estrategia segmentada:

1. Priorizar o cluster 2 para acoes de retencao, relacionamento e ofertas de maior valor. Como esse grupo ja apresenta alto gasto, descontos amplos podem reduzir margem sem necessidade.
2. Desenvolver campanhas de conversao para o cluster 1, testando ofertas, comunicacao e canais que estimulem maior frequencia de compra.
3. Criar campanhas de ativacao para os clusters 0 e 3, com mensagens simples, ofertas de entrada e incentivos de baixo custo para aumentar o primeiro ou o proximo pedido.
4. Medir a taxa de aceitacao de promocoes por cluster, e nao apenas a contagem total de respostas.
5. Testar campanhas diferentes por meio de grupos de controle e experimentos A/B.
6. Avaliar retorno sobre investimento, margem, receita incremental e custo do incentivo antes de ampliar uma campanha.
7. Usar renda, gasto anterior, recencia e comportamento de compras para personalizar a comunicacao, sem tratar atributos pessoais como determinantes isolados.

A prescricao principal e:

> Investir mais na personalizacao das campanhas, preservando clientes de alto valor, convertendo clientes de consumo intermediario e ativando clientes de baixo consumo com incentivos controlados e avaliados por experimento.

## 13. Limitacoes e pontos de melhoria

- A clusterizacao e exploratoria e depende das variaveis, da escala e do algoritmo escolhidos.
- A rotulacao dos clusters e arbitraria e pode mudar quando o modelo for recalculado.
- Os graficos de KDE nao sao ideais para variaveis categoricas ou discretas, como `Education` e `Living_Width`; boxplots e tabelas de proporcao sao mais adequados.
- `Customer_For` deve ser armazenada explicitamente em dias ou meses para evitar interpretacoes incorretas da conversao numerica de `timedelta`.
- O limite de idade e renda precisa de justificativa de negocio e validacao de impacto.
- O projeto ainda nao mede a qualidade da clusterizacao com metricas como silhouette score, Calinski-Harabasz ou Davies-Bouldin.
- Nao foi implementada uma previsao supervisionada nem uma avaliacao temporal de campanhas.
- As conclusoes sobre rentabilidade devem ser confirmadas com receita, margem e custo das campanhas, pois gasto total nao e igual a lucro.

## 14. Proximos passos recomendados

1. Corrigir e documentar a unidade de `Customer_For`.
2. Calcular metricas de qualidade e estabilidade dos clusters.
3. Criar uma tabela de perfil com media, mediana, tamanho e proporcao de cada cluster.
4. Comparar taxas de promocao aceitas dentro de cada cluster.
5. Construir um modelo supervisionado para prever `Response` ou uma nova conversao.
6. Validar as recomendacoes com experimentos controlados.
7. Criar um painel com indicadores de receita, gasto, margem, resposta e retorno por cluster.

## 15. Conclusao

O projeto demonstrou que a base pode ser segmentada em quatro grupos com comportamentos distintos. O cluster 2 se destaca pelo maior gasto total, o cluster 1 apresenta comportamento intermediario e maior variacao em compras promocionais, enquanto os clusters 0 e 3 concentram clientes de menor consumo relativo.

A principal descoberta de negocio e que a adesao geral as promocoes e baixa e que os clientes nao devem receber a mesma estrategia. A recomendacao final e utilizar os clusters como ponto de partida para campanhas personalizadas, mas validar cada acao por meio de proporcoes, metricas financeiras e experimentos.

O resultado atual e uma segmentacao exploratoria com forte utilidade descritiva e diagnostica. Para completar o ciclo analitico, o proximo desenvolvimento deve incluir previsao supervisionada e medicao objetiva do impacto das decisoes recomendadas.
