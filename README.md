# Projeto de Segmentação de Clientes com Aprendizado Não Supervisionado

## 1. Visão geral

Este projeto analisa uma base de clientes de uma campanha de marketing para identificar grupos com comportamentos semelhantes. O objetivo principal foi transformar dados demográficos, familiares e financeiros em segmentos interpretáveis e acionáveis.

A análise foi desenvolvida no notebook `notebook/Unsupervised_Learning_Project.ipynb`, utilizando o arquivo de dados `data/marketing_campaign (2).csv`. Os dados brutos e os documentos da pasta `docs` não são versionados.

O trabalho segue quatro perspectivas analíticas:

1. **Análise descritiva**: o que existe na base e como os clientes se distribuem.
2. **Análise diagnóstica**: quais padrões e diferenças aparecem entre os grupos.
3. **Análise preditiva**: quais possibilidades de previsão podem ser construídas a partir dos segmentos.
4. **Análise prescritiva**: quais decisões de negócio podem ser recomendadas com base nos resultados.

## 2. Problema de negócio

A empresa possui clientes com diferentes níveis de renda, gastos, composição familiar, idade e resposta a promoções. Tratar todos os clientes da mesma maneira pode reduzir a efetividade das campanhas.

A pergunta central do projeto foi:

> Quais perfis de clientes podem ser identificados a partir de suas características e comportamentos de consumo, e como esses perfis podem orientar campanhas de marketing mais eficientes?

## 3. Estrutura do projeto

```text
.
├── data/
│   └── marketing_campaign (2).csv
├── docs/
│   └── documento de apoio da análise
├── notebook/
│   └── Unsupervised_Learning_Project.ipynb
├── .gitignore
└── README.md
```

Os arquivos CSV, a pasta `docs` e os arquivos Word não são versionados, conforme definido no `.gitignore`.

## 4. Tecnologias utilizadas

- Python
- pandas e NumPy para manipulação dos dados
- Matplotlib e Seaborn para visualização
- scikit-learn para pré-processamento, PCA e clusterização
- Yellowbrick para avaliação do método do cotovelo
- Jupyter Notebook para desenvolvimento e registro da análise

## 5. Tratamento e preparação dos dados

O notebook executa as seguintes etapas:

### 5.1 Carregamento

A base é carregada como um arquivo separado por tabulação:

```python
data = pd.read_csv("../data/marketing_campaign (2).csv", sep="\t")
```

### 5.2 Tratamento de valores ausentes

As linhas com valores ausentes são removidas para evitar problemas nas etapas de transformação e modelagem:

```python
data = data.dropna()
```

### 5.3 Conversão de datas

A coluna `Dt_Customer` é convertida para o formato de data. A partir dela, foi criada a variável `Customer_For`, que representa o tempo desde o cadastro do cliente em relação à data mais recente da base.

### 5.4 Engenharia de atributos

Foram criadas novas variáveis para tornar os perfis mais interpretáveis:

- `Age`: idade estimada a partir do ano de nascimento.
- `Spent`: soma dos gastos com vinhos, frutas, carnes, peixes, doces e produtos de ouro.
- `Living_Width`: classificação da situação familiar em `Alone` ou `Partner`.
- `Children`: soma de crianças e adolescentes no domicílio.
- `Family_Size`: tamanho estimado da família.
- `Is_Parent`: indicador de existência de filhos no domicílio.
- `Total_Promos`: quantidade de campanhas aceitas pelo cliente.

As variáveis de gastos também foram renomeadas para nomes mais curtos e claros, como `Wines`, `Fruits`, `Meat`, `Fish`, `Sweets` e `Gold`.

### 5.5 Remoção de colunas redundantes

Foram removidas colunas que não seriam utilizadas diretamente na modelagem, incluindo identificadores, datas originais, variáveis constantes e campos redundantes.

### 5.6 Tratamento de valores extremos

Foram aplicados limites para reduzir a influência de registros extremos de idade e renda:

- idade inferior a 90 anos;
- renda inferior a 600.000.

Esses limites devem ser revisados em uma aplicação produtiva com base em regras de negócio e validação estatística.

### 5.7 Transformação das variáveis

As variáveis categóricas foram convertidas para valores numéricos com `LabelEncoder`. Depois, as variáveis utilizadas na segmentação foram padronizadas com `StandardScaler`, garantindo que atributos com escalas diferentes tenham peso equivalente no agrupamento.

As variáveis diretamente relacionadas à resposta de campanhas, reclamações e resposta final foram retiradas da base usada para criar os clusters:

- `AcceptedCmp1` a `AcceptedCmp5`;
- `Response`;
- `Complain`.

Essa separação evita que o agrupamento seja definido diretamente pelo resultado das campanhas que posteriormente serão analisadas.

## 6. Redução de dimensionalidade

Foi aplicado o PCA, ou Análise de Componentes Principais, para reduzir as variáveis padronizadas a três componentes:

```python
pca = PCA(n_components=3)
pca_data = pca.fit_transform(scaled_ds)
```

Os componentes `PC1`, `PC2` e `PC3` foram utilizados para visualizar os clientes em um espaço tridimensional. Clientes próximos nesse espaço possuem perfis gerais mais semelhantes, considerando as variáveis combinadas.

O PCA facilita a visualização, mas os componentes não devem ser interpretados como variáveis originais isoladas. Cada componente é uma combinação de várias características.

## 7. Definição do número de clusters

O método do cotovelo foi utilizado como apoio para avaliar diferentes quantidades de grupos. A análise registrada no notebook indicou quatro clusters como uma escolha prática, pois a redução da inércia se torna menos pronunciada a partir desse ponto.

O método do cotovelo é uma referência, não uma prova definitiva de que quatro grupos sejam a única solução correta. A escolha também deve considerar estabilidade, interpretabilidade e utilidade para o negócio.

## 8. Segmentação dos clientes

A segmentação final foi realizada com clusterização hierárquica aglomerativa:

```python
AC = AgglomerativeClustering(n_clusters=4)
yhat_AC = AC.fit_predict(PCA_ds)
```

Os grupos foram adicionados ao conjunto de dados original e ao DataFrame do PCA:

```python
PCA_ds["Clusters"] = yhat_AC
data["Clusters"] = yhat_AC
```

Os números dos clusters são identificadores técnicos. O cluster 3 não é necessariamente melhor que o cluster 0; a interpretação deve ser feita a partir das características médias e da distribuição de valores em cada grupo.

## 9. Análise descritiva: o que foi observado

A análise descritiva mostrou que:

- os clientes possuem diferentes níveis de renda e consumo;
- a maior parte dos clientes se concentra em faixas mais baixas ou intermediárias de gasto;
- existe uma relação positiva entre renda e gasto total: clientes com maior renda tendem a apresentar maior consumo;
- a maioria dos clientes possui zero ou uma criança pequena e zero ou um adolescente;
- a maior parte das famílias possui entre um e três membros;
- os clientes estão principalmente em faixas adultas e maduras de idade;
- a maior parte dos clientes não aceitou nenhuma das cinco promoções analisadas;
- a quantidade de compras realizadas em ofertas varia entre os grupos.

Os gráficos de dispersão, boxenplot, countplot e KDE foram utilizados para examinar essas distribuições.

## 10. Análise diagnóstica: diferenças entre os clusters

Com base nos gráficos e nas distribuições observadas, os grupos apresentaram os seguintes perfis gerais:

### Cluster 0

Apresenta menor nível de gasto e, em geral, menor renda. A maior parte dos clientes realiza poucas compras em ofertas. É um grupo de menor consumo relativo.

### Cluster 1

Apresenta comportamento intermediário em gasto, mas se destaca pela maior dispersão no número de compras promocionais. Há clientes com participação mais frequente em ofertas.

### Cluster 2

Concentra os maiores níveis de gasto e uma faixa mais ampla de consumo. É o grupo com maior potencial de valor financeiro, embora também apresente variabilidade interna.

### Cluster 3

Apresenta baixo gasto na distribuição observada. Apesar de aparecer associado a clientes de maior renda em alguns gráficos de renda e gasto, os valores de consumo não são uniformemente altos. Por isso, a renda por si só não explica o comportamento de compra.

### Promoções

A distribuição de `Total_Promos` mostra predominância do valor zero. Portanto, a maioria dos clientes não aceitou nenhuma promoção. Como o gráfico apresenta contagens absolutas, a comparação entre clusters deve considerar o tamanho relativo de cada grupo.

### Compras em ofertas

O cluster 1 apresenta maior dispersão em `NumDealsPurchases`, com alguns clientes realizando muitas compras promocionais. O cluster 2 apresenta menor número típico de compras em ofertas, apesar de possuir maior gasto total, sugerindo que clientes de alto valor fazem poucas compras em desconto.

### Variáveis pessoais

As variáveis `Kidhome`, `Teenhome`, `Children`, `Family_Size`, `Is_Parent`, `Age`, `Education` e `Living_Width` ajudam a descrever os grupos, mas não explicam isoladamente o gasto total. A renda, o histórico de consumo e a recência são mais determinantes.

## 11. Análise preditiva: o que o projeto permite prever

O notebook atual não treina um modelo supervisionado para prever uma nova resposta de campanha, valor de gasto ou probabilidade de compra. Portanto, o resultado preditivo deve ser entendido como uma oportunidade futura.

A segmentação produz variáveis que podem ser usadas em modelos futuros, por exemplo:

- probabilidade de aceitar uma nova campanha;
- probabilidade de realizar uma compra;
- valor esperado de gasto;
- probabilidade de responder a uma oferta promocional;
- risco de baixa participação ou abandono.

Para implementar essa etapa, seria necessário separar treino e teste, definir uma variável-alvo, treinar modelos supervisionados e avaliar métricas como acurácia, precisão, recall, F1, ROC-AUC ou erro médio absoluto.

Um cuidado importante é evitar vazamento de dados. Variáveis de resposta de campanhas só devem ser usadas como alvo ou como atributos históricos válidos, respeitando a ordem temporal do problema.

## 12. Análise prescritiva: recomendações de negócio

A recomendação final é substituir campanhas generalizadas por uma estratégia segmentada:

1. Priorizar o cluster 2 para ações de retenção, relacionamento e ofertas de maior valor. Como esse grupo já apresenta alto gasto, descontos amplos podem reduzir margem sem necessidade.
2. Desenvolver campanhas de conversão para o cluster 1, testando ofertas, comunicação e canais que estimulem maior frequência de compra.
3. Criar campanhas de ativação para os clusters 0 e 3, com mensagens simples, ofertas de entrada e incentivos de baixo custo para aumentar o primeiro ou o próximo pedido.
4. Medir a taxa de aceitação de promoções por cluster, e não apenas a contagem total de respostas.
5. Testar campanhas diferentes por meio de grupos de controle e experimentos A/B.
6. Avaliar retorno sobre investimento, margem, receita incremental e custo do incentivo antes de ampliar uma campanha.
7. Usar renda, gasto anterior, recência e comportamento de compras para personalizar a comunicação, sem tratar atributos pessoais como determinantes isolados.

A prescrição principal é:

> Investir mais na personalização das campanhas, preservando clientes de alto valor, convertendo clientes de consumo intermediário e ativando clientes de baixo consumo com incentivos controlados e avaliados.

## 13. Limitações e pontos de melhoria

- A clusterização é exploratória e depende das variáveis, da escala e do algoritmo escolhidos.
- A rotulação dos clusters é arbitrária e pode mudar quando o modelo for recalculado.
- Os gráficos de KDE não são ideais para variáveis categóricas ou discretas, como `Education` e `Living_Width`; boxplots e tabelas de proporção são mais adequados.
- `Customer_For` deve ser armazenada explicitamente em dias ou meses para evitar interpretações incorretas da conversão numérica de `timedelta`.
- O limite de idade e renda precisa de justificativa de negócio e validação de impacto.
- O projeto ainda não mede a qualidade da clusterização com métricas como silhouette score, Calinski-Harabasz ou Davies-Bouldin.
- Não foi implementada uma previsão supervisionada nem uma avaliação temporal de campanhas.
- As conclusões sobre rentabilidade devem ser confirmadas com receita, margem e custo das campanhas, pois gasto total não é igual a lucro.

## 14. Próximos passos recomendados

1. Corrigir e documentar a unidade de `Customer_For`.
2. Calcular métricas de qualidade e estabilidade dos clusters.
3. Criar uma tabela de perfil com média, mediana, tamanho e proporção de cada cluster.
4. Comparar taxas de promoção aceitas dentro de cada cluster.
5. Construir um modelo supervisionado para prever `Response` ou uma nova conversão.
6. Validar as recomendações com experimentos controlados.
7. Criar um painel com indicadores de receita, gasto, margem, resposta e retorno por cluster.

## 15. Conclusão

O projeto demonstrou que a base pode ser segmentada em quatro grupos com comportamentos distintos. O cluster 2 se destaca pelo maior gasto total, o cluster 1 apresenta comportamento intermediário e maior propensão a aceitar promoções, e os clusters 0 e 3 representam oportunidades de ativação e conversão.

A principal descoberta de negócio é que a adesão geral às promoções é baixa e que os clientes não devem receber a mesma estratégia. A recomendação final é utilizar os clusters como ponto de partida para campanhas personalizadas, com métricas claras de retorno e aprendizado contínuo.

O resultado atual é uma segmentação exploratória com forte utilidade descritiva e diagnóstica. Para completar o ciclo analítico, o próximo desenvolvimento deve incluir previsão supervisionada e medição de impacto das recomendações.
