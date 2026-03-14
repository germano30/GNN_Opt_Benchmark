# GNN Optimizer Benchmark

Benchmark experimental para avaliar o impacto de diferentes **otimizadores de treinamento** em **Graph Neural Networks (GNNs)**.  

O foco principal do projeto é analisar o desempenho do novo otimizador **Muon** em comparação com otimizadores tradicionais amplamente utilizados em treinamento de redes neurais.

O benchmark inclui múltiplas **arquiteturas de GNN**, **tarefas de aprendizado em grafos** e **datasets padrão da literatura**.

---

# Objetivo

O objetivo deste projeto é investigar como diferentes **otimizadores influenciam o treinamento de Graph Neural Networks**.

Apesar de muitos trabalhos focarem em novas arquiteturas, poucos exploram sistematicamente o impacto de **otimização em GNNs**.

Este projeto busca responder perguntas como:

- O **Muon** melhora a convergência em GNNs?
- Existe diferença de comportamento entre **GNNs homogêneas e heterogêneas**?
- Alguns otimizadores funcionam melhor para **tarefas específicas em grafos**?
- O custo computacional de otimizadores avançados compensa o ganho de performance?

---

# Arquiteturas de GNN

Serão avaliadas arquiteturas representativas da literatura.

## Grafos Homogêneos

- **GCN (Graph Convolutional Network)**
- **GAT (Graph Attention Network)**
- **GIN (Graph Isomorphism Network)**

Essas arquiteturas utilizam mecanismos de **message passing**, onde cada nó atualiza sua representação agregando informações dos vizinhos. :contentReference[oaicite:0]{index=0}  

- **GCN** realiza agregação usando a matriz de adjacência normalizada.
- **GAT** introduz mecanismos de atenção para ponderar a importância dos vizinhos.
- **GIN** possui forte poder discriminativo baseado no teste de Weisfeiler–Lehman. :contentReference[oaicite:1]{index=1}  

Referência:  
https://arxiv.org/pdf/2302.13406

---

## Grafos Heterogêneos

- **R-GCN (Relational Graph Convolutional Network)**
- **R-GAT (Relational Graph Attention Network)**

Essas arquiteturas são projetadas para grafos com **múltiplos tipos de relações**, como knowledge graphs.

Referência:  
https://arxiv.org/pdf/2302.13406

---

# Tarefas Avaliadas

O benchmark inclui três tarefas clássicas em aprendizado em grafos:

### Node Classification
Predição de rótulos associados a nós individuais do grafo.

Exemplo:
- classificação de papers em redes de citações.

---

### Link Prediction
Predição da existência de uma aresta entre dois nós.

Aplicações:
- recomendação
- knowledge graphs
- redes sociais

---

### Graph Classification

Classificação de grafos inteiros.

Aplicações:
- química computacional
- bioinformática
- detecção de fraudes

---

# Datasets

Os experimentos serão conduzidos nos seguintes datasets:

| Dataset | Tipo | Tarefa Principal |
|-------|------|------|
| ogbl-collab | grafo de colaboração | link prediction |
| ogbn-proteins | grafo de proteínas | node classification |
| ogbg-ppa | grafo de proteínas | graph classification |
| Cora | rede de citações | node classification |
| WordNet18RR | knowledge graph | link prediction |
| OGB-BioKG | knowledge graph biológico | link prediction |

---

# Otimizadores Avaliados

O foco principal do benchmark é comparar:

| Otimizador | Tipo |
|------|------|
| AdamW | adaptativo |
| SGD | gradiente estocástico |
| Shampoo | segunda ordem aproximada |
| SOAP | otimização adaptativa |
| Muon | otimização baseada em matrizes |

Referência do Muon:  
https://kellerjordan.github.io/posts/muon/

Observação:

- O otimizador **Shampoo** será testado com diferentes frequências de atualização: freq = [10, 32]
