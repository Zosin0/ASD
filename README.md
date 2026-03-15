# Previsão de Estabilidade em Sistemas Distribuídos: Um Modelo de IA para Controlar Recursos e Picos de Carga Causadas por Fatores Externos

## Sobre o Projeto
Este projeto visa desenvolver um sistema de Inteligência Artificial para prever e gerenciar estados de aplicações em sistemas distribuídos. O objetivo principal é alertar quando uma aplicação está sob estresse, permitindo o escalonamento automático de recursos e garantindo a disponibilidade e performance.

### Componentes Principais:
1.  **Modelo Preditivo (Árvore de Decisão/Random Forest):** Para prever o estado atual da saúde da aplicação (Saudável, Em Risco, Necessita de Recursos).
2.  **Rede Neural Recorrente (LSTM):** Para previsão de picos de carga iminentes com base em séries temporais.
3.  **Simulação de Estresse:** Scripts para validar a resiliência do sistema sob cargas elevadas.

---

## Estrutura do Projeto

Abaixo está a organização profissional dos arquivos do projeto:

```bash
ASD/
├── docs/                      # Documentação e Pesquisa
│   ├── paper.pdf              # Versão formatada do artigo
│   ├── plots.py               # Scripts para geração de gráficos
│   ├── research/              # Artigos de referência externa
│   └── drafts/                # Rascunhos e ideias de desenvolvimento
├── models/                    # Scripts de Inteligência Artificial
│   ├── 01_train_decision_tree.py # Treinamento da Árvore de Decisão
│   ├── 02_train_random_forest.py # Treinamento de Random Forest
│   ├── 03_train_lstm.py          # Treinamento de LSTM (v1)
│   ├── 04_train_lstm_advanced.py # Treinamento de LSTM (Avançado)
│   └── trained/               # Modelos exportados (.pkl, .h5, .pkl)
├── notebooks/                 # Desenvolvimento e Análise Exploratória
│   └── model_development.ipynb # Jupyter Notebook principal
├── scripts/                   # Testes e Simulações
│   ├── 01_eval_decision_tree.py # Avaliação do modelo de Árvore de Decisão
│   ├── 02_eval_random_forest.py # Avaliação do modelo Random Forest
│   ├── 03_app_simulation_basic.py # Simulação web básica (v1)
│   ├── 04_app_simulation_integrated.py # Simulação web integrada (v2)
│   └── 05_stress_test.py      # Simulação de picos de carga (Stress Test)
└── data/                      # Conjuntos de Dados
    └── google.csv             # Dataset do Google Cluster Sample 2019
```

---

## 🚀 Como Utilizar

### 1. Preparação dos Dados
O projeto utiliza o dataset **Google Cluster Sample 2019**, disponível em `data/google.csv`.

### 2. Treinamento dos Modelos
Execute os scripts na ordem numérica para treinar e salvar os pesos em `models/trained/`:
*   `python models/01_train_decision_tree.py`
*   `python models/02_train_random_forest.py`
*   `python models/03_train_lstm.py`

### 3. Simulação de Aplicação
Para testar a inteligência do sistema em tempo real:
1. Inicie a aplicação Flask: `python scripts/04_app_simulation_integrated.py`
2. Em outro terminal, execute o teste de estresse: `python scripts/05_stress_test.py`


---

## Resultados e Métricas
*   **Estado da Aplicação:** O modelo Random Forest alcançou um **R² de 0.93**, indicando alta precisão na classificação de saúde do sistema.
*   **Picos de Carga:** A rede LSTM demonstrou capacidade de prever picos iminentes após a acumulação de janelas históricas (Window Size: 10).

---

## Autores
*   Lucas Zoser Nunes Costa
*   Leonardo Areias Rodovalho
*   Leonardo Benttes Almeida Placido dos Santos
*   Pedro Henrique Moreira da Silva

---
**Curso:** Ciência da Computação - UniCEUB
**Disciplina:** Arquitetura de Sistemas Distribuídos
**Professor:** Msc. Fabiano Mariath D'Oliveira
