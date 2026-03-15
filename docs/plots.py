import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Carregar os datasets
google = pd.read_csv('datasets/google.csv')
google['time'] = pd.to_datetime(google['time'], unit='ns')
google.set_index('time', inplace=True)

# Preencher valores ausentes apenas nas colunas numéricas
numeric_cols = google.select_dtypes(include=[np.number]).columns
google[numeric_cols] = google[numeric_cols].fillna(google[numeric_cols].mean())

# Extrair valores de memória dos dicionários em average_usage e maximum_usage
def extract_memory_value(value):
    if isinstance(value, str):
        value_dict = eval(value)
        return value_dict.get('memory', np.nan)
    return value

google['average_usage_memory'] = google['average_usage'].apply(extract_memory_value)
google['maximum_usage_memory'] = google['maximum_usage'].apply(extract_memory_value)
google['resource_request_memory'] = google['resource_request'].apply(extract_memory_value)

# Distribuição das Features
plt.figure(figsize=(12, 8))
google[['average_usage_memory', 'maximum_usage_memory', 'assigned_memory']].hist(bins=50, edgecolor='black')
plt.tight_layout()
plt.show()

# Boxplots das Features
plt.figure(figsize=(12, 8))
sns.boxplot(data=google[['average_usage_memory', 'maximum_usage_memory', 'assigned_memory']])
plt.show()

# Heatmap de Correlação
plt.figure(figsize=(12, 8))
corr_matrix = google[['scheduling_class', 'priority', 'assigned_memory', 'average_usage_memory', 'maximum_usage_memory', 'resource_request_memory']].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.show()

# Análise Temporal das Métricas de Uso
plt.figure(figsize=(12, 8))
google[['average_usage_memory', 'maximum_usage_memory']].resample('D').mean().plot()
plt.title('Métricas de Uso ao Longo do Tempo')
plt.show()

# Comparação entre Eventos
plt.figure(figsize=(12, 8))
sns.boxplot(x='event', y='average_usage_memory', data=google)
plt.title('Comparação da Average Usage Memory entre Eventos')
plt.show()

# Importância das Features no Modelo
import joblib
model = joblib.load('random_forest_model.pkl')
importances = model.feature_importances_
features = ['scheduling_class', 'priority', 'assigned_memory', 'average_usage_memory', 'maximum_usage_memory', 'resource_request_memory', 'cycles_per_instruction', 'memory_accesses_per_instruction']
plt.figure(figsize=(12, 8))
plt.barh(features, importances)
plt.title('Importância das Features')
plt.show()
