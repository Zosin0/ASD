import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

google = pd.read_csv('data/google.csv')
print("\tGoogle:\n", 50*"-")
print(google.head())
print(google.info())

google['time'] = pd.to_datetime(google['time'], unit='ns')
google.set_index('time', inplace=True)

numeric_cols = google.select_dtypes(include=[np.number]).columns
google[numeric_cols] = google[numeric_cols].fillna(google[numeric_cols].mean())

def extract_memory_value(value):
    if isinstance(value, str):
        value_dict = eval(value)
        return value_dict.get('memory', np.nan)
    return value

google['average_usage_memory'] = google['average_usage'].apply(extract_memory_value)
google['maximum_usage_memory'] = google['maximum_usage'].apply(extract_memory_value)

X_google = google[['scheduling_class', 'priority', 'assigned_memory', 'average_usage_memory', 'maximum_usage_memory']]
y_google = google['vertical_scaling']

X_train_google, X_test_google, y_train_google, y_test_google = train_test_split(X_google, y_google, test_size=0.2, random_state=42)

dt_model = DecisionTreeRegressor(random_state=42)
dt_model.fit(X_train_google, y_train_google)

y_pred_google = dt_model.predict(X_test_google)

mse_google = mean_squared_error(y_test_google, y_pred_google)
mae_google = mean_absolute_error(y_test_google, y_pred_google)
r2_google = r2_score(y_test_google, y_pred_google)

print(f'MSE do modelo de árvore de decisão: {mse_google}')
print(f'MAE do modelo de árvore de decisão: {mae_google}')
print(f'R² do modelo de árvore de decisão: {r2_google}')

importances = dt_model.feature_importances_
feature_names = X_google.columns
feature_importances = pd.Series(importances, index=feature_names).sort_values(ascending=False)
print("Importância das features:\n", feature_importances)

joblib.dump(dt_model, 'models/trained/decision_tree_model.pkl')

def predict_application_state(input_data):
    model = joblib.load('models/trained/decision_tree_model.pkl')
    prediction = model.predict(input_data)
    return prediction
