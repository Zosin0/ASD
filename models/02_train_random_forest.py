from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import pandas as pd
import numpy as np

google = pd.read_csv('data/google.csv')

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
google['resource_request_memory'] = google['resource_request'].apply(extract_memory_value)

features = ['scheduling_class', 'priority', 'assigned_memory', 'average_usage_memory', 'maximum_usage_memory', 'resource_request_memory', 'cycles_per_instruction', 'memory_accesses_per_instruction']
X_google = google[features]
y_google = google['vertical_scaling']

tscv = TimeSeriesSplit(n_splits=5)
train_index, test_index = next(tscv.split(X_google))
X_train_google, X_test_google = X_google.iloc[train_index], X_google.iloc[test_index]
y_train_google, y_test_google = y_google.iloc[train_index], y_google.iloc[test_index]

rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
rf_model.fit(X_train_google, y_train_google)

y_pred_google = rf_model.predict(X_test_google)

mse_google = mean_squared_error(y_test_google, y_pred_google)
mae_google = mean_absolute_error(y_test_google, y_pred_google)
r2_google = r2_score(y_test_google, y_pred_google)

print(f'MSE do modelo Random Forest (teste): {mse_google}')
print(f'MAE do modelo Random Forest (teste): {mae_google}')
print(f'R² do modelo Random Forest (teste): {r2_google}')

y_train_pred_google = rf_model.predict(X_train_google)
mse_train_google = mean_squared_error(y_train_google, y_train_pred_google)
mae_train_google = mean_absolute_error(y_train_google, y_train_pred_google)
r2_train_google = r2_score(y_train_google, y_train_pred_google)

print(f'MSE do modelo Random Forest (treinamento): {mse_train_google}')
print(f'MAE do modelo Random Forest (treinamento): {mae_train_google}')
print(f'R² do modelo Random Forest (treinamento): {r2_train_google}')

cv_scores = cross_val_score(rf_model, X_google, y_google, cv=tscv, scoring='r2')
print(f'R² médio na validação cruzada: {cv_scores.mean()}')

importances = rf_model.feature_importances_
feature_names = X_google.columns
feature_importances = pd.Series(importances, index=feature_names).sort_values(ascending=False)
print("Importância das features:\n", feature_importances)

joblib.dump(rf_model, 'models/trained/random_forest_model.pkl')

def predict_application_state(input_data):
    model = joblib.load('models/trained/random_forest_model.pkl')
    prediction = model.predict(input_data)
    return prediction
