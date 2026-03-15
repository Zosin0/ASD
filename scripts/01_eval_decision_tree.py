import pandas as pd
import numpy as np
import joblib
import sys
import os

sys.path.append(os.getcwd())
from models.decision_tree_model import predict_application_state

def test_model():
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

    test_data_fail = google[google['event'] == 'FAIL'][['scheduling_class', 'priority', 'assigned_memory', 'average_usage_memory', 'maximum_usage_memory']].head(3)
    test_data_success = google[google['event'] == 'FINISH'][['scheduling_class', 'priority', 'assigned_memory', 'average_usage_memory', 'maximum_usage_memory']].head(3)

    print(test_data_success)
    print("Testando aplicações saudáveis:")

    predictions_success = predict_application_state(test_data_success)

    for i, prediction in enumerate(predictions_success):
        if prediction < 1:
            print(f'A aplicação {i} precisa de mais recursos: {prediction}')
        elif prediction < 2:
            print(f'A aplicação {i} está em risco: {prediction}')
        else:
            print(f'A aplicação {i} está em estado saudável: {prediction}')

    print("\nTestando aplicações com falha:")

    predictions_fail = predict_application_state(test_data_fail)

    for i, prediction in enumerate(predictions_fail):
        if prediction < 1:
            print(f'A aplicação {i} precisa de mais recursos: {prediction}')
        elif prediction < 2:
            print(f'A aplicação {i} está em risco: {prediction}')
        else:
            print(f'A aplicação {i} está em estado saudável: {prediction}')

if __name__ == "__main__":
    test_model()
