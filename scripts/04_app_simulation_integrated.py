from flask import Flask, request, jsonify
import joblib
import pandas as pd
import threading
import time
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)
rf_model = joblib.load('models/trained/random_forest_model.pkl')
lstm_model = load_model('models/trained/lstm_model.h5', compile=False)
lstm_model.compile(optimizer='adam', loss='mean_squared_error')
scaler = joblib.load('models/trained/scaler.pkl')

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

current_state = google[google['event'] == 'FINISH'][['scheduling_class', 'priority', 'assigned_memory', 'average_usage_memory', 'maximum_usage_memory', 'resource_request_memory', 'cycles_per_instruction', 'memory_accesses_per_instruction']].head(1).to_dict('records')[0]

window_size = 10
historical_data = []

def predict_application_state(input_data):
    prediction = rf_model.predict(input_data)
    return prediction

def create_sequences(data, window_size):
    sequences = []
    for i in range(len(data) - window_size + 1):
        sequences.append(data[i:i + window_size])
    return np.array(sequences)

def predict_traffic_peak(input_data):
    input_data_scaled = scaler.transform(input_data)
    input_data_sequences = create_sequences(input_data_scaled, window_size)
    if input_data_sequences.shape[0] == 0:
        raise ValueError("Os dados de entrada são insuficientes para criar sequências com o tamanho da janela especificado.")
    input_data_sequences = input_data_sequences.reshape((input_data_sequences.shape[0], window_size, input_data.shape[1]))
    prediction = lstm_model.predict(input_data_sequences)
    return prediction

def update_state_and_report():
    global current_state, historical_data
    while True:
        input_data = pd.DataFrame([current_state])
        state_prediction = predict_application_state(input_data)
        historical_data.append(current_state)
        if len(historical_data) > window_size:
            historical_data.pop(0)

        state_message = "Estado desconhecido"
        peak_message = "Previsão desconhecida"

        if state_prediction[0] < 1:
            state_message = 'A aplicação precisa de mais recursos: {:.2f}'.format(state_prediction[0])
        elif state_prediction[0] < 2:
            state_message = 'A aplicação está em risco: {:.2f}'.format(state_prediction[0])
        else:
            state_message = 'A aplicação está em estado saudável: {:.2f}'.format(state_prediction[0])

        if len(historical_data) == window_size:
            historical_df = pd.DataFrame(historical_data)
            try:
                peak_prediction = predict_traffic_peak(historical_df)
                if peak_prediction[0][0] < 1:
                    peak_message = 'A aplicação não terá pico de tráfego iminente'
                else:
                    peak_message = 'A aplicação terá um pico de tráfego: {:.2f}'.format(peak_prediction[0][0])
            except Exception as e:
                peak_message = str(e)

        print(state_message)
        print(peak_message)
        current_state["average_usage_memory"] += 0.1
        current_state["maximum_usage_memory"] += 0.1
        time.sleep(2)

@app.route('/predict', methods=['POST'])
def predict():
    global current_state, historical_data
    data = request.json
    input_data = pd.DataFrame(data)
    state_prediction = predict_application_state(input_data)
    historical_data.append(input_data.iloc[0].to_dict())
    if len(historical_data) > window_size:
        historical_data.pop(0)

    response = []
    for state_pred in state_prediction:
        if state_pred < 1:
            state_message = 'A aplicação precisa de mais recursos: {:.2f}'.format(state_pred)
        elif state_pred < 2:
            state_message = 'A aplicação está em risco: {:.2f}'.format(state_pred)
        else:
            state_message = 'A aplicação está em estado saudável: {:.2f}'.format(state_pred)

        if len(historical_data) == window_size:
            historical_df = pd.DataFrame(historical_data)
            try:
                peak_prediction = predict_traffic_peak(historical_df)
                if peak_prediction[0][0] < 1:
                    peak_message = 'A aplicação não terá pico de tráfego iminente'
                else:
                    peak_message = 'A aplicação terá um pico de tráfego: {:.2f}'.format(peak_prediction[0][0])
            except Exception as e:
                peak_message = str(e)
        else:
            peak_message = 'Dados insuficientes para prever pico de tráfego'

        response.append({'state_message': state_message, 'peak_message': peak_message})
    return jsonify(response)

if __name__ == '__main__':
    threading.Thread(target=update_state_and_report, daemon=True).start()
    app.run(host='0.0.0.0', port=5000)
