from flask import Flask, request, jsonify
import joblib
import pandas as pd
import threading
import time
import numpy as np

app = Flask(__name__)
model = joblib.load('models/trained/random_forest_model.pkl')

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

def predict_application_state(input_data):
    prediction = model.predict(input_data)
    return prediction

def update_state_and_report():
    global current_state
    while True:
        input_data = pd.DataFrame([current_state])
        prediction = predict_application_state(input_data)
        state_message = "Estado desconhecido"
        if prediction[0] < 1:
            state_message = 'A aplicação precisa de mais recursos: {:.2f}'.format(prediction[0])
        elif prediction[0] < 2:
            state_message = 'A aplicação está em risco: {:.2f}'.format(prediction[0])
        else:
            state_message = 'A aplicação está em estado saudável: {:.2f}'.format(prediction[0])
        print(state_message)
        current_state["average_usage_memory"] += 0.1
        current_state["maximum_usage_memory"] += 0.1
        time.sleep(2)

@app.route('/predict', methods=['POST'])
def predict():
    global current_state
    data = request.json
    input_data = pd.DataFrame(data)
    prediction = predict_application_state(input_data)
    response = []
    for pred in prediction:
        if pred < 1:
            response.append('A aplicação precisa de mais recursos: {:.2f}'.format(pred))
        elif pred < 2:
            response.append('A aplicação está em risco: {:.2f}'.format(pred))
        else:
            response.append('A aplicação está em estado saudável: {:.2f}'.format(pred))
    return jsonify(response)

if __name__ == '__main__':
    threading.Thread(target=update_state_and_report, daemon=True).start()
    app.run(host='0.0.0.0', port=5000)
