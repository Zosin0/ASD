import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
import joblib

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

data = pd.concat([X_google, y_google], axis=1).dropna()
X_google = data[features]
y_google = data['vertical_scaling']

scaler = StandardScaler()
X_google_scaled = scaler.fit_transform(X_google)

def create_sequences(data, window_size):
    sequences = []
    for i in range(len(data) - window_size):
        sequences.append(data[i:i + window_size])
    return np.array(sequences)

window_size = 10
X_sequences = create_sequences(X_google_scaled, window_size)
y_sequences = y_google[window_size:].values

X_train, X_test, y_train, y_test = train_test_split(X_sequences, y_sequences, test_size=0.2, random_state=42)

X_train = X_train.reshape((X_train.shape[0], window_size, X_google.shape[1]))
X_test = X_test.reshape((X_test.shape[0], window_size, X_google.shape[1]))

model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(window_size, X_google.shape[1])))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, callbacks=[early_stopping])

y_pred = model.predict(X_test)

if np.isnan(y_pred).any():
    print("y_pred contém NaNs. Ajuste necessário nos dados ou no modelo.")
else:
    mse = mean_squared_error(y_test, y_pred)
    print(f'MSE do modelo LSTM: {mse}')

model.save('models/trained/lstm_model.h5')
joblib.dump(scaler, 'models/trained/scaler.pkl')

def predict_traffic_peak(input_data):
    from tensorflow.keras.models import load_model
    model = load_model('models/trained/lstm_model.h5', compile=False)
    model.compile(optimizer='adam', loss='mean_squared_error')
    scaler = joblib.load('models/trained/scaler.pkl')
    input_data_scaled = scaler.transform(input_data)
    input_data_sequences = create_sequences(input_data_scaled, window_size)
    if input_data_sequences.shape[0] == 0:
        raise ValueError("Os dados de entrada são insuficientes para criar sequências com o tamanho da janela especificado.")
    input_data_sequences = input_data_sequences.reshape((input_data_sequences.shape[0], window_size, X_google.shape[1]))
    prediction = model.predict(input_data_sequences)
    return prediction

new_data = pd.DataFrame({
    'scheduling_class': [2]*15,
    'priority': [0]*15,
    'assigned_memory': [0.5]*15,
    'average_usage_memory': [4.0]*15,
    'maximum_usage_memory': [5.0]*15,
    'resource_request_memory': [3.0]*15,
    'cycles_per_instruction': [2.0]*15,
    'memory_accesses_per_instruction': [1.5]*15
})

try:
    peak_prediction = predict_traffic_peak(new_data)
    print(f'Previsão de pico de tráfego: {peak_prediction[0]}')
except ValueError as e:
    print(f"Erro: {e}")
