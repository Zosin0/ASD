import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import pandas as pd

def create_sequences(data, target, window_size):
    xs, ys = [], []
    for i in range(len(data) - window_size):
        x = data[i:i + window_size]
        y = target[i + window_size]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

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

window_size = 24
X_sequences, y_sequences = create_sequences(X_google_scaled, y_google.values, window_size)

print(X_sequences.shape)
print(y_sequences.shape)

X_train, X_test, y_train, y_test = train_test_split(X_sequences, y_sequences, test_size=0.2, random_state=42)

X_train = X_train.reshape((X_train.shape[0], window_size, X_google.shape[1]))
X_test = X_test.reshape((X_test.shape[0], window_size, X_google.shape[1]))

model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(window_size, X_train.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, callbacks=[early_stopping])

y_pred = model.predict(X_test)

y_test_scaled = scaler.inverse_transform(y_test.reshape(-1, 1))
y_pred_scaled = scaler.inverse_transform(y_pred)

plt.figure(figsize=(12, 6))
plt.plot(y_test_scaled, label='Verdadeiro')
plt.plot(y_pred_scaled, label='Previsto')
plt.legend()
plt.show()

mse = mean_squared_error(y_test_scaled, y_pred_scaled)
print(f'Mean Squared Error: {mse}')

model.save('models/trained/lstm_model.h5')
joblib.dump(scaler, 'models/trained/scaler.pkl')
