import requests
import json
from concurrent.futures import ThreadPoolExecutor
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

test_data_fail = google[google['event'] == 'FAIL'][['scheduling_class', 'priority', 'assigned_memory', 'average_usage_memory', 'maximum_usage_memory', 'resource_request_memory', 'cycles_per_instruction', 'memory_accesses_per_instruction']].head(3).to_dict('records')

url = 'http://localhost:5000/predict'

def send_request(data):
    response = requests.post(url, headers={"Content-Type": "application/json"}, data=json.dumps([data]))
    print(response.json())

num_requests = 100

with ThreadPoolExecutor(max_workers=num_requests) as executor:
    futures = [executor.submit(send_request, data) for data in test_data_fail for _ in range(num_requests // len(test_data_fail))]

for future in futures:
    future.result()
