import pandas as pd
from datetime import datetime
from sklearn.ensemble import IsolationForest

# Leer el archivo CSV que contiene los logs, sin encabezados
df = pd.read_csv('logs_limpios.csv', header=None, names=['raw_log'])

# Extraer el timestamp y las características relevantes
df['timestamp'] = df['raw_log'].str.extract(r'\[(.*?)\]')[0]
df['timestamp'] = pd.to_datetime(df['timestamp'], format='%d/%b/%Y:%H:%M:%S %z')

# Extraer la IP del log
df['ip'] = df['raw_log'].str.extract(r'(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})')

# Crear nuevas características basadas en el timestamp
df['hour'] = df['timestamp'].dt.hour
df['minute'] = df['timestamp'].dt.minute
df['second'] = df['timestamp'].dt.second
df['day_of_week'] = df['timestamp'].dt.dayofweek

# Contar el número de solicitudes y agregar características
request_count = df.groupby([df['timestamp'].dt.floor('s'), 'hour','minute','second', 'day_of_week', 'ip']).size().reset_index(name='count')

# Crear un DataFrame para entrenar el modelo
X = request_count[['count', 'hour','minute','second', 'day_of_week']]

# Entrenar el modelo Isolation Forest
model = IsolationForest(contamination=0.05)  # 5% de los datos se consideran anomalías
model.fit(X)

# Hacer predicciones
request_count['anomaly'] = model.predict(X)

# Marcar anomalías
anomalies = request_count[request_count['anomaly'] == -1]

# Mostrar resultados
if not anomalies.empty:
    print("Posibles ataques DNS detectados en los siguientes segundos:")
    for index, row in anomalies.iterrows():
        print(f"Tiempo: {row['timestamp']}, IP: {row['ip'][0]}, Solicitudes: {row['count']}")
else:
    print("No se detectaron posibles ataques DNS.")

# Opcional: Guardar las anomalías en un archivo CSV
anomalies.to_csv('4.1-anomalies_detected.csv', index=False)