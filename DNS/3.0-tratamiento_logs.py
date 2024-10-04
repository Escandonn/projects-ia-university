import docker
import datetime
import csv

# Crear un cliente Docker
client = docker.from_env()

# Obtener el contenedor que deseas obtener los logs
container_name = "web-server"
container = client.containers.get(container_name)

# Obtener la fecha y hora actual
ahora = datetime.datetime.now()

# Obtener los logs desde hace 1 hora
since = ahora - datetime.timedelta(hours=1)

# Obtener los logs del contenedor
logs = container.logs(stdout=True, stderr=True, since=since, until=None, timestamps=True)

# Convertir los logs a una lista de objetos
log_objects = []
for log in logs.decode("utf-8").splitlines():
    log_object = {
        "timestamp": log[:19],  # obtener la fecha y hora del log
        "message": log[20:]  # obtener el mensaje del log
    }
    log_objects.append(log_object)

# Ordenar la lista de objetos por timestamp
log_objects.sort(key=lambda x: x["timestamp"])

# Crear un archivo CSV y escribir los logs
with open('3.1-logs_limpios.csv', 'w', newline='') as csvfile:
    fieldnames = ['timestamp','message']  # Corregido el error tipográfico
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for log in log_objects:
        writer.writerow(log)

print("Logs guardados en logs_limpios.csv")  # Corregido el nombre del archivo