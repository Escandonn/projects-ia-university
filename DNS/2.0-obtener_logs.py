import docker
import csv

# Conectarse al daemon de Docker
client = docker.from_env()

# Obtener los logs del contenedor
container_name = "web-server"  # El nombre del contenedor
container = client.containers.get(container_name)
logs = container.logs().decode('utf-8')

# Crear un archivo CSV y escribir los logs en él
with open('2.1-logs.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)

    # Procesar los logs y escribirlos en el archivo CSV
    for log in logs.splitlines():
        fecha = log[:19]  # Extraer la fecha del log (suponiendo que está en el formato "YYYY-MM-DD HH:MM:SS")
        log_texto = log[20:]  # Extraer el texto del log
        writer.writerow([fecha, log_texto])

print("Logs guardados en logs.csv")