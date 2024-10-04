import requests
import time
import threading

# Definir URL estática directamente en el código
url = "http://localhost:8080"  # Ruta del servidor web de Docker

# Definir la cantidad de solicitudes y el intervalo directamente en el código
cantidad = 100000  # Número de solicitudes
intervalo = 500  # Intervalo en milisegundos entre las solicitudes

# Configuración de headers basados en el intervalo
if intervalo == 0:
    headers = {'User-Agent': 'UCEVAIADOS'}
else:
    headers = {'User-Agent': 'UCEVAIANORMAL'}

# Función para enviar solicitudes
def enviar_solicitud(numero):
    try:
        # Enviar la solicitud HTTP
        response = requests.get(url, headers=headers)

        # Mostrar el código de respuesta de la solicitud
        print(f"Petición {numero}: {response.status_code}")

    except requests.exceptions.RequestException as e:
        # Manejo de errores durante la solicitud
        print(f"Error en la petición {numero}: {e}")

# Crear y lanzar hilos
hilos = []
for i in range(cantidad):
    hilo = threading.Thread(target=enviar_solicitud, args=(i + 1,))
    hilos.append(hilo)
    hilo.start()
    
    # Esperar el intervalo dado antes de lanzar el siguiente hilo

# Esperar a que todos los hilos terminen
for hilo in hilos:
    hilo.join()
