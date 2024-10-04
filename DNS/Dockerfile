# Usar la imagen oficial de NGINX como base
FROM nginx:alpine

# Copiar el archivo index.html a la carpeta de NGINX
COPY ./index.html /usr/share/nginx/html/index.html

# Exponer el puerto 80 (el puerto donde correrá el servidor NGINX)
EXPOSE 80

# Iniciar NGINX
CMD ["nginx", "-g", "daemon off;"]
