# Imagen base oficial de TensorFlow con Python 3
FROM tensorflow/tensorflow:2.18.0-py3

# Establecer el directorio de trabajo
WORKDIR /app

# Copiar archivos al contenedor
COPY . .

# Instalar dependencias adicionales desde archivo limpio
RUN pip install --no-cache-dir -r requirements_clean.txt

# Puerto por defecto para FastAPI/Uvicorn
EXPOSE 8000

# Comando para arrancar la API
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]