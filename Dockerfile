# Imagen base con Python 3.11
FROM python:3.11-slim

# Establecer el directorio de trabajo dentro del contenedor
WORKDIR /app

# Copiar los archivos al contenedor
COPY . .

# Evitar escritura de bytecode, errores de buffer y encoding
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    LC_ALL=C.UTF-8 \
    LANG=C.UTF-8

# Instalar dependencias de sistema necesarias para compilar
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    gcc \
    && pip install --upgrade pip setuptools wheel \
    && pip install Cython==0.29.36

# Instalar las dependencias del proyecto
RUN pip install --no-cache-dir -r requirements.txt

# Exponer el puerto por defecto
EXPOSE 8000

# Comando para iniciar la API
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
