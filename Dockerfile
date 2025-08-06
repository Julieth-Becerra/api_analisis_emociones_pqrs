# Imagen base con Python 3.11
FROM python:3.11-slim

# Establecer el directorio de trabajo dentro del contenedor
WORKDIR /app

# Copiar los archivos del backend al contenedor
COPY . .

# Evitar escritura de bytecode, errores de buffer y encoding
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    LC_ALL=C.UTF-8 \
    LANG=C.UTF-8

# Actualizar pip y evitar errores con scikit-learn y Cython
RUN pip install --upgrade pip \
 && pip install Cython==0.29.36 \
 && pip install -r requirements.txt

# Puerto por defecto de FastAPI
EXPOSE 8000

# Comando para ejecutar la aplicación
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
