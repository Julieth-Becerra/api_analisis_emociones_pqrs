from fastapi import FastAPI, HTTPException, UploadFile, File
import joblib
import tensorflow as tf
import numpy as np
import pandas as pd
import os
import io
from pydantic import BaseModel
import keras
from keras.models import Model
from keras.layers import Dense, Dropout
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from pgmpy.models import BayesianNetwork
from pgmpy.inference import VariableElimination
from sklearn.decomposition import LatentDirichletAllocation
from gensim.models.ldamodel import LdaModel
from gensim.corpora import Dictionary
import pickle
import re
import string
import spacy
import nltk
from nltk.corpus import stopwords
import json
import unicodedata

# Inicializar FastAPI
app = FastAPI(
    title="API de Análisis de Emociones en PQRS",
    description="Esta API permite analizar emociones en PQRS usando múltiples modelos de inferencia bayesiana.",
    version="1.0.0",
)

# Descargar recursos
nltk.download("stopwords")
nlp = spacy.load("es_core_news_sm")
stopwords_es = stopwords.words("spanish")
stopwords_extra = ['atencion', 'personal', 'servicio', 'paciente', 'usuario', 'hospital']
stopwords_total = set(stopwords_es + stopwords_extra)

stopwords_nltk = set(stopwords.words("spanish"))

# Procesamiento de texto para archivos
def procesar_detalle_api(texto):
    if not isinstance(texto, str):
        return ""
    texto = texto.lower()
    texto = re.sub(r"[^\w\s]", "", texto)  
    texto = re.sub(r"\d+", "", texto)     
    palabras = texto.split()
    palabras_filtradas = [p for p in palabras if p not in stopwords_nltk]
    return " ".join(palabras_filtradas)


# Funciones auxiliares
def limpiar_proceso(nombre):
    nombre = nombre.lower().strip()
    nombre = unicodedata.normalize("NFKD", nombre).encode("ascii", "ignore").decode("utf-8")
    nombre = nombre.replace(" ", "_")
    return nombre

regex_puntuacion = re.compile(f"[{re.escape(string.punctuation)}]")
def limpiar_texto(texto):
    texto = texto.lower()
    texto = regex_puntuacion.sub("", texto)
    doc = nlp(texto)
    tokens = [token.lemma_ for token in doc if token.lemma_ not in stopwords_total and not token.is_space]
    return tokens

# Directorio de modelos
MODELS_DIR = "models"
LDA_MODELS_DIR = os.path.join(MODELS_DIR, "modelos_LDA")
GPR_MODELS_DIR = os.path.join(MODELS_DIR, "modelos_GPR")

# Naive bayes
naive_bayes = joblib.load(os.path.join(MODELS_DIR, "modelo_naivebayes_emociones.pkl"))
tfidf_vectorizer_nb = joblib.load(os.path.join(MODELS_DIR, "vectorizador_tfidf_nb.pkl"))
mlb_nb = joblib.load(os.path.join(MODELS_DIR, "binarizador_emociones_nb.pkl"))

#Red bayesiana
with open(os.path.join(MODELS_DIR, "modelo_red_bayesiana_final.pkl"), "rb") as f:
    modelo_rb = pickle.load(f)
inferencia_rb = VariableElimination(modelo_rb)
le_proceso_rb = joblib.load(os.path.join(MODELS_DIR, "label_encoder_proceso.pkl"))
lista_emociones_rb = joblib.load(os.path.join(MODELS_DIR, "lista_emociones.pkl"))

#BDL
tfidf_vectorizer_bdl = joblib.load(os.path.join(MODELS_DIR, "tfidf_vectorizer_bdl.pkl"))

@keras.saving.register_keras_serializable()
class VariationalLogisticRegression(keras.Model):
    def __init__(self, input_dim=5000, output_dim=8, **kwargs):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dense = keras.Sequential([
            Dense(128, activation="relu", kernel_regularizer=keras.regularizers.l2(0.01)),
            Dropout(0.3),
            Dense(output_dim, activation="sigmoid")
        ])

    def call(self, inputs, training=False):
        return self.dense(inputs, training=training)

    def get_config(self):
        config = super().get_config()
        config.update({"input_dim": self.input_dim, "output_dim": self.output_dim})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

bayesian_deep_learning = keras.models.load_model(
    os.path.join(MODELS_DIR, "modelo_bdl_final.keras"),
    custom_objects={"VariationalLogisticRegression": VariationalLogisticRegression}
)

# Cargar mapeo global y modelos LDA
modelos_lda = {}
diccionarios_lda = {}
mapeos_tema_etiqueta = {}

try:
    with open(os.path.join(MODELS_DIR, "modelos_LDA", "mapeo_tema_etiqueta.json"), "r", encoding="utf-8") as f:
        mapeo_completo = json.load(f)
        urgencias_keys = [k for k in mapeo_completo.keys() if "urgencias" in k]
        #print(f"Claves que contienen 'urgencias': {urgencias_keys}")

except Exception as e:
    print(f"Error cargando mapeo: {e}")
    mapeo_completo = {}

#print("\n--- CARGANDO MAPEOS ---")
for archivo in os.listdir(LDA_MODELS_DIR):
    if archivo.endswith(".gensim") and archivo.startswith("lda_"):
        nombre_base = archivo.replace("lda_", "").replace(".gensim", "")
        proceso = limpiar_proceso(nombre_base)

        ruta_modelo = os.path.join(LDA_MODELS_DIR, archivo)
        ruta_diccionario = os.path.join(LDA_MODELS_DIR, f"diccionario_{nombre_base}.dict")

        try:
            modelos_lda[proceso] = LdaModel.load(ruta_modelo)
            diccionarios_lda[proceso] = Dictionary.load(ruta_diccionario)

            mapeo_filtrado = {}
            for k, v in mapeo_completo.items():
                partes = k.split("_")
                if len(partes) == 2:
                    proceso_key = partes[0].lower().replace(" ", "_")
                    id_tema = partes[1]
                    if proceso_key == proceso:
                        mapeo_filtrado[id_tema] = v

           # print(f"{proceso}: {mapeo_filtrado}")
            mapeos_tema_etiqueta[proceso] = mapeo_filtrado

        except Exception as e:
            print(f"Error cargando LDA para {proceso}: {e}")

#GPR
vectorizador_gpr = joblib.load(os.path.join(GPR_MODELS_DIR, "modelo_vectorizador_tfidf.pkl"))

emociones_gpr = ["anticipación", "asco", "confianza", "enfado", "miedo", "sorpresa", "tristeza", "alegría"]
modelos_gpr = {}

for emocion in emociones_gpr:
    ruta_modelo = os.path.join(GPR_MODELS_DIR, f"modelo_gpr_{emocion}.pkl")
    if os.path.exists(ruta_modelo):
        modelos_gpr[emocion] = joblib.load(ruta_modelo)
    else:
        print(f"Modelo GPR para '{emocion}' no encontrado en {ruta_modelo}")


# Modelos de solicitud
class PQRSRequest(BaseModel):
    texto: str

class BayesianRequest(BaseModel):
    proceso: str
    emocion: str

class GPRRequest(BaseModel):
    texto: str

class BDLRequest(BaseModel):
    texto: str

class LDARequest(BaseModel):
    texto: str
    proceso: str

@app.get("/health", summary="Health Check", description="Verifica el estado de la API.")
def health_check():
    return {"status": "API funcionando correctamente"}

@app.get("/procesos_red_bayesiana", summary="Obtener Procesos Red Bayesiana", description="Obtiene los procesos válidos para la red bayesiana.")
def obtener_procesos_red_bayesiana():
    try:
        procesos_validos = le_proceso_rb.classes_.tolist()
        return {"procesos": procesos_validos}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"No se pudieron obtener los procesos: {str(e)}")

@app.post("/predict/naive_bayes", summary="Predicción Naive Bayes", description="Realiza predicción de emociones usando el modelo Naive Bayes.")
def predict_naive_bayes(request: PQRSRequest):
    try:
        if not request.texto.strip():
            raise HTTPException(status_code=400, detail="El texto no puede estar vacío.")
        vectorized_text = tfidf_vectorizer_nb.transform([request.texto])
        prediction = naive_bayes.predict(vectorized_text)
        predicted_emotions = mlb_nb.inverse_transform(prediction)
        return {"emociones": list(predicted_emotions[0]) if predicted_emotions else []}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/bayesian_deep_learning", summary="Predicción Bayesian Deep Learning", description="Realiza predicción de emociones usando el modelo Bayesian Deep Learning.")
def predict_bayesian_deep_learning(request: BDLRequest):
    try:
        vectorized_text = tfidf_vectorizer_bdl.transform([request.texto]).toarray()
        mc_samples = 10
        preds = [bayesian_deep_learning(vectorized_text, training=True).numpy() for _ in range(mc_samples)]
        pred_mean = np.mean(preds, axis=0)[0]
        emotion_labels = ["anticipación", "asco", "confianza", "enfado", "miedo", "sorpresa", "tristeza", "alegría"]
        emociones_detectadas = {
            emotion: float(prob) for emotion, prob in zip(emotion_labels, pred_mean) if prob > 0.2
        }
        return {"texto": request.texto, "emociones_detectadas": emociones_detectadas}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/bayesian_network", summary="Predicción Redes Bayesianas", description="Realiza predicción de emociones usando el modelo de Redes Bayesianas.")
def predict_bayesian_network(request: BayesianRequest):
    try:
        proceso = request.proceso.strip().lower()
        emocion = request.emocion.strip().lower()
        if emocion not in lista_emociones_rb:
            raise HTTPException(status_code=400, detail=f"Emoción no válida. Debe ser una de: {', '.join(lista_emociones_rb)}")
        if proceso not in le_proceso_rb.classes_:
            raise HTTPException(status_code=400, detail=f"Proceso no reconocido.")
        proceso_codificado = le_proceso_rb.transform([proceso])[0]
        resultado = inferencia_rb.query(variables=[emocion], evidence={"proceso": proceso_codificado})
        probabilidad = float(resultado.values[1])
        return {"proceso": proceso, "emocion": emocion, "probabilidad": round(probabilidad, 4)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/lda", summary="Predicción LDA", description="Realiza predicción de temas usando el modelo LDA.")
def predict_lda(request: LDARequest):
    try:
        texto = request.texto.strip()
        if not texto:
            raise HTTPException(status_code=400, detail="El texto no puede estar vacío.")

        proceso = request.proceso.strip()
        # Validar si existe el modelo exacto, si no usar "general"
        if proceso not in modelos_lda:
            #print(f"No se encontró modelo para '{proceso}', usando 'general'")
            proceso_usado = "general"
        else:
            proceso_usado = proceso

        modelo = modelos_lda[proceso_usado]
        diccionario = diccionarios_lda[proceso_usado]
        tokens = limpiar_texto(texto)
        bow = diccionario.doc2bow(tokens)
        distribucion = modelo.get_document_topics(bow)

        resultados = []
        for tema_id, prob in distribucion:
            etiqueta = mapeos_tema_etiqueta.get(proceso_usado, {}).get(str(tema_id), "tema no etiquetado")
            resultados.append({
                "tema": int(tema_id),
                "probabilidad": round(float(prob), 4),
                "etiqueta": etiqueta
            })

        # Ordenar de mayor a menor y tomar top 2
        resultados_filtrados = sorted(resultados, key=lambda x: x["probabilidad"], reverse=True)[:2]

        return {
            "proceso_utilizado": proceso_usado,
            "texto": request.texto,
            "temas_detectados": resultados_filtrados
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/gpr", summary="Predicción Procesos Gaussianos", description="Realiza predicción de emociones usando el modelo de Procesos Gaussianos.")
def predict_gpr(request: GPRRequest):
    try:
        texto = request.texto.strip()
        if not texto:
            raise HTTPException(status_code=400, detail="El texto no puede estar vacío.")

        X = vectorizador_gpr.transform([texto]).toarray()

        # Predecir con cada modelo GPR
        resultados = {}
        for emocion, modelo in modelos_gpr.items():
            prob = modelo.predict(X)[0]
            resultados[emocion] = round(float(prob), 4)

        # Filtrar emociones con probabilidad > 0.2 
        emociones_detectadas = {k: v for k, v in resultados.items() if v > 0.2}

        return {
            "texto": texto,
            "emociones_detectadas": emociones_detectadas
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict-file/all_models", summary="Análisis de archivo con todos los modelos", description="Analiza un archivo con todos los modelos de inferencia bayesiana.")
async def analizar_archivo_todos_modelos(file: UploadFile = File(...), emocion: str = None):
    try:
        contents = await file.read()
        extension = file.filename.split(".")[-1].lower()

        if extension == "csv":
            try:
                decoded = contents.decode("utf-8")
                if not decoded.strip():
                    raise ValueError("Archivo vacío o sin contenido válido.")
                df = pd.read_csv(io.StringIO(decoded))
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"No se pudo leer el archivo CSV: {str(e)}")
        elif extension in ["xls", "xlsx"]:
            try:
                df = pd.read_excel(io.BytesIO(contents))
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"No se pudo leer el archivo Excel: {str(e)}")
        else:
            raise HTTPException(status_code=400, detail="Formato no soportado. Usa CSV o Excel.")

        df.columns = df.columns.str.lower().str.strip()

        if "detalle" not in df.columns or "proceso" not in df.columns:
            raise HTTPException(status_code=400, detail="El archivo debe contener las columnas 'detalle' y 'proceso'.")

        df["detalle_procesado"] = df["detalle"].apply(procesar_detalle_api)

        resultados = []

        for i, row in df.iterrows():
            texto = row["detalle_procesado"]
            proceso = row["proceso"]

            fila = {
                "fila": i + 1,
                "detalle": row["detalle"],
                "proceso": proceso,
                "detalle_procesado": texto
            }

            # Naive bayes
            try:
                vector = tfidf_vectorizer_nb.transform([texto])
                pred = naive_bayes.predict(vector)
                emociones_nb = mlb_nb.inverse_transform(pred)
                fila["Naïve Bayes"] = list(emociones_nb[0]) if emociones_nb else []
            except Exception as e:
                fila["Naïve Bayes"] = "Error"

            # BDL
            try:
                vector = tfidf_vectorizer_bdl.transform([texto]).toarray()
                mc_samples = 10
                preds = [bayesian_deep_learning(vector, training=True).numpy() for _ in range(mc_samples)]
                pred_mean = np.mean(preds, axis=0)[0]
                emotion_labels = ["anticipación", "asco", "confianza", "enfado", "miedo", "sorpresa", "tristeza", "alegría"]
                emociones_bdl = {
                    emotion: float(prob) for emotion, prob in zip(emotion_labels, pred_mean) if prob > 0.2
                }
                fila["Bayesian Deep Learning"] = emociones_bdl
            except Exception as e:
                fila["Bayesian Deep Learning"] = "Error"

            # GPR
            try:
                X = vectorizador_gpr.transform([texto]).toarray()
                emociones_gpr = {}
                for emo, mod in modelos_gpr.items():
                    prob = mod.predict(X)[0]
                    if prob > 0.2:
                        emociones_gpr[emo] = round(float(prob), 4)
                fila["Procesos Gaussianos"] = emociones_gpr
            except Exception as e:
                fila["Procesos Gaussianos"] = "Error"

            # LDA
            try:
                proc_limpio = proceso.lower()
                if proc_limpio not in modelos_lda:
                    proc_limpio = "general"
                lda_model = modelos_lda[proc_limpio]
                dicc = diccionarios_lda[proc_limpio]
                tokens = limpiar_texto(texto)
                bow = dicc.doc2bow(tokens)
                distribucion = lda_model.get_document_topics(bow)
                temas = [{
                    "tema": int(t),
                    "probabilidad": round(float(p), 4),
                    "etiqueta": mapeos_tema_etiqueta.get(proc_limpio, {}).get(str(t), "tema no etiquetado")
                } for t, p in distribucion]
                fila["LDA"] = ", ".join([
                    f"Tema {t['tema']} - {t['etiqueta']} ({t['probabilidad']:.2f})"
                    for t in sorted(temas, key=lambda x: x["probabilidad"], reverse=True)[:2]
                ])
            except Exception as e:
                fila["LDA"] = "Error"

            # Red Bayesiana
            try:
                proc_limpio = proceso.strip().lower()
                if proc_limpio not in le_proceso_rb.classes_:
                    raise ValueError(f"Proceso no reconocido para red bayesiana: {proc_limpio}")
                proc_encoded = le_proceso_rb.transform([proc_limpio])[0]

                emociones_rb = ["alegría", "asco", "confianza", "enfado", "miedo", "sorpresa", "tristeza", "anticipación"]
                resultados_rb = {}

                for emocion in emociones_rb:
                    try:
                        resultado_rb = inferencia_rb.query(variables=[emocion], evidence={"proceso": proc_encoded})
                        probabilidad = round(float(resultado_rb.values[1]), 4)
                        resultados_rb[emocion] = probabilidad
                    except:
                        resultados_rb[emocion] = None

                fila["Redes Bayesianas"] = resultados_rb
            except Exception as e:
                fila["Redes Bayesianas"] = "Error"

            resultados.append(fila)

        return resultados

    except Exception as e:
        import traceback
        raise HTTPException(status_code=500, detail=str(e))

# Análisis de archivo completo modelo individual 
@app.post("/predict-file/{modelo}", summary="Análisis de archivo completo", description="Analiza un archivo completo usando un modelo específico.")
async def analizar_archivo_completo(modelo: str, file: UploadFile = File(...), emocion: str = None):
    try:
        contents = await file.read()
        extension = file.filename.split(".")[-1].lower()

        if extension == "csv":
            try:
                decoded = contents.decode("utf-8")
                if not decoded.strip():
                    raise ValueError("Archivo vacío o sin contenido válido.")
                df = pd.read_csv(io.StringIO(decoded))
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"No se pudo leer el archivo CSV: {str(e)}")
        elif extension in ["xls", "xlsx"]:
            df = pd.read_excel(io.BytesIO(contents))
        else:
            raise HTTPException(status_code=400, detail="Formato no soportado. Usa CSV o Excel.")

        # Normalizar nombres de columnas
        df.columns = df.columns.str.lower().str.strip()

        if "detalle" not in df.columns or "proceso" not in df.columns:
            raise HTTPException(status_code=400, detail="El archivo debe contener las columnas 'detalle' y 'proceso'.")

        # Preprocesar texto
        df["detalle_procesado"] = df["detalle"].apply(procesar_detalle_api)

        resultados = []

        for i, row in df.iterrows():
            texto = row["detalle_procesado"]
            proceso = row["proceso"]

            fila = {
                "fila": i + 1,
                "detalle": row["detalle"],
                "proceso": proceso,
                "detalle_procesado": texto
            }

            if modelo == "naive_bayes":
                vector = tfidf_vectorizer_nb.transform([texto])
                pred = naive_bayes.predict(vector)
                emociones = mlb_nb.inverse_transform(pred)
                fila["emociones"] = list(emociones[0]) if emociones else []

            elif modelo == "bayesian_deep_learning":
                vector = tfidf_vectorizer_bdl.transform([texto]).toarray()
                mc_samples = 10
                preds = [bayesian_deep_learning(vector, training=True).numpy() for _ in range(mc_samples)]
                pred_mean = np.mean(preds, axis=0)[0]
                emotion_labels = ["anticipación", "asco", "confianza", "enfado", "miedo", "sorpresa", "tristeza", "alegría"]
                emociones = {
                    emotion: float(prob) for emotion, prob in zip(emotion_labels, pred_mean) if prob > 0.2
                }
                fila["emociones_detectadas"] = emociones

            elif modelo == "gpr":
                X = vectorizador_gpr.transform([texto]).toarray()
                emociones = {}
                for emo, mod in modelos_gpr.items():
                    prob = mod.predict(X)[0]
                    if prob > 0.2:
                        emociones[emo] = round(float(prob), 4)
                fila["emociones_detectadas"] = emociones

            elif modelo == "lda":
                proc_limpio = proceso.lower()
                if proc_limpio not in modelos_lda:
                    proc_limpio = "general"

                lda_model = modelos_lda[proc_limpio]
                dicc = diccionarios_lda[proc_limpio]
                tokens = limpiar_texto(texto)
                bow = dicc.doc2bow(tokens)
                distribucion = lda_model.get_document_topics(bow)

                temas = [{
                    "tema": int(t),
                    "probabilidad": round(float(p), 4),
                    "etiqueta": mapeos_tema_etiqueta.get(proc_limpio, {}).get(str(t), "tema no etiquetado")
                } for t, p in distribucion]

                fila["Resultado"] = ", ".join([
                    f"Tema {t['tema']} - {t['etiqueta']} ({t['probabilidad']:.2f})"
                    for t in sorted(temas, key=lambda x: x["probabilidad"], reverse=True)[:2]
                ])

            elif modelo == "bayesian_network":
                proc_limpio = proceso.strip().lower()
                if proc_limpio not in le_proceso_rb.classes_:
                    raise HTTPException(status_code=400, detail="Proceso no reconocido para red bayesiana.")
                proc_encoded = le_proceso_rb.transform([proc_limpio])[0]

                emociones_rb = ["alegría", "asco", "confianza", "enfado", "miedo", "sorpresa", "tristeza", "anticipación"]
                resultados_rb = {}

                for emocion in emociones_rb:
                    try:
                        resultado = inferencia_rb.query(variables=[emocion], evidence={"proceso": proc_encoded})
                        probabilidad = round(float(resultado.values[1]), 4)
                        resultados_rb[emocion] = probabilidad
                    except:
                        resultados_rb[emocion] = None

                fila["Resultado"] = resultados_rb

            else:
                raise HTTPException(status_code=400, detail="Modelo no reconocido.")

            resultados.append(fila)

        # Eliminar columna 'temas_detectados' si existe
        for fila in resultados:
            fila.pop("temas_detectados", None)

        return resultados

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

import os
import uvicorn

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("main:app", host="0.0.0.0", port=port)



