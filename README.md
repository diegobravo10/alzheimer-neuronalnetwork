# alzheimer-neuronalnetwork

## Descripción del Problema
El diagnóstico clínico del Alzheimer suele ser complejo debido a la similitud de síntomas con el envejecimiento natural. Este proyecto utiliza Deep Learning para analizar patrones no lineales entre factores de riesgo (hipertensión, IMC, hábitos), métricas cognitivas (MMSE) y datos demográficos.

El flujo de trabajo incluye desde la limpieza de datos y reducción de dimensionalidad (PCA) hasta la optimización de hiperparámetros (Fine-Tuning) y el seguimiento de experimentos con MLOps.

## Stack Tecnológico

El proyecto fue desarrollado en Google Colab utilizando las siguientes tecnologías:

* **Procesamiento de Datos:** `Pandas`, `NumPy`.
* **Preprocesamiento:** `Scikit-learn` (StandardScaler, OneHotEncoder, PCA).
* **Modelado:** `TensorFlow`, `Keras`.
* **Optimización:** `Keras Tuner` (Hyperband Algorithm).
* **MLOps & Tracking:** `MLflow` (Registro de métricas, parámetros y modelos).
* **Visualización:** `Matplotlib`, `Seaborn`.

## Arquitectura y Fases

El desarrollo se dividió en 4 fases estratégicas:

1.  **Ingesta y EDA:** Análisis exploratorio, mapas de correlación y limpieza de identificadores.
2.  **Preprocesamiento:**
    * Transformación de variables (Estandarización y One-Hot Encoding).
    * **PCA (Principal Component Analysis):** Reducción de dimensionalidad conservando el 95% de la varianza.
3.  **Modelado y Optimización:**
    * *Modelo Base:* Arquitectura simple (RMSProp, Batch 16).
    * *Modelo Optimizado:* Arquitectura profunda (Adam, Batch 64, Dropout).
    * *Fine-Tuning:* Búsqueda automática de hiperparámetros.
4.  **Seguimiento con MLflow:**
    * Registro de experimentos para comparar *Loss* y *Accuracy*.
    * Comparativa de modelos (Base vs. Optimizado).
  
