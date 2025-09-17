# 💤 Sleep Predictor (MVP)

[![Hugging Face Spaces](https://img.shields.io/badge/🤗-Hugging%20Face%20Space-blue)](https://huggingface.co/spaces/codersaren/sleep-predictor)

Modelo de **IA en Python** que predice las **horas de sueño** a partir de variables de entrenamiento y actividad física.  
Entrenado con datos personales de sesiones de Taekwondo y actividad diaria.

---

## ✨ ¿Qué hace?

- **Input**: duración del entrenamiento, RPE (intensidad), hora, pasos, FC promedio, pasos diarios, tipo de entrenamiento.
- **Output**: predicción de `sleep_hours` (en horas).

Este MVP busca explorar la relación entre carga de entrenamiento y descanso nocturno.

---

## 📊 Métricas (holdout test)

- **Baseline (media)**: MAE ≈ 0.33 h
- **Modelo (RandomForestRegressor)**: MAE ≈ 0.29 h
- **R²**: ≈ 0.19

> Mejora frente al baseline → el modelo aprende patrones reales, aunque aún limitado por tamaño del dataset.

---

## 🚀 Demo online

👉 Pruébalo directamente en Hugging Face Spaces:  
🔗 [**Sleep Predictor en HF**](https://huggingface.co/spaces/codersaren/sleep-predictor)

---

## ⚙️ Cómo usarlo localmente

### 1. Clonar el repo

```bash
git clone https://github.com/codersaren/sleep-predictor
cd sleep-predictor
```

### 2. Instalar dependencias

pip install -r requirements.txt

### 3. Ejecutar la app Gradio

python app.py

Abre el link local en tu navegador (http://127.0.0.1:7860
).

🛠️ Tecnologías

Python 3.10+

scikit-learn

pandas / numpy

joblib

Gradio

🗺️ Roadmap

Añadir más features (cafeína, siestas, fin de semana, etc.).

Probar otros modelos (HistGradientBoostingRegressor, redes neuronales).

Validación temporal (entrenar en pasado, predecir futuro).

Recomendaciones automáticas según la predicción de sueño.

⚠️ Disclaimer

Este modelo es un prototipo experimental entrenado con datos personales.
No debe usarse como consejo médico ni deportivo profesional.

✍️ Autor: codersaren
