# Sleep Predictor (MVP)

Modelo de regresión (scikit-learn) que predice **sleep_hours** a partir de la **carga diaria de entrenamiento**.

## Archivos

- `train_model.py`: entrena y guarda `sleep_model.joblib`
- `predict.py`: usa el modelo guardado para predecir
- `workouts.csv`: datos (personales). Considera publicar una muestra.
- `sleep_model.joblib`: modelo entrenado (pipeline completo)
- `requirements.txt`: dependencias

## Uso

```bash
pip install -r requirements.txt
python predict.py --duration_min 90 --rpe 7 --hour 20 --steps 9500 --hr_avg 138 --daily_steps 12000 --type sparring
```
