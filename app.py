# app.py
import gradio as gr
import joblib
import pandas as pd
from pathlib import Path

# 1) Cargar el modelo (pipeline: preprocess + model)
MODEL_PATH = Path(__file__).parent / "sleep_model.joblib"
pipe = joblib.load(MODEL_PATH)

TYPES = ["fuerza", "sparring", "movilidad", "cardio", "rest", "tecnica"]

# 2) Función que Gradio llamará
def predict_sleep(duration_min, rpe, hour, steps, hr_avg, daily_steps, _type):
    # Gradio siempre pasa valores "limpios" desde los inputs
    # Tu modelo espera además 'load' => lo calculamos aquí
    row = {
        "duration_min": duration_min,
        "rpe": rpe,
        "hour": hour,
        "steps": steps,
        "hr_avg": hr_avg,
        "daily_steps": daily_steps,
        "load": duration_min * rpe,
        "type": _type
    }
    X = pd.DataFrame([row])
    pred = pipe.predict(X)[0]
    return round(float(pred), 2)

# 3) Interfaz
with gr.Blocks(title="Sleep Predictor (MVP)") as demo:
    gr.Markdown("# 💤 Sleep Predictor (MVP)\nPredice horas de sueño a partir de la carga del día.")
    with gr.Row():
        duration_min = gr.Slider(0, 240, value=60, step=1, label="duration_min (min)")
        rpe          = gr.Slider(0, 10, value=5, step=1, label="rpe (1-10)")
        hour         = gr.Slider(0, 23, value=20, step=1, label="hour (0-23)")
    with gr.Row():
        steps        = gr.Slider(0, 30000, value=8000, step=100, label="steps")
        hr_avg       = gr.Slider(40, 220, value=130, step=1, label="hr_avg")
        daily_steps  = gr.Slider(0, 30000, value=10000, step=100, label="daily_steps")
    _type            = gr.Dropdown(TYPES, value="fuerza", label="type")
    btn              = gr.Button("Predecir sleep_hours")
    out              = gr.Number(label="Predicción (horas de sueño)")

    # ejemplos clicables (opcional, ayudan mucho)
    gr.Examples(
        examples=[
            [90, 7, 20, 9500, 138, 12000, "sparring"],
            [45, 4, 18, 6000, 120, 8000, "movilidad"],
            [60, 6, 19, 10000, 140, 12000, "fuerza"],
        ],
        inputs=[duration_min, rpe, hour, steps, hr_avg, daily_steps, _type],
        outputs=[out],
        fn=predict_sleep
    )

    btn.click(
        predict_sleep,
        inputs=[duration_min, rpe, hour, steps, hr_avg, daily_steps, _type],
        outputs=out
    )

if __name__ == "__main__":
    # Para local, Gradio abre tu navegador; para servidores usa server_name
    demo.launch()
    # demo.launch(server_name="0.0.0.0", server_port=7860)  # si lo corres en un server/VPS
