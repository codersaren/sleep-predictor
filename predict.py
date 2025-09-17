# predict.py
import argparse
import joblib
import pandas as pd

# 1) Carga del pipeline entrenado
pipe = joblib.load("sleep_model.joblib")

# 2) Argumentos CLI
parser = argparse.ArgumentParser(description="Predicción de sleep_hours")
parser.add_argument("--duration_min", type=float, required=True)
parser.add_argument("--rpe", type=float, required=True)
parser.add_argument("--hour", type=float, default=18)
parser.add_argument("--steps", type=float, required=True)
parser.add_argument("--hr_avg", type=float, required=True)
parser.add_argument("--daily_steps", type=float, required=True)
parser.add_argument("--type", type=str, required=True,
                    choices=["fuerza","sparring","movilidad","cardio","rest","tecnica"])
args = parser.parse_args()

# 3) Construir el registro de entrada (¡recuerda que tu modelo espera 'load'!)
row = {
    "duration_min": args.duration_min,
    "rpe": args.rpe,
    "hour": args.hour,
    "steps": args.steps,
    "hr_avg": args.hr_avg,
    "daily_steps": args.daily_steps,
    "load": args.duration_min * args.rpe,
    "type": args.type
}

X_new = pd.DataFrame([row])

# 4) Predicción
pred = float(pipe.predict(X_new)[0])
print(f"Predicción sleep_hours: {pred:.2f} h")
