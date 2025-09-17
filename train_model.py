import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

from sklearn.dummy import DummyRegressor

# ======================
# 1) Data
# ======================
df = pd.read_csv("workouts.csv")
target = "sleep_hours"
features_num = ["duration_min","rpe","hour","steps","hr_avg","daily_steps","load"]
features_cat = ["type"]

df = df.dropna(subset=[target])
X = df[features_num + features_cat].copy()
y = df[target].astype(float)

# Split holdout
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print("Train:", X_train.shape, " Test:", X_test.shape)

# ======================
# 2) Preprocess
# ======================
num_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())    # no es necesario para árboles, pero no molesta
])

cat_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocess = ColumnTransformer([
    ("num", num_pipe, features_num),
    ("cat", cat_pipe, features_cat)
])

# ======================
# 3) Modelo base
# ======================
base_model = RandomForestRegressor(
    n_estimators=300,
    n_jobs=-1,
    random_state=42
)

pipe = Pipeline([
    ("preprocess", preprocess),
    ("model", base_model)
])

# ======================
# 4) Entrenar + evaluar (baseline)
# ======================
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)


dummmy = DummyRegressor(strategy="mean")
dummmy.fit(X_train, y_train)
y_dummy = dummmy.predict(X_test)
print("Baseline: MAE", mean_absolute_error(y_test, y_dummy))
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
print(f"[BASE] MAE={mae:.2f} h | RMSE={rmse:.2f} h | R²={r2:.3f}")

# Cross-val (en train) para tener una idea de varianza
cv_mae = -cross_val_score(pipe, X_train, y_train, cv=5, scoring="neg_mean_absolute_error").mean()
print(f"[BASE] CV MAE={cv_mae:.2f} h")

# ======================
# 5) GridSearchCV (¡solo en train!)
# ======================
param_grid = {
    "model__n_estimators": [200, 400, 600],
    "model__max_depth": [None, 10, 20]
}

grid = GridSearchCV(
    pipe,
    param_grid=param_grid,
    cv=3,
    scoring="neg_mean_absolute_error",
    n_jobs=-1
)

grid.fit(X_train, y_train)

print("Mejores params:", grid.best_params_)
best_pipe = grid.best_estimator_




# Evalúa BEST en test holdout
y_pred_best = best_pipe.predict(X_test)
mae_best = mean_absolute_error(y_test, y_pred_best)
rmse_best = np.sqrt(mean_squared_error(y_test, y_pred_best))
r2_best = r2_score(y_test, y_pred_best)
print(f"[TUNED] MAE={mae_best:.2f} h | RMSE={rmse_best:.2f} h | R²={r2_best:.3f}")

# ======================
# 6) Importancia de features (para modelos de árbol)
# ======================
rf = best_pipe.named_steps["model"]
ct = best_pipe.named_steps["preprocess"]
ohe = ct.named_transformers_["cat"].named_steps["onehot"]

feature_names = features_num + list(ohe.get_feature_names_out(features_cat))
importances = rf.feature_importances_
top = sorted(zip(importances, feature_names), reverse=True)[:15]

print("\nTop features:")
for imp, name in top:
    print(f"{name:25s} {imp:.3f}")

# ======================
# 7) Guardar modelo final
# ======================
joblib.dump(best_pipe, "sleep_model.joblib")
print("Modelo final guardado en sleep_model.joblib ✅")
