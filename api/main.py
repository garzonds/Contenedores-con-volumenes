from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
import joblib
import numpy as np
from pathlib import Path

app = FastAPI(
    title="Penguin Species API",
    description="Predice la especie de un pingüino usando Random Forest (rf) o Logistic Regression (lr).",
    version="3.0.0"
)

MODELS_DIR = Path("/home/app/models")

SPECIES = {0: "Adelie", 1: "Chinstrap", 2: "Gentoo"}

# Alias cortos para los modelos
MODEL_ALIASES = {
    "rf": "penguins_random_forest.joblib",
    "lr": "penguins_logistic_regression.joblib",
}

DEFAULT_ALIAS = "lr"


def resolve_model_name(model: str) -> str:
    """Convierte alias corto (rf/lr) al nombre real del archivo."""
    return MODEL_ALIASES.get(model.lower(), model)


def load_model(model_name: str):
    model_path = MODELS_DIR / model_name
    if not model_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Modelo '{model_name}' no encontrado. Usa 'rf' o 'lr'."
        )
    return joblib.load(model_path)


def needs_scaling(model_name: str) -> bool:
    return "logistic" in model_name.lower()


# ── Schemas ──────────────────────────────────────────────────────────────────

class PenguinFeatures(BaseModel):
    bill_length_mm: float
    bill_depth_mm: float
    flipper_length_mm: float
    body_mass_g: float

    model_config = {
        "json_schema_extra": {
            "examples": [{
                "bill_length_mm": 45.0,
                "bill_depth_mm": 14.0,
                "flipper_length_mm": 210.0,
                "body_mass_g": 4500.0
            }]
        }
    }

class PredictResponse(BaseModel):
    model_used: str
    species_id: int
    species_name: str


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "message": "Penguin Species API corriendo ✓",
        "modelos_disponibles": MODEL_ALIASES,
        "modelo_por_defecto": DEFAULT_ALIAS,
        "docs": "/docs"
    }


@app.get("/models")
def list_models():
    """Lista los modelos disponibles y sus alias."""
    return {
        "aliases": MODEL_ALIASES,
        "default": DEFAULT_ALIAS,
        "uso": "Usa 'rf' para Random Forest o 'lr' para Logistic Regression en el parámetro ?model="
    }


@app.post("/predict", response_model=PredictResponse)
def predict(
    penguin: PenguinFeatures,
    model: str = Query(
        default=DEFAULT_ALIAS,
        description="Modelo a usar: 'rf' (Random Forest) o 'lr' (Logistic Regression). Por defecto: lr"
    )
):
    """
    Predice la especie de un pingüino.

    **Modelos disponibles:**
    - `lr` → Logistic Regression (default, 100% accuracy)
    - `rf` → Random Forest (97% accuracy)

    **Features requeridas:**
    - `bill_length_mm`: largo del pico en mm
    - `bill_depth_mm`: profundidad del pico en mm
    - `flipper_length_mm`: largo de la aleta en mm
    - `body_mass_g`: masa corporal en gramos
    """
    model_name = resolve_model_name(model)
    clf = load_model(model_name)

    X = np.array([[
        penguin.bill_length_mm,
        penguin.bill_depth_mm,
        penguin.flipper_length_mm,
        penguin.body_mass_g
    ]])

    if needs_scaling(model_name):
        scaler_path = MODELS_DIR / "penguins_scaler.joblib"
        if scaler_path.exists():
            scaler = joblib.load(scaler_path)
            X = scaler.transform(X)

    prediction = int(clf.predict(X)[0])

    return PredictResponse(
        model_used=model_name,
        species_id=prediction,
        species_name=SPECIES[prediction]
    )