from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import traceback
import numpy as np
import pandas as pd
import sklearn
from scipy import sparse
from scipy.sparse import hstack


app = FastAPI(title="Liar Model API")

try:
    with open("best_model.pkl", "rb") as f:
        bundle = pickle.load(f)
except Exception as exc:
    raise RuntimeError(f"Nie udalo sie wczytac best_model.pkl: {exc}") from exc

model = bundle["model"]
threshold = bundle["threshold"]
vectorizer_word = bundle["vectorizer_word"]
vectorizer_char = bundle["vectorizer_char"]
length_scaler = bundle["length_scaler"]
history_scaler = bundle["history_scaler"]
count_features = bundle["count_features"]
party_columns = bundle["party_columns"]
speaker_columns = bundle["speaker_columns"]
top_speakers = set(bundle["top_speakers"])


class PredictRequest(BaseModel):
    statement: str
    subjects: str = ""
    context: str = ""
    party: str = "unknown"
    speaker: str = "unknown"
    barely_true_count: float = 0.0
    false_count: float = 0.0
    half_true_count: float = 0.0
    mostly_true_count: float = 0.0
    pants_fire_count: float = 0.0


def build_features(req: PredictRequest):
    speaker = req.speaker if req.speaker in top_speakers else "other"
    text_all = f"{req.statement} {req.subjects} {req.context}"

    df = pd.DataFrame([{
        "text_all": text_all,
        "length": len(req.statement),
        "party": req.party,
        "speaker": speaker,
        "barely_true_count": req.barely_true_count,
        "false_count": req.false_count,
        "half_true_count": req.half_true_count,
        "mostly_true_count": req.mostly_true_count,
        "pants_fire_count": req.pants_fire_count,
    }])

    x_text = vectorizer_word.transform(df["text_all"])
    x_char = vectorizer_char.transform(df["text_all"])
    x_len = length_scaler.transform(df[["length"]])
    x_hist = history_scaler.transform(df[count_features])

    party_d = pd.get_dummies(df["party"]).reindex(columns=party_columns, fill_value=0).values
    speaker_d = pd.get_dummies(df["speaker"]).reindex(columns=speaker_columns, fill_value=0).values

    return hstack([
        x_text,
        x_char,
        x_len,
        x_hist,
        sparse.csr_matrix(party_d),
        sparse.csr_matrix(speaker_d),
    ])


@app.get("/health")
def health():
    return {
        "status": "ok",
        "sklearn_version": sklearn.__version__,
        "threshold": float(threshold),
    }


@app.post("/predict")
def predict(req: PredictRequest):
    try:
        x = build_features(req)
        scores = model.decision_function(x)
        probs = 1.0 / (1.0 + np.exp(-np.asarray(scores).ravel()))
        prob_true = float(probs[0])
        prediction = int(prob_true >= threshold)

        return {
            "prediction": prediction,
            "prob_true": prob_true,
            "threshold": float(threshold),
        }
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail={
                "error": str(exc),
                "traceback": traceback.format_exc(),
            },
        )
