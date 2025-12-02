import os, numpy as np, pandas as pd
from joblib import load

MODEL_DIR = "models"
_cache = {}

def _load(name):
    path = os.path.join(MODEL_DIR, name)
    if name not in _cache:
        _cache[name] = load(path)
    return _cache[name]

def predict_crop_single(N,P,K,pH,temp,moist):
    model = _load("model_crop.pkl")
    le = _load("le_crop.pkl")
    pred = model.predict(np.array([[N,P,K,pH,temp,moist]]))[0]
    return le.inverse_transform([pred])[0]

def predict_crop_batch(df):
    model = _load("model_crop.pkl"); le = _load("le_crop.pkl")
    X = df[['N','P','K','pH','temperature','moisture']].values
    preds = model.predict(X)
    crops = le.inverse_transform(preds)
    df_out = df.copy(); df_out['predicted_crop'] = crops; return df_out

def predict_fertility_single(N,P,K,pH,temp,moist):
    model = _load("model_fertility.pkl"); le = _load("le_fertility.pkl")
    pred = model.predict(np.array([[N,P,K,pH,temp,moist]]))[0]
    return le.inverse_transform([pred])[0]

def predict_fertility_batch(df):
    model = _load("model_fertility.pkl"); le = _load("le_fertility.pkl")
    X = df[['N','P','K','pH','temperature','moisture']].values
    preds = model.predict(X)
    ferts = le.inverse_transform(preds)
    df_out = df.copy(); df_out['predicted_fertility'] = ferts; return df_out
