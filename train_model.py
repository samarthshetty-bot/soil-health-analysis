import os, pandas as pd
from joblib import dump
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

DATA_PATH = "data/soil_data_large_1000.csv"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

def load_data():
    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.strip().str.lower()
    X = df[['n','p','k','ph','temperature','moisture']]
    y_crop = df['crop']
    y_fert = df['fertility']
    return X, y_crop, y_fert

def train_and_save():
    X, y_crop, y_fert = load_data()
    le_crop = LabelEncoder()
    y_crop_enc = le_crop.fit_transform(y_crop)
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X, y_crop_enc, test_size=0.2, random_state=42, stratify=y_crop_enc)
    crop_model = RandomForestClassifier(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1)
    crop_model.fit(X_train_c, y_train_c)
    print("Crop model accuracy:", accuracy_score(y_test_c, crop_model.predict(X_test_c)))
    print(classification_report(y_test_c, crop_model.predict(X_test_c), target_names=le_crop.classes_))
    dump(crop_model, os.path.join(MODEL_DIR, "model_crop.pkl"))
    dump(le_crop, os.path.join(MODEL_DIR, "le_crop.pkl"))

    le_fert = LabelEncoder()
    y_fert_enc = le_fert.fit_transform(y_fert)
    X_train_f, X_test_f, y_train_f, y_test_f = train_test_split(X, y_fert_enc, test_size=0.2, random_state=42, stratify=y_fert_enc)
    fert_model = RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42, n_jobs=-1)
    fert_model.fit(X_train_f, y_train_f)
    print("Fertility model accuracy:", accuracy_score(y_test_f, fert_model.predict(X_test_f)))
    print(classification_report(y_test_f, fert_model.predict(X_test_f), target_names=le_fert.classes_))
    dump(fert_model, os.path.join(MODEL_DIR, "model_fertility.pkl"))
    dump(le_fert, os.path.join(MODEL_DIR, "le_fertility.pkl"))

if __name__ == "__main__":
    train_and_save()
