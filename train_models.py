# train_models.py
# Trains 5 classifiers on the Adult Income dataset,
# computes 6 metrics each, and saves everything for the Streamlit app.

import os
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
)

# ---------------- SETTINGS ----------------
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_FILE   = os.path.join(BASE_DIR, "adult.csv")
MODEL_DIR   = os.path.join(BASE_DIR, "model")
TARGET      = "income"
TEST_FRAC   = 0.2
SEED        = 42

os.makedirs(MODEL_DIR, exist_ok=True)

# ---------------- 1. LOAD ----------------
data = pd.read_csv(DATA_FILE)
print("Loaded:", data.shape)

# ---------------- 2. CLEAN THE '?' JUNK ----------------
# this dataset marks missing values with '?', turn them into real blanks
data = data.replace("?", np.nan)
print("Missing values after cleaning:\n", data.isna().sum()[data.isna().sum() > 0])

# ---------------- 3. SPLIT INTO FEATURES (X) AND TARGET (y) ----------------
X       = data.drop(columns=[TARGET])
y_text  = data[TARGET]

# turn '<=50K'/'>50K' into 0/1 numbers
encoder = LabelEncoder()
y       = encoder.fit_transform(y_text)
print("Target classes:", list(encoder.classes_))   # e.g. ['<=50K', '>50K']

# ---------------- 4. DETECT COLUMN TYPES ----------------
numeric_cols     = X.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns.tolist()
categorical_cols = X.select_dtypes(include=["object", "string", "category"]).columns.tolist()
print("Numeric columns:", numeric_cols)
print("Categorical columns:", categorical_cols)

# ---------------- 5. PREPROCESSING RECIPE ----------------
# numbers: fill blanks with median, then scale to a common range
numeric_steps = Pipeline([
    ("fill", SimpleImputer(strategy="median")),
    ("scale", StandardScaler()),
])

# text: fill blanks with most common value, then one-hot encode
categorical_steps = Pipeline([
    ("fill", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])

preprocess = ColumnTransformer([
    ("num", numeric_steps, numeric_cols),
    ("cat", categorical_steps, categorical_cols),
], sparse_threshold=0)

# ---------------- 6. TRAIN / TEST SPLIT ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_FRAC, random_state=SEED, stratify=y
)
print("Train size:", X_train.shape, "| Test size:", X_test.shape)

# ---------------- 7. THE 5 MODELS ----------------
model_zoo = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree":       DecisionTreeClassifier(random_state=SEED),
    "kNN":                 KNeighborsClassifier(),
    "Naive Bayes":         GaussianNB(),
    "Random Forest":       RandomForestClassifier(random_state=SEED),
}

# ---------------- 8. METRIC FUNCTION (all 6) ----------------
def score_model(fitted, X_test, y_test):
    y_pred  = fitted.predict(X_test)
    y_proba = fitted.predict_proba(X_test)[:, 1]   # prob of the '>50K' class

    return {
        "Accuracy":  accuracy_score(y_test, y_pred),
        "AUC":       roc_auc_score(y_test, y_proba),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall":    recall_score(y_test, y_pred, zero_division=0),
        "F1":        f1_score(y_test, y_pred, zero_division=0),
        "MCC":       matthews_corrcoef(y_test, y_pred),
    }

# ---------------- 9. TRAIN, SCORE, SAVE EACH MODEL ----------------
all_results = {}
for name, clf in model_zoo.items():
    pipe = Pipeline([("prep", preprocess), ("model", clf)])
    pipe.fit(X_train, y_train)

    all_results[name] = score_model(pipe, X_test, y_test)

    save_name = os.path.join(MODEL_DIR, name.lower().replace(" ", "_") + ".joblib")
    # compress=3: Random Forest / kNN pickle to 100MB+ uncompressed, which
    # GitHub's push limit rejects outright. Compression is lossless.
    joblib.dump(pipe, save_name, compress=3)
    print("Saved", save_name)

# save the label encoder too (app needs it to read class names)
joblib.dump(encoder, os.path.join(MODEL_DIR, "label_encoder.joblib"), compress=3)

# ---------------- 10. COMPARISON TABLE ----------------
table = pd.DataFrame(all_results).T.round(4)
print("\n===== COMPARISON TABLE =====")
print(table)
table.to_csv(os.path.join(MODEL_DIR, "metrics_comparison.csv"))

# ---------------- 11. SAVE TEST DATA (for Streamlit upload) ----------------
test_out = X_test.copy()
test_out[TARGET] = encoder.inverse_transform(y_test)
test_data_path = os.path.join(BASE_DIR, "test_data.csv")
test_out.to_csv(test_data_path, index=False)
print(f"\nSaved {test_data_path}:", test_out.shape)