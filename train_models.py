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

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_FILE   = os.path.join(BASE_DIR, "adult.csv")
MODEL_DIR   = os.path.join(BASE_DIR, "model")
TARGET      = "income"
TEST_FRAC   = 0.2
SEED        = 42

os.makedirs(MODEL_DIR, exist_ok=True)

data = pd.read_csv(DATA_FILE)
print("Loaded:", data.shape)

data = data.replace("?", np.nan)
print("Missing values after cleaning:\n", data.isna().sum()[data.isna().sum() > 0])

X       = data.drop(columns=[TARGET])
y_text  = data[TARGET]

encoder = LabelEncoder()
y       = encoder.fit_transform(y_text)
print("Target classes:", list(encoder.classes_))

numeric_cols     = X.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns.tolist()
categorical_cols = X.select_dtypes(include=["object", "string", "category"]).columns.tolist()
print("Numeric columns:", numeric_cols)
print("Categorical columns:", categorical_cols)

numeric_steps = Pipeline([
    ("fill", SimpleImputer(strategy="median")),
    ("scale", StandardScaler()),
])

categorical_steps = Pipeline([
    ("fill", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])

preprocess = ColumnTransformer([
    ("num", numeric_steps, numeric_cols),
    ("cat", categorical_steps, categorical_cols),
], sparse_threshold=0)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_FRAC, random_state=SEED, stratify=y
)
print("Train size:", X_train.shape, "| Test size:", X_test.shape)

model_zoo = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree":       DecisionTreeClassifier(random_state=SEED),
    "kNN":                 KNeighborsClassifier(),
    "Naive Bayes":         GaussianNB(),
    "Random Forest":       RandomForestClassifier(random_state=SEED),
}

def score_model(fitted, X_test, y_test):
    y_pred  = fitted.predict(X_test)
    y_proba = fitted.predict_proba(X_test)[:, 1]

    return {
        "Accuracy":  accuracy_score(y_test, y_pred),
        "AUC":       roc_auc_score(y_test, y_proba),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall":    recall_score(y_test, y_pred, zero_division=0),
        "F1":        f1_score(y_test, y_pred, zero_division=0),
        "MCC":       matthews_corrcoef(y_test, y_pred),
    }

all_results = {}
for name, clf in model_zoo.items():
    pipe = Pipeline([("prep", preprocess), ("model", clf)])
    pipe.fit(X_train, y_train)

    all_results[name] = score_model(pipe, X_test, y_test)

    save_name = os.path.join(MODEL_DIR, name.lower().replace(" ", "_") + ".joblib")
    joblib.dump(pipe, save_name, compress=3)
    print("Saved", save_name)

joblib.dump(encoder, os.path.join(MODEL_DIR, "label_encoder.joblib"), compress=3)

table = pd.DataFrame(all_results).T.round(4)
print("\n===== COMPARISON TABLE =====")
print(table)
table.to_csv(os.path.join(MODEL_DIR, "metrics_comparison.csv"))

test_out = X_test.copy()
test_out[TARGET] = encoder.inverse_transform(y_test)
test_data_path = os.path.join(BASE_DIR, "test_data.csv")
test_out.to_csv(test_data_path, index=False)
print(f"\nSaved {test_data_path}:", test_out.shape)
