from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils import resample


# Stop if a model does not converge.
warnings.filterwarnings("error", category=ConvergenceWarning)

RANDOM_STATE = 42

url = (
    "https://archive.ics.uci.edu/ml/"
    "machine-learning-databases/statlog/german/german.data"
)

local_file = Path("data/german.data")
source = local_file if local_file.exists() else url

columns = [f"feature_{i}" for i in range(1, 21)] + ["target"]

df = pd.read_csv(
    source,
    sep=r"\s+",
    header=None,
    names=columns,
    na_values="?",
)

print("Dataset shape:", df.shape)
print("Original target counts:")
print(df["target"].value_counts().sort_index())

# Original UCI labels: 1 = Good, 2 = Bad.
assert set(df["target"].unique()) == {1, 2}

X = df.drop(columns="target")
y = df["target"].map({1: 0, 2: 1})

print("\nMapped target counts: 0 = Safe, 1 = Risk")
print(y.value_counts().sort_index())

print("\nMissing feature values:", X.isna().sum().sum())
# Reserve 20% of the data for final testing.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=RANDOM_STATE,
)

# Split the training data into fitting and validation sets.
X_fit, X_val, y_fit, y_val = train_test_split(
    X_train,
    y_train,
    test_size=0.25,
    stratify=y_train,
    random_state=RANDOM_STATE,
)

print("\nFitting rows:", len(X_fit))
print("Validation rows:", len(X_val))
print("Test rows:", len(X_test))

numeric_columns = X.select_dtypes(include="number").columns.tolist()
categorical_columns = X.select_dtypes(exclude="number").columns.tolist()


def make_preprocessor():
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False,
            ),
        ),
    ])

    return ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_columns),
        ("categorical", categorical_pipeline, categorical_columns),
    ])


def make_model(class_weight=None):
    return LogisticRegression(
        max_iter=1000,
        random_state=RANDOM_STATE,
        class_weight=class_weight,
    )
def evaluate(y_true, y_pred):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(
            y_true, y_pred, pos_label=1, zero_division=0
        ),
        "Recall": recall_score(
            y_true, y_pred, pos_label=1, zero_division=0
        ),
        "F1": f1_score(
            y_true, y_pred, pos_label=1, zero_division=0
        ),
    }


validation_model = Pipeline([
    ("preprocessing", make_preprocessor()),
    ("model", make_model()),
])

validation_model.fit(X_fit, y_fit)

risk_column = list(validation_model.classes_).index(1)
validation_probabilities = validation_model.predict_proba(X_val)[
    :, risk_column
]

validation_results = []

for threshold in [0.2, 0.3, 0.4]:
    predictions = (
        validation_probabilities >= threshold
    ).astype(int)

    validation_results.append({
        "Threshold": threshold,
        **evaluate(y_val, predictions),
    })

validation_table = pd.DataFrame(validation_results)

# Select the highest validation F1.
# If F1 values are equal, prefer the higher threshold.
best_row = validation_table.sort_values(
    ["F1", "Threshold"],
    ascending=[False, False],
).iloc[0]

selected_threshold = float(best_row["Threshold"])

print("\nValidation results:")
print(validation_table.round(4).to_string(index=False))
print("\nSelected threshold:", selected_threshold)
# Fit preprocessing on original training rows only.
preprocessor = make_preprocessor()
X_train_processed = preprocessor.fit_transform(X_train)

# 1. Baseline
baseline = make_model()
baseline.fit(X_train_processed, y_train)

# 2. Class weights
weighted_model = make_model(class_weight="balanced")
weighted_model.fit(X_train_processed, y_train)

# 3. Random oversampling
# Retain original rows and add Risk duplicates.
risk_positions = np.flatnonzero(y_train.to_numpy() == 1)
safe_count = int((y_train == 0).sum())

extra_risk_positions = resample(
    risk_positions,
    replace=True,
    n_samples=safe_count - len(risk_positions),
    random_state=RANDOM_STATE,
)

oversampled_positions = np.concatenate([
    np.arange(len(y_train)),
    extra_risk_positions,
])

X_oversampled = X_train_processed[oversampled_positions]
y_oversampled = y_train.iloc[oversampled_positions]

print("\nOversampled training counts:")
print(y_oversampled.value_counts().sort_index())

oversampled_model = make_model()
oversampled_model.fit(X_oversampled, y_oversampled)

# Transform the untouched test set.
X_test_processed = preprocessor.transform(X_test)

risk_column = list(baseline.classes_).index(1)
test_probabilities = baseline.predict_proba(X_test_processed)[
    :, risk_column
]

predictions = {
    "Baseline": baseline.predict(X_test_processed),
    "Class weights": weighted_model.predict(X_test_processed),
    f"Threshold moving ({selected_threshold:.1f})": (
        test_probabilities >= selected_threshold
    ).astype(int),
    "Random oversampling": oversampled_model.predict(
        X_test_processed
    ),
}

test_results = []

for technique, y_pred in predictions.items():
    test_results.append({
        "Technique": technique,
        **evaluate(y_test, y_pred),
    })

comparison_table = pd.DataFrame(test_results)

print("\nTEST RESULTS")
print(comparison_table.round(4).to_string(index=False))

comparison_table.to_csv(
    "comparison_results.csv",
    index=False,
)

validation_table.to_csv(
    "validation_results.csv",
    index=False,
)

print("\nCONFUSION MATRICES")

for technique, y_pred in predictions.items():
    print(f"\n{technique}")
    print(confusion_matrix(y_test, y_pred, labels=[0, 1]))

print("\nCONVERGENCE CHECK")

models = {
    "Validation baseline": validation_model.named_steps["model"],
    "Final baseline": baseline,
    "Class weights": weighted_model,
    "Random oversampling": oversampled_model,
}

for name, model in models.items():
    print(f"{name}: {model.n_iter_[0]} iterations")

print("\nAll models completed without convergence warnings.")
print("max_iter remained 1000.")