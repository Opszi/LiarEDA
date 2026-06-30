import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, mean_absolute_error
from scipy.sparse import hstack
from scipy import sparse

# ======================
# 📥 WCZYTANIE
# ======================

train = pd.read_csv("train.tsv", sep="\t", header=None)
test = pd.read_csv("test.tsv", sep="\t", header=None)
valid = pd.read_csv("valid.tsv", sep="\t", header=None)

columns = [
    "id","label","statement","subjects","speaker","job","state","party",
    "barely_true_count","false_count","half_true_count",
    "mostly_true_count","pants_fire_count","context"
]

train.columns = columns
test.columns = columns
valid.columns = columns

# ======================
# 🧠 LABEL (multiclass)
# ======================

label_map = {
    "pants-fire": 0,
    "false": 1,
    "barely-true": 2,
    "half-true": 3,
    "mostly-true": 4,
    "true": 5
}

for df in [train, valid, test]:
    df["label"] = df["label"].map(label_map)

# ======================
# 🧹 CLEANING
# ======================

for df in [train, valid, test]:
    df.dropna(subset=["statement"], inplace=True)
    df["subjects"] = df["subjects"].fillna("")
    df["context"] = df["context"].fillna("")
    df["length"] = df["statement"].apply(len)
    df["speaker"] = df["speaker"].fillna("unknown")
    df["party"] = df["party"].fillna("unknown")
    df["text_all"] = (
        df["statement"].astype(str) + " " +
        df["subjects"].astype(str) + " " +
        df["context"].astype(str)
    )

# filtr długości
train = train[(train["length"] > 20) & (train["length"] < 500)].copy()
valid = valid[(valid["length"] > 20) & (valid["length"] < 500)].copy()
test = test[(test["length"] > 20) & (test["length"] < 500)].copy()

# ======================
# 🔥 FEATURE ENGINEERING
# ======================

# -------- TEXT (TF-IDF)
vectorizer = TfidfVectorizer(
    max_features=20000,
    ngram_range=(1,2),
    stop_words='english',
    min_df=2,
    max_df=0.9
)

X_train_text = vectorizer.fit_transform(train["text_all"])
X_valid_text = vectorizer.transform(valid["text_all"])
X_test_text = vectorizer.transform(test["text_all"])

# -------- TEXT (char n-grams)
char_vectorizer = TfidfVectorizer(
    analyzer="char_wb",
    ngram_range=(3, 5),
    min_df=2,
    max_features=30000,
)
X_train_char = char_vectorizer.fit_transform(train["text_all"])
X_valid_char = char_vectorizer.transform(valid["text_all"])
X_test_char = char_vectorizer.transform(test["text_all"])

# -------- LENGTH
scaler = StandardScaler()

X_train_len = scaler.fit_transform(train[["length"]])
X_valid_len = scaler.transform(valid[["length"]])
X_test_len = scaler.transform(test[["length"]])

# -------- HISTORY COUNTS (numeryczne)
count_features = [
    "barely_true_count",
    "false_count",
    "half_true_count",
    "mostly_true_count",
    "pants_fire_count",
]

for df in [train, valid, test]:
    for col in count_features:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

history_scaler = StandardScaler()
X_train_hist = history_scaler.fit_transform(train[count_features])
X_valid_hist = history_scaler.transform(valid[count_features])
X_test_hist = history_scaler.transform(test[count_features])

# -------- PARTY (one-hot, fit only on train)
train_party_dummies = pd.get_dummies(train["party"])
X_train_party = train_party_dummies.values
X_valid_party = pd.get_dummies(valid["party"]).reindex(
    columns=train_party_dummies.columns, fill_value=0
).values
X_test_party = pd.get_dummies(test["party"]).reindex(
    columns=train_party_dummies.columns, fill_value=0
).values

# -------- SPEAKER (top-N)
top_n = 50
top_speakers = train["speaker"].value_counts().nlargest(top_n).index

def map_speaker(s):
    return s if s in top_speakers else "other"

train.loc[:, "speaker"] = train["speaker"].apply(map_speaker)
valid.loc[:, "speaker"] = valid["speaker"].apply(map_speaker)
test.loc[:, "speaker"] = test["speaker"].apply(map_speaker)

train_speaker_dummies = pd.get_dummies(train["speaker"])
X_train_speaker = train_speaker_dummies.values
X_valid_speaker = pd.get_dummies(valid["speaker"]).reindex(
    columns=train_speaker_dummies.columns, fill_value=0
).values
X_test_speaker = pd.get_dummies(test["speaker"]).reindex(
    columns=train_speaker_dummies.columns, fill_value=0
).values

# ======================
# 🔗 ŁĄCZENIE CECH
# ======================

X_train = hstack([
    X_train_text,
    X_train_char,
    X_train_len,
    X_train_hist,
    sparse.csr_matrix(X_train_party),
    sparse.csr_matrix(X_train_speaker),
])
X_valid = hstack([
    X_valid_text,
    X_valid_char,
    X_valid_len,
    X_valid_hist,
    sparse.csr_matrix(X_valid_party),
    sparse.csr_matrix(X_valid_speaker),
])
X_test = hstack([
    X_test_text,
    X_test_char,
    X_test_len,
    X_test_hist,
    sparse.csr_matrix(X_test_party),
    sparse.csr_matrix(X_test_speaker),
])

y_train = train["label"]
y_valid = valid["label"]
y_test = test["label"]

# ======================
# 🔄 MULTICLASS → BINARY
# ======================

def to_binary(y, threshold=4):
    return (y >= threshold).astype(int)

# ======================
# 🤖 MODEL (BINARY + TUNING)
# ======================

y_train_bin = to_binary(y_train)
y_valid_bin = to_binary(y_valid)
y_test_bin = to_binary(y_test)

threshold_grid = np.arange(0.35, 0.66, 0.01)

def best_threshold_from_scores(y_true, scores):
    best_thr = 0.5
    best_acc = -1.0
    for thr in threshold_grid:
        pred = (scores >= thr).astype(int)
        acc = accuracy_score(y_true, pred)
        if acc > best_acc:
            best_acc = acc
            best_thr = float(thr)
    return best_thr, best_acc

best_model = None
best_cfg = None
best_threshold = 0.5
best_valid_acc = -1.0

# ------- LinearSVC
svm_param_grid = [
    {"C": 0.25, "loss": "hinge"},
    {"C": 0.5, "loss": "hinge"},
    {"C": 1.0, "loss": "hinge"},
    {"C": 2.0, "loss": "hinge"},
    {"C": 0.25, "loss": "squared_hinge"},
    {"C": 0.5, "loss": "squared_hinge"},
    {"C": 1.0, "loss": "squared_hinge"},
    {"C": 2.0, "loss": "squared_hinge"},
]

for cfg in svm_param_grid:
    candidate = LinearSVC(
        C=cfg["C"],
        loss=cfg["loss"],
        class_weight="balanced",
        max_iter=10000,
        random_state=42,
    )
    candidate.fit(X_train, y_train_bin)
    valid_scores = candidate.decision_function(X_valid)
    valid_scores_prob = 1.0 / (1.0 + np.exp(-valid_scores))
    thr, valid_acc = best_threshold_from_scores(y_valid_bin, valid_scores_prob)
    if valid_acc > best_valid_acc:
        best_valid_acc = valid_acc
        best_cfg = cfg
        best_model = candidate
        best_threshold = thr

print(
    "Best model: LinearSVC",
    "| params:", best_cfg,
    "| threshold:", round(best_threshold, 3),
    "| Validation Accuracy:", best_valid_acc
)

# ======================
# 📊 WALIDACJA
# ======================

valid_scores = best_model.decision_function(X_valid)
valid_scores = 1.0 / (1.0 + np.exp(-valid_scores))

y_pred_valid_bin = (valid_scores >= best_threshold).astype(int)

print("\n=== VALIDATION (BINARY) ===")
print("Accuracy:", accuracy_score(y_valid_bin, y_pred_valid_bin))
print("MAE:", mean_absolute_error(y_valid_bin, y_pred_valid_bin))
print(classification_report(y_valid_bin, y_pred_valid_bin))

# ======================
# 🧪 TEST
# ======================

test_scores = best_model.decision_function(X_test)
test_scores = 1.0 / (1.0 + np.exp(-test_scores))

y_pred_test_bin = (test_scores >= best_threshold).astype(int)

print("\n=== TEST (BINARY) ===")
print("Accuracy:", accuracy_score(y_test_bin, y_pred_test_bin))
print("MAE:", mean_absolute_error(y_test_bin, y_pred_test_bin))
print(classification_report(y_test_bin, y_pred_test_bin))

# ======================
# 💾 ZAPIS MODELU
# ======================

model_bundle = {
    "model": best_model,
    "threshold": best_threshold,
    "label_threshold": 4,
    "vectorizer_word": vectorizer,
    "vectorizer_char": char_vectorizer,
    "length_scaler": scaler,
    "history_scaler": history_scaler,
    "count_features": count_features,
    "party_columns": train_party_dummies.columns.tolist(),
    "speaker_columns": train_speaker_dummies.columns.tolist(),
    "top_speakers": list(top_speakers),
}

with open("best_model.pkl", "wb") as f:
    pickle.dump(model_bundle, f)

print("\nModel zapisany do: best_model.pkl")