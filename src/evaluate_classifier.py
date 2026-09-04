import os
import glob
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# ============================================================
# PROJECT PATH
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DATASET_DIR = os.path.join(
    BASE_DIR,
    "extracted_text"
)

RESUME_DIR = os.path.join(
    DATASET_DIR,
    "resume"
)

NON_RESUME_DIR = os.path.join(
    DATASET_DIR,
    "non_resume"
)

MODEL_FILE = os.path.join(
    BASE_DIR,
    "models",
    "resume_classifier.pkl"
)


# ============================================================
# LOAD TEXT FILES
# ============================================================

def load_text_files(folder, label):

    texts = []
    labels = []

    files = glob.glob(
        os.path.join(folder, "*.txt")
    )

    for file_path in files:

        if file_path.endswith("_metadata.txt"):
            continue

        try:

            with open(
                file_path,
                "r",
                encoding="utf-8",
                errors="ignore"
            ) as file:

                text = file.read()

            text = " ".join(text.split())

            if len(text) < 100:
                continue

            texts.append(text)
            labels.append(label)

        except Exception as e:

            print("ERROR:", file_path)
            print(e)

    return texts, labels


# ============================================================
# START
# ============================================================

print("\n")
print("=" * 70)
print("RESUME CLASSIFIER EVALUATION")
print("=" * 70)


# ============================================================
# LOAD DATA
# ============================================================

resume_texts, resume_labels = load_text_files(
    RESUME_DIR,
    1
)

non_resume_texts, non_resume_labels = load_text_files(
    NON_RESUME_DIR,
    0
)


texts = resume_texts + non_resume_texts
labels = resume_labels + non_resume_labels


print("\nTotal documents:", len(texts))


# ============================================================
# RECREATE SAME TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(

    texts,
    labels,

    test_size=0.20,

    random_state=42,

    stratify=labels
)


print("Testing documents:", len(X_test))


# ============================================================
# LOAD EXISTING MODEL
# ============================================================

print("\nLoading existing model...")

model = joblib.load(MODEL_FILE)

print("Model loaded successfully.")


# ============================================================
# PREDICTION
# ============================================================

print("\nRunning prediction...")

y_pred = model.predict(X_test)


# ============================================================
# CALCULATE METRICS
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    pos_label=1,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    pos_label=1,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    pos_label=1,
    zero_division=0
)


# ============================================================
# DISPLAY PERFORMANCE
# ============================================================

print("\n")
print("=" * 70)
print("MODEL PERFORMANCE")
print("=" * 70)

print(
    f"Accuracy  : {accuracy * 100:.2f}%"
)

print(
    f"Precision : {precision * 100:.2f}%"
)

print(
    f"Recall    : {recall * 100:.2f}%"
)

print(
    f"F1 Score  : {f1 * 100:.2f}%"
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\n")
print("=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

print(
    classification_report(

        y_test,
        y_pred,

        labels=[0, 1],

        target_names=[
            "Non-Resume",
            "Resume"
        ],

        zero_division=0
    )
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

matrix = confusion_matrix(
    y_test,
    y_pred,
    labels=[0, 1]
)

print("\n")
print("=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

print(
    "                    Predicted"
)

print(
    "                 Non-Resume  Resume"
)

print(
    f"Actual Non-Resume    {matrix[0][0]:4d}      {matrix[0][1]:4d}"
)

print(
    f"Actual Resume        {matrix[1][0]:4d}      {matrix[1][1]:4d}"
)


# ============================================================
# LOGISTIC REGRESSION ITERATIONS
# ============================================================

classifier = model.named_steps["classifier"]

print("\n")
print("=" * 70)
print("LOGISTIC REGRESSION")
print("=" * 70)

print(
    "Actual iterations:",
    classifier.n_iter_
)

print("\nEvaluation completed.")