import os
import glob
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier  # Added Random Forest
from sklearn.pipeline import Pipeline

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


# ============================================================
# DATA PATH
# ============================================================

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


# ============================================================
# MODEL PATH
# ============================================================

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


MODEL_FILE = os.path.join(
    MODEL_DIR,
    "resume_classifier.pkl"
)


# ============================================================
# LOAD TEXT FILES
# ============================================================

def load_text_files(
    folder,
    label
):

    texts = []
    labels = []

    files = glob.glob(
        os.path.join(
            folder,
            "*.txt"
        )
    )


    for file_path in files:

        # Ignore metadata files
        if file_path.endswith(
            "_metadata.txt"
        ):

            continue


        try:

            with open(
                file_path,
                "r",
                encoding="utf-8",
                errors="ignore"
            ) as file:

                text = file.read()


            text = " ".join(
                text.split()
            )


            # Ignore extremely poor OCR
            if len(text) < 100:

                print(
                    "SKIPPED:",
                    os.path.basename(file_path),
                    "-> too little text"
                )

                continue


            texts.append(
                text
            )

            labels.append(
                label
            )


        except Exception as e:

            print(
                "ERROR:",
                file_path
            )

            print(
                e
            )


    return texts, labels


# ============================================================
# START
# ============================================================

print("\n")
print("=" * 70)
print("RESUME / NON-RESUME MODEL COMPARISON & TRAINING")
print("=" * 70)


# ============================================================
# CHECK FOLDERS
# ============================================================

if not os.path.exists(
    RESUME_DIR
):

    print(
        "\nERROR: Resume extracted-text folder not found:"
    )

    print(
        RESUME_DIR
    )

    exit()


if not os.path.exists(
    NON_RESUME_DIR
):

    print(
        "\nERROR: Non-resume extracted-text folder not found:"
    )

    print(
        NON_RESUME_DIR
    )

    exit()


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading resumes...")

resume_texts, resume_labels = load_text_files(
    RESUME_DIR,
    1
)


print("Loading non-resumes...")

non_resume_texts, non_resume_labels = load_text_files(
    NON_RESUME_DIR,
    0
)


# ============================================================
# COMBINE
# ============================================================

texts = (
    resume_texts +
    non_resume_texts
)

labels = (
    resume_labels +
    non_resume_labels
)


# ============================================================
# DATASET SUMMARY
# ============================================================

print("\n")
print("=" * 70)
print("DATASET SUMMARY")
print("=" * 70)


print(
    f"Usable resumes     : {len(resume_texts)}"
)

print(
    f"Usable non-resumes : {len(non_resume_texts)}"
)

print(
    f"Total documents    : {len(texts)}"
)


if len(resume_texts) == 0 or len(non_resume_texts) == 0 or len(texts) < 20:

    print(
        "\nERROR: Dataset is missing classes or is too small to split."
    )

    exit()


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

print("\n")
print("=" * 70)
print("TRAIN / TEST SPLIT")
print("=" * 70)


X_train, X_test, y_train, y_test = train_test_split(

    texts,

    labels,

    test_size=0.20,

    random_state=42,

    stratify=labels

)


print(
    f"Training documents : {len(X_train)}"
)

print(
    f"Testing documents  : {len(X_test)}"
)


# ============================================================
# DEFINE MODELS
# ============================================================

print("\n")
print("=" * 70)
print("CREATING PIPELINES")
print("=" * 70)

# Shared vectorizer parameters
vectorizer = TfidfVectorizer(
    lowercase=True,
    strip_accents="unicode",
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True,
    max_features=50000
)

# Pipeline 1: Logistic Regression
lr_model = Pipeline([
    ("tfidf", vectorizer),
    ("classifier", LogisticRegression(
        C=2.0,
        max_iter=2000,
        class_weight="balanced",
        random_state=42
    ))
])

# Pipeline 2: Random Forest
rf_model = Pipeline([
    ("tfidf", vectorizer),
    ("classifier", RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1  # Uses all available CPU cores for speed
    ))
])


# ============================================================
# TRAIN MODELS
# ============================================================

print("\nTraining Logistic Regression model...")
lr_model.fit(X_train, y_train)

print("Training Random Forest model...")
rf_model.fit(X_train, y_train)

print("Training completed.")


# ============================================================
# EVALUATE
# ============================================================

print("\nEvaluating models...")

# Logistic Regression Predictions
y_pred_lr = lr_model.predict(X_test)

accuracy_lr = accuracy_score(y_test, y_pred_lr)
precision_lr = precision_score(y_test, y_pred_lr, pos_label=1, zero_division=0)
recall_lr = recall_score(y_test, y_pred_lr, pos_label=1, zero_division=0)
f1_lr = f1_score(y_test, y_pred_lr, pos_label=1, zero_division=0)

# Random Forest Predictions
y_pred_rf = rf_model.predict(X_test)

accuracy_rf = accuracy_score(y_test, y_pred_rf)
precision_rf = precision_score(y_test, y_pred_rf, pos_label=1, zero_division=0)
recall_rf = recall_score(y_test, y_pred_rf, pos_label=1, zero_division=0)
f1_rf = f1_score(y_test, y_pred_rf, pos_label=1, zero_division=0)


# ============================================================
# COMPARISON REPORT
# ============================================================

print("\n")
print("=" * 70)
print("MODEL COMPARISON SUMMARY")
print("=" * 70)

print(f"{'Metric':<15} | {'Logistic Regression':<20} | {'Random Forest':<20}")
print("-" * 70)
print(f"{'Accuracy':<15} | {accuracy_lr * 100:>18.2f}% | {accuracy_rf * 100:>18.2f}%")
print(f"{'Precision':<15} | {precision_lr * 100:>18.2f}% | {precision_rf * 100:>18.2f}%")
print(f"{'Recall':<15} | {recall_lr * 100:>18.2f}% | {recall_rf * 100:>18.2f}%")
print(f"{'F1 Score':<15} | {f1_lr * 100:>18.2f}% | {f1_rf * 100:>18.2f}%")
print("-" * 70)


# ============================================================
# CHOOSE & SAVE BEST MODEL
# ============================================================

# We use F1-Score as the primary comparison metric
if f1_rf > f1_lr:
    best_model = rf_model
    best_name = "Random Forest"
    best_f1 = f1_rf
else:
    best_model = lr_model
    best_name = "Logistic Regression"
    best_f1 = f1_lr

print(f"\nChoosing {best_name} as the final model (F1-Score: {best_f1 * 100:.2f}%).")

joblib.dump(
    best_model,
    MODEL_FILE
)

print("\n")
print("=" * 70)
print("BEST MODEL SAVED")
print("=" * 70)
print(f"Path: {MODEL_FILE}")
print(f"Model selected: {best_name}")
print("=" * 70)
print("\nTraining run completed.")