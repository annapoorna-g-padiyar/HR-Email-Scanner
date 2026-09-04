import os
import glob
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
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
print("RESUME / NON-RESUME ML TRAINING")
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

    print(
        "\nRun:"
    )

    print(
        "python extract_dataset.py"
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

    print(
        "\nRun:"
    )

    print(
        "python extract_dataset.py"
    )

    exit()


# ============================================================
# LOAD RESUMES
# ============================================================

print("\nLoading resumes...")

resume_texts, resume_labels = load_text_files(
    RESUME_DIR,
    1
)


# ============================================================
# LOAD NON-RESUMES
# ============================================================

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


# ============================================================
# VALIDATION
# ============================================================

if len(resume_texts) == 0:

    print(
        "\nERROR: No usable resumes found."
    )

    exit()


if len(non_resume_texts) == 0:

    print(
        "\nERROR: No usable non-resumes found."
    )

    exit()


if len(texts) < 20:

    print(
        "\nERROR: Dataset is too small."
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
# MODEL
# ============================================================

print("\n")
print("=" * 70)
print("CREATING MODEL")
print("=" * 70)


model = Pipeline([

    (
        "tfidf",

        TfidfVectorizer(

            lowercase=True,

            strip_accents="unicode",

            ngram_range=(1, 2),

            min_df=2,

            max_df=0.95,

            sublinear_tf=True,

            max_features=50000

        )
    ),

    (
        "classifier",

        LogisticRegression(

            C=2.0,

            max_iter=2000,

            class_weight="balanced",

            random_state=42

        )
    )

])


# ============================================================
# TRAIN
# ============================================================

print("\nTraining model...")


model.fit(X_train, y_train)


print(
    "Training completed."
)


# ============================================================
# PREDICTION
# ============================================================

print("\nTesting model...")

y_pred = model.predict(
    X_test
)


# ============================================================
# METRICS
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
# DISPLAY RESULTS
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
# SAVE MODEL
# ============================================================

joblib.dump(
    model,
    MODEL_FILE
)


print("\n")
print("=" * 70)
print("MODEL SAVED")
print("=" * 70)


print(
    MODEL_FILE
)


print("\n")
print(
    "Training completed successfully."
)