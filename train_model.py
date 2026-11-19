"""VISTA: train and evaluate FUNSD field classifier on the official held-out split.

This measures field labels for supplied annotation TEXT (question / answer /
header / other); it does not measure OCR or invoice/ID recognition.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "FUNSD"
MODELS = ROOT / "models"
MODELS.mkdir(parents=True, exist_ok=True)
LABELS = {"question", "answer", "header", "other"}


def load_split(split: str) -> pd.DataFrame:
    folder = DATA / f"{split}_data" / "annotations"
    if not folder.is_dir():
        raise FileNotFoundError(f"FUNSD annotations not found: {folder}")

    rows = []
    for path in sorted(folder.glob("*.json")):
        for field in json.loads(path.read_text(encoding="utf-8")).get("form", []):
            text = (field.get("text") or "").strip()
            label = field.get("label", "other")
            if text and label in LABELS:
                rows.append({"text": text, "label": label, "document": path.stem})
    if not rows:
        raise ValueError(f"No labeled fields found in {folder}")
    return pd.DataFrame(rows)


def train() -> dict:
    train_data = load_split("training")
    test_data = load_split("testing")

    # Character patterns help with noisy/short form text; word n-grams add context.
    # This configuration was chosen on a document-disjoint validation subset of
    # the training data. Official FUNSD testing forms are held out until here.
    features = FeatureUnion([
        ("characters", TfidfVectorizer(
            analyzer="char_wb", ngram_range=(2, 5), min_df=2,
            max_features=24000, sublinear_tf=True)),
        ("words", TfidfVectorizer(
            analyzer="word", ngram_range=(1, 2), min_df=2,
            max_features=14000, sublinear_tf=True)),
    ])
    model = Pipeline([
        ("tfidf", features),
        ("classifier", LogisticRegression(C=3.0, max_iter=400)),
    ])

    model.fit(train_data["text"].tolist(), train_data["label"].tolist())
    predictions = model.predict(test_data["text"].tolist())
    labels = list(model.classes_)
    score = float(accuracy_score(test_data["label"], predictions))
    macro_f1 = float(f1_score(test_data["label"], predictions,
                              average="macro", zero_division=0))
    report = classification_report(test_data["label"], predictions,
                                   labels=labels, zero_division=0, output_dict=True)
    confusion = confusion_matrix(test_data["label"], predictions,
                                 labels=labels).tolist()

    result = {
        "experiment": "FUNSD ground-truth text field classification",
        "model": "Character + word TF-IDF, Logistic Regression",
        "training_documents": int(train_data["document"].nunique()),
        "test_documents": int(test_data["document"].nunique()),
        "train_fields": int(len(train_data)),
        "test_fields": int(len(test_data)),
        "accuracy": score,
        "macro_f1": macro_f1,
        "labels": labels,
        "per_class": {key: report[key] for key in labels},
        "confusion_matrix": confusion,
        "note": "Accuracy is on labeled FUNSD text fields, NOT OCR or document-category classification.",
    }

    joblib.dump(model, MODELS / "funsd_classifier.joblib")
    (MODELS / "evaluation.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    pretty = classification_report(test_data["label"], predictions,
                                   labels=labels, zero_division=0)
    report_text = (
        f"VISTA / FUNSD Field Classification\n"
        f"Model: {result['model']}\n"
        f"Training docs: {result['training_documents']} / fields: {result['train_fields']}\n"
        f"Testing docs: {result['test_documents']} / fields: {result['test_fields']}\n"
        f"Held-out accuracy: {score:.4%}\nMacro F1: {macro_f1:.4f}\n\n"
        f"{pretty}\n"
        "IMPORTANT: This is field classification accuracy using the annotated text; "
        "it is NOT OCR or document-type accuracy.\n"
    )
    (MODELS / "evaluation.txt").write_text(report_text, encoding="utf-8")
    print(report_text)
    return result


if __name__ == "__main__":
    train()
