"""Reproduce the FUNSD text-field classification experiment (document-wise split)."""
from __future__ import annotations
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data' / 'FUNSD'
MODEL_DIR = ROOT / 'models'
MODEL_DIR.mkdir(exist_ok=True)
VALID_LABELS = {'question', 'answer', 'header', 'other'}


def load_split(split: str) -> pd.DataFrame:
    folder = DATA / f'{split}_data' / 'annotations'
    if not folder.is_dir():
        raise FileNotFoundError(f'Cannot find FUNSD dataset folder: {folder}')
    rows = []
    for path in sorted(folder.glob('*.json')):
        form = json.loads(path.read_text(encoding='utf-8')).get('form', [])
        for field in form:
            value = (field.get('text') or '').strip()
            label = field.get('label', 'other')
            if value and label in VALID_LABELS:
                rows.append({'text': value, 'label': label, 'document': path.stem})
    if not rows:
        raise ValueError(f'No training rows found in {folder}')
    return pd.DataFrame(rows)


def train() -> dict:
    train = load_split('training')
    test = load_split('testing')
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 5), min_df=2, max_features=25000)),
        ('classifier', LogisticRegression(max_iter=600, class_weight='balanced')),
    ])
    pipeline.fit(train.text, train.label)
    predicted = pipeline.predict(test.text)
    acc = float(accuracy_score(test.label, predicted))
    labels = list(pipeline.classes_)
    per_label = classification_report(test.label, predicted, labels=labels, zero_division=0, output_dict=True)
    conf = confusion_matrix(test.label, predicted, labels=labels).tolist()
    result = {
        'experiment': 'FUNSD field-label text classification',
        'training_documents': int(train.document.nunique()),
        'test_documents': int(test.document.nunique()),
        'train_fields': int(len(train)),
        'test_fields': int(len(test)),
        'accuracy': acc,
        'per_class': {label: per_label[label] for label in labels},
        'labels': labels,
        'confusion_matrix': conf,
        'note': 'Results use ground-truth text annotations; they do not measure OCR performance or document category classification.',
    }
    joblib.dump(pipeline, MODEL_DIR / 'funsd_classifier.joblib')
    (MODEL_DIR / 'evaluation.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    report = classification_report(test.label, predicted, labels=labels, zero_division=0)
    text = (f'FUNSD annotated text-field classification\nTraining documents: {len(train.document.unique())}\n'
            f'Test documents: {len(test.document.unique())}\nTraining fields: {len(train)}\n'
            f'Test fields: {len(test)}\nAccuracy: {acc:.4f}\n\n{report}\n'
            f'Labels: {labels}\nConfusion matrix:\n{conf}\n\n'
            'NOTES: This accuracy measures annotation text field classification, NOT OCR accuracy.\n')
    (MODEL_DIR / 'evaluation.txt').write_text(text, encoding='utf-8')
    print(text)
    print('Saved trained FUNSD model and evaluation results successfully.')
    return result


if __name__ == '__main__':
    train()
