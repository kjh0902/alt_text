from __future__ import annotations

import csv
from pathlib import Path

from sklearn.metrics import classification_report, confusion_matrix, f1_score

from . import LABELS
from .io import write_json, write_jsonl


def checked_predictions(rows, predictions):
    mapped = {p["record_id"]: p["prediction"] for p in predictions}
    expected = {r["record_id"] for r in rows}
    if len(mapped) != len(predictions) or set(mapped) != expected:
        raise ValueError("Prediction IDs must exactly match input IDs, without duplicates")
    if any(label not in LABELS for label in mapped.values()):
        raise ValueError("Invalid prediction label")
    return mapped


def metrics_for(rows, predictions):
    mapped = checked_predictions(rows, predictions)
    truth = [r["label"] for r in rows]
    predicted = [mapped[r["record_id"]] for r in rows]
    macro = f1_score(truth, predicted, labels=list(LABELS), average="macro", zero_division=0)
    binary = f1_score([x != "적절" for x in truth], [x != "적절" for x in predicted], zero_division=0)
    return {"n_records": len(rows), "macro_f1_7class": float(macro), "binary_f1": float(binary),
            "binary_positive": "부적절", "total_score": float(0.5 * (macro + binary)),
            "per_class": classification_report(truth, predicted, labels=list(LABELS), output_dict=True, zero_division=0),
            "confusion_matrix": confusion_matrix(truth, predicted, labels=list(LABELS)).tolist(),
            "label_order": list(LABELS)}


def save_evaluation(directory, rows, predictions):
    directory = Path(directory)
    result = metrics_for(rows, predictions)
    write_json(directory / "metrics.json", result)
    write_jsonl(directory / "predictions.jsonl", predictions)
    return result


def write_submission(template, destination, rows, predictions):
    template, destination = Path(template), Path(destination)
    if template.resolve() == destination.resolve():
        raise ValueError("Submission output must not overwrite sample_submission.csv")
    mapped = checked_predictions(rows, predictions)
    with template.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fields, submission = reader.fieldnames, list(reader)
    if fields != ["record_id", "label"]:
        raise ValueError("Submission template must have record_id,label columns")
    ids = [r["record_id"] for r in submission]
    if len(ids) != len(set(ids)) or set(ids) != set(mapped):
        raise ValueError("Submission IDs differ from test predictions")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows({"record_id": rid, "label": mapped[rid]} for rid in ids)
