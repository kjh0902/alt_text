from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from .data import DEFAULT_SPLIT, image_path, load_records, read_split, rule_flags, select_records
from .images import load_image
from .io import environment, file_sha, read_json, write_json, write_jsonl


def parser(command):
    p = argparse.ArgumentParser(description=f"KoAltQ {command}", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--train-dir", type=Path)
    p.add_argument("--test-dir", type=Path)
    p.add_argument("--run-dir", type=Path, default=Path("runs/qwen"))
    p.add_argument("--split-file", type=Path, default=DEFAULT_SPLIT)
    p.add_argument("--model-id", default="Qwen/Qwen3-VL-8B-Instruct")
    p.add_argument("--revision", default="main")
    p.add_argument("--model-cache-dir", type=str, default=None)
    p.add_argument("--max-seq-length", type=int, default=8192)
    p.add_argument("--max-image-tokens", type=int, default=512)
    p.add_argument("--l1-max-image-tokens", type=int, default=1024)
    p.add_argument("--l1-max-new-tokens", type=int, default=2048)
    p.add_argument("--checkpoint", type=Path, help="Checkpoint directory containing adapter/ and metadata.json")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lora-rank", type=int, default=8)
    p.add_argument("--lora-alpha", type=int, default=16)
    if command == "run_layer1":
        p.add_argument("--split", choices=("train", "validation", "test"), required=True)
    if command == "train":
        p.add_argument("--epochs", type=int, default=3)
        p.add_argument("--gradient-accumulation", type=int, default=8)
        p.add_argument("--learning-rate", type=float, default=1e-4)
        p.add_argument("--weight-decay", type=float, default=0.01)
        p.add_argument("--warmup-ratio", type=float, default=0.05)
        p.add_argument("--max-grad-norm", type=float, default=1.0)
        p.add_argument("--rho", type=float, default=0.99)
        p.add_argument("--save-steps", type=int, default=25)
        p.add_argument("--resume", action="store_true")
    if command == "predict":
        p.add_argument("--sample-submission", type=Path)
        p.add_argument("--output", type=Path, help="Defaults to RUN_DIR/submission.csv")
    return p


def prepare_data(args):
    from tqdm import tqdm
    if not args.train_dir:
        raise ValueError("--train-dir is required")
    rows = load_records(args.train_dir, labeled=True)
    parts, split_audit = read_split(rows, args.split_file)
    audit = {"split": split_audit, "environment": environment(), "datasets": {}}
    sources = [("train", args.train_dir, rows)]
    if args.test_dir:
        test = load_records(args.test_dir, labeled=False)
        if len(test) != 600:
            raise ValueError("Expected 600 test records")
        sources.append(("test", args.test_dir, test))
    for name, root, records in sources:
        image_audit, failures = [], []
        for row in tqdm(records, desc=f"Audit {name} images"):
            try:
                image, details = load_image(image_path(root, row))
                image.close()
                image_audit.append({"record_id": row["record_id"], **details})
            except Exception as exc:
                failures.append({"record_id": row["record_id"], "error": str(exc)})
        audit["datasets"][name] = {"count": len(records),
            "records_sha256": file_sha(root / "records_with_thumbnail.jsonl"),
            "thumbnail": dict(Counter(str(r["thumbnail"]).lower() for r in records)),
            "image_formats": dict(Counter(r["format"] for r in image_audit)),
            "repaired_iccp": [r["record_id"] for r in image_audit if r["removed_bad_iccp"]],
            "image_failures": failures}
        write_jsonl(args.run_dir / "audit" / f"{name}_images.jsonl", image_audit)
        if name == "test":
            parts["test"] = records
    for name, records in parts.items():
        write_jsonl(args.run_dir / "layer0" / f"{name}.jsonl",
                    [{"record_id": r["record_id"], **rule_flags(r)} for r in records])
    write_json(args.run_dir / "audit" / "dataset.json", audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    if any(a["image_failures"] for a in audit["datasets"].values()):
        raise RuntimeError("Image audit failed; inspect audit/dataset.json")


def run_evaluation(args, *, test=False):
    from .evaluation import save_evaluation, write_submission
    from .model import load_model, load_processor
    from .runtime import infer_rows, l1_contract, l2_contract, pin_model_revision, require_cache
    from .training import checkpoint_path
    pin_model_revision(args)
    split = "test" if test else "validation"
    rows, root = select_records(args, split)
    cache = require_cache(args, split, rows, root)
    checkpoint = checkpoint_path(args, best=True)
    contract = read_json(checkpoint / "metadata.json")["contract"]
    if contract["layer1"] != l1_contract(args) or contract["layer2"] != l2_contract(args):
        raise ValueError("Checkpoint input/model settings differ; pass the same settings used for training")
    model = load_model(args, adapter=checkpoint / "adapter")
    processor = load_processor(args.model_id, args.revision, args.max_image_tokens, args.model_cache_dir)
    predictions = infer_rows(args, model, processor, rows, root, cache)
    if test:
        template = args.sample_submission or root / "sample_submission.csv"
        output = args.output or args.run_dir / "submission.csv"
        write_submission(template, output, rows, predictions)
        write_jsonl(args.run_dir / "test_predictions.jsonl", predictions)
        print(f"Saved {len(predictions)} predictions to {output}")
    else:
        metrics = save_evaluation(args.run_dir / "validation" / "best", rows, predictions)
        print(json.dumps(metrics, ensure_ascii=False, indent=2))


def main(command):
    p = parser(command)
    args = p.parse_args()
    for key in ("max_seq_length", "max_image_tokens", "l1_max_image_tokens", "l1_max_new_tokens",
                "lora_rank", "lora_alpha", "epochs", "gradient_accumulation", "save_steps"):
        if hasattr(args, key) and getattr(args, key) < 1:
            p.error(f"--{key.replace('_', '-')} must be positive")
    if command == "prepare_data":
        prepare_data(args)
    elif command == "run_layer1":
        from .runtime import generate_layer1
        rows, root = select_records(args, args.split)
        generate_layer1(args, args.split, rows, root)
    elif command == "train":
        from .training import train
        rows = load_records(args.train_dir, labeled=True)
        parts, _ = read_split(rows, args.split_file)
        train(args, parts["train"], parts["validation"], args.train_dir)
    elif command == "evaluate":
        run_evaluation(args)
    elif command == "predict":
        run_evaluation(args, test=True)
    elif command == "smoke_test":
        from .smoke import smoke_test
        smoke_test(args)
    else:
        raise ValueError(command)
