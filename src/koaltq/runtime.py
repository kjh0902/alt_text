from __future__ import annotations

import json
from pathlib import Path

import torch
from tqdm import tqdm

from .data import image_path
from .images import load_image
from .io import append_jsonl, digest, file_sha, read_json, read_jsonl, write_json, write_jsonl
from .model import load_model, load_processor, resolve_revision
from .prompts import L1_SYSTEM, L2_SYSTEM, l1_payload, l2_payload, parse_analysis, prepare_input, validate_analysis
from .scoring import label_token_ids, prediction_row, score_candidates

PREPROCESS_VERSION = "rgb_white_first_frame_iccp_v1"
TRUNCATION_VERSION = "longest_field_right_binary_v2"


def pin_model_revision(args):
    path = Path(args.run_dir) / "model_revision.json"
    if path.exists():
        saved = read_json(path)
        if saved["model_id"] != args.model_id or args.revision not in ("main", saved["revision"]):
            raise ValueError("Model identity changed; use a new --run-dir")
        args.revision = saved["revision"]
    else:
        args.revision = resolve_revision(args.model_id, args.revision)
        write_json(path, {"model_id": args.model_id, "revision": args.revision})


def l1_contract(args):
    return {"model_id": args.model_id, "revision": args.revision, "prompt": L1_SYSTEM,
            "max_seq_length": args.max_seq_length, "max_image_tokens": args.l1_max_image_tokens,
            "max_new_tokens": args.l1_max_new_tokens, "preprocess": PREPROCESS_VERSION,
            "truncation": TRUNCATION_VERSION, "quantization": "nf4_double_bf16", "schema": 1}


def l2_contract(args):
    return {"model_id": args.model_id, "revision": args.revision, "prompt": L2_SYSTEM,
            "max_seq_length": args.max_seq_length, "max_image_tokens": args.max_image_tokens,
            "preprocess": PREPROCESS_VERSION, "truncation": TRUNCATION_VERSION,
            "lora_scope": "language_model_only", "scoring": "mean_label_token_logprob_no_eos_v1"}


def l1_key(args, root, row):
    # No alt_text, label, or whole-record hash: their changes cannot affect L1.
    return digest({"contract": l1_contract(args), "input": l1_payload(row),
                   "image_sha256": file_sha(image_path(root, row))})


def cache_path(args, split):
    return Path(args.run_dir) / "layer1" / f"{split}.jsonl"


def read_cache(path):
    if not Path(path).exists():
        return {}
    rows = read_jsonl(path)
    result = {r["record_id"]: r for r in rows}
    if len(result) != len(rows):
        raise ValueError(f"Duplicate cache IDs: {path}")
    return result


def require_cache(args, split, rows, root):
    saved = read_cache(cache_path(args, split))
    result, errors = {}, []
    for row in rows:
        item = saved.get(row["record_id"])
        try:
            if item is None or item["input_sha256"] != l1_key(args, root, row):
                raise ValueError("missing/stale")
            result[row["record_id"]] = validate_analysis(item["analysis"], row["thumbnail"])
        except (ValueError, KeyError) as exc:
            errors.append(f"{row['record_id']}: {exc}")
    if errors:
        raise ValueError(f"Layer 1 {split}: {len(errors)} invalid caches; run run_layer1.py first. {errors[:8]}")
    return result


def log_truncation(args, row, stage, details):
    if details["truncated_fields"]:
        item = {"record_id": row["record_id"], "stage": stage, **details}
        append_jsonl(Path(args.run_dir) / "truncation.jsonl", item)
        tqdm.write(json.dumps(item, ensure_ascii=False))


def to_device(inputs, model):
    # Preserve integer IDs/grids. Model visual.forward handles pixel dtype conversion.
    return {key: value.to(model.device) for key, value in inputs.items()}


def generate_layer1(args, split, rows, root):
    pin_model_revision(args)
    target = cache_path(args, split)
    saved = read_cache(target)
    keys = {r["record_id"]: l1_key(args, root, r) for r in rows}
    valid = {}
    for row in rows:
        item = saved.get(row["record_id"])
        if item and item.get("input_sha256") == keys[row["record_id"]]:
            try:
                validate_analysis(item["analysis"], row["thumbnail"])
                valid[row["record_id"]] = item
            except (ValueError, KeyError):
                pass
    pending = [r for r in rows if r["record_id"] not in valid]
    write_json(Path(args.run_dir) / "layer1" / "contract.json", l1_contract(args))
    if not pending:
        print(f"Layer 1 {split}: reused {len(valid)} valid records")
        return
    model = load_model(args)
    processor = load_processor(args.model_id, args.revision, args.l1_max_image_tokens, args.model_cache_dir)
    failures = []
    for row in tqdm(pending, desc=f"Layer 1 {split}"):
        image, image_info = load_image(image_path(root, row))
        errors = []
        for attempt in range(3):
            inputs, trunc = prepare_input(processor, image, l1_payload(row), stage=1,
                max_seq_length=args.max_seq_length, reserve_tokens=args.l1_max_new_tokens, retry=attempt > 0)
            log_truncation(args, row, "layer1", trunc)
            with torch.inference_mode():
                result = model.generate(**to_device(inputs, model), do_sample=False,
                    max_new_tokens=args.l1_max_new_tokens, use_cache=True)
            text = processor.tokenizer.decode(result[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            try:
                analysis = parse_analysis(text, row["thumbnail"])
                valid[row["record_id"]] = {"record_id": row["record_id"],
                    "input_sha256": keys[row["record_id"]], "analysis": analysis,
                    "image": image_info, "truncation": trunc}
                # Atomic snapshots make interrupted runs resumable, without partial JSONL tails.
                write_jsonl(target, sorted(valid.values(), key=lambda r: r["record_id"]))
                break
            except (ValueError, TypeError, KeyError) as exc:
                errors.append({"attempt": attempt + 1, "error": str(exc), "output": text})
        else:
            failures.append({"record_id": row["record_id"], "attempts": errors})
        image.close()
    write_json(Path(args.run_dir) / "layer1" / f"{split}_errors.json", failures)
    if failures:
        raise RuntimeError(f"Layer 1 {split}: {len(failures)} failed records; valid cache retained. Rerun to retry.")


def prepare_layer2(args, processor, candidates, row, root, analysis):
    image, _ = load_image(image_path(root, row))
    try:
        inputs, details = prepare_input(processor, image, l2_payload(row, analysis), stage=2,
            max_seq_length=args.max_seq_length, reserve_tokens=max(map(len, candidates)))
    finally:
        image.close()
    log_truncation(args, row, "layer2", details)
    return inputs


def infer_rows(args, model, processor, rows, root, cache):
    candidates = label_token_ids(processor.tokenizer)
    was_training = model.training
    model.eval()
    result = []
    try:
        with torch.inference_mode():
            for row in tqdm(rows, desc="Layer 2 candidate scoring"):
                inputs = prepare_layer2(args, processor, candidates, row, root, cache[row["record_id"]])
                betas = score_candidates(model, to_device(inputs, model), candidates)
                result.append(prediction_row(row["record_id"], betas, candidates))
    finally:
        model.train(was_training)
    return result
