from __future__ import annotations

import math
import os
import random
import uuid
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup, set_seed

from . import LABELS
from .evaluation import save_evaluation
from .io import append_jsonl, digest, read_json, write_json
from .model import check_trainable, gpu_environment, load_model, load_processor
from .runtime import infer_rows, l1_contract, l2_contract, pin_model_revision, prepare_layer2, require_cache, to_device
from .scoring import class_weights, classification_loss, label_token_ids, score_candidates


def training_contract(args, train, validation, train_cache, val_cache):
    return {"layer1": l1_contract(args), "layer2": l2_contract(args),
            "training": {key: getattr(args, key) for key in (
                "epochs", "gradient_accumulation", "learning_rate", "weight_decay", "warmup_ratio",
                "max_grad_norm", "seed", "rho", "lora_rank", "lora_alpha")},
            "train_sha256": digest(train), "validation_sha256": digest(validation),
            "train_cache_sha256": digest(train_cache), "validation_cache_sha256": digest(val_cache)}


def save_checkpoint(model, optimizer, scheduler, directory, state, contract):
    directory = Path(directory)
    name = f"step_{state['step']:06d}_epoch_{state['epoch']:02d}_cursor_{state['cursor']:04d}"
    target = directory / "checkpoints" / name
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite checkpoint: {target}")
    staging = target.with_name(name + ".tmp_" + uuid.uuid4().hex[:8])
    staging.mkdir(parents=True)
    model.save_pretrained(staging / "adapter", safe_serialization=True)
    write_json(staging / "metadata.json", {"state": state, "contract": contract,
        "contract_sha256": digest(contract), "trainable": check_trainable(model)})
    torch.save({"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                "python_rng": random.getstate(), "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []},
               staging / "training_state.pt")
    os.replace(staging, target)
    write_json(directory / "last_checkpoint.json", {"path": str(target.relative_to(directory))})
    return target


def load_training_state(path, optimizer, scheduler):
    # Only load checkpoints created by this project from a trusted run directory.
    saved = torch.load(Path(path) / "training_state.pt", map_location="cpu", weights_only=False)
    optimizer.load_state_dict(saved["optimizer"])
    scheduler.load_state_dict(saved["scheduler"])
    random.setstate(saved["python_rng"])
    torch.set_rng_state(saved["torch_rng"])
    if torch.cuda.is_available() and saved["cuda_rng"]:
        torch.cuda.set_rng_state_all(saved["cuda_rng"])


def checkpoint_path(args, *, best=False):
    if args.checkpoint:
        return Path(args.checkpoint)
    pointer = Path(args.run_dir) / ("best_checkpoint.json" if best else "last_checkpoint.json")
    if not pointer.exists():
        raise FileNotFoundError(f"Checkpoint pointer missing: {pointer}")
    return Path(args.run_dir) / read_json(pointer)["path"]


def train(args, train_rows, validation_rows, root):
    pin_model_revision(args)
    train_cache = require_cache(args, "train", train_rows, root)
    val_cache = require_cache(args, "validation", validation_rows, root)
    contract = training_contract(args, train_rows, validation_rows, train_cache, val_cache)
    out = Path(args.run_dir)
    contract_file = out / "training_contract.json"
    if contract_file.exists():
        if read_json(contract_file) != contract:
            raise ValueError("Training inputs/settings changed; use a new --run-dir")
        if not args.resume:
            raise ValueError("Training run already exists; use --resume or a new --run-dir")
    write_json(contract_file, contract)
    weights, counts = class_weights([row["label"] for row in train_rows], args.rho)
    write_json(out / "class_weights.json", {"rho": args.rho, "source": "train_split_only",
        "counts": counts, "weights": dict(zip(LABELS, weights.tolist()))})
    set_seed(args.seed)
    state = {"epoch": 0, "cursor": 0, "step": 0, "best_score": -1.0}
    resume = checkpoint_path(args) if args.resume else None
    if resume:
        metadata = read_json(resume / "metadata.json")
        if metadata["contract"] != contract:
            raise ValueError("Resume checkpoint contract does not match this run")
        state = metadata["state"]
    model = load_model(args, training=True, adapter=resume / "adapter" if resume else None)
    processor = load_processor(args.model_id, args.revision, args.max_image_tokens, args.model_cache_dir)
    candidates = label_token_ids(processor.tokenizer)
    weights = weights.to(model.device)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                 lr=args.learning_rate, weight_decay=args.weight_decay)
    steps = math.ceil(len(train_rows) / args.gradient_accumulation) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(steps * args.warmup_ratio), steps)
    if resume:
        load_training_state(resume, optimizer, scheduler)
    write_json(out / "trainable_parameters.json", check_trainable(model))
    torch.cuda.reset_peak_memory_stats()
    start_epoch = state["epoch"]
    for epoch in range(start_epoch, args.epochs):
        order = list(range(len(train_rows)))
        random.Random(args.seed + epoch).shuffle(order)
        cursor = state["cursor"] if epoch == start_epoch else 0
        if cursor % args.gradient_accumulation:
            raise ValueError("Checkpoint cursor must be on an optimizer-step boundary")
        model.train()
        # The entire visual stack remains frozen and deterministic in training mode.
        model.get_base_model().model.visual.eval()
        for begin in tqdm(range(cursor, len(order), args.gradient_accumulation), desc=f"Epoch {epoch + 1}"):
            batch = order[begin:begin + args.gradient_accumulation]
            optimizer.zero_grad(set_to_none=True)
            step_loss = 0.0
            for index in batch:
                row = train_rows[index]
                inputs = prepare_layer2(args, processor, candidates, row, root, train_cache[row["record_id"]])
                betas = score_candidates(model, to_device(inputs, model), candidates, recompute=True)
                loss = classification_loss(betas, LABELS.index(row["label"]), weights)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite loss: {row['record_id']}")
                (loss / len(batch)).backward()
                step_loss += float(loss.detach()) / len(batch)
                del loss, betas, inputs
            norm = torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad),
                                                  args.max_grad_norm, error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()
            state.update(epoch=epoch, cursor=begin + len(batch), step=state["step"] + 1)
            append_jsonl(out / "training_log.jsonl", {**state, "loss": step_loss,
                "grad_norm": float(norm), "learning_rate": scheduler.get_last_lr()[0]})
            if state["step"] % args.save_steps == 0 and state["cursor"] < len(order):
                save_checkpoint(model, optimizer, scheduler, out, state, contract)
        predictions = infer_rows(args, model, processor, validation_rows, root, val_cache)
        metrics = save_evaluation(out / "validation" / f"epoch_{epoch + 1:02d}", validation_rows, predictions)
        improved = metrics["total_score"] > state["best_score"]
        if improved:
            state["best_score"] = metrics["total_score"]
        state.update(epoch=epoch + 1, cursor=0)
        saved = save_checkpoint(model, optimizer, scheduler, out, state, contract)
        if improved:
            write_json(out / "best_checkpoint.json", {"path": str(saved.relative_to(out)),
                "epoch": epoch + 1, "total_score": state["best_score"]})
        print({key: metrics[key] for key in ("macro_f1_7class", "binary_f1", "total_score")})
        write_json(out / "training_summary.json", {**state, "gpu": gpu_environment()})

