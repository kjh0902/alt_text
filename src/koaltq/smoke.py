from __future__ import annotations

import gc
from pathlib import Path

import torch
from transformers import set_seed

from . import LABELS
from .data import select_records
from .io import write_json
from .model import check_trainable, gpu_environment, load_model, load_processor
from .runtime import generate_layer1, pin_model_revision, prepare_layer2, require_cache, to_device
from .scoring import class_weights, classification_loss, label_token_ids, score_candidates


def smoke_test(args):
    if not torch.cuda.is_available():
        raise RuntimeError("GPU smoke test requires a CUDA GPU; CPU tests do not certify 16GB training")
    # Smoke artifacts cannot be mistaken for a complete production run.
    args.run_dir = Path(args.run_dir) / "gpu_smoke"
    if (args.run_dir / "adapter").exists():
        raise FileExistsError("Smoke adapter already exists; choose a fresh --run-dir")
    set_seed(args.seed)
    pin_model_revision(args)
    all_rows, root = select_records(args, "train")
    selected = {r["record_id"]: r for r in [all_rows[0], all_rows[-1], max(all_rows, key=lambda r: len(r["alt_text"]))]}
    rows = list(selected.values())
    generate_layer1(args, "train", rows, root)
    cache = require_cache(args, "train", rows, root)
    gc.collect()
    torch.cuda.empty_cache()
    model = load_model(args, training=True)
    processor = load_processor(args.model_id, args.revision, args.max_image_tokens, args.model_cache_dir)
    candidates = label_token_ids(processor.tokenizer)
    weights, _ = class_weights([row["label"] for row in all_rows])
    weights = weights.to(model.device)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-4)
    checks = []
    torch.cuda.reset_peak_memory_stats()
    for row in rows:
        model.train()
        model.get_base_model().model.visual.eval()
        inputs = prepare_layer2(args, processor, candidates, row, root, cache[row["record_id"]])
        optimizer.zero_grad(set_to_none=True)
        tracked = next(p for name, p in model.named_parameters() if ".lora_B." in name and p.requires_grad)
        before = tracked.detach().clone()
        betas = score_candidates(model, to_device(inputs, model), candidates, recompute=True)
        loss = classification_loss(betas, LABELS.index(row["label"]), weights)
        loss.backward()
        gradients = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
        if not gradients or not all(torch.isfinite(g).all() for g in gradients) or not any(g.abs().sum() > 0 for g in gradients):
            raise AssertionError("Missing/nonfinite/zero language LoRA gradient")
        if any(p.requires_grad or p.grad is not None for p in model.get_base_model().model.visual.parameters()):
            raise AssertionError("Visual encoder/projection must be frozen")
        torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), 1.0, error_if_nonfinite=True)
        optimizer.step()
        if torch.equal(before, tracked.detach()):
            raise AssertionError("Optimizer did not update language LoRA")
        checks.append({"record_id": row["record_id"], "prompt_tokens": inputs["input_ids"].shape[1],
                       "loss": float(loss.detach()), "language_lora_updated": True})
        del before, betas, loss, gradients, tracked
    model.eval()
    with torch.inference_mode():
        expected = score_candidates(model, to_device(inputs, model), candidates).float().cpu()
    trainable = check_trainable(model)
    model.save_pretrained(args.run_dir / "adapter", safe_serialization=True)
    gpu = gpu_environment()
    del model, optimizer, weights
    gc.collect()
    torch.cuda.empty_cache()
    reloaded = load_model(args, adapter=args.run_dir / "adapter")
    with torch.inference_mode():
        actual = score_candidates(reloaded, to_device(inputs, reloaded), candidates).float().cpu()
    torch.testing.assert_close(actual, expected, atol=0.08, rtol=0.005)
    report = {"status": "GPU_TEST_PASSED", "checks": checks, "gpu": gpu,
              "trainable": trainable, "reload_max_abs_difference": float((actual - expected).abs().max()),
              "model_revision": args.revision, "max_seq_length": args.max_seq_length,
              "max_image_tokens": args.max_image_tokens, "l1_max_image_tokens": args.l1_max_image_tokens}
    write_json(args.run_dir / "result.json", report)
    print(report)
