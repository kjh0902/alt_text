from __future__ import annotations

from collections import Counter

import torch
from torch.utils.checkpoint import checkpoint

from . import LABELS


def label_token_ids(tokenizer):
    result = [tokenizer.encode(label, add_special_tokens=False) for label in LABELS]
    for label, ids in zip(LABELS, result):
        if not ids or tokenizer.decode(ids, skip_special_tokens=False) != label:
            raise ValueError(f"Label does not round-trip through tokenizer: {label}")
    return result


def candidate_beta(model, inputs, tokens):
    """Prompt + y[:-1] predicts all of y; logits include the last prompt position."""
    ids = inputs["input_ids"]
    if ids.shape[0] != 1:
        raise ValueError("Candidate scorer requires sample batch_size=1")
    target = torch.tensor(tokens, device=ids.device, dtype=torch.long)
    kwargs = {k: v for k, v in inputs.items() if k not in ("input_ids", "attention_mask", "token_type_ids")}
    full_ids = torch.cat((ids, target[:-1].view(1, -1)), dim=1)
    attention = torch.cat((inputs.get("attention_mask", torch.ones_like(ids)),
                           torch.ones((1, len(tokens) - 1), device=ids.device, dtype=torch.long)), dim=1)
    # Always compute explicit position_ids; checkpoint recomputation must not depend on mutable rope_deltas.
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    if hasattr(base, "model") and hasattr(base.model, "get_rope_index"):
        kwargs["position_ids"], _ = base.model.get_rope_index(
            full_ids, kwargs.get("image_grid_thw"), attention_mask=attention)
    output = model(input_ids=full_ids, attention_mask=attention, use_cache=False,
                   logits_to_keep=len(tokens), **kwargs)
    logits = output.logits[0].float()
    if logits.shape[0] != len(tokens):
        raise ValueError("Model must honor logits_to_keep for label positions")
    return torch.log_softmax(logits, dim=-1).gather(1, target[:, None]).mean()


def score_candidates(model, inputs, candidates, *, recompute=False):
    betas = []
    # Checkpoint the complete candidate, including vocabulary projection/log_softmax.
    # Non-reentrant checkpoint tracks captured LoRA parameters even with integer input IDs.
    for token_ids in candidates:
        if recompute and torch.is_grad_enabled():
            keys = tuple(inputs)
            def score_one(*values, _tokens=tuple(token_ids), _keys=keys):
                return candidate_beta(model, dict(zip(_keys, values)), _tokens)
            beta = checkpoint(score_one, *(inputs[key] for key in keys), use_reentrant=False,
                              preserve_rng_state=True)
        else:
            beta = candidate_beta(model, inputs, token_ids)
        betas.append(beta)
    return torch.stack(betas)


def class_weights(labels, rho=0.99):
    if not 0 <= rho < 1:
        raise ValueError("rho must satisfy 0 <= rho < 1")
    counts = Counter(labels)
    if set(counts) != set(LABELS):
        raise ValueError("Training split must contain exactly the seven classes")
    n = torch.tensor([counts[label] for label in LABELS], dtype=torch.float64)
    if rho == 0:
        raw = torch.ones_like(n)
    else:
        log_rho = torch.tensor(rho, dtype=torch.float64).log()
        raw = (1 - rho) / (-torch.expm1(n * log_rho))
    return (len(LABELS) * raw / raw.sum()).float(), dict(counts)


def classification_loss(betas, target, weights):
    # F.cross_entropy(weight=..., reduction='mean') would cancel weights at batch_size=1.
    return -weights[target] * torch.log_softmax(betas.float(), dim=-1)[target]


def prediction_row(record_id, betas, candidates):
    beta = betas.detach().float().cpu()
    probability = beta.softmax(dim=-1)
    if not torch.isfinite(beta).all():
        raise FloatingPointError(f"Non-finite candidate scores: {record_id}")
    return {"record_id": record_id, "prediction": LABELS[int(beta.argmax())],
            "classes": {label: {"beta": float(beta[i]), "probability": float(probability[i]),
                                "token_count": len(candidates[i])} for i, label in enumerate(LABELS)}}

