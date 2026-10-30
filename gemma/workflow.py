"""Full-data Joint CE training; each epoch predicts with its current adapter."""
from __future__ import annotations

from contextlib import contextmanager, redirect_stdout, redirect_stderr
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch

from .data import correct_boundary, write_predictions
from .runtime.koaltq.layer3 import apply_layer3
from .runtime.koaltq.scoring import select_label
from .runtime.koaltq.training import backward_group, optimizer_for, rng_state, restore_rng
from .runtime.koaltq.utils import append_event, read_json, save_json, seed_all


class Tee:
    def __init__(self, console, file):
        self.console, self.file = console, file

    def write(self, value):
        self.console.write(value)
        self.file.write(value)
        self.file.flush()
        return len(value)

    def flush(self):
        self.console.flush()
        self.file.flush()

    def isatty(self):
        return False


@contextmanager
def capture_log(path):
    with Path(path).open('a', encoding='utf-8') as f:
        with redirect_stdout(Tee(sys.stdout, f)), redirect_stderr(Tee(sys.stderr, f)):
            yield


def synchronize():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def memory(reset=False):
    if not torch.cuda.is_available():
        return {'peak_allocated_gib': 0., 'peak_reserved_gib': 0.}
    if reset:
        torch.cuda.reset_peak_memory_stats()
    return {'peak_allocated_gib': torch.cuda.max_memory_allocated() / 2**30,
            'peak_reserved_gib': torch.cuda.max_memory_reserved() / 2**30}


def predict_epoch(engine, dataset, encoded, out, candidate_batch, score_method='mean_logprob'):
    """No previous epoch prediction cache and no model reload/selection."""
    if score_method!='mean_logprob':raise ValueError('Joint CE 추론은 mean_logprob만 지원')
    engine.model.eval()
    details, predictions = [], []
    with torch.no_grad():
        # Keep only one decoded input in RAM; batch the seven candidate labels.
        for n, row in enumerate(dataset.test, 1):
            scores = engine.score([encoded.get(row)], candidate_batch)[0]
            label = select_label(scores, score_method)
            predictions.append({'record_id': row['record_id'], 'label': label})
            details.append({'record_id': row['record_id'], 'prediction': label, 'scores': scores})
            if n % 50 == 0 or n == len(dataset.test):
                print(f'Public test: {n}/{len(dataset.test)}', flush=True)
    processed = apply_layer3(dataset.test, details, dataset.policy)
    submission = [{'record_id': r['record_id'], 'label': r['prediction_after']} for r in processed]
    corrected = correct_boundary(submission, dataset.known)
    write_predictions(out / 'model_predictions.csv', predictions, dataset.sample_order)
    write_predictions(out / 'submission.csv', submission, dataset.sample_order)
    write_predictions(out / 'submission_boundary109.csv', corrected, dataset.sample_order)
    save_json(out / 'prediction_details.json', {'score_method': score_method, 'layer2': details, 'layer3': processed,
        'boundary_applied_count': len(dataset.known), 'boundary_changed_count': sum(a != b for a, b in zip(submission, corrected))})
    return {'test_records': len(details), 'boundary_applied_count': len(dataset.known)}


def train_epochs(engine, cfg, dataset, encoded, plan):
    work = Path(cfg['_work'])
    rows = sorted(dataset.train, key=lambda r: r['record_id'])
    spec = cfg['training']
    effective, epochs = spec['effective_batch_size'], spec['epochs']
    weights = read_json(work / 'class_weights.json')['weights']
    opt = optimizer_for(engine, cfg)
    steps_per_epoch = math.ceil(len(rows) / effective)
    total_steps = epochs * steps_per_epoch
    warmup = math.ceil(total_steps * spec['warmup_ratio'])

    def lr_factor(s):
        if warmup and s < warmup:
            return (s + 1) / warmup
        return max(0., (total_steps - s) / max(1, total_steps - warmup))

    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_factor)
    seed_all(cfg['seed'])
    step, seen, summaries = 0, 0, []
    for epoch in range(1, epochs + 1):
        out = work / 'epochs' / f'epoch_{epoch:02d}'
        out.mkdir(parents=True, exist_ok=False)
        with capture_log(out / 'epoch.log'):
            print(f'Epoch {epoch}/{epochs}: {len(rows)} examples, {steps_per_epoch} optimizer steps', flush=True)
            engine.model.train()
            memory(reset=True)
            started = time.perf_counter()
            loss_sums = {key: 0. for key in ('loss', 'type_loss', 'binary_loss')}
            epoch_seen = 0
            order = np.random.default_rng(cfg['seed'] + epoch - 1).permutation(len(rows))
            for start in range(0, len(rows), effective):
                group_rows = [rows[int(i)] for i in order[start:start + effective]]
                group = [(r, encoded.get(r)) for r in group_rows]
                losses = backward_group(engine, group, weights, plan['microbatch'], opt, cfg)
                # An optimizer OOM may partially update parameters: abort, never resume.
                opt.step()
                scheduler.step()
                opt.zero_grad(set_to_none=True)
                synchronize()
                n = len(group)
                for key in loss_sums:
                    loss_sums[key] += losses[key] * n
                epoch_seen += n
                step += 1
                seen += n
                append_event(out / 'train_steps.jsonl', {'epoch': epoch, 'global_step': step,
                    'examples': n, **losses, 'learning_rate': scheduler.get_last_lr()[0],
                    'microbatch': plan['microbatch'], 'effective_batch_size': effective})
                if step % 25 == 0 or epoch_seen == len(rows):
                    print(f'Epoch {epoch}: {epoch_seen}/{len(rows)}, step {step}/{total_steps}, '
                          f'loss {losses["loss"]:.6f} (type {losses["type_loss"]:.6f}, binary {losses["binary_loss"]:.6f})', flush=True)
                del group
            synchronize()
            train_seconds = time.perf_counter() - started
            train_memory = memory()
            if epoch_seen != len(rows):
                raise AssertionError('Epoch training example count mismatch')
            adapter_sha = engine.fingerprint()
            engine.save_adapter(out / 'adapter')
            saved = read_json(out / 'adapter' / 'koaltq_adapter.json')
            from safetensors.torch import load_file
            from .runtime.koaltq.model import tensor_state_hash
            disk_sha = tensor_state_hash(load_file(str(out / 'adapter' / 'adapter_model.safetensors'), device='cpu'))
            if saved['adapter_sha256'] != adapter_sha or disk_sha != adapter_sha:
                raise AssertionError('Saved adapter differs from current epoch weights')
            state = rng_state()
            infer_start = time.perf_counter()
            # Persist the epoch's completed training even if inference fails.
            summary = {'epoch': epoch, 'train_examples': epoch_seen, 'optimizer_steps': steps_per_epoch,
                'global_step': step, 'train_loss': loss_sums['loss'] / epoch_seen,
                'train_type_loss': loss_sums['type_loss'] / epoch_seen,
                'train_binary_loss': loss_sums['binary_loss'] / epoch_seen, 'train_seconds': train_seconds,
                'adapter_sha256': adapter_sha, 'saved_adapter_sha256': disk_sha,
                'status': 'adapter_saved', **train_memory}
            save_json(out / 'epoch_summary.json', summary)
            try:
                prediction = predict_epoch(engine, dataset, encoded, out, plan['candidate_batch'], cfg['score_method'])
                synchronize()
                if engine.fingerprint() != adapter_sha:
                    raise AssertionError('Inference changed the current epoch adapter')
            finally:
                restore_rng(state)
                engine.model.train()
            summary.update(prediction, inference_seconds=time.perf_counter() - infer_start,
                           inference_adapter_sha256=adapter_sha, status='completed')
            save_json(out / 'epoch_summary.json', summary)
            summaries.append(summary)
            print(f'Epoch {epoch} completed: mean train loss {summary["train_loss"]:.6f}; test predictions saved', flush=True)
            save_json(work / 'training_summary.json', {'epochs_completed': epoch, 'epochs_requested': epochs,
                'optimizer_steps': step, 'examples_seen': seen, 'validation_used': False, 'epochs': summaries})
    if step != total_steps or seen != len(rows) * epochs:
        raise AssertionError('Total training amount mismatch')
    return summaries
