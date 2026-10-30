from __future__ import annotations

import copy
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from gemma.data import (EXPECTED_FULL, boundary_labels, correct_boundary, read_records,
                        sample_ids, unique_ids, validate_synthetic, write_predictions)
from gemma.train import arguments, claim_run, configuration
from gemma.runtime.koaltq.utils import LABELS, save_json
from gemma.runtime.koaltq.scoring import class_weights
from gemma.runtime.koaltq.layer3 import build_policy
from gemma.runtime.koaltq.layer1_source import key_identity, seed_cache, validate_cache

ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = {'role': 'unknown', 'visual_type': 'unknown', 'has_informative_text': False,
            'visible_text': [], 'key_visual_information': [], 'context_already_covers': [],
            'additional_information': [], 'uncertain': []}


def row(rid='a', label='적절'):
    return {'record_id': rid, 'label': label, 'page_url': 'https://a/' + rid,
            'image_url': 'https://a/' + rid + '.png', 'image_file': rid + '.png', 'alt_text': '설명',
            'context_text': '문맥', 'page_title': '제목', 'in_link': False, 'link_dest': '', 'heading_path': ''}


def csv_file(path, fields, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fields)
        w.writeheader()
        w.writerows(rows)
    return path


def test_cli_defaults_and_fresh_run(tmp_path):
    args = arguments(['--data-root', str(tmp_path), '--run-dir', str(tmp_path / 'out')])
    cfg = configuration(args)
    assert cfg['training']['epochs'] == 3
    assert cfg['training']['effective_batch_size'] == 8
    assert cfg['training']['gradient_checkpointing']
    assert cfg['training']['loss'] == 'joint_ce'
    assert cfg['training']['binary_loss_weight'] == .2
    assert cfg['score_method'] == 'mean_logprob'
    assert cfg['runtime']['microbatch_trials'] == [1]
    assert cfg['model']['revision'] == 'ee0ef6023621cff504d758262d4e04895a5af4a2'
    assert cfg['layer1']['recovery']['supplemental_protocol'] == 'schema_recovery_v3'
    assert 'checkpoint_selection' not in cfg['training']
    claim_run(args.run_dir, args.data_root)
    with pytest.raises(FileExistsError):
        claim_run(args.run_dir, args.data_root)
    with pytest.raises(ValueError):
        claim_run(tmp_path, tmp_path)
    with pytest.raises(SystemExit):
        arguments(['--data-root', str(tmp_path), '--epochs', '0'])
    assert arguments(['--data-root', str(tmp_path), '--smoke-only']).smoke_only
    with pytest.raises(SystemExit):
        arguments(['--data-root', str(tmp_path), '--validate-only', '--smoke-only'])


def test_records_reject_invalid_label_and_labeled_test(tmp_path):
    p = tmp_path / 'records.jsonl'
    p.write_text(json.dumps(row(label='wrong')), encoding='utf-8')
    with pytest.raises(ValueError, match='label'):
        read_records(tmp_path, True, 1)
    p.write_text(json.dumps(row()), encoding='utf-8')
    with pytest.raises(ValueError, match='unlabeled'):
        read_records(tmp_path, False, 1)


def test_boundary_uses_dataset_label_and_rejects_duplicates(tmp_path):
    fields = ['record_id', '분할', '데이터셋_라벨', '재판정_라벨']
    a = {'record_id': 'a', '분할': 'test', '데이터셋_라벨': '불충분형', '재판정_라벨': '적절'}
    p = csv_file(tmp_path / 'b.csv', fields, [a])
    assert boundary_labels(p, [row()], expected=1) == {'a': '불충분형'}
    csv_file(p, fields, [a, a])
    with pytest.raises(ValueError, match='duplicate'):
        boundary_labels(p, [row()], expected=1)
    csv_file(p, fields, [dict(a, 데이터셋_라벨='invalid')])
    with pytest.raises(ValueError):
        boundary_labels(p, [row()], expected=1)
    csv_file(p, fields, [dict(a, record_id='missing')])
    with pytest.raises(ValueError):
        boundary_labels(p, [row()], expected=1)


def test_submission_order_and_correction(tmp_path):
    predictions = [{'record_id': 'b', 'label': '적절'}, {'record_id': 'a', 'label': '중복형'}]
    corrected = correct_boundary(predictions, {'a': '불충분형'}, expected=1)
    assert corrected[0] == predictions[0]
    assert corrected[1]['label'] == '불충분형'
    p = write_predictions(tmp_path / 'p.csv', corrected, ['a', 'b'])
    assert sample_ids(p, predictions) == ['a', 'b']
    with pytest.raises(ValueError):
        write_predictions(tmp_path / 'bad.csv', predictions + [predictions[0]], ['a', 'b'])
    with pytest.raises(ValueError):
        write_predictions(tmp_path / 'missing.csv', predictions[:1], ['a', 'b'])
    with pytest.raises(ValueError):
        correct_boundary(predictions, {'unknown': '적절'}, expected=1)
    with pytest.raises(FileExistsError):
        write_predictions(p, corrected, ['a', 'b'])


def test_full_real_dataset_composition_and_synthetic_rules(tmp_path, monkeypatch):
    import os
    data = Path(os.environ.get('KOALTQ_DATA_ROOT', ROOT.parent / 'data' / 'koaltq_261014'))
    if not (data / 'koaltq_train/records.jsonl').exists():
        pytest.skip('Local release data not installed')
    from gemma.data_work.augment_rules import RULES, generate
    from collections import Counter
    original = read_records(data / 'koaltq_train', True, 2400)
    test = read_records(data / 'koaltq_test_public', False, 600)
    known = boundary_labels(ROOT / 'resources/경계사례_20261013.csv', test, original)
    synthetic, _ = generate(original, None, {r: 40 for r in RULES}, 42, 3)
    validate_synthetic(synthetic, original)
    assert synthetic == generate(original, None, {r: 40 for r in RULES}, 42, 3)[0]
    real = original + [dict(r, label=known[r['record_id']]) for r in test if r['record_id'] in known]
    full = real + synthetic
    assert len(unique_ids(full, 2669)) == 2669
    assert Counter(r['label'] for r in full) == EXPECTED_FULL
    assert class_weights(full, 3.) == class_weights(real, 3.)
    policy_rows = [r for r in full if not r.get('synthetic')]
    assert build_policy(policy_rows, [r['record_id'] for r in policy_rows], ['filename_only_v1'])['active_rules'] == []
    with pytest.raises(ValueError):
        validate_synthetic(synthetic[:-1], original)
    broken = copy.deepcopy(synthetic)
    broken[0]['image_file'] = 'changed.png'
    with pytest.raises(ValueError, match='changed'):
        validate_synthetic(broken, original)
    # Exercise the actual composition entry; image decoding is covered by validate-only.
    import gemma.data as data_module
    def image_info(task):
        root, record = task
        assert (root / record['image_file']).exists() or (root / 'images' / Path(record['image_file']).name).exists()
        return {'record_id': record['record_id'], 'ok': True, 'image_sha256': 'image'}
    monkeypatch.setattr(data_module, 'audit_image', image_info)
    dataset = data_module.prepare_data(data, ROOT / 'resources/경계사례_20261013.csv',
                                      data / 'koaltq_test_public/sample_submission.csv', tmp_path)
    assert all('label' not in r and Path(r['image_root']) == (data / 'koaltq_test_public').resolve() for r in dataset.test)
    assert len(dataset.train) == 2669


def test_preserved_source_and_training_helpers():
    import ast
    import hashlib
    manifest = json.loads((ROOT / 'METHOD_PRESERVATION.json').read_text(encoding='utf-8'))
    for name, meta in manifest['files'].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == meta['current_sha256']
        if meta['identical']:
            assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == meta['original_sha256']
    training = ast.parse((ROOT / 'runtime/koaltq/training.py').read_text(encoding='utf-8'))
    for name, expected in manifest['preserved_training_functions'].items():
        fn = next(n for n in training.body if isinstance(n, ast.FunctionDef) and n.name == name)
        assert hashlib.sha256(ast.dump(fn).encode()).hexdigest() == expected
    scoring = ast.parse((ROOT / 'runtime/koaltq/scoring.py').read_text(encoding='utf-8'))
    for name, expected in manifest['preserved_scoring_functions'].items():
        fn = next(n for n in scoring.body if isinstance(n, ast.FunctionDef) and n.name == name)
        assert hashlib.sha256(ast.dump(fn).encode()).hexdigest() == expected


def test_l1_cache_integrity_identity_and_missing(tmp_path):
    value = {'key': 'k', 'analysis': ANALYSIS, 'raw': json.dumps(ANALYSIS)}
    validate_cache(value, 'k')
    with pytest.raises(ValueError):
        validate_cache(value, 'wrong')
    with pytest.raises(ValueError):
        validate_cache(dict(value, analysis=dict(ANALYSIS, role='invalid')), 'k')
    with pytest.raises(ValueError):
        validate_cache(dict(value, raw='not JSON'), 'k')
    with pytest.raises(ValueError):
        validate_cache(dict(value, raw=json.dumps(dict(ANALYSIS, role='장식성'))), 'k')
    source = tmp_path / 'old'
    identity = {'model_id': 'gemma', 'revision': 'v1', 'gpu_name': 'old'}
    save_json(source / 'actual_model_api.json', {'identity': identity})
    save_json(source / 'cache/layer1/k.json', value)
    cfg = {'_work': str(tmp_path / 'new'), 'layer1_cache_source': {'run_dir': str(source), 'require_complete': False}}
    assert key_identity(cfg, dict(identity, gpu_name='new')) == identity
    with pytest.raises(ValueError):
        key_identity(cfg, dict(identity, revision='v2'))
    assert seed_cache(cfg, dict(identity, gpu_name='new'), {'a': 'k', 'b': 'missing'}) == 1


def test_l1_deduplicates_content_and_excludes_alt_and_label(tmp_path, monkeypatch):
    from gemma.runtime.koaltq import pipeline
    cfg = {'_work': str(tmp_path), '_package_root': str(ROOT / 'runtime'), 'layer1': {}, 'seed': 42}
    a = row('a')
    b = dict(a, record_id='b', alt_text='different', label='무관형')
    info = {'a': {'image_sha256': 'same'}, 'b': {'image_sha256': 'same'}}
    calls = []
    def unique(engine, cfg, rows, root, info, scope):
        calls.append(len(rows))
        return {r['record_id']: {'analysis': ANALYSIS} for r in rows}, {}
    monkeypatch.setattr(pipeline, '_prepare_layer1_unique', unique)
    result, timing = pipeline.prepare_layer1(SimpleNamespace(identity={}), cfg, [a, b], tmp_path, info, 'test')
    assert calls == [1]
    assert result['a'] is result['b']
    assert timing['duplicate_content_keys_reused'] == 1


def test_encoded_inputs_prepare_once_and_corruption(tmp_path, monkeypatch):
    from gemma.cache import EncodedInputs
    from gemma.runtime.koaltq import pipeline
    cfg = {'_package_root': str(ROOT / 'runtime')}
    a, b = row('a'), dict(row('a'), record_id='b')
    info = {k: {'image_sha256': 'same'} for k in ['a', 'b']}
    analyses = {k: {'analysis': ANALYSIS} for k in ['a', 'b']}
    calls = []
    def stream(engine, root, jobs, cfg):
        calls.append(len(jobs))
        for r, _, _ in jobs:
            yield r, {'input_ids': torch.tensor([[1, 2]]), 'attention_mask': torch.tensor([[1, 1]])}
    monkeypatch.setattr(pipeline, 'encoded_stream', stream)
    cache = EncodedInputs(tmp_path)
    engine = SimpleNamespace(identity={'revision': 'v1'})
    cache.prepare(engine, cfg, [a, b], tmp_path, info, analyses)
    assert cache.new_records == 1
    for _ in range(3):
        assert cache.get(a)['input_ids'].tolist() == [[1, 2]]
        cache.get(b)
    cache.prepare(engine, cfg, [a], tmp_path, info, analyses)
    assert calls == [1, 0]
    cache.verified.clear()
    p = tmp_path / (cache.keys['a'] + '.safetensors')
    p.write_bytes(b'corrupt')
    with pytest.raises(ValueError):
        cache.get(a)


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lora = torch.nn.Parameter(torch.tensor([.5]))
        self.frozen = torch.nn.Parameter(torch.tensor([1.]), requires_grad=False)

    def named_parameters(self, *args, **kwargs):
        return iter([('base_model.model.language_model.lora_A', self.lora), ('frozen', self.frozen)])


class TinyEngine:
    def __init__(self):
        self.model = TinyModel()
        self.saved = []
        self.predicted = []

    def forward_targets(self, pairs):
        assert self.model.training
        sums = torch.stack([-((self.model.lora[0] - .1 * (LABELS.index(lab) + 1)).square() + 1)
                            for _, lab in pairs])
        return sums, sums, torch.ones(len(pairs))

    def adapter_state(self):
        return {'lora': self.model.lora.detach().clone()}

    def fingerprint(self):
        from gemma.runtime.koaltq.model import tensor_state_hash
        return tensor_state_hash(self.adapter_state())

    def save_adapter(self, path):
        from safetensors.torch import save_file
        path.mkdir()
        save_file(self.adapter_state(), str(path / 'adapter_model.safetensors'))
        save_json(path / 'koaltq_adapter.json', {'adapter_sha256': self.fingerprint()})
        self.saved.append(self.fingerprint())

    def score(self, bases, candidate_batch):
        assert not self.model.training
        assert not torch.is_grad_enabled()
        self.predicted.append(self.fingerprint())
        # Consume random values to test restoration after epoch inference.
        torch.rand(1)
        return [{label: {'sum_logprob': -float(i + 1), 'mean_logprob': -float(i + 1), 'n_tokens': 1}
                 for i, label in enumerate(LABELS)} for _ in bases]


@pytest.mark.parametrize('size,expected_steps', [(13, 6), (2669, 1002)])
def test_epochs_use_current_saved_weights_no_best_selection(tmp_path, size, expected_steps):
    from gemma.workflow import train_epochs
    args = arguments(['--data-root', str(tmp_path), '--run-dir', str(tmp_path)])
    cfg = configuration(args)
    train = [row(f'r{i:04d}', LABELS[i % 7]) for i in range(size)]
    # Exercise the production 600-record writer/correction contract.
    test = [row(f't{i:03d}') for i in range(600)]
    for r in test:
        del r['label']
    known = {r['record_id']: '불충분형' for r in test[:109]}
    policy = build_policy(train, [r['record_id'] for r in train], [])
    dataset = SimpleNamespace(train=train, test=test, known=known, sample_order=[r['record_id'] for r in reversed(test)], policy=policy)
    save_json(tmp_path / 'class_weights.json', class_weights(train, 3.))
    engine = TinyEngine()
    encoded = SimpleNamespace(get=lambda row: {'input_ids': torch.tensor([[1]])})
    result = train_epochs(engine, cfg, dataset, encoded, {'microbatch': 1, 'candidate_batch': 1})
    assert engine.model.training
    assert len(set(engine.saved)) == 3
    assert all(engine.predicted[(epoch - 1) * 600:epoch * 600] == [engine.saved[epoch - 1]] * 600 for epoch in [1, 2, 3])
    summary = json.loads((tmp_path / 'training_summary.json').read_text())
    assert summary['optimizer_steps'] == expected_steps
    assert summary['examples_seen'] == size * 3
    assert summary['epochs_completed'] == 3
    for epoch in [1, 2, 3]:
        out = tmp_path / 'epochs' / f'epoch_{epoch:02d}'
        for name in ['model_predictions.csv', 'submission.csv', 'submission_boundary109.csv']:
            assert len(sample_ids(out / name, test)) == 600
        info = result[epoch - 1]
        assert info['adapter_sha256'] == info['saved_adapter_sha256'] == info['inference_adapter_sha256']
        assert info['boundary_applied_count'] == 109
        assert info['train_loss'] == pytest.approx(.8 * info['train_type_loss'] + .2 * info['train_binary_loss'])
        lines = [json.loads(l) for l in (out / 'train_steps.jsonl').read_text().splitlines()]
        assert sum(l['examples'] for l in lines) == size
        assert lines[-1]['examples'] == size % 8
        assert all(l['loss'] == pytest.approx(.8 * l['type_loss'] + .2 * l['binary_loss']) for l in lines)
    assert not list(tmp_path.rglob('*best*'))
