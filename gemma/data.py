"""CPU-only data composition and submission contracts; never modify source data."""
from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import csv
from dataclasses import dataclass
import json
from pathlib import Path

from .data_work.augment_rules import RULES, generate
from .data_work.check_augment import audit_row, load_rule_flags, real_alt_label_counts
from .runtime.koaltq.images import audit_image
from .runtime.koaltq.layer3 import build_policy
from .runtime.koaltq.utils import LABELS, CONTEXT_KEYS, digest, file_sha, save_json

EXPECTED_FULL = dict(zip(LABELS, [1619, 65, 66, 187, 30, 647, 55]))
EXPECTED_REAL = dict(zip(LABELS, [1619, 25, 26, 147, 30, 647, 15]))
EXPECTED_TRAIN = dict(zip(LABELS, [1599, 24, 26, 146, 26, 564, 15]))
REQUIRED = ['record_id', 'page_url', 'image_url', 'image_file', 'alt_text', *CONTEXT_KEYS]


def unique_ids(rows, expected=None):
    ids = [r['record_id'] for r in rows]
    if any(not isinstance(i, str) or not i for i in ids) or len(ids) != len(set(ids)):
        raise ValueError('Empty or duplicate record_id')
    if expected is not None and len(ids) != expected:
        raise ValueError(f'Expected {expected} records; got {len(ids)}')
    return set(ids)


def read_records(root, labeled, expected):
    root = Path(root)
    path = root / 'records.jsonl'
    if not path.is_file():
        raise FileNotFoundError(f'Expected extracted dataset: {path}')
    rows = [json.loads(line) for line in path.read_text(encoding='utf-8-sig').split('\n') if line.strip()]
    unique_ids(rows, expected)
    for r in rows:
        for k in REQUIRED:
            if k not in r or (type(r[k]) is not bool if k == 'in_link' else not isinstance(r[k], str)):
                raise ValueError(f'Invalid {k}: {r["record_id"]}')
        if labeled and r.get('label') not in LABELS:
            raise ValueError(f'Invalid label: {r["record_id"]}')
        if not labeled and 'label' in r:
            raise ValueError('Test records must remain unlabeled')
    return rows


def boundary_labels(path, test, train=(), expected=109):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        if not {'record_id', '분할', '데이터셋_라벨'} <= set(reader.fieldnames or []):
            raise ValueError('Boundary CSV columns missing')
        rows = list(reader)
    unique_ids(rows)
    train_labels = {r['record_id']: r['label'] for r in train}
    for r in rows:
        if r['분할'] not in ('train', 'test') or r['데이터셋_라벨'] not in LABELS:
            raise ValueError('Invalid boundary split or label')
        if r['분할'] == 'train' and train_labels and train_labels.get(r['record_id']) != r['데이터셋_라벨']:
            raise ValueError('Boundary train label mismatch')
    known = {r['record_id']: r['데이터셋_라벨'] for r in rows if r['분할'] == 'test'}
    if len(known) != expected or not set(known) <= unique_ids(test):
        raise ValueError(f'Boundary labels must identify exactly {expected} public test records')
    return known


def sample_ids(path, test):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != ['record_id', 'label']:
            raise ValueError('Submission columns must be record_id,label')
        rows = list(reader)
    if unique_ids(rows) != unique_ids(test):
        raise ValueError('Sample submission IDs differ from test')
    return [r['record_id'] for r in rows]


def validate_synthetic(rows, train):
    unique_ids(rows, 160)
    if Counter(r['aug_rule'] for r in rows) != {k: 40 for k in RULES}:
        raise ValueError('Synthetic quotas must be 40 per rule')
    bases = {r['record_id']: r for r in train}
    used = set()
    flags, alts = load_rule_flags(), real_alt_label_counts(train)
    for r in rows:
        base = bases.get(r['base_record_id'])
        if base is None or base['label'] != '적절' or not base['alt_text'].strip():
            raise ValueError('Synthetic base is not a nonempty appropriate train example')
        if base['record_id'] in used:
            raise ValueError('Synthetic base reused')
        used.add(base['record_id'])
        if r.get('synthetic') is not True or r['label'] != RULES[r['aug_rule']]:
            raise ValueError('Invalid synthetic label or marker')
        if r['orig_alt'] != base['alt_text'] or any(r[k] != base[k] for k in [*CONTEXT_KEYS, 'image_file', 'image_url', 'page_url']):
            raise ValueError('Synthetic image/context/base changed')
        issues = audit_row(r, flags, alts)
        if issues:
            raise ValueError(f'Synthetic audit failed: {r["record_id"]}: {issues}')


@dataclass
class Dataset:
    original: list
    train: list
    test: list
    real: list
    synthetic: list
    known: dict
    sample_order: list
    image_info: dict
    policy: dict
    audit: dict


def prepare_data(data_root, boundary_csv, sample, run_dir):
    root, work = Path(data_root).resolve(), Path(run_dir)
    train_root, test_root = root / 'koaltq_train', root / 'koaltq_test_public'
    original = read_records(train_root, True, 2400)
    test = [{**r, 'image_root': str(test_root)} for r in read_records(test_root, False, 600)]
    if Counter(r['label'] for r in original) != EXPECTED_TRAIN:
        raise ValueError('Train label distribution differs from the corrected v1.1 release')
    overlaps = {}
    for k in ('record_id', 'page_url', 'image_url'):
        overlaps[k] = len({r[k] for r in original if r[k]} & {r[k] for r in test if r[k]})
    if any(overlaps.values()):
        raise ValueError(f'Unexpected released train/test overlap: {overlaps}')
    known = boundary_labels(boundary_csv, test, original)
    order = sample_ids(sample, test)
    extra = [{**r, 'label': known[r['record_id']], 'source': 'public_test_109', 'image_root': str(test_root)}
             for r in test if r['record_id'] in known]
    originals = [{**r, 'source': 'train_2400', 'image_root': str(train_root)} for r in original]
    synthetic, rejected = generate(original, None, {r: 40 for r in RULES}, 42, 3)
    validate_synthetic(synthetic, original)
    synthetic = [{**r, 'source': 'synthetic', 'image_root': str(train_root)} for r in synthetic]
    real, full = originals + extra, originals + extra + synthetic
    unique_ids(full, 2669)
    if Counter(r['label'] for r in real) != EXPECTED_REAL or Counter(r['label'] for r in full) != EXPECTED_FULL:
        raise ValueError('Final training class distribution mismatch')
    content_keys = [digest({k: r[k] for k in ['image_file', 'alt_text', *CONTEXT_KEYS]}) for r in full]
    if len(set(content_keys)) != len(content_keys):
        raise ValueError('Duplicate complete model input')
    image_info = {}
    for rows, directory in [(original, train_root), (test, test_root)]:
        print(f'CPU image audit: {directory.name}, {len(rows)} records', flush=True)
        with ThreadPoolExecutor(max_workers=4) as pool:
            info = list(pool.map(audit_image, ((directory, r) for r in rows)))
        bad = [r for r in info if not r['ok']]
        if bad:
            save_json(work / 'image_errors.json', bad)
            raise ValueError(f'Image audit failed: {len(bad)} records')
        image_info.update({r['record_id']: r for r in info})
    for r in synthetic:
        image_info[r['record_id']] = {**image_info[r['base_record_id']], 'record_id': r['record_id']}
    policy = build_policy(real, [r['record_id'] for r in real], ['filename_only_v1'], 3, 2)
    audit = {
        'counts': {'original_train': 2400, 'public_test_labeled': 109, 'synthetic': 160, 'full_train': 2669, 'test': 600},
        'class_counts': dict(Counter(r['label'] for r in full)),
        'real_class_counts': dict(Counter(r['label'] for r in real)),
        'synthetic_rules': dict(Counter(r['aug_rule'] for r in synthetic)),
        'duplicate_ids': 0, 'duplicate_complete_inputs': 0, 'image_errors': 0,
        'released_train_test_overlap': overlaps,
        'shared_page_occurrences': len(full) - len({r['page_url'] for r in full}),
        'shared_image_url_occurrences': len(full) - len({r['image_url'] for r in full}),
        'synthetic_rejections': rejected,
        'source_sha256': {'train': file_sha(train_root / 'records.jsonl'), 'test': file_sha(test_root / 'records.jsonl'),
                          'boundary': file_sha(boundary_csv), 'sample': file_sha(sample)},
        'training_records_sha256': digest(full), 'test_records_sha256': digest(test),
        'public_test_labels_used_count': 109, 'synthetic_excluded_from_weights_and_policy': True,
        'validation_used': False,
    }
    save_json(work / 'data_audit.json', audit)
    save_json(work / 'layer3_policy.json', policy)
    (work / 'synthetic_train.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in synthetic), encoding='utf-8')
    return Dataset(originals, full, test, real, synthetic, known, order, image_info, policy, audit)


def write_predictions(path, predictions, order):
    if unique_ids(predictions) != set(order) or len(order) != len(set(order)):
        raise ValueError('Prediction IDs missing, duplicated or unexpected')
    by_id = {r['record_id']: r['label'] for r in predictions}
    if any(label not in LABELS for label in by_id.values()):
        raise ValueError('Invalid prediction label')
    path = Path(path)
    with path.open('x', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['record_id', 'label'])
        writer.writerows((rid, by_id[rid]) for rid in order)
    # Verify the serialized file too.
    sample_ids(path, predictions)
    return path


def correct_boundary(predictions, known, expected=109):
    ids = unique_ids(predictions)
    if len(known) != expected or not set(known) <= ids or any(v not in LABELS for v in known.values()):
        raise ValueError('Invalid boundary correction IDs or labels')
    corrected = [{'record_id': r['record_id'], 'label': known.get(r['record_id'], r['label'])} for r in predictions]
    if any(a != b for a, b in zip(predictions, corrected) if a['record_id'] not in known):
        raise AssertionError('Non-boundary predictions changed')
    return corrected
