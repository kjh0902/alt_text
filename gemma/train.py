"""Run from Linux/Windows terminals: python gemma/train.py --help."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))

from gemma.runtime.koaltq.utils import digest, file_sha, save_json, read_json


def arguments(argv=None):
    p = argparse.ArgumentParser(description='Gemma Joint CE: full 2669-example training and epoch test predictions')
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--epochs', type=int, default=3)
    p.add_argument('--run-dir', type=Path, default=None)
    p.add_argument('--layer1-cache-dir', type=Path)
    p.add_argument('--boundary-csv', type=Path, default=ROOT / 'resources' / '경계사례_20261013.csv')
    p.add_argument('--sample-submission', type=Path)
    p.add_argument('--model-cache-dir', type=Path, default=ROOT.parent / '.cache' / 'gemma_models')
    mode = p.add_mutually_exclusive_group()
    mode.add_argument('--validate-only', action='store_true')
    mode.add_argument('--smoke-only', action='store_true', help='Validate data and run real GPU Joint CE checks, then exit')
    args = p.parse_args(argv)
    if args.epochs < 1:
        p.error('--epochs must be positive')
    if args.run_dir is None:
        args.run_dir = ROOT.parent / 'runs' / ('gemma_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    for k in ('data_root', 'run_dir', 'boundary_csv', 'model_cache_dir', 'layer1_cache_dir', 'sample_submission'):
        value = getattr(args, k)
        if value is not None:
            setattr(args, k, value.expanduser().resolve())
    if args.sample_submission is None:
        candidates = [args.data_root / 'koaltq_test_public' / 'sample_submission.csv', args.data_root / 'sample_submission.csv']
        args.sample_submission = next((p for p in candidates if p.is_file()), candidates[0])
    if args.layer1_cache_dir is not None:
        source = args.layer1_cache_dir
        if not (source / 'actual_model_api.json').is_file() or not (source / 'cache/layer1').is_dir():
            p.error('--layer1-cache-dir must contain actual_model_api.json and cache/layer1/')
    return args


def configuration(args):
    cfg = read_json(ROOT / 'baseline_config.json')
    # These are the notebook bootstrap overrides, with batches fixed to one for 16GB.
    cfg['layer1']['recovery']['supplemental_protocol'] = 'schema_recovery_v3'
    cfg['runtime'].update(layer1_batch_trials=[1], candidate_batch_trials=[1], microbatch_trials=[1])
    cfg['training']['epochs'] = args.epochs
    cfg['training'].pop('checkpoint_selection', None)
    cfg['training'].pop('checkpoint_every_steps', None)
    cfg['split'] = None
    cfg['sources'] = {'train': str(args.data_root / 'koaltq_train'), 'test': str(args.data_root / 'koaltq_test_public'),
                      'sample_submission': str(args.sample_submission), 'boundary_csv': str(args.boundary_csv)}
    cfg.pop('optional', None)
    cfg.pop('documents', None)
    cfg['project_dir'] = str(args.data_root)
    cfg['work_dir'] = str(args.run_dir)
    cfg['model_cache_dir'] = str(args.model_cache_dir)
    cfg['_package_root'] = str(ROOT / 'runtime')
    cfg['_work'] = str(args.run_dir)
    cfg['fresh_start_only'] = True
    cfg['smoke_only'] = args.smoke_only
    if args.layer1_cache_dir:
        cfg['layer1_cache_source'] = {'run_dir': str(args.layer1_cache_dir), 'require_complete': False}
    cfg['method_changes'] = {
        'training_data': '2400 original + 109 public labels + 160 synthetic; no validation or selection',
        'class_weights_and_layer3': '2509 real labeled examples only; original formulas and rule eligibility',
        'epochs': 'CLI count controls the original warmup/linear scheduler total steps',
        'batching': 'microbatch=1, accumulation=8, effective batch=8; retain final partial group',
        'saving': 'every epoch adapter and test predictions; no optimizer checkpoints or resume',
        'preprocessing': 'base-only Layer1 prepared once; deterministic CPU Layer2 inputs saved once',
        'loss': '0.8 weighted 7-class CE + 0.2 unweighted binary CE; mean label logprobs; example-count mean',
        'backward': 'two-pass candidate recomputation with RNG replay; one model graph at a time',
        'inference': '7-class mean_logprob argmax, matching training scores',
    }
    return cfg


def claim_run(path, data_root):
    path = Path(path)
    if path.resolve() == Path(data_root).resolve() or path.is_symlink():
        raise ValueError('Run directory must be separate from data and cannot be a symlink')
    # Existing even empty run directories are refused; never delete or restore files.
    path.mkdir(parents=True, exist_ok=False)


def run(args):
    # No CUDA/model-library imports or downloads are needed for validation.
    from gemma.data import prepare_data
    from gemma.runtime.koaltq.scoring import class_weights
    from gemma.runtime.koaltq.layer0 import rule_flags
    from gemma.workflow import capture_log

    claim_run(args.run_dir, args.data_root)
    cfg = configuration(args)
    save_json(args.run_dir / 'config.json', cfg)
    with capture_log(args.run_dir / 'run.log'):
        try:
            dataset = prepare_data(args.data_root, args.boundary_csv, args.sample_submission, args.run_dir)
            weights = class_weights(dataset.train, 3.)
            if weights != class_weights(dataset.real, 3.) or sum(weights['counts'].values()) != 2509:
                raise AssertionError('Synthetic examples affected class weights')
            save_json(args.run_dir / 'class_weights.json', weights)
            save_json(args.run_dir / 'layer0_flags.json', {r['record_id']: rule_flags(r) for r in dataset.train + dataset.test})
            print(json.dumps(dataset.audit, ensure_ascii=False, indent=2), flush=True)
            if args.validate_only:
                save_json(args.run_dir / 'validation_result.json', {'status': 'CPU_VALIDATION_PASSED', **dataset.audit})
                print('CPU_VALIDATION_PASSED: 2669 training examples; 3000 images; 600 unique submission IDs', flush=True)
                return
            from gemma.cache import EncodedInputs
            from gemma.runtime.koaltq.model import GemmaEngine, check_gpu_kernels
            from gemma.runtime.koaltq.pipeline import prepare_layer1
            from gemma.runtime.koaltq.smoke import run_smoke
            from gemma.workflow import train_epochs

            save_json(args.run_dir / 'gpu_kernels.json', check_gpu_kernels())
            # Fail early on the longest training samples before a full Layer1 pass.
            # The smoke engine attaches/restores LoRA, so close it before base-only Layer1.
            with GemmaEngine(cfg) as engine:
                contract = digest({'config': cfg, 'dataset': dataset.audit, 'model': engine.identity,
                    'pipeline_code': {str(p.relative_to(ROOT)): file_sha(p) for p in sorted(ROOT.rglob('*.py'))}})
                save_json(args.run_dir / 'run_contract.json', {'sha256': contract})
                smoke = run_smoke(engine, cfg, dataset.train, args.data_root, dataset.image_info, contract, 'full_joint_ce')
                if smoke['status'] != 'GPU_TEST_PASSED' or not smoke['smoke_adapter_restored']:
                    raise RuntimeError('GPU preflight did not restore the initial adapter')
            if args.smoke_only:
                print('GPU smoke complete; full training was not requested:', args.run_dir, flush=True)
                return
            with GemmaEngine(cfg) as engine:
                # A fresh base model generates only missing Layer1 keys, including smoke hits.
                base_rows = dataset.original + dataset.test
                layer1, _ = prepare_layer1(engine, cfg, base_rows, args.data_root, dataset.image_info, 'full_train_and_public_test')
                for r in dataset.synthetic:
                    layer1[r['record_id']] = layer1[r['base_record_id']]
                engine.attach_lora()
                encoded = EncodedInputs(args.run_dir / 'cache' / 'encoded_inputs')
                encoded.prepare(engine, cfg, dataset.train + dataset.test, args.data_root, dataset.image_info, layer1)
                train_epochs(engine, cfg, dataset, encoded, smoke['plan'])
            print('Training complete:', args.run_dir, flush=True)
        except BaseException as error:
            save_json(args.run_dir / 'failure.json', {'error': repr(error), 'traceback': traceback.format_exc(), 'resume_supported': False})
            traceback.print_exc()
            if not args.validate_only:
                try:
                    from gemma.runtime.koaltq.gpu_memory import collect_diagnostics
                    save_json(args.run_dir / 'failure_gpu_diagnostics.json', collect_diagnostics())
                except Exception:
                    pass
            raise


if __name__ == '__main__':
    run(arguments())
