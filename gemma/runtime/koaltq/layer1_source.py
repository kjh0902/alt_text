"""완료된 다른 PC 실행의 Layer 1 결과 재사용. cfg['layer1_cache_source']가 있을 때만 동작."""
from __future__ import annotations
import functools
import shutil
from pathlib import Path
from .utils import read_json, save_json
from .prompts import validate_layer1
from .l1_recovery import parse_output

# 실행 환경 항목만 원본 실행값으로 대체. 프롬프트·이미지·생성 설정·seed는 키에 그대로 남음
ENVIRONMENT_FIELDS = ('python','platform','torch_cuda','gpu_name','compute_capability',
                      'implementation_sha256','packages')


def source_run(cfg: dict) -> Path | None:
    source = cfg.get('layer1_cache_source')
    return Path(source['run_dir']) if source else None


@functools.lru_cache(maxsize=None)
def _source_identity(run_dir: str) -> dict:
    return read_json(Path(run_dir)/'actual_model_api.json')['identity']


def _model_fields(identity: dict) -> dict:
    return {k:v for k,v in identity.items() if k not in ENVIRONMENT_FIELDS}


def key_identity(cfg: dict, identity: dict) -> dict:
    run = source_run(cfg)
    if run is None:return identity
    source = _source_identity(str(run))
    if _model_fields(source) != _model_fields(identity):
        diff = sorted(k for k in set(source)|set(identity)
                      if k not in ENVIRONMENT_FIELDS and source.get(k) != identity.get(k))
        raise ValueError(f'Layer 1 원본 실행과 모델 조건 불일치: {diff}')
    return source


def validate_cache(value, key):
    if value.get('key') != key:
        raise ValueError('Layer 1 cache key mismatch')
    validate_layer1(value.get('analysis'))
    raw = value.get('raw')
    if not isinstance(raw, str) or parse_output(raw)[0] != value['analysis']:
        raise ValueError('Layer 1 cached raw/analysis mismatch')
    return value


def seed_cache(cfg: dict, identity: dict, keys: dict) -> int:
    """요청 레코드의 원본 캐시 파일만 현재 실행 cache/layer1로 복사. 기존 파일은 덮어쓰지 않음."""
    run = source_run(cfg)
    if run is None:return 0
    source = _source_identity(str(run))
    work = Path(cfg['_work']);dest = work/'cache'/'layer1';dest.mkdir(parents=True,exist_ok=True)
    record = work/'layer1_cache_source.json'
    if not record.exists():
        save_json(record,{'run_dir':str(run),'model_fields_equal':True,
                          'environment_differences':{k:{'source':source.get(k),'current':identity.get(k)}
                                                     for k in ENVIRONMENT_FIELDS if source.get(k) != identity.get(k)}})
    copied = 0
    for key in sorted(set(keys.values())):
        src = run/'cache'/'layer1'/(key+'.json');out = dest/(key+'.json')
        if out.exists() or not src.exists():continue
        validate_cache(read_json(src), key)
        shutil.copyfile(src,out);copied += 1
    return copied
