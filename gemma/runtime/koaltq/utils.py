"""공통 입출력·해시·실행 기록."""
from __future__ import annotations
import hashlib, json, os, platform, random, sys, time, uuid
from pathlib import Path
from importlib import metadata
from typing import Any

LABELS = ['적절','무의미형','파일명형','중복형','장식오용형','불충분형','무관형']
CONTEXT_KEYS = ['page_title','context_text','heading_path','in_link','link_dest']


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode('utf-8')).hexdigest()


def file_sha(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(b)
    return h.hexdigest()


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def save_json(path: str | Path, value: Any) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(temp, path)


def save_csv(path: str | Path, frame, encoding: str = 'utf-8-sig', index: bool = False) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    frame.to_csv(temp, index=index, encoding=encoding)
    os.replace(temp, path)


def append_event(path: str | Path, event: dict) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    event = {'time_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **event}
    with path.open('a', encoding='utf-8') as f:
        f.write(canonical(event) + '\n')


def package_versions() -> dict:
    out = {}
    for name in ['torch','torchvision','transformers','peft','bitsandbytes','accelerate',
                 'huggingface-hub','tokenizers','safetensors','numpy','pandas','scipy',
                 'scikit-learn','Pillow','resvg-py','filelock','nbformat']:
        try: out[name] = metadata.version(name)
        except metadata.PackageNotFoundError: out[name] = None
    return out


def environment() -> dict:
    out = {'python': sys.version, 'executable': sys.executable, 'platform': platform.platform(),
           'cwd': str(Path.cwd()), 'packages': package_versions()}
    try:
        import torch
        out.update(torch_version=torch.__version__, cuda_available=torch.cuda.is_available(),
                   torch_cuda=torch.version.cuda)
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            out.update(gpu_name=p.name, gpu_total_gib=p.total_memory/2**30,
                       compute_capability=list(torch.cuda.get_device_capability(0)),
                       compiled_arches=torch.cuda.get_arch_list())
    except ImportError: out['cuda_available'] = False
    return out


def seed_all(seed: int) -> None:
    import numpy as np
    import torch
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def code_hash(root: Path) -> str:
    paths = sorted((root/'koaltq').glob('*.py')) + sorted((root/'prompts').glob('*.txt'))
    return digest({str(p.relative_to(root)): file_sha(p) for p in paths})


def load_config(path: str | Path = 'config.json') -> dict:
    path = Path(path).resolve(); cfg = read_json(path)
    package = Path(cfg.get('package_root', path.parent)).expanduser()
    cfg['_package_root'] = str((path.parent/package).resolve() if not package.is_absolute() else package.resolve())
    root = Path(cfg['project_dir']).expanduser()
    cfg['_project_root'] = str((path.parent/root).resolve() if not root.is_absolute() else root.resolve())
    work = Path(cfg['work_dir']).expanduser()
    cfg['_work'] = str((path.parent/work).resolve() if not work.is_absolute() else work.resolve())
    if Path(cfg['_work']) == Path(cfg['_project_root']):
        raise ValueError('work_dir는 원본 자료 폴더와 다른 별도 하위/외부 폴더가 필요합니다.')
    if cfg['score_method'] not in ('sum_logprob','mean_logprob'): raise ValueError('score_method 오류')
    if cfg['model']['compute_dtype'] != 'bfloat16': raise ValueError('이 구현은 BF16 공통 조건입니다.')
    if cfg['training']['effective_batch_size'] < 1: raise ValueError('effective_batch_size 오류')
    return cfg


def locked_contract(path: Path, contract: dict) -> str:
    h = digest(contract)
    if path.exists() and read_json(path).get('sha256') != h:
        raise RuntimeError(f'공통 조건 변경 감지: {path}. 새 work_dir에서 세 실험을 다시 실행하세요.')
    if not path.exists(): save_json(path, {'sha256': h, 'contract': contract})
    return h
