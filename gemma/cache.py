"""Content-addressed CPU processor outputs, shared across training epochs and test."""
from __future__ import annotations

from pathlib import Path

from .runtime.koaltq.images import IMAGE_POLICY
from .runtime.koaltq.prompts import layer2_text, load_prompts
from .runtime.koaltq.utils import digest, file_sha, read_json, save_json


class EncodedInputs:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.keys = {}
        self.verified = set()
        self.new_records = 0

    def prepare(self, engine, cfg, rows, root, image_info, layer1):
        from safetensors.torch import save_file
        from .runtime.koaltq.pipeline import encoded_stream

        prompts = load_prompts(Path(cfg['_package_root']))
        jobs, missing = [], set()
        for r in rows:
            rid = r['record_id']
            prompt = layer2_text(r, layer1[rid]['analysis'], prompts)
            sha = image_info[rid]['image_sha256']
            key = digest({'stage': 'layer2_cpu_input_v1', 'model': engine.identity, 'prompt': prompt,
                          'image_sha256': sha, 'image_policy': IMAGE_POLICY})
            if rid in self.keys and self.keys[rid] != key:
                raise ValueError('Record processor input changed: ' + rid)
            self.keys[rid] = key
            path = self.directory / (key + '.safetensors')
            if path.exists():
                self.get(r)
            elif key not in missing:
                missing.add(key)
                jobs.append((r, prompt, sha))
        for r, base in encoded_stream(engine, root, jobs, cfg):
            key = self.keys[r['record_id']]
            path = self.directory / (key + '.safetensors')
            save_file({k: v.cpu().contiguous() for k, v in base.items()}, str(path), metadata={'key': key})
            save_json(path.with_suffix('.json'), {'key': key, 'sha256': file_sha(path)})
            self.new_records += 1
        save_json(self.directory / 'record_index.json', {'keys': self.keys, 'new_records': self.new_records})

    def get(self, row):
        from safetensors import safe_open
        from safetensors.torch import load_file

        key = self.keys[row['record_id']]
        path = self.directory / (key + '.safetensors')
        if key not in self.verified:
            meta = read_json(path.with_suffix('.json'))
            if meta['key'] != key or meta['sha256'] != file_sha(path):
                raise ValueError('Encoded input cache integrity failure')
            with safe_open(str(path), framework='pt', device='cpu') as f:
                if f.metadata().get('key') != key:
                    raise ValueError('Encoded input cache key mismatch')
            self.verified.add(key)
        return load_file(str(path), device='cpu')
