"""같은 시점의 CUDA·nvidia-smi 메모리 기록. 판정 기준 자동 변경 없음."""
from __future__ import annotations
import csv
import io
import math
import os
import subprocess
import time


def _number(value):
    try:
        value = float(value)
        return value if math.isfinite(value) and value >= 0 else None
    except (TypeError, ValueError):
        return None


def parse_gpu_csv(text):
    names = ['index', 'uuid', 'name', 'driver_version', 'total_mib', 'used_mib', 'free_mib']
    result = []
    for fields in csv.reader(io.StringIO(text)):
        if not fields:
            continue
        if len(fields) != len(names):
            raise ValueError('nvidia-smi GPU 열 개수 오류')
        row = dict(zip(names, (s.strip() for s in fields)))
        for key in ('total_mib', 'used_mib', 'free_mib'):
            row[key] = _number(row[key])
        result.append(row)
    return result


def _run_query(query):
    try:
        p = subprocess.run(['nvidia-smi', query, '--format=csv,noheader,nounits'],
                           capture_output=True, text=True, encoding='utf-8', errors='replace',
                           timeout=5, check=False)
        if p.returncode:
            return {'ok':False, 'error':p.stderr.strip()[:500] or f'exit={p.returncode}'}
        return {'ok':True, 'stdout':p.stdout}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {'ok':False, 'error':str(exc)[:500]}


def snapshot(device=0):
    import torch
    torch.cuda.synchronize(device)
    free, total = torch.cuda.mem_get_info(device)
    return {'time_ns':time.time_ns(), 'free_gib':free / 2**30,
            'driver_total_gib':total / 2**30,
            'allocated_gib':torch.cuda.memory_allocated(device) / 2**30,
            'reserved_gib':torch.cuda.memory_reserved(device) / 2**30}


def select_device(rows, properties):
    uuid = str(getattr(properties, 'uuid', '') or '')
    if uuid:
        normalized = uuid.removeprefix('GPU-').lower()
        matches = [r for r in rows if r['uuid'].removeprefix('GPU-').lower() == normalized]
        return (matches[0], 'uuid') if len(matches) == 1 else (None, 'uuid_unmatched')
    # CUDA_VISIBLE_DEVICES 때문에 index=0만으로 다른 GPU를 고르지 않음
    if len(rows) == 1:
        row = rows[0]
        expected = properties.total_memory / 2**20
        if row['name'] == properties.name and row['total_mib'] is not None and abs(row['total_mib']-expected) < 128:
            return row, 'single_gpu_name_and_total'
    return None, 'unmatched'


def collect_diagnostics(device=0):
    import torch
    before = snapshot(device)
    gpu = _run_query('--query-gpu=index,uuid,name,driver_version,memory.total,memory.used,memory.free')
    selected = None
    match = 'query_unavailable'
    if gpu['ok']:
        try:
            gpu['gpus'] = parse_gpu_csv(gpu.pop('stdout'))
            selected, match = select_device(gpu['gpus'], torch.cuda.get_device_properties(device))
        except ValueError as exc:
            gpu = {'ok':False, 'error':str(exc)}
    after = snapshot(device)
    processes = _run_query('--query-compute-apps=pid,process_name')
    return {'pid':os.getpid(), 'cuda_device':device, 'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),
            'cuda_before_smi':before, 'cuda_after_smi':after, 'nvidia_smi':gpu,
            'selected_gpu':selected, 'device_match':match, 'compute_processes':processes,
            'automatic_fallback':False,
            'note':'같은 모델이 살아 있는 동안 연속 측정. nvidia-smi 값으로 CUDA 부족 판정을 덮어쓰지 않음.'}


def memory_decision(peak_reserved, total, free, fraction, minimum):
    values = (peak_reserved, total, free, fraction, minimum)
    if not all(isinstance(x, (int,float)) and math.isfinite(x) for x in values):
        raise ValueError('GPU 메모리 측정값/기준이 유효하지 않음')
    if peak_reserved < 0 or total <= 0 or not 0 <= free <= total or not 0 < fraction <= 1 or minimum <= 0:
        raise ValueError('GPU 메모리 측정값/기준 범위 오류')
    peak_ok = peak_reserved <= total * fraction
    free_ok = free >= minimum
    reasons = []
    if not peak_ok:
        reasons.append('peak_reserved_above_fraction_limit')
    if not free_ok:
        reasons.append('free_after_cache_release_below_minimum')
    return peak_ok, free_ok, reasons


def print_diagnostics(value):
    cuda_free = value['cuda_after_smi']['free_gib']
    selected = value['selected_gpu']
    smi_free = None if selected is None else selected['free_mib']
    smi = '확인 불가' if smi_free is None else f'{smi_free/1024:.3f} GiB'
    print(f"현재 PID: {value['pid']} | CUDA 여유: {cuda_free:.3f} GiB | nvidia-smi 여유: {smi}")
    if cuda_free == 0 and smi_free is not None and smi_free > 0:
        print('두 측정값이 다름. Windows/드라이버 메모리 보고 확인 필요; 실제 여유 확보로 간주하지 않음.')
    processes = value['compute_processes']
    if processes['ok']:
        lines = processes.get('stdout', '').strip().splitlines()
        print('GPU 계산 프로세스:', ' / '.join(lines[:8]) or '조회 결과 없음')
