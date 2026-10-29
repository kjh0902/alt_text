"""내용 기반 캐시·CPU 전처리 선행 준비·GPU 배치 판정·실행 비용 분리."""
from __future__ import annotations
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import gc,json,time,uuid
import torch
from tqdm.auto import tqdm
from .utils import LABELS, canonical,digest,read_json,save_json,append_event,code_hash,locked_contract
from .prompts import context_for,rule_flags,load_prompts,layer1_text,layer2_text,validate_layer1
from .images import load_image,image_path,IMAGE_POLICY
from .scoring import select_label
from .l1_recovery import parse_output, load_seed, recover_one
from .layer1_source import key_identity, seed_cache, validate_cache


LAYER1_RECOVERY_SUFFIX = '''\n\n[JSON 형식 복구 재시도]\n같은 이미지와 같은 문맥을 다시 분석하되 내용 설명을 덧붙이지 말고 JSON 객체 하나만 출력하라.\n키는 role, visual_type, has_informative_text, visible_text, key_visual_information, context_already_covers, additional_information, uncertain 여덟 개를 모두 정확히 사용한다.\nvisible_text, key_visual_information, context_already_covers, additional_information, uncertain은 반드시 JSON 배열이다.\n각 배열은 핵심만 최대 4개, 각 문자열은 120자 이하로 간결하게 쓴다. 불확실하거나 해당 사항이 없으면 빈 배열 []을 사용한다.\n마크다운 코드펜스, 설명문, 주석은 출력하지 않는다. 기존 alt와 정답 label은 사용하지 않는다.'''
_LAYER1_LIST_FIELDS = {'visible_text','key_visual_information','context_already_covers','additional_information','uncertain'}


def l1_key(cfg: dict,identity: dict,prompts: dict,row: dict,image_sha: str) -> str:
    return digest({'model':key_identity(cfg,identity),'stage':'layer1','prompt':layer1_text(row,prompts),
                   'image_sha256':image_sha,'image_policy':IMAGE_POLICY,'generation':cfg['layer1'],
                   'recovery_prompt':LAYER1_RECOVERY_SUFFIX,'seed':cfg['seed']})


def _encoded(engine,root: Path,row: dict,text: str,expected_sha: str) -> dict:
    # PIL 디코딩·EXIF·색 공간 변환·processor 실행: Windows 안전 스레드 작업
    image,meta=load_image(root,row)
    try:
        if meta['image_sha256']!=expected_sha:raise RuntimeError('데이터 검사 이후 이미지 내용 변경')
        return engine.encode(text,image)
    finally:image.close()


def encoded_stream(engine,root: Path,jobs: list[tuple[dict,str,str]],cfg: dict):
    # 메인 GPU 계산 중 다음 입력 전처리, 제출 순서 보존
    with ThreadPoolExecutor(max_workers=cfg['runtime']['cpu_workers']) as pool:
        pending=deque();it=iter(jobs)
        def refill():
            while len(pending)<cfg['runtime']['prefetch_images']:
                try:row,text,sha=next(it)
                except StopIteration:break
                pending.append((row,pool.submit(_encoded,engine,root,row,text,sha)))
        refill()
        try:
            while pending:
                row,f=pending.popleft();value=f.result();refill();yield row,value
        finally:
            for _,f in pending:f.cancel()


def begin_measure() -> float:
    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();return time.perf_counter()


def finish_measure(start: float) -> dict:
    torch.cuda.synchronize();return {'wall_seconds':time.perf_counter()-start,
        'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,
        'peak_reserved_gib':torch.cuda.max_memory_reserved()/2**30}


def _parse_layer1_output(raw: str) -> tuple[dict,dict]:
    return parse_output(raw)


def _recovery_layer1_text(row: dict,prompts: dict) -> str:
    return layer1_text(row,prompts)+LAYER1_RECOVERY_SUFFIX


def _load_layer1_seed(cfg, identity, prompts, rows, image_info):
    return load_seed(cfg, identity, prompts, rows, image_info, l1_key)


def choose_layer1_batch(engine,cfg,root,jobs):
    # GPU 프로세스 복제 없이 동일 모델 내부 배치. 출력 일치와 메모리 실측.
    from .smoke import memory_ok
    trials=cfg['runtime'].get('layer1_batch_trials',[1])
    if not jobs or trials==[1]:return 1
    path=Path(cfg['_work'])/'layer1_batch_plan.json'
    signature=digest({'identity':engine.identity,'layer1':cfg['layer1'],'trials':trials,
                      'limits':{k:cfg['runtime'][k] for k in ('max_gpu_fraction','min_free_gib')}})
    if path.exists():
        previous=read_json(path)
        if previous['signature']!=signature:raise ValueError('Layer 1 배치 검사 조건 변경')
        return previous['batch_size']
    sample=sorted(jobs,key=lambda job:(-len(job[1]),job[0]['record_id']))[:max(trials)]
    bases=[_encoded(engine,root,*job) for job in sample]
    print(f'Layer 1 배치 검사: 표본 {len(sample)}건, 후보 {trials}',flush=True)
    reference=[];reports=[]
    for bs in trials:
        if bs>len(bases):continue
        gc.collect();torch.cuda.empty_cache();start=begin_measure()
        try:
            outputs=[]
            for index in range(0,len(bases),bs):outputs.extend(engine.generate_layer1(bases[index:index+bs]))
            report=finish_measure(start)
            if bs==1:reference=outputs
            def normalized(raw):
                try:return parse_output(raw)[0]
                except (ValueError,TypeError):return raw
            match=len(reference)==len(outputs) and all(normalized(a)==normalized(b) for a,b in zip(reference,outputs))
            mem=memory_ok(cfg,report)
            reports.append({'batch_size':bs,**report,'outputs_match':match,'memory_ok':mem})
        except torch.OutOfMemoryError:
            gc.collect();torch.cuda.empty_cache()
            reports.append({'batch_size':bs,'oom':True,'outputs_match':False,'memory_ok':False})
        print(f"배치 {bs}: 출력 일치={reports[-1]['outputs_match']}, 메모리={reports[-1]['memory_ok']}, OOM={reports[-1].get('oom',False)}",flush=True)
    valid=[r for r in reports if r['outputs_match'] and r['memory_ok']]
    if not valid:
        save_json(path.with_name('layer1_batch_failed.json'),{'reports':reports})
        raise RuntimeError('Layer 1 단건/배치 GPU 검사 미통과. 원본 캐시 보존.')
    best=min(valid,key=lambda r:r['wall_seconds'])['batch_size']
    save_json(path,{'signature':signature,'batch_size':best,'reports':reports,
                    'sample_ids':[j[0]['record_id'] for j in sample],
                    'scope':'sample_only; full_generation_has_OOM_backoff',
                    'probe_compute_seconds':sum(r.get('wall_seconds',0) for r in reports)})
    print(f'Layer 1 선택 배치: {best} | 미통과/느린 큰 배치는 제외',flush=True)
    return best


def _prepare_layer1_unique(engine,cfg: dict,rows: list[dict],root: Path,image_info: dict,scope: str) -> tuple[dict,dict]:
    if engine.adapter_active:raise RuntimeError('공통 Layer1에는 기본 Gemma만 허용')
    prompts=load_prompts(Path(cfg['_package_root']));work=Path(cfg['_work']);cache=work/'cache'/'layer1'
    cache.mkdir(parents=True,exist_ok=True)
    recovery_cfg=cfg['layer1'].get('recovery',{})
    recovery_enabled=bool(recovery_cfg.get('enabled',False))
    if recovery_enabled and recovery_cfg.get('protocol')!='strict_json_retry_v1':raise ValueError('Layer1 recovery protocol 오류')
    _load_layer1_seed(cfg,engine.identity,prompts,rows,image_info)  # rejects old-run cache settings
    byid={};first_jobs=[];recovery_jobs=[];keys={};hits=0;reused_seconds=0.;errors=[];compute=0.;actual_batches=[]
    migrated_valid=0;safe_repairs=0;recovery_gpu=0;first_gpu=0
    started=begin_measure()
    for r in rows:
        rid=r['record_id'];keys[rid]=l1_key(cfg,engine.identity,prompts,r,image_info[rid]['image_sha256'])
    seeded=seed_cache(cfg,engine.identity,keys)
    if seeded:print(f'Layer 1: 원본 실행 캐시 {seeded}건 복사 ({cfg["layer1_cache_source"]["run_dir"]})')
    for r in rows:
        rid=r['record_id'];key=keys[rid]
        p=cache/(key+'.json')
        if p.exists():
            value=read_json(p)
            validate_cache(value,key)
            byid[rid]=value;hits+=1;reused_seconds+=value.get('new_compute_seconds',0);continue
        first_jobs.append((r,layer1_text(r,prompts),image_info[rid]['image_sha256']))

    print(f'Layer 1: 요청 {len(rows)}건 / 현재 캐시 {hits}건 / 재추론 대상 {len(recovery_jobs)}건 / 신규 {len(first_jobs)}건')
    source=cfg.get('layer1_cache_source')
    if first_jobs and source and source.get('require_complete',True):
        # 원본 캐시를 쓰기로 한 실행에서 몇 시간짜리 Layer 1 재생성을 조용히 시작하지 않음
        missing=[r['record_id'] for r,_,_ in first_jobs]
        raise RuntimeError(f'Layer 1 원본 캐시 누락 {len(missing)}건 (예: {missing[:5]}). GPU 생성 전에 중단')
    batch_size=choose_layer1_batch(engine,cfg,root,first_jobs) if first_jobs else 1
    pending=[]
    def save_success(r,raw,elapsed,mode='first_pass',extra=None):
        rid=r['record_id'];key=keys[rid];analysis,repair=_parse_layer1_output(raw)
        value={'key':key,'analysis':analysis,'raw':raw,'new_compute_seconds':elapsed}
        if mode!='first_pass' or repair['actions']:
            value['recovery']={'mode':mode,**repair,**(extra or {})}
        save_json(cache/(key+'.json'),value);byid[rid]=value

    def flush_first(part):
        nonlocal compute,batch_size,first_gpu
        position=0
        while position<len(part):
            chunk=part[position:position+batch_size]
            torch.cuda.synchronize();t=time.perf_counter()
            try:raws=engine.generate_layer1([b for _,b in chunk]);torch.cuda.synchronize()
            except torch.OutOfMemoryError:
                gc.collect();torch.cuda.empty_cache()
                if batch_size<=1:raise RuntimeError('Layer1 batch=1 OOM. 입력 조건 자동 변경 없음; 새 공통 실험 설정 검토 필요') from None
                batch_size=max(1,batch_size//2);append_event(work/'oom_events.jsonl',{'stage':'layer1','new_batch_size':batch_size});continue
            elapsed=time.perf_counter()-t;compute+=elapsed;actual_batches.append(len(chunk));first_gpu+=len(chunk)
            if len(raws)!=len(chunk):raise AssertionError('Layer1 출력 개수 불일치')
            for (r,_),raw in zip(chunk,raws):
                try:save_success(r,raw,elapsed/len(chunk),'first_pass')
                except (ValueError,TypeError,json.JSONDecodeError) as e:
                    if recovery_enabled:
                        recovery_jobs.append((r,_recovery_layer1_text(r,prompts),image_info[r['record_id']]['image_sha256'],
                                              {'key':keys[r['record_id']],'raw':raw,'error':repr(e)}))
                    else:
                        error={'record_id':r['record_id'],'key':keys[r['record_id']],'raw':raw,'error':repr(e)};errors.append(error)
                        save_json(work/'errors'/'layer1'/(keys[r['record_id']]+'.json'),error)
            position+=len(chunk)

    try:
        for row,base in tqdm(encoded_stream(engine,root,first_jobs,cfg),total=len(first_jobs),desc=f"신규 추론 {scope}"):
            pending.append((row,base))
            if len(pending)>=batch_size:flush_first(pending);pending=[]
        if pending:flush_first(pending);pending=[]

        if recovery_jobs:
            for row, _text, sha, old in tqdm(recovery_jobs,total=len(recovery_jobs),desc=f"남은 건 복구 {scope}"):
                result=recover_one(engine,cfg,row,root,sha,_encoded)
                elapsed=result['seconds'];compute+=elapsed
                actual_batches.extend([1]*result['calls']);recovery_gpu+=1
                if 'error' not in result:
                    save_success(row,result['raw'],elapsed,'supplemental_schema_recovery',
                                 {'source_key':old.get('key'), 'protocol':result['protocol'],
                                  'strategy':result['mode'], 'generation_calls':result['calls'],
                                  'attempts':result['attempts'],
                                  'raw_is_assembled':result.get('raw_is_assembled',False)})
                else:
                    key=keys[row['record_id']]
                    error={'record_id':row['record_id'],'key':key,'raw':result['raw'],
                           'error':result['error'],'recovery_attempted':True,
                           'previous_error':old.get('error'),'attempts':result['attempts']}
                    errors.append(error);save_json(work/'errors'/'layer1'/(key+'.json'),error)
    except BaseException as e:
        append_event(work/'errors'/'execution.jsonl',{'scope':scope,'stage':'layer1','error':repr(e)})
        raise
    timing={**finish_measure(started),'scope':scope,'stage':'layer1','requested_records':len(rows),
        'cache_hits':hits,'legacy_valid_migrated':migrated_valid,'safe_structural_repairs':safe_repairs,
        'first_pass_gpu_records':first_gpu,'recovery_gpu_records':recovery_gpu,
        'new_records':first_gpu+recovery_gpu,'errors':len(errors),
        'new_compute_seconds':compute,'cache_reused_historical_compute_seconds':reused_seconds,
        'actual_batch_sizes':actual_batches,
        'new_compute_records_per_second':(first_gpu+recovery_gpu-len(errors))/compute if compute else None}
    append_event(work/'layer1_runs.jsonl',timing)
    save_json(work/'layer1_exports'/f'{scope}.json',{'records':byid,'timing':timing})
    if errors or len(byid)!=len(rows):raise RuntimeError(f'Layer1 정상 {len(byid)}/{len(rows)}, 남은 오류 {len(errors)}. 정상 캐시 저장 완료. errors/layer1 및 recovery_attempts 확인.')
    return byid,timing



def prepare_layer1(engine, cfg, rows, root, image_info, scope):
    """Generate one analysis per content key, then map it back to all record IDs."""
    prompts = load_prompts(Path(cfg['_package_root']))
    representatives, aliases = {}, {}
    for row in rows:
        rid = row['record_id']
        if rid in aliases:
            raise ValueError('Duplicate Layer1 record ID: ' + rid)
        key = l1_key(cfg, engine.identity, prompts, row, image_info[rid]['image_sha256'])
        representatives.setdefault(key, row)
        aliases[rid] = key
    values, timing = _prepare_layer1_unique(engine, cfg, list(representatives.values()), root, image_info, scope)
    by_key = {key: values[row['record_id']] for key, row in representatives.items()}
    result = {rid: by_key[key] for rid, key in aliases.items()}
    timing.update(requested_records=len(rows), unique_content_keys=len(representatives),
                  duplicate_content_keys_reused=len(rows)-len(representatives))
    save_json(Path(cfg['_work'])/'layer1_exports'/f'{scope}.json', {'records':result, 'timing':timing})
    return result, timing
