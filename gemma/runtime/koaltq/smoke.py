"""동일 학습 표본의 순차/배치 점수 비교·실제 LoRA optimizer 갱신·메모리 실측."""
from __future__ import annotations
from pathlib import Path
import gc,time
import numpy as np
import pandas as pd
import torch
from .utils import LABELS,seed_all,save_json,save_csv,read_json,digest
from .prompts import load_prompts,layer2_text
from .pipeline import prepare_layer1,encoded_stream,begin_measure,finish_measure
from .training import backward_group,optimizer_for
from .model import tensor_state_hash
from .scoring import select_label,class_weights,candidate_tasks
from .gpu_memory import snapshot, memory_decision, collect_diagnostics, print_diagnostics

MEMORY_PREFLIGHT_PATCH = "koaltq_smoke_memory_probe_v2_strict"


def memory_ok(cfg: dict, report: dict) -> bool:
    """피크 예약량 유지 + 미사용 캐시 반환 전후 실제 여유 기록."""
    rt = cfg['runtime']
    fraction = float(rt['max_gpu_fraction'])
    min_free = float(rt['min_free_gib'])
    if not 0 < fraction <= 1 or min_free <= 0:
        raise ValueError('GPU 메모리 안전 기준 오류')
    started = time.perf_counter()
    device = 0
    gib = 2 ** 30

    before = snapshot()
    # 살아 있는 모델·gradient·optimizer는 유지, 미사용 캐시만 반환
    gc.collect()
    torch.cuda.empty_cache()
    after = snapshot()
    total_gib = torch.cuda.get_device_properties(device).total_memory / gib
    limit_gib = total_gib * fraction
    peak_ok, free_ok, reasons = memory_decision(report['peak_reserved_gib'], total_gib,
                                                after['free_gib'], fraction, min_free)
    report['memory_check'] = {
        'policy': 'peak_reserved_and_measured_free_after_unused_cache_release_v1',
        'device': device,
        'device_total_gib': total_gib,
        'max_gpu_fraction': fraction,
        'peak_reserved_limit_gib': limit_gib,
        'min_free_gib': min_free,
        'before_cache_release': before,
        'after_cache_release': after,
        'reserved_released_gib': before['reserved_gib'] - after['reserved_gib'],
        'peak_reserved_fraction_ok': peak_ok,
        'free_after_cache_release_ok': free_ok,
        'legacy_check_would_pass': bool(peak_ok and before['free_gib'] >= min_free),
        'failure_reasons': reasons,
        'measurement_seconds': time.perf_counter() - started,
        'scope': 'after_trial_boundary_not_minimum_free_during_forward_backward',
        'note': '전체 피크 통계 보존; 반환 후 실제 free 측정; 캐시량의 임의 합산 없음',
    }
    print(
        '메모리 검사:',
        f"피크 예약 {report['peak_reserved_gib']:.3f}/{limit_gib:.3f} GiB,",
        f"캐시 반환 전/후 여유 {before['free_gib']:.3f}/{after['free_gib']:.3f} GiB,",
        f"기준 {min_free:.3f} GiB -> {'통과' if peak_ok and free_ok else '미통과'}",
    )
    if not (peak_ok and free_ok):
        diagnostics = collect_diagnostics(device)
        report['memory_check']['diagnostics'] = diagnostics
        print_diagnostics(diagnostics)
        target = Path(cfg['_work'])/'gpu_memory_latest.json'
        save_json(target, report)
        print('메모리 진단 파일:', target)
    return peak_ok and free_ok

def compare_scores(reference: list[dict],candidate: list[dict],ids: list[str],cfg: dict) -> tuple[bool,list[dict]]:
    if len(reference)!=len(candidate) or len(reference)!=len(ids):raise ValueError('비교 레코드 수 불일치')
    details=[];passed=True
    for rid,ref,bat in zip(ids,reference,candidate):
        if set(ref)!=set(LABELS) or set(bat)!=set(LABELS):raise AssertionError('라벨 후보 연결 오류')
        same=select_label(ref,cfg['score_method'])==select_label(bat,cfg['score_method'])
        for lab in LABELS:
            for method in ['sum_logprob','mean_logprob']:
                a,b=ref[lab][method],bat[lab][method]
                ok=bool(np.isclose(a,b,atol=cfg['runtime']['score_atol'],rtol=cfg['runtime']['score_rtol']))
                passed=passed and ok and same
                details.append({'record_id':rid,'label':lab,'score_method':method,'reference':a,'batched':b,
                    'absolute_difference':abs(a-b),'score_close':ok,'prediction_equal':same})
    return passed,details


def run_smoke(engine,cfg: dict,train: list[dict],root: Path,image_info: dict,contract: str,experiment: str) -> dict:
    out=Path(cfg['_work'])/'smoke'/experiment;out.mkdir(parents=True,exist_ok=True)
    save_json(out/'smoke_result.json', {'status':'RUNNING', 'common_contract_sha256':contract})
    print('[1/5] 모델 로딩 후 여유 메모리 확인')
    initial_memory = {'phase':'after_model_load', 'peak_reserved_gib':torch.cuda.memory_reserved()/2**30}
    if not memory_ok(cfg, initial_memory):
        save_json(out/'initial_memory.json', initial_memory)
        save_json(out/'smoke_result.json', {'status':'FAILED_MEMORY_BEFORE_INFERENCE', 'common_contract_sha256':contract})
        raise RuntimeError('모델 로딩 후 메모리 기준 미통과. 추론·학습은 시작하지 않음. 위 CUDA/nvidia-smi 여유와 현재 PID 확인.')
    print('[2/5] Layer 1 분석: 학습 표본 3건 또는 같은 조건의 캐시')
    # 검증 성능과 무관한 선택: 긴 문맥 우선 + 고정 ID 순서
    chosen=sorted(train,key=lambda r:(-len(r['context_text']),r['record_id']))[:cfg['runtime']['smoke_records']]
    if not chosen:raise RuntimeError('smoke 내부 학습 표본 없음')
    # 긴 문맥을 유지하면서 실제 Binary CE 양쪽 정답을 검사.
    if len(chosen)>1:
        for positive in (True,False):
            if not any((r['label']=='적절')==positive for r in chosen):
                eligible=[r for r in train if (r['label']=='적절')==positive]
                if eligible:chosen[-1]=min(eligible,key=lambda r:(-len(r['context_text']),r['record_id']))
    l1,l1time=prepare_layer1(engine,cfg,chosen,root,image_info,'smoke_train')
    prompts=load_prompts(Path(cfg['_package_root']));jobs=[(r,layer2_text(r,l1[r['record_id']]['analysis'],prompts),image_info[r['record_id']]['image_sha256']) for r in chosen]
    encoded=list(encoded_stream(engine,root,jobs,cfg));bases=[b for _,b in encoded];ids=[r['record_id'] for r,_ in encoded]
    if ids!=[r['record_id'] for r in chosen]:raise AssertionError('병렬 전처리 순서 변경')
    print('[3/5] Layer 2 점수 검사: 기본 모델, 7개 라벨 × 표본 수')
    start=begin_measure();reference=engine.score_reference(bases,progress=True);reference_time=finish_measure(start)
    trials=[];details=[]
    for bs in cfg['runtime']['candidate_batch_trials']:
        gc.collect();torch.cuda.empty_cache();start=begin_measure()
        try:
            scores=engine.score(bases,bs,progress=True);report=finish_measure(start)
            ok,diffs=compare_scores(reference,scores,ids,cfg);details.extend([{**x,'candidate_batch':bs} for x in diffs])
            mem_ok=memory_ok(cfg,report)
            trials.append({'candidate_batch':bs,**report,'numeric_and_prediction_match':ok,'memory_ok':mem_ok})
        except torch.OutOfMemoryError:
            gc.collect();torch.cuda.empty_cache();trials.append({'candidate_batch':bs,'oom':True,'numeric_and_prediction_match':False,'memory_ok':False})
    valid=[t for t in trials if t['numeric_and_prediction_match'] and t['memory_ok']]
    save_json(out/'inference_trials.json',trials)
    save_csv(out/'score_differences.csv',pd.DataFrame(details))
    for trial in trials:
        print(f"후보 배치 {trial['candidate_batch']}: 점수 일치={trial['numeric_and_prediction_match']}, 메모리={trial['memory_ok']}, OOM={trial.get('oom',False)}")
    if not valid:
        save_json(out/'smoke_result.json', {'status':'FAILED_INFERENCE', 'common_contract_sha256':contract})
        raise RuntimeError('추론 검사 미통과. 위의 점수 일치·메모리·OOM 항목과 inference_trials.json 확인.')
    print('[4/5] Joint CE LoRA 학습 검사: Type CE 단위 가중치 / 기존 클래스 가중치')
    best_infer=min(valid,key=lambda t:t['wall_seconds'])['candidate_batch']
    engine.attach_lora();initial=engine.adapter_state();initial_sha=engine.fingerprint();backward_trials=[]
    weights=class_weights(train,cfg['training']['class_weight_cap'])['weights']
    # 두 loss 방식 모두 실제 forward/backward/optimizer 확인; 각 시험 전 동일 초기화 복원
    for micro in cfg['runtime']['microbatch_trials']:
        for weighted in [False,True]:
            engine.load_adapter_state(initial);seed_all(cfg['seed']);opt=optimizer_for(engine,cfg)
            frozen_versions={n:p._version for n,p in engine.model.named_parameters() if not p.requires_grad}
            gc.collect();torch.cuda.empty_cache();start=begin_measure()
            try:
                losses=backward_group(engine,encoded,weights if weighted else {l:1. for l in LABELS},micro,opt,cfg)
                opt.step();torch.cuda.synchronize();report=finish_measure(start);updated=engine.fingerprint()!=initial_sha
                fixed=all(p._version==frozen_versions[n] for n,p in engine.model.named_parameters() if not p.requires_grad)
                if not updated or not fixed or losses['grad_norm']<=0:raise RuntimeError('LoRA 갱신/기본 모델 고정 검사 실패')
                mem_ok=memory_ok(cfg,report)
                backward_trials.append({'microbatch':micro,'weighted':weighted,**losses,'adapter_updated':updated,
                    'frozen_parameters_unchanged':fixed,**report,'memory_ok':mem_ok})
            except torch.OutOfMemoryError:
                backward_trials.append({'microbatch':micro,'weighted':weighted,'oom':True,'memory_ok':False})
            finally:
                opt.zero_grad(set_to_none=True);del opt;gc.collect();torch.cuda.empty_cache();engine.load_adapter_state(initial)
    choices=[]
    for micro in cfg['runtime']['microbatch_trials']:
        ts=[t for t in backward_trials if t['microbatch']==micro]
        if len(ts)==2 and all(t['memory_ok'] for t in ts):choices.append((sum(t['wall_seconds'] for t in ts),micro))
    if not choices:
        save_json(out/'backward_trials.json',backward_trials)
        for trial in backward_trials:
            print('학습 smoke 상세:', trial)
        raise RuntimeError(
            '학습 smoke 미통과: backward_trials.json의 adapter_updated, oom, '
            'memory_check.failure_reasons 확인. 메모리 기준 자동 완화 없음.'
        )
    selected_micro=min(choices)[1]
    # LoRA 래퍼 초기 상태에서도 같은 숫자/배치 정렬 확인
    print('[5/5] LoRA 적용 후 점수 일치 검사')
    adapter_reference=engine.score_reference(bases,progress=True);adapter_scores=engine.score(bases,best_infer,progress=True)
    ok,ad=compare_scores(adapter_reference,adapter_scores,ids,cfg)
    if not ok:raise RuntimeError('LoRA 래퍼의 배치 점수 일치 검사 실패')
    # 학습 경로는 eval 전환이나 score()의 Python 변환을 거치지 않음.
    training_scores=[{} for _ in bases];engine.model.train()
    with torch.no_grad():
        for i,lab,base in candidate_tasks(bases):
            sums,means,counts=engine.forward_targets([(base,lab)])
            training_scores[i][lab]={'sum_logprob':float(sums[0]),'mean_logprob':float(means[0]),'n_tokens':int(counts[0])}
    train_match,td=compare_scores(adapter_reference,training_scores,ids,cfg)
    if not train_match:raise RuntimeError('학습·추론 Mean Log Probability 불일치')
    # 이미 확정된 배치 계획은 다른 노트북에서 임의 변경 금지
    plan_path=Path(cfg['_work'])/'batch_plan.json'
    candidate={'common_contract_sha256':contract,'candidate_batch':best_infer,'microbatch':selected_micro,
               'effective_batch_size':cfg['training']['effective_batch_size']}
    if plan_path.exists():
        plan=read_json(plan_path)
        if plan['common_contract_sha256']!=contract:raise RuntimeError('배치 계획 공통 조건 불일치')
        if not any(t['candidate_batch']==plan['candidate_batch'] for t in valid):raise RuntimeError('기존 후보 배치 smoke 실패')
        if plan['microbatch'] not in [m for _,m in choices]:raise RuntimeError('기존 학습 배치 smoke 실패')
    else:plan=candidate;save_json(plan_path,plan)
    result={'status':'GPU_TEST_PASSED','experiment':experiment,'common_contract_sha256':contract,
        'record_ids':ids,'record_labels':[r['label'] for r in chosen],
        'input_token_counts':[int(b['input_ids'].shape[1]) for b in bases],
        'score_method':cfg['score_method'],'loss':'joint_ce','binary_loss_weight':0.2,
        'training_inference_match':train_match,'source':'internal_train_only','reference':reference_time,'layer1':l1time,
        'inference_trials':trials,'backward_trials':backward_trials,'plan':plan,
        'smoke_adapter_restored':engine.fingerprint()==initial_sha,'adapter_not_used_for_layer1':True,
        'unweighted_and_weighted_actual_optimizer_step':True}
    save_csv(out/'score_differences.csv',pd.DataFrame(details));save_csv(out/'adapter_score_differences.csv',pd.DataFrame(ad))
    save_csv(out/'training_score_differences.csv',pd.DataFrame(td))
    save_json(out/'smoke_result.json',result);print('GPU_TEST_PASSED: 소량 추론·LoRA 갱신·메모리 검사 통과');return result
