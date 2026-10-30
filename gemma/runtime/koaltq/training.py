"""QLoRA Joint CE: 후보별 재계산으로 한 번에 하나의 모델 그래프만 유지."""
from __future__ import annotations
from pathlib import Path
from itertools import islice
import gc,math,random,time,uuid
import numpy as np
import torch
from tqdm.auto import tqdm
from .utils import LABELS,digest,save_json,read_json,append_event,seed_all,locked_contract
from .prompts import load_prompts,layer2_text
from .scoring import joint_ce_loss,candidate_tasks


def rng_state() -> dict:
    return {'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),
            'cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state: dict) -> None:
    random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['torch'])
    if state['cuda']:torch.cuda.set_rng_state_all(state['cuda'])


def trainable_parameters(engine) -> list:
    pairs=[(n,p) for n,p in engine.model.named_parameters() if p.requires_grad]
    if not pairs or any('lora_' not in n or '.language_model.' not in n for n,p in pairs):
        raise AssertionError('지정 LoRA 이외 학습 파라미터 발견')
    return [p for _,p in pairs]


def optimizer_for(engine,cfg: dict):
    return torch.optim.AdamW(trainable_parameters(engine),lr=cfg['training']['learning_rate'],
        weight_decay=cfg['training']['weight_decay'],foreach=False)


def backward_group(engine,group: list[tuple[dict,dict]],weights: dict,micro: int,optimizer,cfg: dict) -> dict:
    if not group or micro<1:raise ValueError('빈 학습 그룹 또는 잘못된 microbatch')
    if cfg['training']['binary_loss_weight']!=0.2:raise ValueError('Binary CE lambda는 0.2로 고정')
    engine.model.train();optimizer.zero_grad(set_to_none=True)
    totals={key:0. for key in ('loss','type_loss','binary_loss')}
    for start in range(0,len(group),micro):
        part=group[start:start+micro]
        tasks=candidate_tasks([base for _,base in part])
        values=[];states=[]
        # score()는 eval 모드 + Python 숫자 변환을 하므로 학습에는 사용하지 않음.
        with torch.no_grad():
            for _,lab,base in tasks:
                states.append(rng_state())
                _,means,_=engine.forward_targets([(base,lab)])
                values.append(means)
        after_scores=rng_state()
        scores=torch.cat(values).reshape(len(part),len(LABELS)).detach().requires_grad_(True)
        targets=torch.tensor([LABELS.index(row['label']) for row,_ in part],device=scores.device)
        w=torch.tensor([weights[lab] for lab in LABELS],device=scores.device,dtype=scores.dtype)
        losses=joint_ce_loss(scores,targets,w)
        if not all(bool(torch.isfinite(loss)) for loss in losses):raise FloatingPointError('학습 loss NaN/Inf')
        fraction=len(part)/len(group)
        coefficients=torch.autograd.grad(losses[0]*fraction,scores)[0].reshape(-1)
        for key,loss in zip(totals,losses):totals[key]+=float(loss.detach())*fraction
        try:
            for j,(_,lab,base) in enumerate(tasks):
                # stochastic 연산이 있어도 1차 채점과 같은 RNG를 재현.
                restore_rng(states[j])
                sums,means,counts=engine.forward_targets([(base,lab)])
                if not torch.allclose(means.detach(),scores.detach().reshape(-1)[j:j+1],
                                      atol=cfg['runtime']['score_atol'],rtol=cfg['runtime']['score_rtol']):
                    raise RuntimeError('후보 점수 재계산 불일치')
                means.backward(coefficients[j:j+1])
                del sums,means,counts
        finally:
            # 재계산으로 RNG 스트림을 두 번 소비하지 않음.
            restore_rng(after_scores)
    params=trainable_parameters(engine)
    if not any(p.grad is not None for p in params):raise RuntimeError('LoRA gradient 없음')
    norm=torch.nn.utils.clip_grad_norm_(params,cfg['training']['max_grad_norm'])
    if not bool(torch.isfinite(norm)):raise FloatingPointError('LoRA gradient NaN/Inf')
    if any(p.grad is not None for p in engine.model.parameters() if not p.requires_grad):raise AssertionError('기본 모델 gradient 발견')
    return {**totals,'grad_norm':float(norm)}
