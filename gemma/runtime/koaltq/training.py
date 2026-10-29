"""독립 QLoRA 학습·예제 가중 loss·체크포인트 저장·새 실행 시작."""
from __future__ import annotations
from pathlib import Path
from itertools import islice
import gc,math,random,time,uuid
import numpy as np
import torch
from tqdm.auto import tqdm
from .utils import LABELS,digest,save_json,read_json,append_event,seed_all,locked_contract
from .prompts import load_prompts,layer2_text
from .scoring import example_loss,class_weights


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


def backward_group(engine,group: list[tuple[dict,dict]],weights: dict,micro: int,optimizer,cfg: dict) -> float:
    engine.model.train();optimizer.zero_grad(set_to_none=True);total=0.
    for start in range(0,len(group),micro):
        part=group[start:start+micro]
        sums,means,counts=engine.forward_targets([(base,row['label']) for row,base in part])
        w=torch.tensor([weights[row['label']] for row,_ in part],device=sums.device,dtype=torch.float32)
        loss=example_loss(sums,counts,w)*(len(part)/len(group))
        if not bool(torch.isfinite(loss)):raise FloatingPointError('학습 loss NaN/Inf')
        loss.backward();total+=float(loss.detach())
    params=trainable_parameters(engine)
    if not any(p.grad is not None for p in params):raise RuntimeError('LoRA gradient 없음')
    norm=torch.nn.utils.clip_grad_norm_(params,cfg['training']['max_grad_norm'])
    if not bool(torch.isfinite(norm)):raise FloatingPointError('LoRA gradient NaN/Inf')
    if any(p.grad is not None for p in engine.model.parameters() if not p.requires_grad):raise AssertionError('기본 모델 gradient 발견')
    return total
