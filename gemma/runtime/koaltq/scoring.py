"""한국어 라벨 전체 토큰 likelihood·타깃 마스킹·예제별 가중 loss."""
from __future__ import annotations
from collections import Counter
import math
import torch
import torch.nn.functional as F
import numpy as np
from .utils import LABELS

SEQ_KEYS = {'input_ids','attention_mask','mm_token_type_ids','token_type_ids'}
VISION_KEYS = {'pixel_values','image_position_ids'}


def append_target(base: dict, token_ids: list[int]) -> tuple[dict,torch.Tensor]:
    if not token_ids: raise ValueError('빈 라벨 토큰')
    if base['input_ids'].shape[0]!=1: raise ValueError('append_target은 전처리된 단일 레코드 전용')
    target=torch.tensor([token_ids],dtype=torch.long)
    out={}
    for key,value in base.items():
        if key=='input_ids': out[key]=torch.cat([value,target],dim=1)
        elif key=='attention_mask':out[key]=torch.cat([value,torch.ones_like(target)],dim=1)
        elif key in ('mm_token_type_ids','token_type_ids'):out[key]=torch.cat([value,torch.zeros_like(target)],dim=1)
        elif key in VISION_KEYS:out[key]=value
        else:raise ValueError(f'검증하지 않은 processor 출력 키: {key}')
    labels=torch.full_like(out['input_ids'],-100);labels[:,-len(token_ids):]=target
    return out,labels


def collate_inputs(items: list[dict], pad_id: int, targets: list[torch.Tensor] | None = None):
    if not items:raise ValueError('빈 배치')
    if len({tuple(sorted(x)) for x in items})!=1:raise ValueError('processor 출력 키 불일치')
    width=max(x['input_ids'].shape[1] for x in items);out={}
    for key in items[0]:
        if key in SEQ_KEYS:
            fill=pad_id if key=='input_ids' else 0
            out[key]=torch.cat([F.pad(x[key],(width-x[key].shape[1],0),value=fill) for x in items],dim=0)
        elif key in VISION_KEYS:
            values=[x[key] for x in items]
            if any(v.shape[0]!=1 for v in values):raise ValueError('현재 구현은 레코드당 이미지 1장')
            # 패치 수가 다른 processor 구현의 명시적 padding
            max_patches=max(v.shape[1] for v in values)
            fill=-1 if key=='image_position_ids' else 0
            padded=[]
            for value in values:
                if value.shape[1]!=max_patches:
                    shape=list(value.shape);shape[1]=max_patches-value.shape[1]
                    value=torch.cat([value,torch.full(shape,fill,dtype=value.dtype)],dim=1)
                padded.append(value)
            out[key]=torch.cat(padded,dim=0)
        else:raise ValueError(f'검증하지 않은 배치 입력 키: {key}')
    out['position_ids']=(out['attention_mask'].long().cumsum(-1)-1).clamp_min(0)
    labels=None
    if targets is not None:
        if len(targets)!=len(items):raise ValueError('target 개수 불일치')
        labels=torch.cat([F.pad(t,(width-t.shape[1],0),value=-100) for t in targets],dim=0)
        if bool(((labels!=-100)&(out['attention_mask']==0)).any()):raise AssertionError('padding에 loss target 존재')
    return out,labels


def target_logprobs(logits: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor,torch.Tensor,torch.Tensor]:
    """logits_to_keep=max_target_tokens+1, 마지막 로짓 제외 후 다음 토큰과 정렬."""
    counts=(labels!=-100).sum(-1)
    if bool((counts==0).any()):raise ValueError('정답 토큰이 없는 예제')
    max_n=int(counts.max().item())
    if logits.shape[1]!=max_n+1:raise AssertionError(f'로짓 정렬 실패: {logits.shape[1]} != {max_n+1}')
    tail=labels[:,-max_n:];mask=tail!=-100
    if not torch.equal(mask.sum(-1),counts):raise AssertionError('전체 라벨 토큰이 꼬리 위치에 포함되지 않음')
    # prompt/image/padding 위치의 어휘 확률 계산 제외
    selected=logits[:,:-1,:][mask].float();truth=tail[mask]
    values=-F.cross_entropy(selected,truth,reduction='none')
    if not bool(torch.isfinite(values).all()):raise FloatingPointError('라벨 logprob NaN/Inf')
    index=torch.arange(labels.shape[0],device=labels.device)[:,None].expand_as(tail)[mask]
    sums=torch.zeros(labels.shape[0],device=logits.device,dtype=torch.float32).scatter_add(0,index,values)
    return sums,sums/counts,counts


def example_loss(sum_logprob: torch.Tensor, token_count: torch.Tensor,
                 example_weights: torch.Tensor) -> torch.Tensor:
    # 예제별 전체 라벨 토큰 평균 NLL → 예제별 가중 → 예제 수로 평균
    # 배치별 weight 합으로 나누지 않음: microbatch=1에서도 클래스 가중 효과 유지
    if sum_logprob.shape!=example_weights.shape:raise ValueError('가중치/예제 순서 오류')
    return ((-sum_logprob/token_count)*example_weights).mean()


def class_weights(rows: list[dict], cap: float) -> dict:
    rows = [r for r in rows if not r.get('synthetic')]  # 합성 레코드는 가중치 계산에서 제외: 실제 레코드 빈도 기준
    if cap<1:raise ValueError('평균 1 가중치의 상한은 1 이상 필요')
    c=Counter(r['label'] for r in rows);n=np.array([c[x] for x in LABELS],dtype=np.float64)
    N=n.sum()
    if not N:raise ValueError('가중치 계산용 학습 데이터 없음')
    present=n>0;raw=np.zeros(len(LABELS));raw[present]=np.sqrt(N/(present.sum()*n[present]))
    # 상한 적용 후에도 train 예제 가중 평균=1: 단조식 이분 탐색
    lo,hi=0.,1.
    while float(np.dot(n,np.minimum(cap,hi*raw))/N)<1-1e-12:hi*=2
    for _ in range(100):
        mid=(lo+hi)/2
        if float(np.dot(n,np.minimum(cap,mid*raw))/N)<1:lo=mid
        else:hi=mid
    weights=np.minimum(cap,hi*raw);weights[~present]=0.
    return {'method':'inverse_sqrt_frequency_capped_train_mean_one',
            'formula':'raw_c=sqrt(N/(K_present*n_c)); w_c=min(cap,s*raw_c); sum(n_c*w_c)/N=1',
            'counts':dict(zip(LABELS,n.astype(int).tolist())), 'raw':dict(zip(LABELS,raw.tolist())),
            'weights':dict(zip(LABELS,weights.tolist())),'cap':cap,'scale':hi,
            'train_example_weight_mean':float(np.dot(n,weights)/N),
            'source':'internal_train_only','missing_labels':[x for x in LABELS if not c[x]]}


def select_label(scores: dict, method: str) -> str:
    if set(scores)!=set(LABELS):raise ValueError('7개 실제 라벨 이름과 점수 연결 실패')
    if method not in ('sum_logprob','mean_logprob'):raise ValueError('점수 방식 오류')
    for lab in LABELS:
        value=scores[lab]
        if type(value.get('n_tokens')) is not int or value['n_tokens']<1:raise ValueError('라벨 전체 토큰 수 오류')
        if not all(isinstance(value.get(k),(int,float)) and math.isfinite(value[k]) for k in ('sum_logprob','mean_logprob')):
            raise ValueError('라벨 점수 누락/NaN/Inf')
        if not math.isclose(value['sum_logprob']/value['n_tokens'],value['mean_logprob'],rel_tol=1e-5,abs_tol=1e-5):
            raise ValueError('sum/mean/token count 연결 불일치')
    return max(LABELS,key=lambda lab:scores[lab][method])
