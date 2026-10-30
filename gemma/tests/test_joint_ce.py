from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from gemma.runtime.koaltq.scoring import (append_target, candidate_tasks, collate_inputs,
                                         joint_ce_loss, target_logprobs)
from gemma.runtime.koaltq.training import backward_group, optimizer_for
from gemma.runtime.koaltq.utils import LABELS, read_json
from gemma.train import arguments, configuration


def test_joint_loss_matches_probability_formula_and_weighting():
    scores=torch.tensor([[-.2,-3.,-1.,-2.,-4.,-1.5,-.5],[-4.,-.2,-.5,-2.,-1.,-.8,-3.]],requires_grad=True)
    targets=torch.tensor([0,2])
    weights=torch.tensor([.7,1.1,3.,1.2,2.,2.1,2.5])
    total,typ,binary=joint_ce_loss(scores,targets,weights)
    p=scores.softmax(-1)
    expected_type=(-weights[targets]*p[torch.arange(2),targets].log()).mean()
    expected_binary=-(p[0,0].log()+(1-p[1,0]).log())/2
    assert torch.allclose(typ,expected_type)
    assert torch.allclose(binary,expected_binary)
    assert torch.allclose(total,.8*expected_type+.2*expected_binary)
    unweighted=joint_ce_loss(scores,targets,torch.ones(7))
    assert torch.equal(binary,unweighted[2])
    assert not torch.allclose(typ,unweighted[1])
    # A one-example microbatch must retain w_y rather than divide it out.
    assert joint_ce_loss(scores[:1],targets[:1],weights)[1].item()==pytest.approx(-.7*p[0,0].log().item())
    binary_grad=torch.autograd.grad(binary,scores)[0]
    assert bool((binary_grad.abs()>0).all())  # Both classes depend on every candidate.


@pytest.mark.parametrize('positive_score',[10000.,-10000.])
def test_extreme_scores_have_finite_loss_and_gradients(positive_score):
    scores=torch.tensor([[positive_score]+[-positive_score]*6]*2,requires_grad=True)
    losses=joint_ce_loss(scores,torch.tensor([0,6]),torch.ones(7))
    grad=torch.autograd.grad(losses[0],scores)[0]
    assert all(bool(torch.isfinite(loss)) for loss in losses)
    assert bool(torch.isfinite(grad).all())


def test_joint_ce_gradcheck():
    scores=torch.linspace(-3.,-.1,14,dtype=torch.float64).reshape(2,7).requires_grad_()
    assert torch.autograd.gradcheck(lambda s:joint_ce_loss(s,torch.tensor([0,5]),torch.arange(1.,8.,dtype=s.dtype))[0],(scores,))


def test_target_mask_alignment_mean_normalization_and_gradient():
    # Different prompt lengths, different label lengths, left padding and last-logit exclusion.
    a={'input_ids':torch.tensor([[8,9]]),'attention_mask':torch.ones(1,2,dtype=torch.long)}
    b={'input_ids':torch.tensor([[7,8,9]]),'attention_mask':torch.ones(1,3,dtype=torch.long)}
    items,targets=zip(*(append_target(base,ids) for base,ids in [(a,[1]),(b,[2,3,4])]))
    batch,labels=collate_inputs(list(items),0,list(targets))
    logits=torch.linspace(-1.,1.,2*4*10).reshape(2,4,10).requires_grad_()
    sums,means,counts=target_logprobs(logits,labels)
    expected=torch.stack([F.log_softmax(logits[0,2],-1)[1],
                          sum(F.log_softmax(logits[1,i],-1)[t] for i,t in enumerate([2,3,4]))])
    assert counts.tolist()==[1,3]
    assert torch.allclose(sums,expected)
    assert torch.allclose(means,expected/counts)
    means.sum().backward()
    assert torch.equal(logits.grad[0,:2],torch.zeros_like(logits.grad[0,:2]))
    assert torch.equal(logits.grad[:,-1],torch.zeros_like(logits.grad[:,-1]))
    assert not bool(((labels!=-100)&(batch['attention_mask']==0)).any())


class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lora=torch.nn.Parameter(torch.linspace(-.4,.6,21,dtype=torch.float64).reshape(3,7))
        self.frozen=torch.nn.Parameter(torch.ones(3,dtype=torch.float64),requires_grad=False)

    def named_parameters(self,*args,**kwargs):
        return iter([('base_model.model.language_model.lora_A',self.lora),('frozen',self.frozen)])


class ToyEngine:
    def __init__(self,stochastic=False):
        self.model=ToyModel()
        self.stochastic=stochastic
        self.calls=[]

    def forward_targets(self,pairs):
        assert self.model.training
        self.calls.extend((base['id'],lab,torch.is_grad_enabled()) for base,lab in pairs)
        scores=[];counts=[]
        for base,lab in pairs:
            j=LABELS.index(lab)
            value=base['x']@self.model.lora[:,j]
            if self.stochastic:value=value*torch.rand((),dtype=value.dtype)
            scores.append(value)
            counts.append(j+1)
        means=torch.stack(scores)
        counts=torch.tensor(counts)
        return means*counts,means,counts


@pytest.mark.parametrize('size,micro',[(1,1),(5,1),(5,2),(8,3)])
@pytest.mark.parametrize('stochastic',[False,True])
def test_recomputation_matches_full_graph_gradients_and_optimizer(size,micro,stochastic,tmp_path):
    cfg=configuration(arguments(['--data-root',str(tmp_path)]))
    cfg['training']['max_grad_norm']=1e6
    engine=ToyEngine(stochastic)
    direct=copy.deepcopy(engine)
    group=[({'label':LABELS[i%7]}, {'id':i,'x':torch.tensor([1.,i/5.,-.7],dtype=torch.float64)}) for i in range(size)]
    weights={lab:.7+.3*j for j,lab in enumerate(LABELS)}
    opt=optimizer_for(engine,cfg);reference_opt=optimizer_for(direct,cfg)
    torch.manual_seed(17)
    full_scores=torch.cat([direct.forward_targets([(base,lab)])[1] for _,lab,base in candidate_tasks([b for _,b in group])]).reshape(size,7)
    losses=joint_ce_loss(full_scores,torch.tensor([LABELS.index(r['label']) for r,_ in group]),
                         torch.tensor(list(weights.values()),dtype=torch.float64))
    losses[0].backward()
    expected_rng=torch.get_rng_state()
    expected_gradient=direct.model.lora.grad.clone()
    torch.manual_seed(17)
    actual=backward_group(engine,group,weights,micro,opt,cfg)
    assert torch.equal(expected_rng,torch.get_rng_state())
    for key,value in zip(('loss','type_loss','binary_loss'),losses):
        assert actual[key]==pytest.approx(value.item(),rel=1e-10,abs=1e-10)
    assert torch.allclose(engine.model.lora.grad,expected_gradient,rtol=1e-10,atol=1e-10)
    assert engine.model.frozen.grad is None
    assert torch.equal(engine.model.frozen,direct.model.frozen)
    opt.step();reference_opt.step()
    assert torch.allclose(engine.model.lora,direct.model.lora,rtol=1e-10,atol=1e-10)
    # Every example has exactly one score pass and one differentiable pass per label.
    assert len(engine.calls)==size*14
    assert all(sum(call==(i,lab,enabled) for call in engine.calls)==1
               for i in range(size) for lab in LABELS for enabled in (False,True))


def test_epoch_prediction_uses_mean_argmax(tmp_path):
    from gemma.runtime.koaltq.layer3 import build_policy
    from gemma.workflow import predict_epoch
    rows=[{'record_id':f't{i}','alt_text':'설명'} for i in range(600)]
    # sum winner=적절 (-1), mean winner=무의미형 (-.5).
    scores={lab:{'sum_logprob':-10.,'mean_logprob':-10.,'n_tokens':1} for lab in LABELS}
    scores['적절']={'sum_logprob':-1.,'mean_logprob':-1.,'n_tokens':1}
    scores['무의미형']={'sum_logprob':-1.5,'mean_logprob':-.5,'n_tokens':3}
    engine=SimpleNamespace(model=torch.nn.Linear(1,1),score=lambda bases,bs:[scores])
    dataset=SimpleNamespace(test=rows,policy=build_policy([],[],[]),known={r['record_id']:'적절' for r in rows[:109]},
                            sample_order=[r['record_id'] for r in rows])
    predict_epoch(engine,dataset,SimpleNamespace(get=lambda r:{}),tmp_path,1)
    details=read_json(tmp_path/'prediction_details.json')
    assert details['score_method']=='mean_logprob'
    assert all(row['prediction']=='무의미형' for row in details['layer2'])


def test_recomputation_rejects_score_mismatch(tmp_path):
    engine=ToyEngine()
    original=engine.forward_targets
    def mismatch(pairs):
        sums,means,counts=original(pairs)
        return sums,means+int(torch.is_grad_enabled()),counts
    engine.forward_targets=mismatch
    cfg=configuration(arguments(['--data-root',str(tmp_path)]))
    group=[({'label':'적절'},{'id':0,'x':torch.ones(3,dtype=torch.float64)})]
    with pytest.raises(RuntimeError,match='재계산 불일치'):
        backward_group(engine,group,dict.fromkeys(LABELS,1.),1,optimizer_for(engine,cfg),cfg)
