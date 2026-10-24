import copy
from types import SimpleNamespace

import pytest
import torch

from koaltq import LABELS
from koaltq.model import attach_language_lora, check_trainable
from koaltq.scoring import (candidate_beta, class_weights, classification_loss,
                           joint_classification_loss, prediction_row, score_candidates)

CANDIDATES = [[10], [11, 12], [13, 14, 15], [16, 17], [18], [19, 20, 21], [22, 23]]


def test_weight_formula_and_single_sample_gradient():
    labels = [label for i, label in enumerate(LABELS, 1) for _ in range(i * 3)]
    weights, counts = class_weights(labels)
    raw = torch.tensor([(1 - .99) / (1 - .99 ** counts[label]) for label in LABELS])
    torch.testing.assert_close(weights, raw * 7 / raw.sum())
    assert weights.mean() == pytest.approx(1.0)
    betas = torch.arange(7, dtype=torch.float32, requires_grad=True)
    loss = classification_loss(betas, 0, weights)
    loss.backward()
    expected = weights[0] * (betas.detach().softmax(0) - torch.eye(7)[0])
    torch.testing.assert_close(betas.grad, expected)
    assert (betas.grad != 0).all()  # every candidate participates, including negatives


def test_full_logits_oracle_matches_shift_and_mean(tiny_qwen, multimodal_input):
    model = tiny_qwen.eval()
    with torch.no_grad():
        for candidate in CANDIDATES:
            ids = torch.cat((multimodal_input["input_ids"], torch.tensor([candidate])), dim=1)
            kwargs = {**multimodal_input, "input_ids": ids, "attention_mask": torch.ones_like(ids)}
            logits = model(**kwargs, use_cache=False).logits.float()
            start = multimodal_input["input_ids"].shape[1] - 1
            independent = torch.stack([logits[0, start + i].log_softmax(0)[token]
                                       for i, token in enumerate(candidate)]).mean()
            torch.testing.assert_close(candidate_beta(model, multimodal_input, candidate), independent)


def test_checkpoint_gradient_equivalence_and_freeze(tiny_qwen, multimodal_input):
    model = attach_language_lora(tiny_qwen, rank=2, alpha=4)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.get_base_model().model.visual.gradient_checkpointing_disable()
    model.train()
    visual_before = {name: p.detach().clone() for name, p in model.get_base_model().model.visual.named_parameters()}
    weights = torch.tensor([1., 2., 3., 4., 5., 6., 7.])
    plain = score_candidates(model, multimodal_input, CANDIDATES)
    joint_classification_loss(plain, 2, weights)[0].backward()
    expected = {name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None}
    model.zero_grad(set_to_none=True)
    recomputed = score_candidates(model, multimodal_input, CANDIDATES, recompute=True)
    assert recomputed.requires_grad and recomputed.grad_fn is not None
    joint_classification_loss(recomputed, 2, weights)[0].backward()
    torch.testing.assert_close(recomputed, plain)
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            assert ".language_model." in name and ".lora_" in name
            torch.testing.assert_close(parameter.grad, expected[name], atol=2e-6, rtol=1e-4)
        else:
            assert parameter.grad is None
    assert any(g.abs().sum() > 0 for g in expected.values())
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-3)
    optimizer.step()
    for name, parameter in model.get_base_model().model.visual.named_parameters():
        assert torch.equal(parameter, visual_before[name])
    assert check_trainable(model)["visual_projection_frozen"]


def test_adapter_save_reload(tiny_qwen, multimodal_input, tmp_path):
    from peft import PeftModel
    base_weights = copy.deepcopy(tiny_qwen.state_dict())
    cfg = tiny_qwen.config
    model = attach_language_lora(tiny_qwen, rank=2, alpha=4).eval()
    optimizer = torch.optim.SGD((p for p in model.parameters() if p.requires_grad), lr=.1)
    classification_loss(score_candidates(model, multimodal_input, CANDIDATES), 1, torch.ones(7)).backward()
    optimizer.step()
    expected = score_candidates(model, multimodal_input, CANDIDATES).detach()
    model.save_pretrained(tmp_path / "adapter")
    from transformers import Qwen3VLForConditionalGeneration
    fresh = Qwen3VLForConditionalGeneration(cfg)
    fresh.load_state_dict(base_weights)
    reloaded = PeftModel.from_pretrained(fresh, tmp_path / "adapter").eval()
    actual = score_candidates(reloaded, multimodal_input, CANDIDATES).detach()
    torch.testing.assert_close(actual, expected)


def test_prediction_all_scores():
    beta = torch.tensor([1., -1., 2., 0., 3., -4., -2.])
    result = prediction_row("sample", beta, CANDIDATES)
    assert result["prediction"] == LABELS[4]
    assert set(result["classes"]) == set(LABELS)
    assert sum(r["probability"] for r in result["classes"].values()) == pytest.approx(1)
    assert [r["token_count"] for r in result["classes"].values()] == list(map(len, CANDIDATES))


@pytest.mark.parametrize("target", range(7))
def test_binary_auxiliary_probability_and_gradient_oracle(target):
    betas = torch.tensor([.1, -.4, .7, -.8, .2, 1.1, -.3], requires_grad=True)
    weights = torch.tensor([.3, 1.4, 1.3, .4, 1.3, .25, 2.0])
    total, type_loss, binary_loss = joint_classification_loss(betas, target, weights)
    probability = betas.softmax(0)
    oracle_type = -weights[target] * probability[target].log()
    oracle_binary = -(probability[0] if target == 0 else probability[1:].sum()).log()
    oracle = .5 * (oracle_type + oracle_binary)
    torch.testing.assert_close(type_loss, oracle_type)
    torch.testing.assert_close(binary_loss, oracle_binary)
    torch.testing.assert_close(total, oracle)
    gradient = torch.autograd.grad(total, betas, retain_graph=True)[0]
    oracle_gradient = torch.autograd.grad(oracle, betas)[0]
    torch.testing.assert_close(gradient, oracle_gradient)
    assert torch.isfinite(gradient).all() and (gradient != 0).all()


def test_binary_auxiliary_zero_preserves_old_loss_and_gradients_exactly():
    scores = torch.randn(7, requires_grad=True)
    weights = torch.arange(1., 8.)
    old = classification_loss(scores, 3, weights)
    new = joint_classification_loss(scores, 3, weights, binary_weight=0)[0]
    assert torch.equal(old, new)
    assert torch.equal(torch.autograd.grad(old, scores, retain_graph=True)[0],
                       torch.autograd.grad(new, scores)[0])


@pytest.mark.parametrize("scores,target", [([10000.] + [-10000.] * 6, 1),
                                           ([-10000.] + [10000.] * 6, 0)])
def test_binary_auxiliary_extreme_scores_do_not_underflow(scores, target):
    betas = torch.tensor(scores, requires_grad=True)
    losses = joint_classification_loss(betas, target, torch.ones(7))
    assert all(torch.isfinite(loss) for loss in losses)
    losses[0].backward()
    assert torch.isfinite(betas.grad).all()


@pytest.mark.parametrize("weight", [-.1, 1.1, float("nan"), float("inf")])
def test_binary_auxiliary_invalid_weight(weight):
    with pytest.raises(ValueError, match="binary_weight"):
        joint_classification_loss(torch.zeros(7), 0, torch.ones(7), binary_weight=weight)
