"""Actual PEFT/Qwen dtype preparation; the CUDA download boundary is mocked on CPU."""
import copy
from types import SimpleNamespace

import torch

from koaltq.model import load_model, model_numeric_profile
from koaltq.scoring import classification_loss, score_candidates


def test_training_resume_and_inference_share_base_dtypes(tiny_qwen, multimodal_input, tmp_path, monkeypatch):
    import transformers
    from peft import PeftModel
    template = tiny_qwen.to(dtype=torch.bfloat16)
    args = SimpleNamespace(model_id="tiny-qwen", revision="a" * 40, model_cache_dir=None,
                           lora_rank=2, lora_alpha=4)
    loads = []

    def from_pretrained(*a, **kw):
        loads.append(kw)
        return copy.deepcopy(template)

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    # Only the CUDA loading boundary is simulated; adapter tensors remain on CPU.
    monkeypatch.setattr("peft.peft_model.infer_device", lambda: "cpu")
    monkeypatch.setattr("peft.utils.save_and_load.infer_device", lambda: "cpu")
    monkeypatch.setattr(transformers, "BitsAndBytesConfig", lambda **kw: SimpleNamespace(**kw))
    monkeypatch.setattr(transformers.Qwen3VLForConditionalGeneration, "from_pretrained", from_pretrained)
    trained = load_model(args, training=True)
    assert trained.get_base_model().model.language_model.norm.weight.dtype == torch.float32
    assert trained.get_base_model().model.visual.merger.linear_fc1.weight.dtype == torch.bfloat16
    assert trained.get_base_model().model.language_model.embed_tokens.weight.dtype == torch.bfloat16
    assert trained.get_base_model().lm_head.weight.dtype == torch.bfloat16
    assert all(p.dtype == torch.float32 for n, p in trained.named_parameters() if ".lora_" in n)
    candidates = [[10 + i, 20 + i] for i in range(7)]
    # Dense CPU layers stand in for 4-bit CUDA layers; explicit autocast enables BF16 compute here.
    optimizer = torch.optim.SGD((p for p in trained.parameters() if p.requires_grad), lr=.1)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        loss = classification_loss(score_candidates(trained, multimodal_input, candidates), 2, torch.ones(7))
    loss.backward()
    optimizer.step()
    # Open a new autocast scope after updating weights so its cast cache cannot be stale.
    with torch.autocast("cpu", dtype=torch.bfloat16), torch.inference_mode():
        expected = score_candidates(trained, multimodal_input, candidates).detach()
    trained.save_pretrained(tmp_path / "adapter")
    # Reproduce the original missing preparation: norms remain BF16, unlike training.
    old_reload = PeftModel.from_pretrained(copy.deepcopy(template), tmp_path / "adapter").eval()
    assert old_reload.get_base_model().model.language_model.norm.weight.dtype == torch.bfloat16
    assert model_numeric_profile(old_reload) != model_numeric_profile(trained)
    actual = load_model(args, adapter=tmp_path / "adapter")
    resumed = load_model(args, training=True, adapter=tmp_path / "adapter")
    assert model_numeric_profile(actual) == model_numeric_profile(trained) == model_numeric_profile(resumed)
    saved_parameters = dict(trained.named_parameters())
    for name, parameter in actual.named_parameters():
        torch.testing.assert_close(parameter, saved_parameters[name], atol=0, rtol=0)
    assert all(not p.requires_grad for p in actual.parameters())
    for model in (trained, resumed):
        assert all(".language_model." in n and ".lora_" in n for n, p in model.named_parameters() if p.requires_grad)
        assert all(not p.requires_grad for p in model.get_base_model().model.visual.parameters())
    with torch.autocast("cpu", dtype=torch.bfloat16), torch.inference_mode():
        reloaded = score_candidates(actual, multimodal_input, candidates)
    torch.testing.assert_close(reloaded, expected, atol=0.0, rtol=0.0)
    configs = [vars(item["quantization_config"]) for item in loads]
    assert all(cfg == configs[0] for cfg in configs)
    zero_shot = load_model(args)
    assert zero_shot.model.language_model.norm.weight.dtype == torch.bfloat16
    assert all(not p.requires_grad for p in zero_shot.parameters())
