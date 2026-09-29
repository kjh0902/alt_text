import copy
import json
from types import SimpleNamespace

import pytest
import torch

from koaltq.io import read_json, write_jsonl
from koaltq.model import attach_language_lora
from koaltq.prompts import l2_payload, prepare_input
from koaltq.runtime import cache_path, l1_key, require_cache
from koaltq.scoring import classification_loss, score_candidates
from koaltq.training import load_training_state, save_checkpoint


class CharacterTokenizer:
    def encode(self, text, add_special_tokens=False):
        return list(text.encode("utf-8"))

    def decode(self, ids, **kwargs):
        return bytes(ids).decode("utf-8", errors="replace")


class MinimalProcessor:
    tokenizer = CharacterTokenizer()

    def apply_chat_template(self, messages, **kwargs):
        self.payload = json.loads(messages[1]["content"][1]["text"])
        return messages[1]["content"][1]["text"]

    def __call__(self, text, images, **kwargs):
        ids = torch.tensor([[999] * 4 + self.tokenizer.encode(text[0])])
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids), "image_grid_thw": torch.tensor([[1, 4, 4]])}


def test_truncation_preserves_image_structure_rules_and_continues(record):
    processor = MinimalProcessor()
    record["alt_text"] = "long description " * 500
    payload = l2_payload(record, {"visual_role": "informational", "thumbnail_type": None, "visible_text": "text"})
    original = copy.deepcopy(payload)
    inputs, info = prepare_input(processor, object(), payload, stage=2, max_seq_length=900, reserve_tokens=5)
    assert info["original_token_length"] > 900 >= info["final_token_length"]
    assert "alt_text" in info["truncated_fields"]
    assert payload == original  # caller/cache is not mutated
    assert inputs["input_ids"][0, :4].tolist() == [999] * 4
    assert processor.payload["rule_flags"] == original["rule_flags"]
    assert processor.payload["visual_analysis"]["visual_role"] == "informational"
    record["alt_text"] = "short"
    _, next_info = prepare_input(processor, object(), l2_payload(record, payload["visual_analysis"]),
                                 stage=2, max_seq_length=900, reserve_tokens=5)
    assert not next_info["truncated_fields"]


def test_cache_missing_and_stale_are_rejected(tmp_path, record):
    from PIL import Image
    (tmp_path / "images").mkdir()
    Image.new("RGB", (2, 2)).save(tmp_path / "images/a.png")
    args = SimpleNamespace(run_dir=tmp_path / "run", model_id="qwen", revision="a" * 40,
                           max_seq_length=8192, l1_max_image_tokens=1024, l1_max_new_tokens=2048)
    with pytest.raises(ValueError, match="missing/stale"):
        require_cache(args, "train", [record], tmp_path)
    analysis = {"visual_role": "informational", "thumbnail_type": None, "visible_text": ""}
    write_jsonl(cache_path(args, "train"), [{"record_id": record["record_id"],
        "input_sha256": l1_key(args, tmp_path, record), "analysis": analysis}])
    assert require_cache(args, "train", [record], tmp_path) == {record["record_id"]: analysis}
    args.l1_max_image_tokens = 512
    with pytest.raises(ValueError, match="missing/stale"):
        require_cache(args, "train", [record], tmp_path)


def test_checkpoint_optimizer_scheduler_rng_restore(tiny_qwen, multimodal_input, tmp_path):
    import random
    from peft import PeftModel
    from transformers import Qwen3VLForConditionalGeneration, get_linear_schedule_with_warmup
    base_state = copy.deepcopy(tiny_qwen.state_dict())
    cfg = tiny_qwen.config
    model = attach_language_lora(tiny_qwen, rank=2, alpha=4)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=.01)
    scheduler = get_linear_schedule_with_warmup(optimizer, 0, 10)
    candidates = [[10 + i, 20 + i] for i in range(7)]

    def update(m, opt, sch):
        opt.zero_grad(set_to_none=True)
        loss = classification_loss(score_candidates(m, multimodal_input, candidates, recompute=True), 2, torch.ones(7))
        loss.backward()
        opt.step()
        sch.step()

    update(model, optimizer, scheduler)
    path = save_checkpoint(model, optimizer, scheduler, tmp_path,
                           {"epoch": 0, "cursor": 8, "step": 1, "best_score": -1}, {"test": "trusted"})
    expected_random = (random.random(), torch.rand(3))
    update(model, optimizer, scheduler)
    expected = {name: p.clone() for name, p in model.named_parameters() if p.requires_grad}
    base = Qwen3VLForConditionalGeneration(cfg)
    base.load_state_dict(base_state)
    restored = PeftModel.from_pretrained(base, path / "adapter", is_trainable=True)
    restored_optimizer = torch.optim.AdamW((p for p in restored.parameters() if p.requires_grad), lr=.01)
    restored_scheduler = get_linear_schedule_with_warmup(restored_optimizer, 0, 10)
    load_training_state(path, restored_optimizer, restored_scheduler)
    assert random.random() == expected_random[0]
    torch.testing.assert_close(torch.rand(3), expected_random[1])
    update(restored, restored_optimizer, restored_scheduler)
    for name, parameter in restored.named_parameters():
        if parameter.requires_grad:
            torch.testing.assert_close(parameter, expected[name])
    assert read_json(tmp_path / "last_checkpoint.json")["path"]
