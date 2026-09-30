"""Regression cases for the reported Layer 1 failures and successful-cache reuse."""
import copy
import json
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from koaltq.io import read_json, read_jsonl, write_jsonl
from koaltq.prompts import parse_analysis
from koaltq.runtime import (GeneratedTextRepetitionControl, cache_path, generate_layer1,
                            l1_key, require_cache, validated_cache_item)


def analysis(text="그림에 있는 글자"):
    return {"visual_role": "text-heavy", "thumbnail_type": None, "visible_text": text}


@pytest.mark.parametrize("character", ["\t", "\n", "\r", "\x00"])
def test_raw_control_chars_preserved_and_serialized_safely(character):
    expected = analysis("첫째" + character + "둘째")
    raw = json.dumps(expected, ensure_ascii=False).replace(json.dumps(character)[1:-1], character)
    parsed, recovery = parse_analysis(raw, False, return_recovery=True)
    assert parsed == expected and recovery == "raw_control_characters"
    # Cached JSON is strict-valid again; no OCR text is discarded or invented.
    assert json.loads(json.dumps(parsed, ensure_ascii=False)) == expected


@pytest.mark.parametrize("raw", [
    '{"visual_role":"text-heavy","thumbnail_type":null,"visible_text":"반복 반복 반복 ',
    '{"visual_role":"text-heavy","thumbnail_type":null,"visible_text":"탭\t미완성',
    '{"visual_role":"text-heavy","thumbnail_type":null,"visible_text":"bad\\q"}',
    '{"visual_role":"invalid","thumbnail_type":null,"visible_text":"탭\t글자"}',
])
def test_unterminated_or_invalid_json_is_not_fabricated(raw):
    with pytest.raises(ValueError):
        parse_analysis(raw, False)


def setup_cache(tmp_path, record):
    (tmp_path / "images").mkdir()
    Image.new("RGB", (4, 4)).save(tmp_path / "images/a.png")
    args = SimpleNamespace(run_dir=tmp_path / "run", model_id="qwen", revision="a" * 40,
        max_seq_length=8192, l1_max_image_tokens=1024, l1_max_new_tokens=2048, model_cache_dir=None)
    item = {"record_id": record["record_id"], "input_sha256": l1_key(args, tmp_path, record,
        truncation_version="longest_field_right_v1"), "analysis": analysis(),
        "truncation": {"original_token_length": 2400, "final_token_length": 2400, "truncated_fields": []}}
    return args, item


def test_legacy_untruncated_cache_only_migrates_if_all_other_inputs_match(tmp_path, record):
    args, item = setup_cache(tmp_path, record)
    updated = validated_cache_item(args, tmp_path, record, item)
    assert updated["analysis"] == item["analysis"]
    assert updated["cache_migration"]["from_input_sha256"] == item["input_sha256"]
    assert item["input_sha256"] != updated["input_sha256"]  # original metadata not mutated
    bad = copy.deepcopy(item)
    bad["truncation"] = {"original_token_length": 9000, "final_token_length": 8180, "truncated_fields": ["context_text"]}
    with pytest.raises(ValueError, match="missing/stale"):
        validated_cache_item(args, tmp_path, record, bad)
    with pytest.raises(ValueError, match="missing/stale"):
        validated_cache_item(args, tmp_path, {**record, "context_text": "changed"}, item)
    Image.new("RGB", (4, 4), "red").save(tmp_path / "images/a.png")
    with pytest.raises(ValueError, match="missing/stale"):
        validated_cache_item(args, tmp_path, record, item)


def test_preserve_1917_successes_and_retry_only_three_reported_failures(tmp_path, record, monkeypatch):
    import koaltq.runtime as runtime
    args, legacy = setup_cache(tmp_path, record)
    good_rows = [{**record, "record_id": f"cached_{i:04d}"} for i in range(1917)]
    bad_ids = ["r046bf6ae3c", "r84a4c397ac", "r59a621e5bb"]
    failed_rows = [{**record, "record_id": rid} for rid in bad_ids]
    saved = [{**legacy, "record_id": row["record_id"]} for row in good_rows]
    write_jsonl(cache_path(args, "train"), saved)
    repeated = '{"visual_role":"text-heavy","thumbnail_type":null,"visible_text":"' + "문구 반복 " * 80
    complete = json.dumps(analysis(), ensure_ascii=False)
    raw_tab = complete.replace("그림에 있는 글자", "그림\t글자")
    outputs = iter([repeated, complete, repeated, complete, raw_tab])
    calls = []

    class Tokenizer:
        def decode(self, tokens, **kwargs):
            return bytes(tokens.tolist()).decode("utf-8")

    class Model:
        device = torch.device("cpu")

        def generate(self, **kwargs):
            calls.append(kwargs)
            return torch.cat((kwargs["input_ids"], torch.tensor([list(next(outputs).encode("utf-8"))])), dim=1)

    monkeypatch.setattr(runtime, "pin_model_revision", lambda a: None)
    monkeypatch.setattr(runtime, "load_model", lambda a: Model())
    monkeypatch.setattr(runtime, "load_processor", lambda *a: SimpleNamespace(tokenizer=Tokenizer()))
    retries = []

    def prepare(*a, **kw):
        retries.append(kw["retry"])
        return {"input_ids": torch.tensor([[1, 2]])}, {"original_token_length": 2400,
            "final_token_length": 2400, "truncated_fields": []}

    monkeypatch.setattr(runtime, "prepare_input", prepare)
    generate_layer1(args, "train", good_rows + failed_rows, tmp_path)
    result = {r["record_id"]: r for r in read_jsonl(cache_path(args, "train"))}
    assert len(result) == 1920 and len(calls) == 5
    assert retries == [0, 1, 0, 1, 0]
    assert "logits_processor" not in calls[0]
    retry_settings = calls[1]["logits_processor"][0].settings
    assert retry_settings["repetition_penalty"] > 1 and retry_settings["no_repeat_ngram_size"] == 16
    for row in good_rows:
        assert result[row["record_id"]]["analysis"] == legacy["analysis"]
        assert result[row["record_id"]]["cache_migration"]["reason"] == "untruncated_v1_input_unchanged"
    assert result[bad_ids[2]]["analysis"]["visible_text"] == "그림\t글자"
    assert result[bad_ids[2]]["generation"]["parse_recovery"] == "raw_control_characters"
    assert read_json(args.run_dir / "layer1/train_errors.json") == []
    assert len(require_cache(args, "train", good_rows + failed_rows, tmp_path)) == 1920
    monkeypatch.setattr(runtime, "load_model", lambda a: pytest.fail("Valid cache must not load the model"))
    generate_layer1(args, "train", good_rows + failed_rows, tmp_path)


def test_failed_retries_remain_failures_and_keep_diagnostics(tmp_path, record, monkeypatch):
    import koaltq.runtime as runtime
    args, _ = setup_cache(tmp_path, record)
    broken = '{"visible_text":"unterminated'
    calls = []

    class Model:
        device = torch.device("cpu")

        def generate(self, **kwargs):
            calls.append(kwargs)
            return torch.cat((kwargs["input_ids"], torch.tensor([list(broken.encode())])), dim=1)

    monkeypatch.setattr(runtime, "pin_model_revision", lambda a: None)
    monkeypatch.setattr(runtime, "load_model", lambda a: Model())
    monkeypatch.setattr(runtime, "load_processor", lambda *a: SimpleNamespace(
        tokenizer=SimpleNamespace(decode=lambda t, **k: bytes(t.tolist()).decode())))
    monkeypatch.setattr(runtime, "prepare_input", lambda *a, **kw: (
        {"input_ids": torch.tensor([[1]])}, {"truncated_fields": []}))
    with pytest.raises(RuntimeError, match="1 failed records"):
        generate_layer1(args, "train", [record], tmp_path)
    assert len(calls) == 3 and calls[1]["logits_processor"][0].settings != calls[2]["logits_processor"][0].settings
    assert not cache_path(args, "train").exists()
    errors = read_json(args.run_dir / "layer1/train_errors.json")
    assert len(errors[0]["attempts"]) == 3
    assert errors[0]["attempts"][0]["output"] == broken


def test_repetition_control_does_not_ban_text_seen_only_in_input():
    control = GeneratedTextRepetitionControl(4, {"repetition_penalty": 1.1, "no_repeat_ngram_size": 4})
    prompt_only = torch.tensor([[1, 2, 3, 4]])
    torch.testing.assert_close(control(prompt_only, torch.ones(1, 10)), torch.ones(1, 10))
    first_copy = torch.tensor([[1, 2, 3, 4, 1, 2, 3]])
    scores = control(first_copy, torch.ones(1, 10))
    assert scores[0, 4] == 1  # prompt's 1,2,3,4 does not ban OCR's first 1,2,3,4
    repeat_in_output = torch.tensor([[1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3]])
    assert control(repeat_in_output, torch.ones(1, 10))[0, 4].isneginf()
