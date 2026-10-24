import pytest

from koaltq.cli import parser
from koaltq.io import read_json
from koaltq.reproducibility import record_invocation
from koaltq.training import train, training_contract


def test_weight_counts_are_from_training_rows_only(tmp_path, monkeypatch):
    args = parser("train").parse_args(["--run-dir", str(tmp_path)])
    training = [{"label": "적절"}, {"label": "무관형"}]
    validation = [{"label": "파일명형"}] * 20
    monkeypatch.setattr("koaltq.training.pin_model_revision", lambda args: None)
    monkeypatch.setattr("koaltq.training.require_cache", lambda *args: {})

    def capture(labels, rho):
        assert labels == ["적절", "무관형"] and rho == .99
        raise RuntimeError("verified train-only weight inputs")

    monkeypatch.setattr("koaltq.training.class_weights", capture)
    with pytest.raises(RuntimeError, match="verified train-only"):
        train(args, training, validation, tmp_path)


def test_objective_weight_is_part_of_resume_contract():
    args = parser("train").parse_args([])
    original = training_contract(args, [], [], {}, {})
    args.binary_loss_weight = 0
    changed = training_contract(args, [], [], {}, {})
    assert original != changed
    assert original["objective"]["binary_loss_weight"] == .5


def test_invocation_preserves_source_and_effective_settings(tmp_path):
    args = parser("train").parse_args(["--run-dir", str(tmp_path), "--binary-loss-weight", "0.25"])
    record_invocation(args, "train")
    report = read_json(next((tmp_path / "invocations").glob("train_*.json")))
    assert report["arguments"]["binary_loss_weight"] == .25
    assert "src/koaltq/scoring.py" in report["source_sha256"]
    assert report["split_sha256"] == "e34881f018d26eaf21ad8ca40ecedf9e095fffecbb8c112cfbf3eedb921ca862"
