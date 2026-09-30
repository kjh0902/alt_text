import copy
import csv
import io
import json
import struct
import zlib
from types import SimpleNamespace

import pytest
from PIL import Image

from koaltq.data import load_records, rule_flags
from koaltq.evaluation import metrics_for, write_submission
from koaltq.images import clean_png_profile, load_image
from koaltq.io import read_jsonl, write_jsonl
from koaltq.prompts import l1_payload, parse_analysis
from koaltq.runtime import l1_key


def test_exact_feature_set_and_empty(record):
    row = {**record, "alt_text": " ", "context_text": " ", "link_dest": " "}
    flags = rule_flags(row)
    assert set(flags) == {"filename_pattern", "alt_equals_context", "alt_in_context",
                          "alt_equals_link_dest", "alt_in_link_dest", "alt_empty"}
    assert sum(flags.values()) == 1 and flags["alt_empty"]


@pytest.mark.parametrize("alt,expected", [("a.JPG", True), ("../a", True), ("C:\\images\\a", True),
    ("https://a/b", True), ("계획_최종", False), ("복제_안내", False), ("설명", False)])
def test_filename_general_only(record, alt, expected):
    assert rule_flags({**record, "alt_text": alt})["filename_pattern"] is expected


def test_link_requires_in_link(record):
    row = {**record, "alt_text": "문맥", "context_text": "문맥", "link_dest": "문맥", "in_link": False}
    flags = rule_flags(row)
    assert flags["alt_equals_context"] and flags["alt_in_context"]
    assert not flags["alt_equals_link_dest"] and not flags["alt_in_link_dest"]


def test_thumbnail_null_and_unicode_line_separator(tmp_path, record):
    (tmp_path / "images").mkdir()
    Image.new("RGB", (2, 2)).save(tmp_path / "images/a.png")
    row = {**record, "thumbnail": None, "context_text": "첫째\u2028둘째\u2029셋째"}
    write_jsonl(tmp_path / "records_with_thumbnail.jsonl", [row])
    result = load_records(tmp_path, labeled=True)
    assert len(result) == 1 and result[0]["thumbnail"] == "unknown"
    assert result[0]["context_text"] == row["context_text"]
    (tmp_path / "records.jsonl").write_text("invalid file must never be used")
    assert load_records(tmp_path, labeled=True) == result


def test_layer1_alt_label_independence(tmp_path, record):
    (tmp_path / "images").mkdir()
    Image.new("RGB", (2, 2)).save(tmp_path / "images/a.png")
    args = SimpleNamespace(model_id="qwen", revision="a" * 40, max_seq_length=8192,
                           l1_max_image_tokens=1024, l1_max_new_tokens=2048)
    other = {**record, "alt_text": "SENTINEL ALT", "label": "SENTINEL LABEL"}
    assert l1_payload(other) == l1_payload(record)
    assert l1_key(args, tmp_path, other) == l1_key(args, tmp_path, record)
    changed = {**record, "context_text": "changed"}
    assert l1_key(args, tmp_path, changed) != l1_key(args, tmp_path, record)


def test_layer1_schema():
    valid = {"visual_role": "informational", "thumbnail_type": None, "visible_text": "글자"}
    assert parse_analysis("```json\n" + json.dumps(valid) + "\n```", False) == valid
    with pytest.raises(ValueError):
        parse_analysis(json.dumps(valid), True)
    valid["thumbnail_type"] = "photo"
    with pytest.raises(ValueError):
        parse_analysis(json.dumps(valid), "unknown")
    assert parse_analysis(json.dumps(valid), True) == valid


def test_png_bad_profile_recovery_without_pixel_change(tmp_path):
    buffer = io.BytesIO()
    Image.new("RGBA", (4, 2), (255, 0, 0, 128)).save(buffer, format="PNG")
    raw = buffer.getvalue()
    data = b"bad profile"
    corrupt_profile = struct.pack(">I", len(data)) + b"iCCP" + data + b"\x00\x00\x00\x00"
    damaged = raw[:33] + corrupt_profile + raw[33:]
    cleaned, repaired = clean_png_profile(damaged)
    assert repaired and cleaned == raw
    path = tmp_path / "not_png.do"
    path.write_bytes(damaged)
    image, details = load_image(path)
    assert image.size == (4, 2) and image.getpixel((0, 0)) == (255, 127, 127)
    assert details["removed_bad_iccp"]
    broken = bytearray(raw)
    broken[29] ^= 1
    with pytest.raises(ValueError, match="checksum"):
        clean_png_profile(bytes(broken))


def test_svg_preserves_aspect_ratio(tmp_path):
    path = tmp_path / "vector.svg"
    path.write_text('<svg xmlns="http://www.w3.org/2000/svg" width="100" height="50"><rect width="100" height="50" fill="red"/></svg>')
    image, _ = load_image(path, svg_max_side=200)
    assert image.size == (200, 100)


def test_png_with_svg_metadata_is_still_raster(tmp_path):
    from PIL.PngImagePlugin import PngInfo
    metadata = PngInfo()
    metadata.add_text("source", '<svg xmlns="http://www.w3.org/2000/svg"/>')
    path = tmp_path / "raster.png"
    Image.new("RGB", (4, 3), "blue").save(path, pnginfo=metadata)
    image, details = load_image(path)
    assert image.size == (4, 3) and details["format"] == "PNG"


def test_submission_order_and_input_unchanged(tmp_path):
    template = tmp_path / "sample_submission.csv"
    raw = "record_id,label\nb,적절\na,적절\n"
    template.write_text(raw, encoding="utf-8")
    rows = [{"record_id": "a"}, {"record_id": "b"}]
    preds = [{"record_id": "a", "prediction": "불충분형"}, {"record_id": "b", "prediction": "무관형"}]
    output = tmp_path / "submission.csv"
    write_submission(template, output, rows, preds)
    with output.open(encoding="utf-8", newline="") as f:
        assert list(csv.DictReader(f)) == [{"record_id": "b", "label": "무관형"}, {"record_id": "a", "label": "불충분형"}]
    assert template.read_text(encoding="utf-8") == raw
    with pytest.raises(ValueError):
        write_submission(template, output, rows, preds + preds[:1])


def test_binary_positive_is_inappropriate_and_seven_class_macro():
    rows = [{"record_id": str(i), "label": label} for i, label in enumerate(["적절", "불충분형", "무관형"])]
    predictions = [{"record_id": str(i), "prediction": label} for i, label in enumerate(["적절", "불충분형", "적절"])]
    metrics = metrics_for(rows, predictions)
    assert metrics["binary_f1"] == pytest.approx(2 / 3)
    assert metrics["macro_f1_7class"] == pytest.approx((2 / 3 + 1) / 7)
    assert metrics["total_score"] == pytest.approx(0.5 * (metrics["binary_f1"] + metrics["macro_f1_7class"]))
