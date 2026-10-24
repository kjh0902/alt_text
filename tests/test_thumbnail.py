"""Offline checks of the supplied algorithm; no live sites or browser install."""
import importlib.util
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def thumbnail():
    path = Path(__file__).resolve().parents[1] / "scripts" / "add_thumbnail_feature.py"
    spec = importlib.util.spec_from_file_location("thumbnail_script", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_thumbnail_paths_follow_input_directory(thumbnail, tmp_path):
    args = thumbnail.parse_args(["--input", str(tmp_path / "records.jsonl")])
    assert args.output == tmp_path / "records_with_thumbnail.jsonl"
    assert args.audit == tmp_path / "thumbnail_audit.jsonl"


def test_original_graphic_repeat_exception_and_ui_exclusion(thumbnail, record):
    row = {**record, "alt_text": "교통과 이미지"}
    small = {"anchor_found": True, "anchor_text": "교통과 교통과", "rendered_width": 20,
             "rendered_height": 20, "natural_width": 20, "natural_height": 20}
    assert thumbnail.decide_thumbnail(row, small) == (True, "alt_repeats_link_title")
    assert thumbnail.decide_thumbnail(row, {**small, "in_nav": True}) == (False, "site_chrome_link")
    assert thumbnail.decide_thumbnail(row, {**small, "anchor_text": ""}) == (False, "image_only_link")


def test_original_metadata_fallback_and_ambiguous_dom(thumbnail, record):
    row = {**record, "alt_text": "행사 안내", "link_dest": "행사 안내 행사 안내"}
    assert thumbnail.fallback_thumbnail_from_metadata(row) == (True, "fallback_alt_repeats_link_title")
    assert thumbnail.fallback_thumbnail_from_metadata({**row, "link_dest": "#"})[0] is False
    image = {"urls": [row["image_url"]], "alt": row["alt_text"], "anchor_text": row["link_dest"]}
    match, method = thumbnail.match_image([image, dict(image)], row, row["page_url"])
    assert match is None and method == "exact_url_ambiguous"


def test_thumbnail_rejects_source_overwrite_before_browser_import(thumbnail, tmp_path):
    source = tmp_path / "records.jsonl"
    source.write_text("keep")
    # Drive the pre-await validation without opening a Windows event loop/socket.
    operation = thumbnail.process(source, source, tmp_path / "audit.jsonl", None, False, .5)
    try:
        with pytest.raises(ValueError, match="separate"):
            operation.send(None)
    finally:
        operation.close()
    assert source.read_text() == "keep"
