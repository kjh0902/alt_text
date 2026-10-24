import copy

import pytest

from koaltq.data import DEFAULT_SPLIT
from koaltq.dataset_update import merge_thumbnail, prepare_release
from koaltq.io import file_sha


def test_historical_split_file_is_byte_identical():
    assert file_sha(DEFAULT_SPLIT) == "e34881f018d26eaf21ad8ca40ecedf9e095fffecbb8c112cfbf3eedb921ca862"


def test_merge_uses_new_labels_and_id_not_row_order(record):
    old = [{**record, "record_id": "a", "thumbnail": None},
           {**record, "record_id": "b", "thumbnail": True}]
    original = copy.deepcopy(old)
    new = [{k: v for k, v in row.items() if k != "thumbnail"} for row in reversed(old)]
    new[0]["label"] = "불충분형"
    merged, changes = merge_thumbnail(new, old)
    assert old == original
    assert [r["record_id"] for r in merged] == ["b", "a"]
    assert merged[0]["label"] == "불충분형" and merged[0]["thumbnail"] is True
    assert merged[1]["thumbnail"] is None
    assert changes == [{"record_id": "b", "old_label": "적절", "new_label": "불충분형"}]


@pytest.mark.parametrize("field,value", [("alt_text", "different"), ("context_text", "changed"),
                                         ("image_url", "https://other/image.png")])
def test_merge_rejects_input_drift(record, field, value):
    with pytest.raises(ValueError, match="Non-label"):
        merge_thumbnail([{**record, field: value}], [record])


def test_merge_rejects_missing_or_duplicate_ids_and_missing_feature(record):
    with pytest.raises(ValueError, match="duplicates"):
        merge_thumbnail([record, record], [record])
    with pytest.raises(ValueError, match="IDs differ"):
        merge_thumbnail([{**record, "record_id": "other"}], [record])
    with pytest.raises(ValueError, match="thumbnail"):
        merge_thumbnail([record], [{k: v for k, v in record.items() if k != "thumbnail"}])


def test_migration_never_overwrites_existing_destination(tmp_path):
    (tmp_path / "keep.txt").write_text("keep")
    with pytest.raises(FileExistsError):
        prepare_release(tmp_path, tmp_path, tmp_path, tmp_path)
    assert (tmp_path / "keep.txt").read_text() == "keep"
