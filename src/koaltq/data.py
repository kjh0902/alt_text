from __future__ import annotations

import csv
import re
from collections import Counter
from pathlib import Path

from . import LABELS
from .io import file_sha, read_jsonl

TEXT_FIELDS = ("alt_text", "page_title", "heading_path", "context_text", "link_dest")
L1_FIELDS = ("page_title", "heading_path", "context_text", "in_link", "link_dest", "thumbnail")
DEFAULT_SPLIT = Path(__file__).resolve().parents[2] / "splits" / "team_split.csv"
# General filename/path shapes from the supplied notebook; corpus-specific Korean suffixes omitted.
FILENAME_PATTERN = re.compile(
    r"\.(?:jpg|jpeg|png|gif|webp|svg|bmp|tif|tiff)(?:$|[?#])|^(?:[A-Z]:[\\/]|https?://|(?:\.\./)+)",
    re.IGNORECASE,
)


def rule_flags(row):
    alt, ctx, link = (row[k].strip() for k in ("alt_text", "context_text", "link_dest"))
    return {
        "filename_pattern": bool(FILENAME_PATTERN.search(alt)),
        "alt_equals_context": bool(alt) and alt == ctx,
        "alt_in_context": bool(alt) and alt in ctx,
        "alt_equals_link_dest": bool(alt) and row["in_link"] and alt == link,
        "alt_in_link_dest": bool(alt) and row["in_link"] and alt in link,
        "alt_empty": not bool(alt),
    }


def image_path(root, row):
    base = (Path(root) / "images").resolve()
    path = (base / row["image_file"]).resolve()
    if not path.is_relative_to(base):
        raise ValueError(f"Image outside data directory: {row['record_id']}")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def load_records(root, *, labeled):
    rows = read_jsonl(Path(root) / "records_with_thumbnail.jsonl")
    seen = set()
    for row in rows:
        rid = row.get("record_id")
        if not isinstance(rid, str) or not rid or rid in seen:
            raise ValueError(f"Invalid/duplicate record_id: {rid!r}")
        seen.add(rid)
        for key in (*TEXT_FIELDS, "page_url", "image_url", "image_file"):
            if not isinstance(row.get(key), str):
                raise ValueError(f"{rid}: {key} must be a string")
        if type(row.get("in_link")) is not bool:
            raise ValueError(f"{rid}: in_link must be boolean")
        if "thumbnail" not in row:
            raise ValueError(f"{rid}: thumbnail is required")
        value = row["thumbnail"]
        if value is None or value == "unknown":
            row["thumbnail"] = "unknown"
        elif type(value) is not bool:
            raise ValueError(f"{rid}: thumbnail must be true/false/null/unknown")
        if labeled and row.get("label") not in LABELS:
            raise ValueError(f"{rid}: invalid label")
        image_path(root, row)
    return rows


def read_split(rows, path=DEFAULT_SPLIT):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        manifest = list(csv.DictReader(f))
    mapping = {r["record_id"]: r["split"] for r in manifest}
    if len(mapping) != len(manifest) or set(mapping) != {r["record_id"] for r in rows}:
        raise ValueError("Split has duplicate, missing, or extra record IDs")
    if Counter(mapping.values()) != Counter({"train": 1920, "validation": 480}):
        raise ValueError("Expected exactly train=1920 and validation=480")
    parts = {part: [r for r in rows if mapping[r["record_id"]] == part]
             for part in ("train", "validation")}
    overlaps = {}
    for field in ("page_url", "image_url"):
        a, b = ({r[field] for r in parts[p] if r[field]} for p in ("train", "validation"))
        overlaps[field] = len(a & b)
    if any(overlaps.values()):
        raise ValueError(f"Split leakage: {overlaps}")
    counts = {p: dict(Counter(r["label"] for r in rs)) for p, rs in parts.items()}
    if any(set(c) != set(LABELS) for c in counts.values()):
        raise ValueError("Every class must occur in train and validation")
    return parts, {"counts": {p: len(rs) for p, rs in parts.items()}, "per_class": counts,
                   "overlap": overlaps, "split_sha256": file_sha(Path(path))}


def select_records(args, split):
    if split == "test":
        if not args.test_dir:
            raise ValueError("--test-dir is required for test")
        rows = load_records(args.test_dir, labeled=False)
        if len(rows) != 600:
            raise ValueError("Expected 600 test records")
        return rows, Path(args.test_dir)
    if not args.train_dir:
        raise ValueError("--train-dir is required for train/validation")
    rows = load_records(args.train_dir, labeled=True)
    parts, _ = read_split(rows, args.split_file)
    return parts[split], Path(args.train_dir)

