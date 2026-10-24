"""Prepare the 2026-10-14 release without changing the historical split or features."""
from collections import Counter
from contextlib import contextmanager
import json
import hashlib
import io
from pathlib import Path, PurePosixPath
import shutil
import uuid
from zipfile import ZipFile

from .data import DEFAULT_SPLIT, image_path, read_split
from .io import file_sha, read_jsonl, write_json, write_jsonl

RELEASE = "KoAltQ_261014_guideline_v1.1"
EXPECTED_CHANGES = {"train": 41, "validation": 18}
SPLIT_SHA256 = "e34881f018d26eaf21ad8ca40ecedf9e095fffecbb8c112cfbf3eedb921ca862"


def indexed(rows):
    result = {r["record_id"]: r for r in rows}
    if len(result) != len(rows) or any(not isinstance(k, str) or not k for k in result):
        raise ValueError("Record IDs must be nonempty strings without duplicates")
    return result


def merge_thumbnail(rows, previous):
    """Only carry thumbnail forward. Reject any input drift beyond the label."""
    old, new = indexed(previous), indexed(rows)
    if old.keys() != new.keys():
        raise ValueError("Previous and release record IDs differ")
    merged, changes = [], []
    for row in rows:
        rid = row["record_id"]
        prior = old[rid]
        current_input = {k: v for k, v in row.items() if k not in ("label", "thumbnail")}
        previous_input = {k: v for k, v in prior.items() if k not in ("label", "thumbnail")}
        if current_input != previous_input:
            raise ValueError(f"Non-label fields changed: {rid}; thumbnail reuse is unsafe")
        if "thumbnail" not in prior or (prior["thumbnail"] is not None
                and prior["thumbnail"] != "unknown" and type(prior["thumbnail"]) is not bool):
            raise ValueError(f"Invalid/missing previous thumbnail: {rid}")
        if row.get("label") != prior.get("label"):
            changes.append({"record_id": rid, "old_label": prior.get("label"), "new_label": row.get("label")})
        merged.append({**row, "thumbnail": prior["thumbnail"]})
    return merged, changes


@contextmanager
def release_files(path):
    """Read only named files; never extract arbitrary ZIP paths."""
    path = Path(path)
    if path.is_dir():
        base = path.resolve()

        def read(name):
            target = (base / name).resolve()
            if not target.is_relative_to(base):
                raise ValueError("File outside release directory")
            return target.read_bytes()
        yield read, {"records_sha256": file_sha(base / "records.jsonl")}
    else:
        with ZipFile(path) as archive:
            names = [n for n in archive.namelist() if n.endswith("/records.jsonl")]
            if len(names) != 1 or len(archive.namelist()) != len(set(archive.namelist())):
                raise ValueError("Ambiguous release archive")
            prefix = names[0].removesuffix("records.jsonl")

            def read(name):
                posix = PurePosixPath(name)
                if posix.is_absolute() or ".." in posix.parts or "\\" in name:
                    raise ValueError("Unsafe release member")
                return archive.read(prefix + name)
            yield read, {"archive_sha256": file_sha(path)}


def prepare_release(release_dir, previous_train, previous_test, output_dir, *, split_file=DEFAULT_SPLIT):
    """Build a standalone copy; publish it only after all checks pass."""
    release_dir, output_dir = Path(release_dir), Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"Output already exists; choose a new directory: {output_dir}")
    if file_sha(split_file) != SPLIT_SHA256:
        raise ValueError("The 261014 migration requires the unchanged historical team split")
    staging = output_dir.with_name(output_dir.name + ".tmp_" + uuid.uuid4().hex)
    report = {"release": RELEASE, "split_sha256": file_sha(split_file), "datasets": {}}
    staged = False
    try:
        for name, previous_root, count in (("koaltq_train", previous_train, 2400),
                                           ("koaltq_test_public", previous_test, 600)):
            archive = release_dir / f"{name}.zip"
            source = archive if archive.is_file() else release_dir / name
            previous_root = Path(previous_root)
            previous_file = previous_root / "records_with_thumbnail.jsonl"
            previous = read_jsonl(previous_file)
            with release_files(source) as (read, fingerprint):
                # JSONL records may contain U+2028/U+2029 inside string values.
                raw_records = read("records.jsonl")
                rows = [json.loads(line) for line in io.StringIO(raw_records.decode("utf-8-sig")) if line.strip()]
                if len(rows) != count:
                    raise ValueError(f"{name}: expected {count} records")
                merged, changes = merge_thumbnail(rows, previous)
                if name == "koaltq_train":
                    parts, split_audit = read_split(merged, split_file)
                    changed_ids = {r["record_id"] for r in changes}
                    changed_by_split = {p: sum(r["record_id"] in changed_ids for r in rs) for p, rs in parts.items()}
                    if changed_by_split != EXPECTED_CHANGES:
                        raise ValueError(f"Expected label corrections {EXPECTED_CHANGES}; got {changed_by_split}")
                    report["split"] = split_audit
                    report["changed_by_split"] = changed_by_split
                elif changes or any("label" in r for r in rows):
                    raise ValueError("Public test must be unlabeled and unchanged")
                # Do not overlap the source directories. A failed run leaves sources intact.
                for old_root in (Path(previous_train).resolve(), Path(previous_test).resolve(), release_dir.resolve()):
                    if output_dir.is_relative_to(old_root) or old_root.is_relative_to(output_dir):
                        raise ValueError("Output must be separate from all source directories")
                target = staging / name
                (target / "images").mkdir(parents=True, exist_ok=True)
                staged = True
                for row in rows:
                    filename = row["image_file"]
                    if not isinstance(filename, str) or Path(filename).name != filename or "/" in filename or "\\" in filename:
                        raise ValueError(f"Unsafe image filename: {filename!r}")
                    raw = read(f"images/{filename}")
                    if hashlib.sha256(raw).hexdigest() != file_sha(image_path(previous_root, row)):
                        raise ValueError(f"Image changed: {row['record_id']}; existing thumbnail/cache must be reviewed")
                    (target / "images" / filename).write_bytes(raw)
                (target / "records.jsonl").write_bytes(raw_records)
                (target / "records.csv").write_bytes(read("records.csv"))
                (target / "README.md").write_bytes(read("README.md"))
                if name == "koaltq_test_public":
                    (target / "sample_submission.csv").write_bytes(read("sample_submission.csv"))
                write_jsonl(target / "records_with_thumbnail.jsonl", merged)
                provenance = {"release": RELEASE, **fingerprint,
                    "previous_thumbnail_records_sha256": file_sha(previous_file),
                    "prepared_records_sha256": file_sha(target / "records_with_thumbnail.jsonl"),
                    "record_count": count, "changed_label_count": len(changes),
                    "thumbnail_counts": dict(Counter(str(r["thumbnail"]).lower() for r in merged)),
                    "images_verified_unchanged": count}
                write_json(target / "dataset_provenance.json", provenance)
                report["datasets"][name] = provenance
                if changes:
                    write_jsonl(staging / "label_changes.jsonl", changes)
        # Check train/test grouping too, using the exact records that will be published.
        train = read_jsonl(staging / "koaltq_train" / "records_with_thumbnail.jsonl")
        test = read_jsonl(staging / "koaltq_test_public" / "records_with_thumbnail.jsonl")
        overlap = {k: len({r[k] for r in train if r[k]} & {r[k] for r in test if r[k]})
                   for k in ("page_url", "image_url")}
        if any(overlap.values()):
            raise ValueError(f"Train/test group leakage: {overlap}")
        report["train_test_overlap"] = overlap
        write_json(staging / "migration_report.json", report)
        staging.rename(output_dir)
    except BaseException:
        # This is an exclusively-created sibling staging folder, never a user source.
        if staged and staging.is_dir() and staging.parent == output_dir.parent and ".tmp_" in staging.name:
            shutil.rmtree(staging)
        raise
    return report
