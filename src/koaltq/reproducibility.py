"""Record effective CLI settings and source fingerprints without changing numerics."""
from pathlib import Path
from uuid import uuid4

from .io import environment, file_sha, write_json


def record_invocation(args, command):
    root = Path(__file__).resolve().parents[2]
    files = [p for folder in ("src", "scripts") for p in (root / folder).rglob("*.py")]
    files += list(root.glob("requirements*.txt")) + [root / "pyproject.toml"]
    sources = {p.relative_to(root).as_posix(): file_sha(p) for p in sorted(files)}
    inputs = {}
    for part in ("train", "test"):
        directory = getattr(args, f"{part}_dir", None)
        if directory:
            for name in ("records_with_thumbnail.jsonl", "dataset_provenance.json"):
                path = Path(directory) / name
                if path.is_file():
                    inputs[f"{part}/{name}"] = file_sha(path)
    split = Path(args.split_file)
    report = {"command": command,
              "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              "environment": environment(), "source_sha256": sources, "input_sha256": inputs,
              "split_sha256": file_sha(split)}
    write_json(Path(args.run_dir) / "invocations" / f"{command}_{uuid4().hex}.json", report)
