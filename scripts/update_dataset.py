"""Carry existing thumbnail features into the corrected release, offline."""
import argparse
import json
from pathlib import Path

from koaltq.data import DEFAULT_SPLIT
from koaltq.dataset_update import prepare_release


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-dir", type=Path, required=True, help="Directory containing release ZIPs or extracted dataset folders")
    parser.add_argument("--previous-train-dir", type=Path, required=True)
    parser.add_argument("--previous-test-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split-file", type=Path, default=DEFAULT_SPLIT)
    args = parser.parse_args()
    report = prepare_release(args.release_dir, args.previous_train_dir, args.previous_test_dir,
                             args.output_dir, split_file=args.split_file)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
