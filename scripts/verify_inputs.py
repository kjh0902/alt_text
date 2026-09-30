"""CPU verification with the actual Qwen tokenizer/processor, without 8B weights."""
import json

from koaltq.cli import parser
from koaltq.data import image_path, load_records
from koaltq.images import load_image
from koaltq.io import write_json
from koaltq.model import load_processor
from koaltq.prompts import l2_payload, prepare_input, prompt_messages
from koaltq.runtime import pin_model_revision
from koaltq.scoring import label_token_ids


def main():
    args = parser("verify_inputs").parse_args()
    pin_model_revision(args)
    processor = load_processor(args.model_id, args.revision, args.max_image_tokens, args.model_cache_dir)
    candidates = label_token_ids(processor.tokenizer)
    results = []
    for name, root in (("train", args.train_dir), ("test", args.test_dir)):
        if root is None:
            continue
        rows = load_records(root, labeled=name == "train")
        selected = {r["record_id"]: r for r in (
            rows[0], max(rows, key=lambda r: len(r["alt_text"])),
            next(r for r in rows if r["thumbnail"] == "unknown"),
            next(r for r in rows if r["image_file"].endswith(".svg")))}
        for row in selected.values():
            image, _ = load_image(image_path(root, row))
            # Placeholder visual analysis tests plumbing only, never prediction quality.
            analysis = {"visual_role": "informational", "thumbnail_type": "photo" if row["thumbnail"] is True else None,
                        "visible_text": "CPU processor verification placeholder"}
            rendered = processor.apply_chat_template(prompt_messages(2, l2_payload(row, analysis)),
                                                       tokenize=False, add_generation_prompt=True)
            prefix = processor.tokenizer.encode(rendered, add_special_tokens=False)
            for tokens in candidates:
                label = processor.tokenizer.decode(tokens, skip_special_tokens=False)
                assert processor.tokenizer.encode(rendered + label, add_special_tokens=False) == prefix + tokens
            inputs, details = prepare_input(processor, image, l2_payload(row, analysis), stage=2,
                max_seq_length=args.max_seq_length, reserve_tokens=max(map(len, candidates)))
            image_tokens = int((inputs["input_ids"] == processor.image_token_id).sum())
            expected = int(inputs["image_grid_thw"].prod() // processor.image_processor.merge_size ** 2)
            assert image_tokens == expected and image_tokens <= args.max_image_tokens
            results.append({"split": name, "record_id": row["record_id"], "image_tokens": image_tokens, **details})
            image.close()
        longest = max(rows, key=lambda r: len(r["alt_text"]))
        image, _ = load_image(image_path(root, longest))
        payload = l2_payload({**longest, "alt_text": longest["alt_text"] * 40}, analysis)
        _, truncated = prepare_input(processor, image, payload, stage=2,
            max_seq_length=2048, reserve_tokens=max(map(len, candidates)))
        assert truncated["truncated_fields"] and 2000 <= truncated["final_token_length"] <= 2048
        results.append({"split": name, "test": "forced_long_input", **truncated})
        image.close()
    report = {"scope": "actual processor/tokenizer only; no model predictions", "revision": args.revision,
              "candidate_token_counts": list(map(len, candidates)), "checks": results}
    write_json(args.run_dir / "processor_verification.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
