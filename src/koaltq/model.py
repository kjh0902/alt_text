from __future__ import annotations

import torch

from .io import environment

LANGUAGE_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def language_lora_targets(model):
    names = [name for name, module in model.named_modules()
             if ".language_model." in name and name.rsplit(".", 1)[-1] in LANGUAGE_TARGETS]
    if not names or any("visual" in name or "merger" in name for name in names):
        raise ValueError("Could not resolve language-only LoRA modules")
    return names


def check_trainable(model):
    names = [name for name, p in model.named_parameters() if p.requires_grad]
    invalid = [name for name in names if ".language_model." not in name or ".lora_" not in name]
    if not names or invalid:
        raise ValueError(f"Only language_model LoRA may be trainable. Invalid: {invalid}")
    return {"trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "trainable_names": names, "visual_encoder_frozen": True, "visual_projection_frozen": True}


def attach_language_lora(model, *, rank=8, alpha=16):
    from peft import LoraConfig, get_peft_model
    for p in model.parameters():
        p.requires_grad_(False)
    cfg = LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=0.0, bias="none",
                     task_type="CAUSAL_LM", target_modules=language_lora_targets(model))
    model = get_peft_model(model, cfg)
    check_trainable(model)
    return model


def resolve_revision(model_id, revision):
    from huggingface_hub import HfApi
    from pathlib import Path
    if Path(model_id).is_dir():
        raise ValueError("Use the Hugging Face model ID and --model-cache-dir; unversioned local model directories are unsupported")
    if len(revision) == 40 and all(c in "0123456789abcdef" for c in revision):
        return revision
    return HfApi().model_info(model_id, revision=revision).sha


def load_processor(model_id, revision, max_image_tokens, cache_dir=None):
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(model_id, revision=revision, cache_dir=cache_dir)
    image_processor = processor.image_processor
    unit = int(image_processor.patch_size) * int(image_processor.merge_size)
    maximum = int(max_image_tokens) * unit * unit
    # Qwen3-VL uses shortest_edge/longest_edge as pixel-count bounds.
    image_processor.size = {"shortest_edge": 4 * unit * unit, "longest_edge": maximum}
    if hasattr(image_processor, "min_pixels"):
        image_processor.min_pixels = 4 * unit * unit
        image_processor.max_pixels = maximum
    return processor


def load_model(args, *, training=False, adapter=None):
    from transformers import BitsAndBytesConfig, Qwen3VLForConditionalGeneration
    if not torch.cuda.is_available():
        raise RuntimeError("The actual Qwen 8B pipeline requires CUDA. Use requirements-cpu.txt and pytest for local tests.")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 compute is required")
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                              bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16,
                              llm_int8_skip_modules=["visual", "lm_head"])
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_id, revision=args.revision, cache_dir=args.model_cache_dir,
        quantization_config=quant, torch_dtype=torch.bfloat16,
        device_map={"": torch.cuda.current_device()}, attn_implementation="sdpa")
    for p in model.parameters():
        p.requires_grad_(False)
    if training:
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True,
                                               gradient_checkpointing_kwargs={"use_reentrant": False})
        # PEFT upcasts every non-quantized floating tensor. Keep large frozen modules in BF16.
        model.model.visual.to(dtype=torch.bfloat16)
        model.model.language_model.embed_tokens.to(dtype=torch.bfloat16)
        model.lm_head.to(dtype=torch.bfloat16)
        model.model.visual.gradient_checkpointing_disable()
        model.model.visual.requires_grad_(False)
        if adapter:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, str(adapter), is_trainable=True)
        else:
            model = attach_language_lora(model, rank=args.lora_rank, alpha=args.lora_alpha)
        check_trainable(model)
        model.config.use_cache = False
    elif adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(adapter), is_trainable=False)
        for p in model.parameters():
            p.requires_grad_(False)
    model.eval()
    return model


def gpu_environment():
    result = environment()
    result.update(cuda_runtime=torch.version.cuda, gpu=torch.cuda.get_device_name(),
                  capability=list(torch.cuda.get_device_capability()),
                  total_vram_bytes=torch.cuda.get_device_properties(0).total_memory,
                  peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                  peak_reserved_bytes=torch.cuda.max_memory_reserved())
    return result
