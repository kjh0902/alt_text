import pytest
import torch


@pytest.fixture
def record():
    return {"record_id": "example", "page_url": "https://example.org/a", "image_url": "https://example.org/a.png",
            "image_file": "a.png", "alt_text": "사진 설명", "page_title": "페이지 제목",
            "heading_path": "상위 > 하위", "context_text": "문맥", "in_link": True,
            "link_dest": "도착지", "thumbnail": False, "label": "적절"}


@pytest.fixture
def tiny_qwen():
    from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration
    torch.set_num_threads(2)
    torch.manual_seed(11)
    cfg = Qwen3VLConfig(
        text_config={"vocab_size": 128, "hidden_size": 32, "intermediate_size": 64,
                     "num_hidden_layers": 3, "num_attention_heads": 2, "num_key_value_heads": 2,
                     "head_dim": 16, "max_position_embeddings": 512,
                     "rope_scaling": {"rope_type": "default", "mrope_section": [2, 3, 3]}},
        vision_config={"depth": 3, "hidden_size": 32, "intermediate_size": 64, "num_heads": 4,
                       "patch_size": 2, "spatial_merge_size": 2, "temporal_patch_size": 2,
                       "out_hidden_size": 32, "num_position_embeddings": 16,
                       "deepstack_visual_indexes": [0, 1, 2]},
        image_token_id=120, video_token_id=121, vision_start_token_id=122, vision_end_token_id=123,
    )
    model = Qwen3VLForConditionalGeneration(cfg)
    model.set_attn_implementation("sdpa")
    return model


@pytest.fixture
def multimodal_input():
    torch.manual_seed(21)
    ids = torch.tensor([[1, 2, 122, 120, 123, 3, 4]])
    return {"input_ids": ids, "attention_mask": torch.ones_like(ids),
            "pixel_values": torch.randn(4, 3 * 2 * 2 * 2), "image_grid_thw": torch.tensor([[1, 2, 2]])}
