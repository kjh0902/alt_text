"""Shared paths/domain helper from the original augmentation tools."""
from pathlib import Path
from urllib.parse import urlparse

REPO_ROOT = Path(__file__).resolve().parents[1]


def domain(row: dict) -> str:
    return urlparse(row['page_url']).netloc
