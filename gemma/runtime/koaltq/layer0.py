"""Layer 0 — 판정 전 문자열 규칙 후보 추출."""
import re

def rule_flags(row: dict) -> dict:
    alt = row['alt_text'].strip(); ctx = row['context_text'].strip(); link = row['link_dest'].strip()
    return {
        'empty_alt':not bool(alt),
        'filename_candidate':bool(re.search(r'(?i)\.(jpg|jpeg|png|gif|webp|svg|bmp|tif|tiff)(?:$|[?#])|_(수정|원본|복제본|최종|v\d+)(?:_|$)|^복제_|^(?:[A-Z]:[\\/]|https?://|(?:\.\./)+)',alt)),
        'generic_phrase_candidate':alt in {'이미지','사진','배너','배너이미지','로고','*'},
        'alt_in_link_text':bool(alt) and row['in_link'] and alt in link,
        'alt_in_context':bool(alt) and alt in ctx,
        'alt_equals_context':bool(alt) and alt == ctx,
        'alt_equals_link_text':bool(alt) and row['in_link'] and alt == link,
    }

