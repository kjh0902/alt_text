"""합성 레코드 점검: 가이드라인 우선순위 충돌, 우연히 '적절'이 되는 경우, 표본 검토용 CSV.

우선순위(가이드라인 v1.1 3-2): 파일명형 > 무관형 > 무의미형 > 장식오용형 > 중복형 > 불충분형.
Layer 0 규칙은 팀 코드의 koaltq/layer0.py를 파일 경로로 직접 불러 쓴다(정규식 복제 없음, 무거운 import 없음).

자동 판정은 '충돌 후보'를 걸러낼 뿐이고, 최종 판단은 --review-csv로 뽑은 표본을 사람이 확인한다.
"""
from __future__ import annotations
import importlib.util, re, sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import REPO_ROOT

LAYER0_PATH = REPO_ROOT / 'runtime' / 'koaltq' / 'layer0.py'
PRIORITY = ['파일명형', '무관형', '무의미형', '장식오용형', '중복형', '불충분형']
# 생성기와 공유하는 무의미형 어휘 (Layer 0의 generic 집합 + 실제 데이터에서 보인 형태)
GENERIC_WORDS = ['이미지', '사진', '배너', '배너이미지', '로고', '*', '메인이미지', '비주얼1', '이미지1', '배너1']
SINGLE_CHARS = ['1', '2', '3', 'ㅁ', 'ㅇ', 'a', 'x']
RELATED_BIGRAM_JACCARD = 0.12  # 무관형 후보와 문맥의 문자 bigram 유사도 상한
_TOKEN = re.compile(r'[0-9A-Za-z가-힣]+')


def load_rule_flags():
    if not LAYER0_PATH.is_file():
        raise FileNotFoundError(f'Layer 0 규칙 파일 없음: {LAYER0_PATH}')
    spec = importlib.util.spec_from_file_location('koaltq_layer0', LAYER0_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.rule_flags


def bigrams(text: str) -> set[str]:
    compact = re.sub(r'\s+', '', text)
    return {compact[i:i + 2] for i in range(len(compact) - 1)}


def jaccard(a: str, b: str) -> float:
    x, y = bigrams(a), bigrams(b)
    return len(x & y) / len(x | y) if x and y else 0.0


def tokens(text: str) -> set[str]:
    return {t for t in _TOKEN.findall(text) if len(t) >= 2}


def context_blob(row: dict) -> str:
    return ' '.join(str(row.get(k, '')) for k in ('page_title', 'context_text', 'link_dest', 'heading_path'))


def is_meaningless(alt: str, flags: dict) -> bool:
    alt = alt.strip()
    return flags['generic_phrase_candidate'] or alt in GENERIC_WORDS or len(alt) <= 1


def real_alt_label_counts(train: list[dict]) -> dict[str, Counter]:
    by_alt = defaultdict(Counter)
    for r in train:
        by_alt[r['alt_text'].strip()][r['label']] += 1
    return by_alt


def audit_row(row: dict, rule_flags, real_alts: dict[str, Counter] | None = None) -> list[str]:
    """합성 레코드 1건의 문제 코드 목록. 비어 있으면 자동 점검 통과."""
    label, alt = row['label'], row['alt_text']
    issues = []
    if label not in PRIORITY:
        return [f'unexpected_label:{label}']
    if not row.get('synthetic'):
        issues.append('not_marked_synthetic')
    if alt.strip() == row['orig_alt'].strip():
        issues.append('alt_unchanged')
    flags = rule_flags(row)
    filename, meaningless = flags['filename_candidate'], is_meaningless(alt, flags)
    rank = PRIORITY.index(label)

    if label == '파일명형':
        if not filename:
            issues.append('filename_not_detected_by_layer0')
        return issues
    if filename:
        issues.append('priority_conflict:파일명형')  # 파일명형이 항상 앞선다
    if rank > PRIORITY.index('무의미형') and meaningless:
        issues.append('priority_conflict:무의미형')

    if label == '무의미형':
        if not meaningless:
            issues.append('meaningless_not_detected')
        seen = (real_alts or {}).get(alt.strip())
        if seen and seen['적절'] > seen['무의미형']:
            issues.append(f"real_data_mostly_적절({seen['적절']}/{sum(seen.values())})")
    elif label == '중복형':
        if not (flags['alt_in_link_text'] or flags['alt_equals_link_text']
                or flags['alt_in_context'] or flags['alt_equals_context']):
            issues.append('duplicate_not_supported_by_text')
    elif label == '무관형':
        blob = context_blob(row) + ' ' + row['orig_alt']
        if flags['alt_in_context'] or flags['alt_in_link_text']:
            issues.append('irrelevant_but_repeats_text')
        shared = tokens(alt) & tokens(blob)
        if shared:
            issues.append('maybe_related:shared_token:' + sorted(shared)[0])
        if jaccard(alt, blob) >= RELATED_BIGRAM_JACCARD:
            issues.append('maybe_related:bigram_similarity')
    return issues


def issue_kind(code: str) -> str:
    """집계용으로 코드의 가변 부분(토큰, 건수)을 떼어낸다."""
    if code.startswith('maybe_related:shared_token'):
        return 'maybe_related:shared_token'
    return code.split('(')[0]
