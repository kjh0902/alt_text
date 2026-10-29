"""소수 클래스 규칙 기반 증강 (CPU 전용, 학습 전용 -- 검증/평가에 쓰지 말 것).

베이스: 라벨=적절이고 alt가 비어 있지 않은 학습 레코드. 빈 alt가 아닌 '적절'은 장식 이미지가 아니므로(장식이면 빈 alt가
정답) 장식오용형 우선순위와 부딪히지 않는다. 베이스의 이미지·문맥은 그대로 두고 alt_text와 label만 바꾼다.

규칙 (aug_rule -> 라벨):
  irrelevant   -> 무관형   다른 도메인 적절 레코드의 alt로 교체 (문맥과 겹치면 제외)
  duplicate    -> 중복형   alt를 link_dest(링크 텍스트) 또는 context_text 앞부분으로 교체
  filename     -> 파일명형 image_url 파일명 / _수정·_v2 접미 / 복제_ 접두 / 카메라식 이름 / "원본 그림의 이름:" 형태
  meaningless  -> 무의미형 일반어(이미지·배너·로고 등) 또는 한 글자
장식오용형은 이미지 역할 판단이 필요해 규칙으로 만들지 않는다.

각 합성 레코드에는 synthetic=True, aug_rule, base_record_id, base_fold, orig_alt가 기록된다.
전체 학습에서는 fold_of=None이며 분할 파일을 읽지 않는다.
"""
from __future__ import annotations
import random, re, sys
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import unquote, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_augment import (GENERIC_WORDS, SINGLE_CHARS, audit_row, context_blob, issue_kind, jaccard, load_rule_flags,
                           real_alt_label_counts, tokens, RELATED_BIGRAM_JACCARD)
from common import domain

RULES = {'irrelevant': '무관형', 'duplicate': '중복형', 'filename': '파일명형', 'meaningless': '무의미형'}
ORDER = ['duplicate', 'filename', 'irrelevant', 'meaningless']  # 베이스 조건이 까다로운 규칙부터
DEFAULT_PER_RULE = 40
IMAGE_EXT = re.compile(r'\.(jpg|jpeg|png|gif|webp|svg|bmp|tif|tiff)$', re.I)
FILENAME_SUFFIX = ['_수정', '_원본', '_최종', '_v2', '_v3', '_복제본']
URL_LIKE = re.compile(r'^(https?:|javascript:|/|#)', re.I)
RELATIVE_URL = re.compile(r'^\S+\.(do|jsp|php|html?|aspx?|es)(\?\S*)?$|[?&]\w+=', re.I)
REPEATED_PREFIX = re.compile(r'^(\S.{1,}?) \1(?: |$)')


def valid_link_text(link: str) -> bool:
    link = link.strip()
    return len(link) >= 2 and not URL_LIKE.match(link) and not RELATIVE_URL.search(link)


def repeated_link_text(link: str) -> str | None:
    """링크 텍스트가 'X X' 또는 'X X 홈페이지'처럼 앞부분이 두 번 반복되면 X를 돌려준다(메뉴·아이콘형 링크의 전형 패턴)."""
    m = REPEATED_PREFIX.match(re.sub(r'\s+', ' ', link).strip())
    return m.group(1) if m else None


def make_duplicate(base: dict, rnd: random.Random, _pool) -> tuple[str, dict] | None:
    link = base['link_dest']
    if base['in_link'] and valid_link_text(link):
        # 사진·포스터 썸네일 카드에서 제목만 반복하면 가이드라인 3-3상 중복형이 아니라 불충분형이다.
        # 이미지 종류를 알 수 없으므로 링크 텍스트가 반복되는 메뉴·아이콘형 링크에서만 만든다.
        text = repeated_link_text(link)
        return (text[:120], {'source_field': 'link_dest'}) if text else None
    ctx = re.sub(r'\s+', ' ', base['context_text']).strip()
    if len(ctx) < 8:
        return None
    cut = ctx[:80]
    if len(ctx) > 80 and ' ' in cut:
        cut = cut.rsplit(' ', 1)[0]
    # 문맥이 이미지와 같은 대상을 말할 때만 쓴다: 원래 alt의 핵심 문구가 그 문맥 조각 안에 그대로 있어야 한다.
    # 무관한 문맥(슬라이드 네비게이션, 푸터 주소, '현재 총 N개의 소식' 등)을 넣으면 중복형이 아니라
    # 무관형 우선순위와 충돌한다.
    core = re.sub(r'\s*(이미지|사진|아이콘)\s*$', '', re.sub(r'\s+', ' ', base['alt_text']).strip())
    if len(core) < 3 or core not in cut:
        return None
    return cut, {'source_field': 'context_text'}


def make_filename(base: dict, rnd: random.Random, _pool) -> tuple[str, dict] | None:
    name = Path(unquote(urlparse(base['image_url']).path)).name
    has_ext = bool(IMAGE_EXT.search(name))
    kinds = ['camera', 'prefix', 'wizard'] + (['as_is', 'suffix'] * 2 if has_ext else [])
    kind = rnd.choice(kinds)
    stem, ext = (name[:name.rfind('.')], name[name.rfind('.'):]) if has_ext else (f'IMG_{rnd.randint(1000, 9999)}', '.jpg')
    if kind == 'as_is':
        alt = name
    elif kind == 'suffix':
        alt = stem + rnd.choice(FILENAME_SUFFIX) + ext
    elif kind == 'prefix':
        alt = '복제_' + stem + ext
    elif kind == 'camera':
        alt = rnd.choice([f'IMG_{rnd.randint(1000, 9999)}.jpg', f'DSC{rnd.randint(10000, 99999)}.JPG', f'{rnd.randint(1, 99):02d}_{rnd.randint(0, 9)}.png'])
    else:  # 한글 문서 붙여넣기 형태
        alt = '그림입니다.\r\n\r\n원본 그림의 이름: ' + stem + ext
    return alt, {'filename_kind': kind}


def make_meaningless(base: dict, rnd: random.Random, _pool) -> tuple[str, dict] | None:
    if rnd.random() < 0.6:
        return rnd.choice(GENERIC_WORDS), {'kind': 'generic_word'}
    return rnd.choice(SINGLE_CHARS), {'kind': 'single_char'}


def make_irrelevant(base: dict, rnd: random.Random, pool: list[dict]) -> tuple[str, dict] | None:
    blob = context_blob(base) + ' ' + base['alt_text']
    base_tokens = tokens(blob)
    for _ in range(40):
        donor = rnd.choice(pool)
        if domain(donor) == domain(base) or donor['page_url'] == base['page_url']:
            continue
        alt = donor['alt_text'].strip()
        if tokens(alt) & base_tokens or jaccard(alt, blob) >= RELATED_BIGRAM_JACCARD:
            continue
        return alt, {'donor_record_id': donor['record_id']}
    return None


MAKERS = {'duplicate': make_duplicate, 'filename': make_filename, 'irrelevant': make_irrelevant,
          'meaningless': make_meaningless}


def generate(train: list[dict], fold_of: dict[str, int] | None, quotas: dict[str, int], seed: int, max_per_page: int):
    rule_flags = load_rule_flags()
    real_alts = real_alt_label_counts(train)
    rnd = random.Random(seed)
    bases = [r for r in sorted(train, key=lambda r: r['record_id']) if r['label'] == '적절' and r['alt_text'].strip()]
    donors = [r for r in bases
              if 6 <= len(r['alt_text'].strip()) <= 60 and not rule_flags(r)['filename_candidate']
              and not rule_flags(r)['generic_phrase_candidate'] and r['alt_text'].strip() not in GENERIC_WORDS]
    used_base, page_count = set(), Counter()
    out, rejected, per_rule = [], Counter(), Counter()
    for rule in ORDER:
        want = quotas.get(rule, 0)
        by_domain = defaultdict(list)
        for r in bases:
            by_domain[domain(r)].append(r)
        for lst in by_domain.values():
            rnd.shuffle(lst)
        domains = sorted(by_domain)
        # 도메인을 돌아가며 뽑아 특정 기관에 쏠리지 않게 한다
        while per_rule[rule] < want and any(by_domain[d] for d in domains):
            for d in domains:
                if per_rule[rule] >= want:
                    break
                if not by_domain[d]:
                    continue
                base = by_domain[d].pop()
                if base['record_id'] in used_base or page_count[base['page_url']] >= max_per_page:
                    continue
                made = MAKERS[rule](base, rnd, donors)
                if made is None:
                    rejected[f'{rule}:no_candidate'] += 1
                    continue
                alt, extra = made
                syn = {**base, 'record_id': f"syn_{rule}_{base['record_id']}", 'alt_text': alt, 'label': RULES[rule],
                       'synthetic': True, 'aug_rule': rule, 'base_record_id': base['record_id'],
                       'base_fold': fold_of[base['record_id']] if fold_of is not None else None, 'orig_alt': base['alt_text'], 'aug_detail': extra}
                problems = audit_row(syn, rule_flags, real_alts)
                if problems:
                    for code in problems:
                        rejected[f'{rule}:{issue_kind(code)}'] += 1
                    continue
                used_base.add(base['record_id'])
                page_count[base['page_url']] += 1
                per_rule[rule] += 1
                out.append(syn)
    return out, dict(rejected)
