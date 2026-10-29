"""Layer 3: 내부 학습에서 검증한 규칙의 정책 고정과 예측 보정."""
from __future__ import annotations
import re, copy
from .utils import LABELS, digest
PRIORITY=['파일명형','무관형','무의미형','장식오용형','중복형','불충분형','적절']

RULE_SPEC = {
    'filename_only_v1': {
        'target': '파일명형',
        'pattern': r'[A-Za-z0-9가-힣_-][A-Za-z0-9가-힣_.-]*\.(?:jpg|jpeg|png|gif|webp|svg|bmp|tif|tiff)',
        'reason': 'alt 전체가 공백 없는 이미지 파일명 형태',
    }
}

def matches_rule(rule_id, row):
    spec = RULE_SPEC[rule_id]
    alt = row['alt_text']
    if not isinstance(alt, str):
        raise ValueError('alt는 문자열이어야 함. 빈 문자열은 허용')
    return re.fullmatch(spec['pattern'], alt.strip(), flags=re.IGNORECASE) is not None

def build_policy(train_rows, expected_train_ids, requested, min_matches=3, min_pages=2):
    actual = [r['record_id'] for r in train_rows]
    expected = list(expected_train_ids)
    if len(actual) != len(set(actual)) or len(expected) != len(set(expected)) or set(actual) != set(expected):
        raise ValueError('규칙 검증에 내부 학습 ID만 정확히 사용해야 함')
    if min_matches < 1 or min_pages < 1 or len(requested) != len(set(requested)):
        raise ValueError('규칙 검증 설정 오류')
    if set(requested) - set(RULE_SPEC):
        raise ValueError('정의되지 않은 규칙')
    rows, active = [], []
    for rid in requested:
        hit = [r for r in train_rows if matches_rule(rid, r)]
        bad = [r['record_id'] for r in hit if r['label'] != RULE_SPEC[rid]['target']]
        pages = {r['page_url'] for r in hit if r['page_url']}
        eligible = len(hit) >= min_matches and len(pages) >= min_pages and not bad
        if eligible:
            active.append(rid)
        rows.append({'rule_id': rid, 'target': RULE_SPEC[rid]['target'], 'matches': len(hit),
                     'pages': len(pages), 'false_positives': len(bad), 'false_positive_ids': bad,
                     'precision': (len(hit)-len(bad))/len(hit) if hit else None, 'active': eligible})
    policy = {'version': 'layer3-v1', 'rules': copy.deepcopy(RULE_SPEC), 'priority': PRIORITY,
              'requested_rules': list(requested), 'active_rules': active,
              'min_matches': min_matches, 'min_pages': min_pages,
              'train_ids_sha256': digest(sorted(actual)),
              'train_records_sha256': digest(sorted(train_rows, key=lambda r: r['record_id'])),
              'audit': rows, 'selection_data': 'internal_train_only'}
    policy['policy_sha256'] = digest(policy)
    return policy

def check_policy(policy):
    body = {k: v for k, v in policy.items() if k != 'policy_sha256'}
    if policy.get('policy_sha256') != digest(body):
        raise ValueError('정책 내용/해시 불일치')
    if policy['rules'] != RULE_SPEC or policy['priority'] != PRIORITY:
        raise ValueError('정책과 현재 규칙 코드 불일치')
    if set(policy['active_rules']) - set(RULE_SPEC):
        raise ValueError('정책 내 알 수 없는 규칙')

def select_rule_match(rule_ids):
    return min(rule_ids, key=lambda rid: (PRIORITY.index(RULE_SPEC[rid]['target']), rid)) if rule_ids else None

def apply_layer3(feature_rows, raw_predictions, policy):
    check_policy(policy)
    ids = [r['record_id'] for r in feature_rows]
    pids = [r['record_id'] for r in raw_predictions]
    if len(ids) != len(set(ids)) or len(pids) != len(set(pids)) or set(ids) != set(pids):
        raise ValueError('Layer 3 입력/예측 ID 불일치·중복')
    # 정답이나 모델 내부 점수를 규칙 함수에 전달하지 않는 입력 제한
    features = {r['record_id']: {'alt_text': r['alt_text']} for r in feature_rows}
    result = []
    for pred in raw_predictions:
        rid, before = pred['record_id'], pred['prediction']
        if before not in LABELS:
            raise ValueError('7개 라벨 이외의 원래 예측')
        matched = [key for key in policy['active_rules'] if matches_rule(key, features[rid])]
        selected = select_rule_match(matched)
        after = RULE_SPEC[selected]['target'] if selected else before
        result.append({'record_id': rid, 'prediction_before': before, 'prediction_after': after,
                       'matched_rules': '|'.join(matched), 'applied_rule': selected or '',
                       'rule_reason': RULE_SPEC[selected]['reason'] if selected else '',
                       'changed': before != after, 'policy_sha256': policy['policy_sha256']})
    return result
