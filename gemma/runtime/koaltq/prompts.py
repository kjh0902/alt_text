"""v0.9 판정 프롬프트·규칙 플래그·입력 화이트리스트."""
from __future__ import annotations
import re
from pathlib import Path
from .utils import canonical, CONTEXT_KEYS


def _normalize_context_value(key: str, value):
    # 크롤링 과정에서 page_title에 섞인 대량의 개행/들여쓰기만 제거; 문자 내용·순서 유지
    if key == 'page_title' and isinstance(value, str):
        return re.sub(r'\s+', ' ', value).strip()
    return value


def context_for(row: dict) -> dict:
    return {k:_normalize_context_value(k,row[k]) for k in CONTEXT_KEYS}


from .layer0 import rule_flags


def load_prompts(package_root: Path) -> dict:
    return {k:(package_root/'prompts'/f'{k}_prompt.txt').read_text(encoding='utf-8') for k in ('layer1','layer2')}


def layer1_text(row: dict, prompts: dict) -> str:
    # 기존 alt·정답·record_id·경로·URL 미포함
    return prompts['layer1'] + '\n문맥:\n' + canonical(context_for(row))


def layer2_text(row: dict, analysis: dict, prompts: dict) -> str:
    return prompts['layer2'] + '\n평가 자료:\n' + canonical({
        'context':context_for(row),'alt_text':row['alt_text'],
        'layer1':analysis,'candidate_flags':rule_flags(row)})


def validate_layer1(value: dict) -> dict:
    required = {'role','visual_type','has_informative_text','visible_text','key_visual_information',
                'context_already_covers','additional_information','uncertain'}
    if not isinstance(value,dict) or set(value)!=required: raise ValueError('Layer 1 JSON 키 오류')
    if value['role'] not in ['기능성','정보성','텍스트 포함','장식성','unknown']: raise ValueError('Layer 1 role 오류')
    if value['visual_type'] not in ['촬영사진','그래픽','혼합','unknown']: raise ValueError('Layer 1 visual_type 오류')
    if type(value['has_informative_text']) is not bool: raise ValueError('Layer 1 bool 오류')
    for key in required-{'role','visual_type','has_informative_text'}:
        if not isinstance(value[key],list) or not all(isinstance(x,str) for x in value[key]):
            raise ValueError(f'Layer 1 문자열 목록 오류: {key}')
    return value
