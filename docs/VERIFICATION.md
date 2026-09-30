# 검증 기록

## 로컬 검증 완료 — 2026-09-30

- Windows / Python 3.12.14 / torch 2.9.1+cpu 환경에서 **25개 테스트 통과**. `pip check`도 통과.
- 작은 실제 Qwen3-VL 모델로 full-logits 기준 후보 token mean log probability, 일곱 후보 역전파, checkpoint 사용 전후 gradient 동등성 확인.
- language_model LoRA만 학습 가능하며 optimizer step 이후 vision encoder와 최종/DeepStack visual merger의 모든 파라미터가 변경되지 않는 것을 확인.
- adapter 저장/복원 및 optimizer·scheduler·RNG 복원 후 다음 학습 step의 파라미터 일치 검사 통과.
- class-balanced weight, batch 1 가중치 반영, 제출 순서, 이진 F1 양성 클래스, Layer 1 alt/label 비노출, 캐시 무효화, 자동 절삭 후 다음 레코드 처리 검사 통과.
- **train 2,400장 + test 600장 전부 디코딩 성공**. SVG 84장, AVIF 4장, MPO 4장 포함. PNG 1건의 손상된 iCCP 프로필만 메모리에서 제외하며 원본 파일은 변경하지 않음.
- 실제 고정 분할은 train 1,920 / validation 480. 페이지·이미지 URL 중복 모두 0건. thumbnail unknown은 train 10건, test 6건.

## 실제 Qwen tokenizer/processor 검사

- 모델 revision: `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`.
- 후보 토큰 수는 라벨 순서대로 `[2, 4, 4, 3, 5, 4, 3]`. assistant prefix와 라벨을 연결한 토큰화 경계도 확인.
- 실제 데이터 8건에서 이미지 grid와 image token 수 일치, 최대 512 visual tokens 제한 확인.
- 장문 강제 입력 2건에서 최대 2,048토큰으로 자동 절삭, 실제 전후 길이 기록 확인. 고정 지시문·이미지·후보는 보존하고 가능한 텍스트 prefix를 유지.
- 이 검사는 실제 tokenizer/processor와 테스트용 시각 분석 문자열만 사용함. 8B 모델 추론이나 실제 Layer 1 분석 결과가 아님.

요약 수치와 환경은 [verification.json](verification.json)에 기록했습니다.

## 실행하지 않은 범위

RTX 5070 Ti에서 8B 모델 로딩·NF4 CUDA kernel·peak VRAM·전체 학습·실제 validation 점수·최종 test 예측은 아직 실행하지 않았습니다.
원격 서버 실행 명령과 `scripts/smoke_test.py`를 제공합니다. CPU 테스트 성공을 GPU 학습 성공으로 간주하지 않습니다.
