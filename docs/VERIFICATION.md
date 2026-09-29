# 검증 기록

## 초기 push 시점

- Windows / Python 3.12 / torch 2.9.1+cpu 환경에서 21개 테스트 통과.
- 작은 실제 Qwen3-VL 모델로 후보 토큰 mean log probability, 일곱 후보 역전파, checkpoint 사용 전후 gradient 동등성 확인.
- language_model LoRA만 학습 가능하며 vision encoder와 최종/DeepStack visual merger가 변경되지 않는 것을 확인.
- adapter 저장/복원, class-balanced weight, 제출 순서, 이진 F1 양성 클래스, Layer 1 alt/label 비노출 검사 통과.
- 실제 데이터 이미지 검사 및 추가 재개·절삭·실제 tokenizer 검증은 후속 검사 진행 중.

## 실행하지 않은 범위

RTX 5070 Ti에서 8B 모델 로딩·NF4 CUDA kernel·peak VRAM·전체 학습·실제 validation 점수·최종 test 예측은 아직 실행하지 않았습니다.
원격 서버 실행 명령과 `scripts/smoke_test.py`를 제공합니다. CPU 테스트 성공을 GPU 학습 성공으로 간주하지 않습니다.
