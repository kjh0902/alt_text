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

## adapter reload / Layer 1 복구 수정 검증

- 최신 전체 CPU 테스트 **38개 통과**, `pip check` 및 `git diff --check` 통과.
- 기존 loader의 학습 경로에서는 PEFT 준비가 language RMSNorm을 FP32로 올리지만 추론 adapter 재로딩은 이를 건너뛰어 BF16으로 남는 차이를 재현했습니다. 양자화 생성 인자는 기존에도 같았으며, 준비 경로가 달랐습니다.
- 실제 PEFT와 작은 BF16 Qwen 모델에서 수정된 학습·재개·재로딩의 모든 파라미터 dtype/형태가 일치하고 저장/복원된 파라미터와 일곱 후보 점수가 일치하는 것을 확인했습니다. 이 CPU 테스트는 CUDA 다운로드 경계를 대체하고 dense CPU 연산을 사용하므로 NF4 CUDA 커널 검증은 아닙니다.
- 실제 GPU smoke의 기존 허용 오차 assert를 삭제하거나 완화하지 않았습니다. 검사 전에 numeric profile 일치를 추가 확인하고 실패 진단을 파일로 남깁니다.
- raw tab/newline/control character 보존, strict-valid JSON 재직렬화, 미완성 JSON·잘못된 escape·잘못된 역할 거부 테스트 통과.
- 1,917개 성공 캐시 + 보고된 세 ID 형태의 실패를 구성한 회귀 테스트에서 기존 분석을 보존하고 세 레코드만 재시도했습니다. 재실행 때 1,920개 모두 캐시에서 읽고 모델을 로드하지 않는 것도 확인했습니다. 이는 서버의 실제 실패 출력 재실행 결과가 아닌 재현용 fixture 테스트입니다.
- 오래된 캐시는 절삭이 없고 다른 입력·이미지·설정 해시까지 맞을 때만 호환 처리합니다. 이미지/문맥이 바뀌거나 이전 입력이 절삭된 경우에는 재생성합니다.
- 반복 억제가 prompt의 문구를 처음 출력하는 것은 막지 않고, 생성된 출력의 반복에만 적용되는 것을 확인했습니다.

## 실행하지 않은 범위

이 로컬 세션에서는 RTX 5070 Ti의 8B 모델 로딩·NF4 CUDA kernel·peak VRAM·전체 학습·실제 validation 점수·최종 test 예측을 실행하지 않았습니다.
사용자가 보고한 기존 서버 smoke 실패(최대 점수 차이 약 1.687)와 Layer 1 세 건 실패에 대해 코드를 수정했으며, 수정본의 실제 서버 재검증은 필요합니다.
원격 서버 실행 명령과 `scripts/smoke_test.py`를 제공합니다. CPU 테스트 성공을 GPU 학습 성공이나 실제 세 레코드 복구 성공으로 간주하지 않습니다.
