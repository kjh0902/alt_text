# 검증 기록 — 2026-10-29

검증 환경: Windows, Python 3.12.14, PyTorch 2.9.1+cpu.
Linux/RTX 5070 Ti의 실제 Gemma 학습은 이 환경에서 실행하지 않았습니다.

## 실제 데이터 CPU 검사

```bash
python gemma/train.py --data-root data/koaltq_261014 --run-dir runs/gemma_cpu_final --validate-only
```

결과: **CPU_VALIDATION_PASSED**.

| 검사 | 결과 |
|---|---:|
| 원본 Train | 2,400 |
| 공개 Test 라벨 학습 추가 | 109 |
| 규칙 합성 | 160, 규칙별 40 |
| 최종 학습 | 2,669 |
| Test / 제출 양식 고유 ID | 600 / 600, 집합 일치 |
| 원본·Test 이미지 디코딩·해시 검사 | 3,000건, 오류 0 |
| 합성 규칙 검사 오류 | 0 |
| 중복 ID / 동일 모델 입력 | 0 / 0 |
| 원본 Train ↔ Test ID·page_url·image_url 겹침 | 모두 0 |
| 클래스 가중치 계산 대상 | 실제 라벨 2,509건; 합성 추가 전후 동일 |
| Layer 3 정책 대상 / 활성 규칙 | 실제 라벨 2,509건 / `[]` |

최종 클래스 분포: 적절 1,619, 불충분형 647, 중복형 187, 파일명형 66,
무의미형 65, 장식오용형 30, 무관형 55.
합성으로 인한 이미지 재사용 및 기존 데이터의 페이지 공유는 별도 집계하며 중복 ID로 취급하지 않습니다.

공식 입력 파일 SHA256:

```text
Train records.jsonl: d6c475eb0c07aae9478594fcd118dd827d22fd2acbc9c5f5fdd0f75389dbe572
Test records.jsonl:  d143e8aa528662ae16198d8f7af7475264e4cf03f6fc3376d23d16d4681d0c67
경계사례 CSV:       b0b94778789a5ec615ccafb3d3d430c025e0561ba80355114b0adc3f92caddef
```

## 자동 테스트와 보존 검사

```bash
python -m pytest gemma/tests tests -q
```

결과: **78 passed** (Gemma 12개 + 기존 Qwen 66개). 기존 PEFT 테스트의 설정 파일 안내 경고 3건 외 실패는 없습니다.

Windows의 임시 경로 권한 제한을 피하기 위해 실제 검증에는 다음 옵션을 사용했습니다.

```text
--basetemp=runs/pytest_gemma_final -o cache_dir=runs/pytest_cache_gemma
```

Gemma 테스트는 데이터·공개 라벨 오류, 합성 목표 수량·원본 연결, 제출 ID·정답 적용 범위,
Layer 1 캐시 키·JSON·모델 조건 불일치, CPU 입력 캐시 손상과 중복 전처리 방지,
실행 폴더 재사용 거부를 검사합니다.

CPU 모의 모델로 실제 학습 루프와 제출 파일 생성을 실행했습니다.
기본 2,669건 × 3 Epoch에서 **334 step/Epoch, 총 1,002 step·8,007개 예제 처리**,
마지막 5건 배치, Epoch별 가중치 변경, 저장 파일·추론 Adapter 해시 일치,
평가 모드 전환·학습 복귀, Epoch별 CSV 600행을 확인했습니다.

원본 파일 13개가 바이트 단위로 같고, RNG 저장·복원, LoRA 파라미터 검사,
AdamW 생성, 가중 NLL 역전파 함수 5개가 AST 기준으로 같습니다.
세부 해시는 `METHOD_PRESERVATION.json`에 있습니다.

## 의존성과 GPU 검사 범위

고정한 Hugging Face 패키지 조합과 공통 의존성은 `pip install --dry-run`으로
충돌 없이 해결됨을 확인했습니다. 이 확인은 설치된 CPU PyTorch를 사용했으며
CUDA PyTorch wheel의 전체 설치·Linux GPU 실행을 대체하지 않습니다.
PyTorch 2.7/CUDA 12.8 및 bitsandbytes 0.50.2의 Blackwell 지원은 README의 공식 자료를 근거로 선택했습니다.

실제 서버 실행은 CUDA·BF16·NF4 forward/backward, 긴 문맥 표본의 LoRA 갱신,
기본 모델 동결, 추론 점수 일치와 메모리 여유 검사를 먼저 수행합니다.
실측 결과는 실행 폴더의 `gpu_kernels.json`, `smoke/full_nll/`, `batch_plan.json`에 저장됩니다.
