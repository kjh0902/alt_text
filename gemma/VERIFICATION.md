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
python -m pytest -q
```

Gemma 전용 저장소 정리 후 결과: **12 passed**. 기본 테스트 경로는 `gemma/tests`입니다.

Windows의 임시 경로 권한 제한을 피하기 위해 실제 검증에는 다음 옵션을 사용했습니다.

```text
--basetemp=runs/pytest_gemma_only -o cache_dir=runs/pytest_cache_gemma_only
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

## Gemma 전용 저장소 정리

학습·추론·합성·캐시 코드, 기본 설정, 프롬프트, 공개 라벨, 기존 테스트 28개 파일의
SHA256을 정리 전후 비교해 모두 동일함을 확인했습니다. 실행 경로도 `python gemma/train.py`로 유지합니다.
실제 import 의존성을 확인하고 루트의 이전 모델 전용 코드·스크립트·분할·테스트·문서를 제거했습니다.
Gemma가 사용하는 이미지·가중 NLL·점수 계산·Layer 0~3·GPU 검사·캐시·합성 공통 코드는 유지했습니다.

루트 Requirements는 기존 Gemma 버전을 그대로 사용하고, `gemma/requirements*.txt`는
루트 파일을 참조합니다. 두 설치 경로의 의존성이 동일하게 풀리는 것을 확인했습니다.
패키지 이름·탐색 경로와 자동 테스트도 Gemma 기준으로 수정했습니다.

`koaltq-gemma` wheel을 네트워크 없이 빌드하고, 소스 폴더 밖의 별도 경로에서
import·기본 설정·합성 규칙·프롬프트 로딩을 검사했습니다. wheel에 기존 Gemma 코드와
필수 설정·프롬프트·공개 CSV가 원본 해시 그대로 포함되고, 삭제한 코드와 테스트가 포함되지 않음을 확인했습니다.

## 의존성과 GPU 검사 범위

고정한 Hugging Face 패키지 조합과 공통 의존성은 `pip install --dry-run`으로
충돌 없이 해결됨을 확인했습니다. 이 확인은 설치된 CPU PyTorch를 사용했으며
CUDA PyTorch wheel의 전체 설치·Linux GPU 실행을 대체하지 않습니다.
PyTorch 2.7/CUDA 12.8 및 bitsandbytes 0.50.2의 Blackwell 지원은 README의 공식 자료를 근거로 선택했습니다.

실제 서버 실행은 CUDA·BF16·NF4 forward/backward, 긴 문맥 표본의 LoRA 갱신,
기본 모델 동결, 추론 점수 일치와 메모리 여유 검사를 먼저 수행합니다.
실측 결과는 실행 폴더의 `gpu_kernels.json`, `smoke/full_nll/`, `batch_plan.json`에 저장됩니다.
