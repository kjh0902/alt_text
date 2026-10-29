# KoAltQ Gemma NLL 전체 데이터 학습

`KoAltQ_Gemma4_3Epoch_Improved.ipynb`의 NLL Baseline을 터미널용으로 옮겼습니다.
기존 Train 2,400 + 공개 Test 정답 109 + 규칙 합성 160 = **2,669건 전체**를 학습합니다.
Validation·K-Fold·Best Checkpoint 선택·Early Stopping·학습 재개는 사용하지 않습니다.

## 설치: Linux + RTX 5070 Ti 16GB

저장소 루트에서 Python 3.11과 CUDA 12.8을 지원하는 NVIDIA 드라이버를 사용합니다.
별도 CUDA Toolkit 또는 flash-attn 설치는 필요하지 않습니다. 최초 실행은 Hugging Face 모델 다운로드가 필요합니다.
Gemma 전용 가상환경 이름은 `.venv-gemma`입니다. 기존 `.venv` 대신 이 환경을 활성화한 뒤 설치·실행합니다.

```bash
python3.11 -m venv .venv-gemma
source .venv-gemma/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip check
```

주요 버전은 원본과 같은 torch **2.7.1+cu128**, torchvision **0.22.1+cu128**,
transformers **5.17.0**, peft **0.21.0**, bitsandbytes **0.50.2**,
accelerate **1.15.0**, huggingface-hub **1.32.0**입니다.
Blackwell 지원은 [PyTorch 2.7](https://pytorch.org/blog/pytorch-2-7/),
`sm_120`은 [NVIDIA 표](https://developer.nvidia.com/cuda/gpus),
NF4/CUDA 12.8 빌드는 [bitsandbytes 문서](https://huggingface.co/docs/bitsandbytes/v0.50.2/installation)를 참고하세요.

## 단일 명령 실행

데이터는 다음과 같이 압축 해제되어 있어야 합니다. 원본 파일은 수정하지 않습니다.
입력은 `records.jsonl`과 이미지이며 추가 썸네일 파일은 필요하지 않습니다.

```text
data/koaltq_261014/
  koaltq_train/records.jsonl
  koaltq_train/images/
  koaltq_test_public/records.jsonl
  koaltq_test_public/images/
  koaltq_test_public/sample_submission.csv
```

```bash
python gemma/train.py \
  --data-root /home/junhyung/Documents/vscode/alt_text/data/koaltq_261014 \
  --epochs 3 \
  --run-dir /home/junhyung/Documents/vscode/alt_text/runs/gemma_full_3ep
```

`--run-dir`는 아직 존재하지 않는 새 폴더여야 합니다. 중단 후에도 새 폴더에서 처음부터 시작합니다.
생략하면 저장소의 `runs/gemma_<시각>/`을 만듭니다. `--epochs`는 1 이상의 정수이고,
그 Epoch 수를 기준으로 기존 warmup·선형 감소 스케줄을 계산합니다.

추가 옵션:

| 옵션 | 기본값 / 용도 |
|---|---|
| `--layer1-cache-dir /path/to/previous_run` | 이전 실행의 `actual_model_api.json`과 `cache/layer1/` 재사용 |
| `--boundary-csv /path/to/경계사례_20261013.csv` | 기본은 동봉한 사무국 공개 CSV |
| `--sample-submission /path/to/sample_submission.csv` | 기본은 Test 폴더, 없으면 데이터 루트에서 탐색 |
| `--model-cache-dir /path/to/models` | 기본은 저장소 `.cache/gemma_models/` |
| `--validate-only` | GPU나 모델 다운로드 없이 데이터·이미지·합성·제출 ID 검사 후 종료 |

Layer 1 캐시는 모델 revision·양자화·processor·프롬프트·이미지 내용·생성 설정·JSON 형식을 확인합니다.
GPU/OS/패키지 버전/코드 해시 차이는 기존 가져오기 방식대로 기록하며, 모델 조건이 다르면 중단합니다.
유효한 항목만 가져오고 누락분은 기본 Gemma로 생성합니다. 원본 캐시를 덮어쓰지 않습니다.
Train/Test 전체 Layer 1을 준비한 후 합성본에는 원본의 분석을 연결합니다.
Layer 2의 CPU processor 출력도 내용 기반 캐시에 한 번 저장하므로 매 Epoch 이미지 처리·토큰화를 반복하지 않습니다.

## 유지한 학습법과 변경 이유

| 항목 | 설정 |
|---|---|
| 모델 / revision | `google/gemma-4-E4B-it` / `ee0ef6023621cff504d758262d4e04895a5af4a2` |
| Loss | 기존 예제별 정답 라벨 토큰 평균 NLL × 클래스 가중치, 예제 수로 평균 |
| LoRA | 언어 self-attention `q_proj`, `o_proj`; r=8, alpha=16, dropout=0 |
| 양자화 / dtype | NF4 4-bit, double quantization, BF16 |
| 최적화 | AdamW, lr=1e-4, weight decay=0.01, warmup 5%, gradient clip=1 |
| 배치 | microbatch 1 × accumulation 8, 유효 배치 8; 마지막 5건도 학습 |
| 메모리 | 기존 비재진입 gradient checkpointing 유지, 모델 전체 FP32 변환 없음 |
| 추론 | 기존 7개 라벨 전체 토큰 `sum_logprob`; 기존 Layer 3 후처리 |
| Layer 1 | 기존 alt·정답을 제외하는 프롬프트, `schema_recovery_v3` 유지 |

원본 Notebook이 시험하던 batch 1/2 중 **1을 고정**해 16GB 메모리 여유를 확보하며 유효 배치와 Loss는 유지합니다.
입력 문맥·이미지 해상도·LoRA 구조는 자동 축소하지 않습니다. CUDA/BF16/NF4 검사 후,
긴 문맥 표본에서 실제 LoRA 갱신과 메모리를 확인합니다. 이 검사에 쓴 어댑터는 버리고
동일 초기값의 새 어댑터로 전체 학습합니다. OOM이나 입력 상한 초과 시 진단과 이전 Epoch 결과를 보존하고 종료합니다.

학습 데이터 변경에 따라 step 수와 클래스 가중치는 달라집니다.
기본값은 Epoch당 **334 step**, 총 **1,002 step / 8,007개 예제 처리**입니다.
클래스 가중치와 Layer 3 정책은 합성본을 제외한 실제 2,509건으로 결정합니다.
공개 Test 109건은 동봉 CSV의 `데이터셋_라벨`만 사용하며 `재판정_라벨`을 사용하지 않습니다.
합성은 원본 Train에서 seed 42, 페이지당 최대 3건으로 만들고
중복형·파일명형·무관형·무의미형을 각 40건 검증합니다. 생성 목표에 못 미치면 학습하지 않습니다.
최종 분포는 적절 1,619 / 불충분형 647 / 중복형 187 / 파일명형 66 /
무의미형 65 / 장식오용형 30 / 무관형 55입니다.

## 저장 결과

```text
RUN_DIR/
  config.json                  # 실제 설정과 변경 이유
  data_audit.json              # 개수·분포·중복·입력 해시·이미지 검사
  class_weights.json
  synthetic_train.jsonl
  layer0_flags.json
  layer3_policy.json
  cache/layer1/
  cache/encoded_inputs/
  run.log
  training_summary.json
  epochs/epoch_01/              # epoch_02, epoch_03 등
    adapter/                   # 해당 Epoch LoRA, 해시 포함
    model_predictions.csv      # Layer 2 원본, 600행
    submission.csv             # 기존 Layer 3 적용, 600행
    submission_boundary109.csv  # submission의 109건만 공개 정답 적용
    prediction_details.json    # 7개 후보 점수와 후처리 내역
    train_steps.jsonl
    epoch_summary.json         # 예제 수 가중 평균 Train Loss·시간·메모리·해시
    epoch.log
```

매 Epoch 저장한 Adapter 파일의 실제 해시와 추론 가중치 해시를 대조합니다.
이전 Epoch 예측을 재사용하지 않으며 세 CSV 모두 `record_id,label` 두 열,
sample submission 순서의 고유 ID 600개를 검증합니다. 109건 보정 전 결과도 보존합니다.
Test 정답 109건을 학습에 사용하므로 해당 항목의 결과를 독립 검증 성능으로 보고하지 않습니다.

## CPU 검증과 테스트

GPU 환경 설치 대신 CPU 검증 환경을 구성할 때도 `.venv-gemma`를 사용하고,
루트의 `requirements-cpu.txt`를 설치합니다.

```bash
python3.11 -m venv .venv-gemma
source .venv-gemma/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-cpu.txt
python -m pip check
python gemma/train.py --data-root data/koaltq_261014 --run-dir runs/gemma_cpu_check --validate-only
python -m pytest -q
```

CPU 검증 범위와 실제 실행 결과는 [검증 기록](gemma/VERIFICATION.md),
원본 코드·프롬프트 해시는 [METHOD_PRESERVATION.json](gemma/METHOD_PRESERVATION.json)에 기록합니다.
CPU 검사와 모의 모델 검사는 실제 RTX 5070 Ti에서의 학습 완료를 의미하지 않습니다.

## 저장소 구조

```text
README.md                  # 설치·실행 안내
requirements.txt           # Gemma GPU 환경
requirements-cpu.txt       # CPU 검증·테스트 환경
requirements-common.txt    # 공통 의존성
pyproject.toml             # Gemma 패키지와 기본 테스트 설정
gemma/
  train.py                 # 기존 실행 진입점
  data.py, cache.py, workflow.py
  runtime/koaltq/           # 모델·NLL·Layer 0~3·이미지·GPU 검사
  runtime/prompts/          # 원본 프롬프트
  data_work/               # 합성 생성·검증 공통 코드
  resources/               # 공개 라벨 CSV
  tests/                   # Gemma CPU 테스트
```

`gemma/requirements.txt`와 `gemma/requirements-cpu.txt`는 기존 설치 명령 호환용으로
루트 Requirements를 읽습니다. 의존성 수정은 루트 파일에서만 합니다.
자동 테스트는 Linux와 Windows에서 CPU Requirements와 `python -m pytest -q`를 사용합니다.
