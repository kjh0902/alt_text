# KoAltQ: Qwen3-VL 한국어 대체텍스트 품질 판정

`Qwen/Qwen3-VL-8B-Instruct`의 **language_model에만 QLoRA**를 적용합니다.
Vision encoder, 최종 visual merger, DeepStack visual merger를 포함한 모든 visual projection은 동결합니다.

## 파이프라인

1. `records_with_thumbnail.jsonl`과 이미지 검사, 고정 train 1,920 / validation 480 분할.
2. Layer 0: 요청된 문자열 feature 여섯 개만 추출.
3. Layer 1: alt와 정답을 제외한 이미지·문맥으로 zero-shot 시각 분석. 별도 명령으로 캐시 생성.
4. Layer 2: 이미지·alt·문맥·feature·시각 분석을 입력하고 일곱 후보 라벨의 **평균 token log probability**를 비교.

라벨 순서는 `적절, 무의미형, 파일명형, 중복형, 장식오용형, 불충분형, 무관형`입니다.
`beta[c] = mean(log p(label_token | input, previous_label_tokens))`이고,
`P = softmax(beta)`, `loss = -class_weight[y] * log_softmax(beta)[y]`입니다.
정답 라벨의 생성 손실로 대체하지 않으며, 부정 후보까지 모두 autograd에 연결합니다.
EOS·prompt·padding은 후보 토큰 평균에 포함하지 않습니다.

## 설치: Ubuntu + RTX 5070 Ti 16GB

Python 3.12 권장(최소 3.11). NVIDIA driver 580 계열 이상, CUDA 13.0 PyTorch wheel을 사용합니다.
별도 CUDA toolkit이나 flash-attn 소스 빌드는 필요하지 않습니다.

```bash
git clone https://github.com/kjh0902/alt_text.git
cd alt_text
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip check
```

주요 버전: torch 2.9.1+cu130, torchvision 0.24.1+cu130, transformers 4.57.6,
peft 0.18.1, bitsandbytes 0.48.2, accelerate 1.12.0.
GPU wheel: [PyTorch 공식 배포](https://pytorch.org/get-started/previous-versions/),
Blackwell 지원: [bitsandbytes 0.48.2](https://huggingface.co/docs/bitsandbytes/v0.48.2/installation).

## 데이터 위치

각 `--train-dir` / `--test-dir` 바로 아래에 다음 파일이 있어야 합니다.

```text
records_with_thumbnail.jsonl
images/
sample_submission.csv         # test 디렉터리
```

`records.jsonl`은 읽지 않습니다. thumbnail의 JSON null은 unknown으로 정규화합니다.
분할은 제공된 Gemma 노트북의 내장 CSV를 그대로 추출한 것으로, ID와 page_url/image_url 누출을 실행 시 검증합니다.
노트북의 학습법·프롬프트·Layer 3 정책은 가져오지 않았습니다.

이미지는 확장자가 아닌 내용으로 디코딩합니다. SVG는 resvg로 렌더링하고 첫 프레임·흰 배경 RGB로 통일합니다.
PNG의 손상된 iCCP 체크섬만 메모리에서 복구하며, 픽셀 데이터 손상은 오류로 보고합니다. 원본 파일은 수정하지 않습니다.

## 실행 순서

아래 경로 변수만 서버 환경에 맞춰 변경하세요. 공통 설정을 변경하면 모든 단계에 같은 인자를 전달하세요.
하나의 run 디렉터리를 여러 프로세스가 동시에 수정하지 않도록 하세요.

```bash
TRAIN=/absolute/path/koaltq_train
TEST=/absolute/path/koaltq_test_public
RUN=runs/qwen

python scripts/prepare_data.py --train-dir "$TRAIN" --test-dir "$TEST" --run-dir "$RUN"

# 실제 GPU 역전파 및 저장/복원 검사. 전체 실행 전에 별도 디렉터리에서 수행합니다.
python scripts/smoke_test.py --train-dir "$TRAIN" --run-dir runs/preflight --max-seq-length 8192

python scripts/run_layer1.py --split train --train-dir "$TRAIN" --run-dir "$RUN"
python scripts/run_layer1.py --split validation --train-dir "$TRAIN" --run-dir "$RUN"
python scripts/run_layer1.py --split test --test-dir "$TEST" --run-dir "$RUN"

python scripts/train.py --train-dir "$TRAIN" --run-dir "$RUN" --epochs 3 --max-seq-length 8192
python scripts/evaluate.py --train-dir "$TRAIN" --run-dir "$RUN"
python scripts/predict.py --test-dir "$TEST" --run-dir "$RUN" \
  --sample-submission "$TEST/sample_submission.csv" --output "$RUN/submission.csv"
```

학습 재개는 동일한 설정에 `--resume`을 추가합니다. 재개 가능한 체크포인트는 optimizer step 경계에서 저장합니다.
모델은 첫 실행에서 Hugging Face revision SHA로 고정하며 `--model-cache-dir`로 다운로드 위치를 지정할 수 있습니다.
Layer 1 캐시는 모델·프롬프트·이미지·입력 문맥·전처리가 일치할 때만 재사용합니다. alt·정답은 Layer 1 입력과 캐시 키에 포함하지 않습니다.
JSON 생성은 최대 3회 시도하고, 실패 레코드는 기록한 뒤 재실행 시 재시도합니다. 누락되거나 오래된 캐시로 학습하지 않습니다.

## 기본 학습 설정과 메모리

| 설정 | 값 |
|---|---|
| 기반 언어 모델 | 4-bit NF4, double quantization, BF16 compute |
| 학습 파라미터 | language_model의 q/k/v/o/gate/up/down projection LoRA만 |
| LoRA | rank 8, alpha 16, dropout 0 |
| 시각 모듈 | encoder와 모든 projection 동결, LoRA 없음 |
| batch / gradient accumulation | 1 / 8 |
| optimizer / learning rate | AdamW / 1e-4 |
| weight decay / warmup / clipping | 0.01 / 5% / 1.0 |
| epochs / seed | 3 / 42 |
| Layer 1 / Layer 2 이미지 예산 | 최대 1024 / 512 visual tokens |
| checkpoint 선택 | validation total score 최고, 동점이면 먼저 나온 epoch |

Class-Balanced weight는 train split의 빈도만 사용합니다.
`raw[c]=(1-rho)/(1-rho**n[c])`, `w[c]=7*raw[c]/sum(raw)`, 기본 rho는 0.99입니다.
batch 1에서 가중치가 소거되는 weighted CE mean reduction을 사용하지 않습니다.

일곱 후보를 순차 계산하고, 후보 전체의 non-reentrant checkpoint와 decoder checkpoint를 사용해 activation 메모리를 줄입니다.
label 예측 위치의 logits만 계산하고, 학습에서는 KV cache를 끕니다. 학습 속도보다 16GB 메모리 사용을 우선한 구성입니다.
16GB 실사용 적합성은 서버의 `smoke_test.py` 성공 및 peak VRAM 기록으로 확인해야 합니다.
OOM은 조용히 샘플을 건너뛰지 않고 오류로 표시합니다. 필요하면 새 run에서 `--max-image-tokens` 또는 `--max-seq-length`를 낮추세요.

### 입력 길이 초과

`--max-seq-length 8192`로 최대 길이를 설정합니다. 한도를 넘으면 **자동 절삭 후 학습을 계속**합니다.
가장 긴 텍스트 필드부터 뒤를 줄이고, 이미지 토큰·고정 지시문·구조화 feature·범주·후보 라벨은 보존합니다.
일곱 후보는 같은 절삭된 입력을 사용하며 Layer 0 feature는 절삭 전 원문에서 계산합니다.
`truncation.jsonl`과 콘솔에 record_id, 절삭 전/후 실제 토큰 길이, 절삭 필드를 기록합니다.
고정 지시문·이미지 자체조차 들어가지 않는 비정상적으로 작은 한도는 오류입니다.

## 결과 파일

- `layer0/{train,validation,test}.jsonl`: 여섯 feature.
- `layer1/{train,validation,test}.jsonl`: visual_role, thumbnail_type, visible_text 및 provenance.
- `class_weights.json`: train class counts와 정규화 가중치.
- `trainable_parameters.json`: 실제 trainable 이름과 개수, 시각 모듈 동결 확인.
- `checkpoints/`, `best_checkpoint.json`, `last_checkpoint.json`: adapter와 재개 상태.
- `validation/epoch_*/metrics.json`, `validation/best/metrics.json`: 평가 결과.
- `test_predictions.jsonl`: record_id, prediction, 각 클래스의 beta·probability·token_count.
- `submission.csv`: sample_submission의 ID와 순서를 유지한 새 파일. 원본은 보존.

Macro-F1은 일곱 클래스 고정 평균이고, Binary F1은 **부적절(여섯 클래스)을 양성**으로 계산합니다.
`total_score = 0.5 * binary_f1 + 0.5 * macro_f1_7class`입니다.
규칙에 따른 라벨 강제 변경이나 추가 후처리는 없습니다.
