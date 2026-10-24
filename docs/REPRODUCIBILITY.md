# 학습·추론 재현 및 코드 제출

## 환경

권장 실제 학습 환경은 기존과 동일한 Ubuntu x86_64, Python 3.12, RTX 5070 Ti 16GB,
NVIDIA driver 580 이상이다. 저장소를 clone한 뒤 README의 가상환경·requirements.txt 설치를 따른다.
GPU smoke 성공이 실제 8B NF4 학습의 선행 조건이며 CPU 테스트는 이를 대신하지 않는다.

한글 경로와 외부 패키지의 기본 텍스트 인코딩을 위해 UTF-8 모드를 권장한다.

```bash
export PYTHONUTF8=1
```

Windows PowerShell에서는 다음과 같이 설정한다. GPU requirements는 Ubuntu용이므로 Windows는
별도의 CPU 가상환경에서 requirements-cpu.txt로 로컬 검증한다.

```powershell
$env:PYTHONUTF8 = "1"
```

## 데이터 준비: 과거 thumbnail 보존

`scripts/update_dataset.py`는 수정 배포 ZIP과 기존 thumbnail 레코드를 읽는다. 내부 분할
`splits/team_split.csv`는 변경하지 않고, 내부 학습 41건·검증 18건의 라벨 정정을 검증한다.
모든 비라벨 입력과 이미지가 기존과 같은지 확인한 뒤 신규 레코드에 기존 thumbnail만 결합한다.
기존 이미지와 원본 파일은 수정하지 않으며, 출력 폴더가 이미 있으면 중단한다.

```bash
python scripts/update_dataset.py \
  --release-dir /absolute/path/to/data_261014/공개_데이터셋_20260917 \
  --previous-train-dir /absolute/path/to/old/koaltq_train \
  --previous-test-dir /absolute/path/to/old/koaltq_test_public \
  --output-dir data/koaltq_261014
```

출력은 다음과 같다. `dataset_provenance.json`에는 원본 ZIP 또는 레코드 해시와 이전 thumbnail 파일 해시를 남긴다.

```text
data/koaltq_261014/
  migration_report.json
  label_changes.jsonl
  koaltq_train/
    records.jsonl
    records.csv
    records_with_thumbnail.jsonl
    dataset_provenance.json
    images/
  koaltq_test_public/
    records.jsonl
    records.csv
    records_with_thumbnail.jsonl
    dataset_provenance.json
    sample_submission.csv
    images/
```

학습·평가 실행 명령은 README의 순서를 따른다. 기본 데이터 경로는 위 디렉터리이고, 새로운
`RUN=runs/qwen_261014_aux`를 사용한다. 모델 revision은 기존처럼 첫 실행에서 SHA로 고정한다.
이전 Layer 1 결과를 활용할 때에는 이전 `model_revision.json`과 `layer1/`만 복사하고
train/validation/test의 run_layer1.py를 각각 재실행해 해시 검증을 통과한 캐시만 사용한다.
모든 단계에 이미지 예산·입력 길이·모델 설정을 동일하게 전달한다.

## 기존 thumbnail 파일이 없는 환경

원래 스크립트를 `scripts/add_thumbnail_feature.py`로 포함했다. DOM 후보 매칭, alt를 이용한 후보
선택·제목 반복 예외, metadata fallback, 페이지 접속 실패 시 null 처리 등 원래 판정 로직은 유지했다.
개인 PC의 절대 기본 경로만 제거하고 CLI 경로·원본 덮어쓰기 방지를 보완했다.

```bash
python -m pip install -r requirements-thumbnail.txt
python -m playwright install chromium
python scripts/add_thumbnail_feature.py --input /path/to/koaltq_train/records.jsonl
python scripts/add_thumbnail_feature.py --input /path/to/koaltq_test_public/records.jsonl
```

기본 출력은 입력 파일 옆의 `records_with_thumbnail.jsonl`과 `thumbnail_audit.jsonl`이다.
`--output`, `--audit`, `--limit`, `--headful`, `--delay`를 지원한다. 기존 출력은 덮어쓰지 않는다.
원격 페이지는 시간이 지나면 바뀌므로, **동일 실험 재현에는 과거에 저장한 thumbnail 파일과 audit를 함께 제공**해야 한다.
원본 배포본을 외부에서 다시 수집한 결과가 과거 thumbnail과 완전히 같다고 보장할 수는 없다.

과거 원본·thumbnail이 없고 신규 배포본만 있다면 신규 records.jsonl에 이 스크립트를 직접 실행한 뒤
prepare_data.py부터 진행할 수 있다. 이 경우 과거 59건 변경 대조가 완료된 경로와 구분해 기록한다.
공식 수정 라벨은 그대로 사용하며 thumbnail 생성은 label을 읽어 판정하지 않는다.

## 손실과 실행 기록

기본값은 `L_total=0.5*L_type+0.5*L_bin`이다. 이진 확률은 기존 7범주 확률의 합으로 계산한다.
유형 손실과 train-only Class-Balanced 공식은 유지하며, 이진 손실에는 클래스 가중치를 적용하지 않는다.
`--binary-loss-weight 0`이면 이전 유형 손실과 그 gradient가 정확히 동일하다. 이는 손실 비교 옵션이며
v1.1 L2 프롬프트까지 과거 버전으로 되돌리는 옵션은 아니다.

학습 로그에는 혼합 loss·type_loss·binary_loss를 각각 기록한다. loss 비중은 resume contract에 포함돼
다른 설정으로 기존 optimizer 상태를 이어 쓰는 것을 막는다. 최종 예측은 계속 7개 beta의 argmax이다.
이진 F1도 최종 7범주 예측을 적절/부적절로 묶어 계산하며 별도 이진 threshold는 없다.

`invocations/`에는 실행별 CLI 인자·패키지 버전·소스 및 requirements 해시·입력/분할 해시가 남는다.
`model_revision.json`, `training_contract.json`, 데이터 migration report와 함께 보관한다.

## CPU 검증

새 가상환경에 requirements-cpu.txt를 설치한 뒤 실행한다. GPU 환경 requirements와 섞어 설치하지 않는다.

```bash
python -m pip check
python -m pytest -q --basetemp runs/pytest -o cache_dir=runs/pytest_cache
python scripts/prepare_data.py \
  --train-dir data/koaltq_261014/koaltq_train \
  --test-dir data/koaltq_261014/koaltq_test_public --run-dir runs/local_audit_261014
```

`scripts/verify_inputs.py`는 Qwen tokenizer/processor만 사용하는 별도 검증이다. 모델 가중치를 받지 않지만
토크나이저 파일은 필요하다. 이미지 토큰 정렬·후보 토큰 경계·입력 절삭을 확인한다.
CI는 Ubuntu와 Windows의 CPU 회귀 테스트를 실행하도록 구성했다. CI 결과는 원격 실행에서 확인한다.

## 코드 제출에 포함할 것

- 저장소의 commit과 모든 tracked source, requirements, README·docs, 고정 splits.
- 데이터 배포 버전·원본 체크섬, migration report, 실제 사용한 thumbnail 입력과 생성 audit.
- 실제 실행의 invocations·model revision·training contract·class weights·검증 결과·최종 adapter.
- test 600건 모두의 예측 파일과 sample_submission 순서를 유지한 submission.csv.

데이터 이미지·가중치·캐시·실행 결과는 Git에 올리지 않으며 저장소의 ignore 규칙을 유지한다.
대회에 제출할 묶음에는 실제 사용한 필요한 파일을 별도로 포함한다. 경계 사례 CSV의 test 라벨을
읽거나 예측을 덮어쓰는 경로는 없다. 하드웨어·CUDA 라이브러리가 달라질 때 비트 단위 동일성까지
보장하지 않으므로 실제 환경과 GPU smoke 결과를 함께 기록한다.
