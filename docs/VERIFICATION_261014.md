# 2026-10-24 수정 데이터 반영 검증

## 실제 데이터

- 공식 배포 ZIP에서 별도 `data/koaltq_261014` 입력본을 생성했다. 원본 데이터는 보존했다.
- 공개 학습 2,400건 중 59건의 label 변경을 확인했다. 고정 내부 학습 1,920건에서는 41건,
  검증 480건에서는 18건이다. ID 집합과 모든 비라벨 입력 필드는 기존과 동일하다.
- `splits/team_split.csv` SHA-256은 기존과 같은
  `e34881f018d26eaf21ad8ca40ecedf9e095fffecbb8c112cfbf3eedb921ca862`이다.
- 기존 thumbnail 값 3,000건을 그대로 보존했다. 평가 레코드 600건은 정답이 없으며 변경도 없다.
- 기존·신규 이미지 3,000개의 SHA-256 일치를 확인했고, prepare_data.py로 전 이미지를 실제 디코딩했다.
  이미지 실패 0건이며 기존 iCCP 메모리 복구 1건을 동일하게 처리했다.
- 내부 학습/검증, 공개 학습/평가의 page_url·image_url 겹침은 모두 0건이다.
- 실제 배포·준비 파일 해시와 신규 분포는 [dataset_261014.json](dataset_261014.json)에 기록했다.

## 방법 보존

기존 scoring의 후보 평균 로그확률, class_weights, classification_loss 함수는 유지했다.
분할·data.py·model.py·images.py·runtime.py·evaluation.py·io.py는 기존 main ab52e44와 동일하다.
Layer 1 프롬프트·캐시 키·출력 스키마도 유지한다.

썸네일 원본 SHA-256은 `80a972144fa3356ed224b04c2448b57c57a6d983a12152f77c0596f7d791ef10`이다.
원본과 추가 스크립트의 함수 AST를 비교해 26개 판정·매칭·브라우저 보조 함수가 동일함을 확인했다.
변경된 함수는 CLI 경로를 받는 parse_args, 경로·출력 보호를 추가한 process,
도움말 파싱을 Windows 이벤트 루프 생성보다 먼저 처리하도록 한 main이다.

이진 보조 손실은 기존 7개 후보 확률에서 적절/부적절 확률을 계산한다. 유형 가중치를 이진 손실에
중복 적용하지 않으며 최종 추론은 기존 7범주 argmax다. 검증에는 loss/gradient oracle, 극단적인
score의 수치 안정성, binary 비중 0일 때 기존 loss/gradient의 정확한 일치, checkpoint 재계산과
저장/복원·optimizer 재개, 시각 모듈 동결, train-only 클래스 가중치 입력을 포함했다.

## 실행 범위

최종 CPU 회귀 테스트 **66개 통과**. 작은 실제 Qwen 구조와 PEFT를 사용한 수치 검증을 포함한다.
PEFT의 테스트용 config 파일 부재 경고 3건은 기존 테스트에서도 발생하는 경고이며 실패는 없다.
추가한 썸네일·데이터 갱신·학습 CLI의 --help, pip check, git diff --check도 통과했다.

실제 Qwen tokenizer/processor 검증도 고정 revision
`0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`의 로컬 캐시로 통과했다. 학습·평가 대표 8건에서
후보 토큰 경계와 이미지 토큰 정렬을 검사했고, 긴 입력 2건이 2,048 토큰으로 정상 절삭됐다.
새 L2 프롬프트를 포함한 최대 길이 학습 대표 입력은 6,088 토큰으로 기본 8,192 이내였다.
이 검사는 placeholder 시각 분석을 사용한 입력 구성 검증이며 모델 예측 성능 측정은 아니다.

로컬은 Windows 11 / Python 3.12.14 / torch 2.9.1+cpu다. UTF-8 모드와 프로젝트 내부 pytest 임시
디렉터리를 사용한다. 기본 Windows 인코딩 및 샌드박스 밖 임시 경로에서 발생한 초기 환경 오류는
이 실행 설정으로 해소했다. pip check도 통과했다.

8B NF4 CUDA 로딩·실제 GPU 학습·16GB VRAM 적합성·실제 validation 성능·test 600건 예측 생성은
이 CPU 환경에서 수행하지 않았다. GPU smoke와 실제 학습 명령은 README에 제공하며 기존 허용 오차를
완화하지 않았다. test 전체를 모델로 예측하는 코드를 유지하고 공개 경계 사례 정답은 사용하지 않는다.

## 수정·추가 파일 목록

| 구분 | 파일 |
|---|---|
| 기존 학습 코드 수정 | src/koaltq/cli.py, prompts.py, scoring.py, smoke.py, training.py |
| 데이터·실행 기록 모듈 추가 | src/koaltq/dataset_update.py, reproducibility.py |
| 실행 도구 추가 | scripts/update_dataset.py, scripts/add_thumbnail_feature.py |
| 기존 회귀 테스트 수정 | tests/test_scoring.py, tests/test_runtime.py |
| 회귀 테스트 추가 | tests/test_dataset_update.py, tests/test_thumbnail.py, tests/test_training_objective.py |
| 문서·실측 기록 | README.md, docs/METHOD_REVIEW.md, docs/REPRODUCIBILITY.md, docs/VERIFICATION_261014.md, docs/dataset_261014.json |
| 선택 의존성·CI | requirements-thumbnail.txt, .github/workflows/tests.yml |

총 21개 파일이며 기존 splits와 데이터 이미지·학습 가중치는 커밋 대상에 포함하지 않는다.
