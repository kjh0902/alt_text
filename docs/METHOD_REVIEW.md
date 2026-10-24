# 코드 검토와 실험 방법 보존

검토 기준: 기존 main ab52e44 및 2026-10-14 배포 데이터. 변경은 데이터 준비·L2 지침·요청한 이진
보조 손실·제출용 재현 기록에 한정한다.

| 코드 | 검토 내용과 처리 |
|---|---|
| data.py / splits | 기존 1,920/480 membership, 순서와 페이지·이미지 그룹 검사 유지 |
| images.py | 실제 바이트 디코딩·SVG·첫 프레임·흰 배경·PNG iCCP 복구 유지 |
| model.py | Qwen3-VL-8B NF4/BF16, 언어 모듈 QLoRA, 시각 전체 동결, dtype·adapter reload 유지 |
| prompts.py | L1·입력 payload·절삭 유지, L2에 v1.1 기준 추가 |
| runtime.py | L1 캐시 키·검증·재시도·OCR schema·이미지 예산·추론 유지 |
| scoring.py | 일곱 후보 평균 token logprob·class_weights·기존 classification_loss 유지, grouped binary CE 추가 |
| training.py | optimizer·scheduler·seed·배치·누적·clip·epoch·저장/복원 유지. 결합 loss·성분 로그·objective contract 추가 |
| evaluation.py | 7범주 고정 macro-F1·부적절 양성 binary F1·0.5:0.5 종합·제출 순서 유지 |
| smoke.py | 실제 학습과 같은 결합 loss로 GPU 역전파 검사. 기존 reload 허용 오차 유지 |
| cli.py / scripts | 기존 명령 유지. loss 비중과 실행 기록, 오프라인 데이터 갱신 명령 추가 |
| io.py | 기존 JSONL 물리 줄 읽기·atomic 쓰기 유지 |
| add_thumbnail_feature.py | 사용자 원본 판정 알고리즘 유지, portable CLI·출력 보호 추가 |
| tests | 기존 oracle·동결·save/reload·복구 테스트 유지, 신규 loss·병합·분할·train-only 가중치 회귀 검증 추가 |

클래스 가중치는 원래부터 train(args, train_rows, validation_rows, ...)에서 train_rows의 label만
class_weights에 넘겨 매번 계산한다. 과거 분포가 하드코딩되거나 class_weights.json에서 재로딩되지 않는다.
따라서 공식과 해당 경로를 수정하지 않았다. rho=0.99, raw=(1-rho)/(1-rho**n), 평균 1 정규화가 유지된다.

이진 loss는 새로운 분류 헤드나 '부적절' 후보 토큰을 추가하지 않고 기존 확률을 합한다. 모든 후보가
autograd에 연결되고 후보 토큰 평균에서 EOS·prompt·padding을 제외하는 원래 방식도 유지된다.
이진 loss 0.5와 유형 loss 0.5의 조합은 이번에 요청된 학습 목표 변경이다. 점수의 50:50 비중으로부터
이 loss 혼합이 최적이라는 결론을 내리지 않으며, 실제 성능 개선은 새 검증 정답으로 비교해야 한다.

수정된 검증에서 무관형은 2건이다. 전체 480건 성능과 per-class support·혼동행렬을 함께 해석한다.
경계 사례를 이유로 라벨을 바꾸거나 test 정답을 입력하는 코드, 규칙으로 최종 라벨을 덮어쓰는 후처리는 없다.
