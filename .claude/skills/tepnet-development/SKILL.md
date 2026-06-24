---
name: tepnet-development
description: TEP-Net 소스코드(src/nn/, src/utils/, configs/) 수정·기능 추가·버그 수정·리팩터링 절차와 컨벤션. 모델/백본/디코더/loss 변경, dataset·trainer·interface·evaluate·visualization 수정, config↔model↔inference 경계면 정합성 검증을 다룬다. 코드 수정·구현·버그·리팩터링·모델 변경·loss·dataset·구조 변경 요청 시 반드시 사용.
---

# tepnet-development — TEP-Net 코드 개발

`code-developer` 에이전트가 소스를 안전하게 변경하기 위한 컨벤션과 검증 절차다.

## 1. 코드 지도

```
configs/
  global.yaml          # 데이터 경로, input_shape, seed, *_prop, batch_size, epochs, lr, scheduler, 증강
  {method}.yaml        # method별 아키텍처/손실 파라미터
src/nn/
  model.py             # RegressionNet, ClassificationNet, SegmentationNet
  backbone.py          # EfficientNetBackbone, ResNetBackbone
  decoder.py           # UNetDecoder (segmentation)
  loss.py              # TrainEgoPathRegressionLoss, CrossEntropyLoss, BinaryDiceLoss
src/utils/
  interface.py         # Detector — 추론 핵심 계약
  trainer.py           # train() 학습 루프
  dataset.py           # PathsDataset
  evaluate.py          # IoUEvaluator, LatencyEvaluator
  autocrop.py / postprocessing.py / visualization.py / common.py
```
진입점: `train.py`, `detect.py`, `eval.py`, `demo.py`, `gui_app.py`.

## 2. 보존해야 할 계약 (이유: 깨지면 진입점 전체가 실패)

- **`Detector(model_path, crop_coords, runtime, device)`** — 추론 인터페이스. 명확한 필요 없이 시그니처 변경 금지. 변경 시 `detect.py`, `demo.py`, `gui_app.py`, `IoUEvaluator`/`LatencyEvaluator` 사용처를 모두 갱신.
- **모델 생성자 ↔ config 키** — `train.py`는 머지된 config의 키(`anchors`, `pool_channels`, `fc_hidden_size`, `classes`, `decoder_channels`, `input_shape`, `pretrained` 등)를 모델 생성자에 그대로 넘긴다. 생성자 인자를 바꾸면 `configs/*.yaml`와 `train.py` 주입부를 함께 바꾼다.
- **method 분기 일관성** — method를 추가/변경하면 `train.py`의 모델 선택 분기, loss 선택 분기, `configs/{method}.yaml`을 모두 갱신.
- **재현성** — `torch.use_deterministic_algorithms(True)`, seed 설정, `CUBLAS_WORKSPACE_CONFIG`를 깨지 않는다.

## 3. 경계면 교차 검증 (QA 핵심)

ML 파이프라인 버그는 모듈 "경계"에서 난다. 변경 후 아래 경계를 **양쪽 동시에 Read**해 shape/키/타입 일관성을 확인한다:

| # | 경계 | 확인 |
|---|------|------|
| 1 | config 키 ↔ `train.py`/모델 생성자 | 키 이름·존재 일치 |
| 2 | 모델 출력 ↔ loss 입력 | 텐서 shape·타입 |
| 3 | `Detector.detect()` 출력 ↔ `draw_egopath`/postprocessing 입력 | 좌표/포맷 |
| 4 | `PathsDataset` 산출 ↔ DataLoader ↔ 모델 입력 | 배치 shape, method별 타깃 |

검증은 "존재 확인"이 아니라 **양쪽을 함께 읽고 대조**하는 것이다.

## 4. 변경 원칙

- 최소 변경. 주변 코드의 스타일·명명·주석 밀도에 맞춘다. 무관한 리팩터링 금지.
- method별 로직은 `src/nn/`·`configs/`에 분리 유지(진입점에 분기 누적 금지).
- 데이터/가중치 경로 보존.

## 5. 변경 후 후속 검증 연계

- 모델/config 변경 → `experiment-runner`에 재학습 필요 여부 통지.
- 추론 경로/`Detector` 변경 → `inference-runner`에 샘플 재추론 요청.
- 성능 영향 가능 변경 → `model-evaluator`에 회귀 검증 요청(변경 전 baseline 대비).

## 6. 보고 형식

변경 파일 + diff 요약 → 영향 경계면과 교차검증 결과 → 권고 후속 검증(train/detect/eval) → 리스크/롤백 옵션.
