---
name: evaluate-models
description: TEP-Net 모델 성능을 eval.py로 정량 평가하는 절차. weights/의 모델들을 테스트셋에서 IoU·latency 측정, output/eval.csv 생성·해석, 모델 간 성능 비교, 코드/config 변경 전후 회귀(regression) 검증을 다룬다. 평가·evaluate·eval·IoU·latency·벤치마크·성능 비교·회귀 검증 요청 시 반드시 사용. 학습(train)·추론(detect)과 혼동하지 말 것.
---

# evaluate-models — TEP-Net 모델 평가

`model-evaluator` 에이전트가 모델 성능을 측정·비교하기 위한 절차다.

## 1. 실행

```bash
python eval.py
```
인자가 없다. `eval.py`는 `weights/`의 모든 모델 디렉토리를 자동 순회한다. 평가 대상/메트릭/runtime을 바꾸려면 `eval.py` 상단의 리스트를 편집한다:
- `methods`, `backbones` — 평가에 포함할 method/backbone 필터.
- `runtimes` — `["pytorch", "tensorrt"]`. **TensorRT는 모델에 `best.trt`가 있을 때만 유효**하므로, 없으면 `["pytorch"]`로 한정.
- `metrics` — `["iou", "latency"]`. latency는 모델마다 `time.sleep(30)` cooldown이 있어 느리다. **IoU만 필요하면 `["iou"]`로 한정**해 시간을 크게 절약.

## 2. 동작 원리 (이유 이해)

- 각 모델의 `weights/{model}/config.yaml`에서 method/backbone을 읽는다.
- 테스트셋은 `configs/global.yaml`의 `seed`와 `train/val/test_prop`로 `split_dataset`이 결정한다(IoU는 segmentation 방식으로 통일 측정).
- **공정 비교 전제:** 비교하려는 모델들이 동일 `seed`/`*_prop`로 학습·평가돼야 test 분할이 같다. 불일치 발견 시 경고하고 비교의 한계를 명시.

## 3. 산출물 — output/eval.csv

컬럼: `runtime, backbone, precision, method, model[, latency][, iou]`
- `precision`: tensorrt면 `amx`, 아니면 `fp32`.
- 결과는 정렬되어 저장된다.

## 4. baseline / 회귀 검증

코드·config 변경 후에는 변경 전 수치와 비교해 퇴행을 잡는다. 공개된 참고 baseline(README, PyTorch IoU):

| 모델 | backbone | method | IoU |
|---|---|---|---|
| fortuitous-goat-12 | EfficientNet-B3 | classification | 0.9673 |
| chromatic-laughter-5 | EfficientNet-B3 | regression | 0.9753 |
| twinkling-rocket-21 | EfficientNet-B3 | segmentation | 0.9769 |
| fortuitous-pig-8 | ResNet-18 | classification | 0.9629 |
| logical-tree-1 | ResNet-18 | regression | 0.9695 |
| brilliant-horse-15 | ResNet-18 | segmentation | 0.9737 |

> 단, 위 수치는 원저자 환경/분할 기준이다. 로컬 `seed`/데이터가 다르면 절대값보다 **변경 전후 상대 비교**를 신뢰한다.

## 5. 보고 형식

실행 설정(metrics/runtimes/대상) → `output/eval.csv` 경로 → 핵심 수치 표 → baseline/이전 대비 증감 → 이상치·누락(실패한 모델은 삭제하지 않고 표기) → 다음 단계.
