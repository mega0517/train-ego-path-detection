---
name: run-training
description: TEP-Net 모델을 train.py로 학습하는 절차. method(regression/classification/segmentation)와 backbone(resnet18/34/50, efficientnet-b0~b3) 선택, configs/global.yaml + configs/{method}.yaml 편집, device 설정, W&B 추적, weights/{run}/ 산출물 관리를 다룬다. 학습·재학습·하이퍼파라미터 조정·실험 설정·config 변경 요청 시 반드시 사용. 추론(detect)·평가(eval)와 혼동하지 말 것.
---

# run-training — TEP-Net 학습 실행

`experiment-runner` 에이전트가 학습 실험을 안전하고 재현 가능하게 수행하기 위한 절차다.

## 1. 명령 구조

```bash
python train.py {method} {backbone} --device {device}
```

- `method`: `regression` | `classification` | `segmentation` (위치 인자, 필수)
- `backbone`: `resnet18` | `resnet34` | `resnet50` | `efficientnet-b0` | `efficientnet-b1` | `efficientnet-b2` | `efficientnet-b3` (위치 인자, 필수)
- `--device`: `cpu` | `cuda` | `cuda:N` | `mps` (기본 `cuda`)

예: `python train.py regression resnet18 --device cuda`

## 2. 실행 전 점검 (이유: 무의미한 장시간 실패 방지)

1. **device 가용성** — `cuda` 요청 시 GPU가 실제로 있는지 확인. 없으면 `cpu`/`mps` 안내(학습은 CPU에서 매우 느림).
2. **데이터 경로** — `configs/global.yaml`의 `images_path`, `annotations_path`가 실재하는지 확인. README의 `configs/training/global.yaml` 경로는 구버전 표기이며 **실제 코드는 `configs/global.yaml`을 읽는다**.
3. **W&B** — `train.py`는 `wandb.login()`을 먼저 호출한다. 로그인/키가 없으면 즉시 실패하므로, 비대화 환경에서는 `WANDB_MODE=offline` 또는 `WANDB_API_KEY`를 안내.
4. **결정성** — `torch.use_deterministic_algorithms(True)`와 `CUBLAS_WORKSPACE_CONFIG`가 설정되어 있다. seed 고정으로 재현성이 보장되므로 임의 변경 금지.

## 3. config 편집 원칙

- 공통 설정은 `configs/global.yaml`(데이터 경로, `input_shape`, `seed`, `train/val/test_prop`, `batch_size`, `epochs`, `learning_rate`, `scheduler`, `workers`, 증강 등).
- method별 설정은 `configs/{method}.yaml`(아키텍처·손실 관련: `anchors`, `pool_channels`, `fc_hidden_size`, `classes`, `decoder_channels`, `ylimit_loss_weight`, `perspective_weight_limit_percentile` 등). `train.py`가 두 파일을 머지해 모델 생성자에 키로 주입하므로, **config 키 이름과 모델 생성자 인자가 일치해야 한다**.
- 수정 전 현재 값을 Read하고, 무엇을 왜 바꾸는지 기록(변경 1건 = 가설 1건).
- 절대 임의 변경 금지 항목: `seed`, `*_prop`, `images_path`, `annotations_path`(사용자 승인 필요).

## 4. 산출물

- 각 학습은 W&B run 이름으로 `weights/{run-name}/` 디렉토리를 만들고 `best.pt` + `config.yaml`을 저장한다.
- `test_prop > 0`이면 학습 후 테스트셋 IoU를 자동 평가해 로그/`wandb.log({"test_iou": ...})`에 남긴다.
- 학습 완료 후 `model-evaluator`에 `weights/{run-name}` 경로를 전달해 정량 평가를 잇는다.

## 5. 보고 형식

실행 명령 → config diff(있으면) → 학습 진행/완료 상태 → 산출 경로 → test IoU(있으면) → 다음 단계 제안. 비용이 큰 전체 학습은 실행 전 사용자에게 예상 시간·자원을 고지하고 승인받는다.
