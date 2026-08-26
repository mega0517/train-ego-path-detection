---
name: run-inference
description: TEP-Net으로 ego-path를 추론하는 절차. detect.py로 이미지/비디오 추론, demo.py 예제, gui_app.py GUI 구동, crop 모드(auto/manual 좌표/none) 선택, 비디오 start/end 구간, --show-crop, device 설정, 결과 시각화 저장을 다룬다. 추론·탐지·detect·비디오 처리·crop·GUI·데모·시각화 요청 시 반드시 사용. 학습(train)·평가(eval)와 혼동하지 말 것.
---

# run-inference — TEP-Net 추론 실행

`inference-runner` 에이전트가 학습된 모델로 ego-path를 탐지·시각화하기 위한 절차다.

## 1. detect.py — 이미지/비디오 추론

```bash
python detect.py {model} {input} --output output --crop {auto|x_left,y_top,x_right,y_bottom|none} [--start S] [--end E] [--show-crop] --device {device}
```

- `model`: `weights/` 하위 디렉토리명 (예: `chromatic-laughter-5`, `logical-tree-1`). 위치 인자, 필수.
- `input`: 입력 파일. 이미지 `.jpg/.jpeg/.png` 또는 비디오 `.mp4/.avi`. 위치 인자, 필수.
- `--output`: 출력 디렉토리(미지정 시 입력과 같은 위치). 출력명은 `{입력명}_out{확장자}`.
- `--crop`: **세 가지 의미를 정확히 구분**
  - `auto` — 자동 크롭(기본값). 이미지+auto는 내부적으로 50회 반복해 크롭을 수렴.
  - `x_left,y_top,x_right,y_bottom` — 수동, **절대좌표, 양끝 포함(inclusive)**. 예: `580,270,1369,1079`.
  - `none` — 크롭 비활성.
- `--start` / `--end`: 비디오 추론 구간(초). 긴 비디오는 구간 한정 권장.
- `--show-crop`: 시각 출력에 크롭 경계 표시.
- `--device`: `cpu` | `cuda` | `cuda:N` | `mps` (기본 `cuda`).

예(이미지): `python detect.py chromatic-laughter-5 sample.jpg --output output --crop auto --device cuda`
예(비디오 구간): `python detect.py logical-tree-1 ride.mp4 --output output --crop 580,270,1369,1079 --start 10 --end 70 --device cuda`

## 2. demo.py — 기본 사용 예제

```bash
python demo.py
```
`Detector` 클래스의 PyTorch 추론 예시. TensorRT 예시는 기본 비활성(파일 내 해당 라인 주석 해제 시 활성). 모델 가중치가 `weights/`에 준비돼 있어야 한다.

## 3. gui_app.py — 대화형 GUI

```bash
./launch_gui.sh   # 또는 launch_gui.bat (Windows), 또는 python gui_app.py
```
PyQt5 GUI에서 파일·모델·device·crop 모드를 선택해 추론. 데스크톱 환경(디스플레이)이 필요하므로 헤드리스 서버에서는 detect.py를 쓴다.

## 4. 핵심 계약 (보존)

`Detector(model_path, crop_coords, runtime, device)`:
- `crop_coords`: 고정 튜플 | `"auto"` | `None`.
- `runtime`: `"pytorch"`(기본) | `"tensorrt"`. TensorRT는 모델에 `best.trt`와 `tensorrt`/`pycuda` 설치가 있어야 유효.
- 출력은 `draw_egopath`로 시각화되어 저장된다.

## 5. 실행 전 점검 / 보고

1. 모델 디렉토리(`weights/{model}`)와 입력 파일 존재 확인.
2. 입력 확장자 지원 여부 확인(.jpg/.jpeg/.png/.mp4/.avi 외는 미지원).
3. `cuda` 불가 시 `cpu`/`mps` 폴백 제안.
보고: 실행 명령 → 출력 경로 → (비디오면) 처리 프레임/진행 → 시각 이상 여부 → 다음 단계.
