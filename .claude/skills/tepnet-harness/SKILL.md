---
name: tepnet-harness
description: TEP-Net 프로젝트의 오케스트레이터. 학습·추론·평가·코드개발 작업을 전문 에이전트 팀(experiment-runner, inference-runner, model-evaluator, code-developer)으로 분배·조율한다. TEP-Net 관련 작업(학습/추론/평가/코드 변경)이나 여러 단계가 얽힌 ML 워크플로우 요청 시 사용. "다시 실행", "재실행", "업데이트", "보완", "이전 결과 기반으로", "파이프라인 돌려줘" 같은 후속 요청에도 사용. 단순 단일 질문은 직접 응답 가능.
---

# tepnet-harness — TEP-Net 오케스트레이터

TEP-Net(철도 ego-path 탐지) 작업을 전문 에이전트 팀으로 조율한다. "누가 언제 어떤 순서로 협업하는가"를 정의한다.

**실행 모드:** 에이전트 팀(기본). 모든 Agent 호출은 `model: "opus"`.

## 팀 구성

| 에이전트 | 역할 | 스킬 |
|---|---|---|
| `experiment-runner` | 학습 실행·config·W&B | `run-training` |
| `inference-runner` | 추론·crop·비디오·GUI·시각화 | `run-inference` |
| `model-evaluator` | IoU·latency 평가·회귀 검증 | `evaluate-models` |
| `code-developer` | src/ 코드 변경·경계면 QA | `tepnet-development` |

> 팀 크기 가이드: 한 요청에 보통 1~2개 에이전트면 충분하다. 단일 작업이면 해당 에이전트만 서브로 호출하고, 2단계 이상 의존(예: 코드변경→학습→평가)일 때 팀으로 묶는다.

## Phase 0: 컨텍스트 확인 (후속 작업 판별)

워크플로우 시작 시 `_workspace/` 존재로 실행 모드를 결정한다:
- `_workspace/` 존재 + 사용자가 부분 수정 요청 → **부분 재실행**(해당 에이전트만 재호출)
- `_workspace/` 존재 + 새 입력 제공 → **새 실행**(기존 `_workspace/`를 `_workspace_prev/`로 이동)
- `_workspace/` 미존재 → **초기 실행**

## Phase 1: 요청 분류 → 작업 라우팅

사용자 요청을 다음으로 분류해 담당 에이전트를 정한다:

| 요청 유형 | 1차 담당 | 후속 연계 |
|---|---|---|
| 학습/실험/config | experiment-runner | → model-evaluator(평가) |
| 추론/탐지/비디오/GUI | inference-runner | (이상 시) → model-evaluator |
| 평가/IoU/비교/벤치마크 | model-evaluator | (퇴행 시) → 변경 주체 |
| 코드 수정/버그/기능 | code-developer | → 영향 따라 train/detect/eval |

## Phase 2: 실행

**단일 작업(서브 에이전트):**
```
Agent(subagent_type="<해당 에이전트>", model="opus", prompt=<작업+해당 스킬 사용 지시>)
```

**다단계 의존(에이전트 팀):** 대표 흐름 — 코드 변경 후 검증 파이프라인
```
TeamCreate(team, [code-developer, experiment-runner, model-evaluator])
TaskCreate(코드 변경 → (필요 시)재학습 → 회귀 평가, 의존성 명시)
팀원이 SendMessage로 자체 조율:
  code-developer → (모델 영향?) experiment-runner → model-evaluator
결과 수집·종합 → 팀 정리
```

## Phase 3: 데이터 전달 프로토콜

- **태스크 기반**(조율): `TaskCreate`/`TaskUpdate`로 의존·진행 관리.
- **메시지 기반**(소통): 에이전트 간 `SendMessage`로 경로·이상·요청 전달.
- **파일 기반**(산출물): `_workspace/{phase}_{agent}_{artifact}.{ext}`에 중간 산출물 저장. 최종 산출물(학습 weights, `output/` 결과, `output/eval.csv`)은 프로젝트 표준 경로 유지. `_workspace/`는 사후 감사용으로 보존.

## 에러 핸들링

- 실패 시 1회 재시도. 재실패면 해당 결과 없이 진행하되 **보고서에 누락 명시**.
- 상충/불완전 데이터(예: 일부 모델 평가 실패)는 삭제하지 않고 출처와 함께 병기.
- 장시간/고비용 작업(전체 학습, latency 평가)은 실행 전 사용자에게 비용 고지 후 진행.

## 테스트 시나리오

**정상 흐름 — "resnet18 regression 모델 학습하고 성능 평가해줘"**
1. Phase 0: `_workspace/` 없음 → 초기 실행.
2. Phase 1: 학습 → experiment-runner, 후속 평가 → model-evaluator. 2단계 의존 → 팀 구성.
3. experiment-runner: device·데이터 경로·W&B 점검 → `python train.py regression resnet18 --device <dev>` → `weights/{run}/` 생성 → 경로를 model-evaluator에 전달.
4. model-evaluator: `eval.py`로 IoU 측정 → baseline 대비 보고.
5. 종합 보고 + 피드백 요청.

**에러 흐름 — 데이터 경로 미설정으로 학습 실패**
1. experiment-runner가 실행 전 `images_path`/`annotations_path` 부재 감지.
2. 학습을 시작하지 않고 경로 설정 방법(README 다운로드 절차) 안내.
3. 사용자 경로 제공 후 재개. 끝까지 미해결이면 누락 명시하고 평가 단계 생략.

## Phase 4: 진화 (실행 후)

작업 완료 후 개선점·팀 구성 피드백을 1회 요청한다(강요하지 않음). 피드백 유형별 반영:
- 결과 품질 → 해당 스킬 수정 / 에이전트 역할 → 에이전트 `.md` / 워크플로우 순서 → 이 오케스트레이터 / 트리거 누락 → description 확장.
- 모든 변경은 `CLAUDE.md`의 **변경 이력** 테이블에 기록한다.
