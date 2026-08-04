"""검수 큐 정리 발표 그림 — 왜 844프레임이 417로 줄었는가.

큐가 왜 컸는지, 무엇을 빼도 되는지, 빼고 나서 무엇을 잃지 않았는지를 각각 한 장의
그림으로 만든다. 표로는 "약한 모델이 침묵해서 큐가 부풀었다"가 잘 전달되지 않는다 —
침묵률과 오차율을 같은 평면에 놓으면 두 모델이 다른 다섯 개와 떨어져 있는 것이
바로 보이고, 그것이 이 작업의 근거 전부다.

모든 수치는 실측이다. 침묵률은 캐시된 3981프레임, 오차율은 손라벨 2268프레임,
검출력은 손라벨이 있는 분기 이벤트 6개(486프레임)에서 측정했다.

  python experiments/make_fig_review_queue.py
"""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt, font_manager, rcParams
from matplotlib.patches import Patch

FP = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
font_manager.fontManager.addfont(FP)
rcParams["font.family"] = font_manager.FontProperties(fname=FP).get_name()
rcParams["axes.unicode_minus"] = False
rcParams["figure.dpi"] = 200

REPO = "/data3/bhkim/workspace/train-ego-path-detection"
WT = os.path.join(REPO, ".claude/worktrees/prior-channel")
SW = "/data3/bhkim/datasets/Rail_switch_crawling/switch_events"
OUT = os.path.join(REPO, "figures")

INK = "#1a1a1a"
MUTED = "#8a8a8a"
BAD = "#e53935"       # 뺀 것 / 문제
GOOD = "#2e9e2e"      # 남긴 것 / 개선
BLUE = "#1e88e5"
AMBER = "#f9a825"

MODELS = ["brilliant-horse-15", "chromatic-laughter-5", "fortuitous-goat-12",
          "fortuitous-pig-8", "logical-tree-1", "osdar23-seg-r18", "twinkling-rocket-21"]
DROPPED = {"osdar23-seg-r18", "brilliant-horse-15"}

cache = json.load(open(os.path.join(WT, "output/ensemble_spread_div.json")))
far = json.load(open(os.path.join(WT, "output/ensemble_far.json")))
now = json.load(open(os.path.join(REPO, "output/review_list_ensemble.json")))
old = json.load(open(os.path.join(REPO, "output/review_backup_20260730/review_list_ensemble.json")))


def save(fig, name):
    p = os.path.join(OUT, name)
    fig.savefig(p, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", p)
    return p


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=INK, labelsize=11)


# ---------------------------------------------------------------- 1. 큐 구성 반전
def fig_composition():
    """검수 사유의 구성이 뒤집힌 것을 보인다. 크기보다 이쪽이 본질이다."""
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    before = {"일부 모델이 경로를 못 냄": 719, "실제 모델 불일치": 125, "자동 라벨 없음": 0}
    after = {"일부 모델이 경로를 못 냄": 126, "실제 모델 불일치": 279, "자동 라벨 없음": 12}
    colors = [BAD, GOOD, AMBER]
    for i, (label, d) in enumerate([("이전\n844 프레임", before), ("이후\n417 프레임", after)]):
        left = 0
        for (k, v), c in zip(d.items(), colors):
            if v == 0:
                continue
            ax.barh(i, v, left=left, color=c, height=0.52, edgecolor="white", linewidth=1.5)
            if v > 60:
                ax.text(left + v / 2, i, f"{v}", ha="center", va="center",
                        color="white", fontsize=13, fontweight="bold")
            left += v
        ax.text(-24, i, label, ha="right", va="center", fontsize=12, color=INK)
    ax.annotate("", xy=(430, 0.62), xytext=(430, 0.38),
                arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.4))
    ax.set_yticks([])
    ax.set_ylim(-0.55, 1.55)
    ax.set_xlim(0, 880)
    ax.invert_yaxis()
    ax.set_xlabel("사람이 봐야 하는 프레임 수", fontsize=11, color=INK)
    style(ax)
    ax.legend(handles=[Patch(facecolor=c, label=k) for k, c in
                       zip(before.keys(), colors)],
              loc="lower right", frameon=False, fontsize=10.5)
    ax.set_title("검수 큐의 85%는 프레임이 어려워서가 아니라 약한 모델이 침묵해서 올라와 있었다",
                 fontsize=12.5, color=INK, pad=14, loc="left")
    return save(fig, "fig_review_composition.png")


# ---------------------------------------------------- 2. 원인: 침묵률 x 오차율
def fig_models():
    """두 모델이 다른 다섯과 떨어져 있다는 것 하나만 보이면 되는 그림."""
    # 두 축을 독립으로 둔다: x는 침묵률, y는 "답을 냈을 때" 틀린 비율. 침묵을 오답에도
    # 넣으면 같은 실패를 두 축에 겹쳐 세게 되고, 두 모델이 나빠 보이는 이유가 침묵
    # 하나인지 정확도까지인지 그림이 답하지 못한다.
    silence, wrong = {}, {}
    for m in MODELS:
        silence[m] = 100 * sum(1 for v in cache.values() if m not in v) / len(cache)
        e = [abs(v["pred"][m] - v["gt"]) for v in far.values()
             if v["pred"].get(m) is not None]
        wrong[m] = 100 * sum(1 for x in e if x > 0.05) / len(e)

    # 점이 몰린 왼쪽 아래에서 이름표가 겹치지 않도록 모델마다 방향을 지정한다.
    NUDGE = {"osdar23-seg-r18": (-0.9, 0.5, "right"),
             "brilliant-horse-15": (-0.9, 0.5, "right"),
             "fortuitous-pig-8": (0.8, 0.25, "left"),
             "logical-tree-1": (0.8, 0.3, "left"),
             "fortuitous-goat-12": (0.8, -0.75, "left"),
             "chromatic-laughter-5": (0.8, -0.75, "left"),
             "twinkling-rocket-21": (0.8, 0.25, "left")}

    fig, ax = plt.subplots(figsize=(8.8, 5.4))
    for m in MODELS:
        drop = m in DROPPED
        ax.scatter(silence[m], wrong[m], s=250 if drop else 160,
                   color=BAD if drop else GOOD, zorder=4,
                   edgecolor="white", linewidth=1.6, alpha=0.95)
        dx, dy, ha = NUDGE[m]
        ax.text(silence[m] + dx, wrong[m] + dy, m, fontsize=10.5, va="center",
                color=INK if drop else MUTED, ha=ha,
                fontweight="bold" if drop else "normal")
    ax.axvspan(9, 28, color=BAD, alpha=0.06, zorder=0)
    ax.text(27.2, 1.0, "앙상블에서 제외", fontsize=11.5, color=BAD,
            ha="right", fontweight="bold")
    ax.set_xlabel("경로를 아예 못 내는 비율  (캐시 3981프레임 중, %)", fontsize=11, color=INK)
    ax.set_ylabel("답을 냈을 때 정답과 0.05 이상\n어긋난 비율 (손라벨 기준, %)", fontsize=11, color=INK)
    ax.set_xlim(-2.0, 28)
    ax.set_ylim(0, 18)
    style(ax)
    ax.grid(axis="both", color="#eeeeee", zorder=0)
    ax.set_title("침묵이 잦은 두 모델은 정확도도 가장 나쁘다 — 빼도 잃을 의견이 아니었다",
                 fontsize=12.5, color=INK, pad=14, loc="left")
    return save(fig, "fig_review_models.png")


# --------------------------------------------- 3. 규칙별 큐 크기 대 검출력
def fig_tradeoff():
    """큐만 줄이면 되는 게 아니라는 것 — 다수결 규칙이 왜 탈락했는지."""
    rules = ["현재 규칙\n(7개 전부 필요)", "약한 2개 제외\n(5개 전부 필요)",
             "7개 중 5개만\n있으면 통과", "7개 중 4개만\n있으면 통과"]
    queue = [844, 416, 230, 220]
    recall = [100, 100, 79, 71]
    picked = 1

    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    bars = ax.bar(range(4), queue, width=0.56,
                  color=[GOOD if i == picked else "#d8d8d8" for i in range(4)],
                  zorder=3)
    for i, (b, q) in enumerate(zip(bars, queue)):
        # 검출률 선이 100%로 지나가는 왼쪽 두 칸은 막대 안에 값을 넣어야 겹치지 않는다.
        inside = q > 380
        ax.text(b.get_x() + b.get_width() / 2, q - 48 if inside else q + 22,
                f"{q}", ha="center", fontsize=12.5,
                color="white" if inside and i == picked else INK,
                va="top" if inside else "baseline",
                fontweight="bold" if i == picked else "normal")
    ax.set_xticks(range(4))
    ax.set_xticklabels(rules, fontsize=10.5, color=INK)
    ax.set_ylabel("검수 프레임 수", fontsize=11, color=INK)
    ax.set_ylim(0, 980)
    style(ax)

    ax2 = ax.twinx()
    ax2.plot(range(4), recall, "o-", color=BAD, lw=2.2, ms=9, zorder=4)
    for i, r in enumerate(recall):
        ax2.text(i, r + 3.2, f"{r}%", ha="center", fontsize=11.5,
                 color=BAD, fontweight="bold")
    ax2.set_ylabel("심각한 오류(폭의 0.10 초과) 검출률", fontsize=11, color=BAD)
    ax2.set_ylim(0, 118)
    ax2.tick_params(colors=BAD, labelsize=11)
    for s in ("top", "left"):
        ax2.spines[s].set_visible(False)
    ax2.spines["right"].set_color(BAD)
    ax2.spines["bottom"].set_color(MUTED)

    ax.annotate("채택", xy=(picked, queue[picked] + 120), ha="center",
                fontsize=12, color=GOOD, fontweight="bold")
    ax.set_title("큐를 더 줄이는 규칙은 있었지만, 심각한 오류를 놓쳐서 탈락했다",
                 fontsize=12.5, color=INK, pad=14, loc="left")
    return save(fig, "fig_review_tradeoff.png")


# --------------------------------------------------------- 4. 감소 내역 워터폴
def fig_waterfall():
    """줄어든 428프레임의 절반은 사라진 게 아니라 다른 결정으로 옮겨간 것."""
    steps = [("이전 큐", 844, MUTED), ("약한 모델\n침묵분 제거", -265, GOOD),
             ("이벤트 통째로\n제외 (9건)", -208, BLUE), ("제외됐던 이벤트\n복귀", +45, AMBER),
             ("자동 라벨\n없는 프레임", +1, AMBER), ("현재 큐", 417, GOOD)]
    fig, ax = plt.subplots(figsize=(9.2, 5.0))
    run = 0
    for i, (label, v, c) in enumerate(steps):
        if i == 0 or i == len(steps) - 1:
            ax.bar(i, v, width=0.6, color=c, zorder=3)
            ax.text(i, v + 22, f"{v}", ha="center", fontsize=12.5,
                    color=INK, fontweight="bold")
            run = v
        else:
            bottom = run + v if v < 0 else run
            ax.bar(i, abs(v), bottom=bottom, width=0.6, color=c, zorder=3)
            ax.text(i, max(run, run + v) + 22, f"{v:+d}", ha="center",
                    fontsize=12, color=INK)
            ax.plot([i - 0.72, i + 0.3], [run, run], color=MUTED, lw=0.9,
                    ls=":", zorder=2)
            run += v
    ax.set_xticks(range(len(steps)))
    ax.set_xticklabels([s[0] for s in steps], fontsize=10.5, color=INK)
    ax.set_ylabel("검수 프레임 수", fontsize=11, color=INK)
    ax.set_ylim(0, 980)
    style(ax)
    ax.text(2, 300, "프레임이 맞아진 게 아니라\n'이 이벤트를 쓸 것인가'라는\n별개 판단으로 옮겨간 몫",
            fontsize=10, color=BLUE, ha="center", va="center")
    ax.set_title("줄어든 428프레임의 내역 — 절반은 제거가 아니라 이관이다",
                 fontsize=12.5, color=INK, pad=14, loc="left")
    return save(fig, "fig_review_waterfall.png")


# ------------------------------------------------------- 5. 이벤트 단위 제외
def fig_events():
    """프레임 검수로는 감당이 안 되는 이벤트가 따로 있다는 것."""
    rows = sorted(now, key=lambda r: -r["median_spread"])
    med = [r["median_spread"] for r in rows]
    unrel = [r.get("unreliable", False) for r in rows]
    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    ax.bar(range(len(med)), med, width=0.82,
           color=[BAD if u else GOOD for u in unrel], zorder=3)
    ax.axhline(0.05, color=INK, lw=1.3, ls="--", zorder=4)
    ax.text(len(med) - 1, 0.056, "제외 기준 0.05", ha="right", fontsize=11, color=INK)
    ax.text(13, 0.108, f"{sum(unrel)}개 이벤트\n학습·검수에서 제외",
            fontsize=11.5, color=BAD, fontweight="bold", ha="left")
    ax.set_xlabel("분기 이벤트 (불일치 큰 순)", fontsize=11, color=INK)
    ax.set_ylabel("이벤트 내 모델 불일치 중앙값", fontsize=11, color=INK)
    ax.set_xlim(-1, len(med))
    style(ax)
    ax.set_title("모델들이 이벤트 내내 갈리는 9건은 프레임 검수가 아니라 제외 대상이다",
                 fontsize=12.5, color=INK, pad=14, loc="left")
    return save(fig, "fig_review_events.png")


# ------------------------------------------- 6. 실제 프레임에서의 불일치 예시
def fig_frame():
    """두 종류의 프레임을 나란히 — 무엇을 뺐고, 무엇이 사람에게 남는가.

    모델의 위치를 전부 세로선으로 그리면 일치하는 모델끼리 겹쳐 한 줄로 보인다.
    모델마다 높이를 달리한 눈금으로 찍어야 '다섯이 같은 곳을 가리킨다'와 '다섯이
    갈린다'가 같은 그림 문법으로 구분된다.
    """
    from PIL import Image
    KEPT = [m for m in MODELS if m not in DROPPED]

    full = [k for k, v in far.items() if all(v["pred"].get(m) is not None for m in MODELS)]
    # 왼쪽: 남긴 다섯이 정답 위에 있는데 제외한 둘만 크게 빗나간 프레임. 다섯이 서로
    # 일치하는 것만으로는 부족하다 -- 함께 틀린 프레임도 일치는 하므로, 정답과의
    # 거리로 걸러야 "남긴 다섯은 맞았다"는 설명이 그림과 맞는다.
    def err(k, ms):
        return float(np.mean([abs(far[k]["pred"][m] - far[k]["gt"]) for m in ms]))
    left = max((k for k in full if err(k, KEPT) < 0.01),
               key=lambda k: err(k, DROPPED))
    # 오른쪽: 남긴 다섯끼리 갈리는 프레임 — 이것이 큐에 남아 사람에게 간다.
    right = max(full, key=lambda k: float(np.std([far[k]["pred"][m] for m in KEPT])))

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.5))
    for ax, key, head in [
            (axes[0], left, "제외한 두 모델은 이렇게 틀렸다\n남긴 다섯은 정답에 모여 있다"),
            (axes[1], right, "남긴 다섯이 갈리는 프레임\n이것이 사람에게 남는 417장이다")]:
        ev, name = key.split("/")
        im = Image.open(os.path.join(SW, ev, name)).convert("RGB")
        W, H = im.size
        rec = far[key]
        ax.imshow(im)
        # 두 프레임의 가로세로비가 달라 그대로 두면 제목 높이가 어긋난다. 위쪽으로
        # 붙여야 두 패널을 나란히 읽을 수 있다.
        ax.set_anchor("N")
        ax.axvline(rec["gt"] * W, color=GOOD, lw=2.2, alpha=0.95, zorder=3)
        # 모델마다 높이를 달리한 눈금. 겹쳐도 각각 보인다.
        top, bot = H * 0.30, H * 0.72
        for i, m in enumerate(MODELS):
            y = top + (bot - top) * i / (len(MODELS) - 1)
            drop = m in DROPPED
            ax.plot([rec["pred"][m] * W] * 2, [y - H * 0.028, y + H * 0.028],
                    color=BAD if drop else BLUE, lw=3.2 if drop else 2.6,
                    solid_capstyle="butt", zorder=4, alpha=0.95)
        sp = float(np.std([rec["pred"][m] for m in KEPT]))
        ax.set_title(f"{head}\n{ev.split('__')[0]} / {name} · 남긴 다섯의 불일치 {sp:.3f}",
                     fontsize=11.5, color=INK, pad=10, loc="left")
        ax.set_xticks([])
        ax.set_yticks([])
    axes[1].legend(handles=[
        plt.Line2D([], [], color=GOOD, lw=2.5, label="사람이 그린 정답"),
        plt.Line2D([], [], color=BLUE, lw=2.5, label="남긴 5개 모델"),
        plt.Line2D([], [], color=BAD, lw=3, label="제외한 2개 모델")],
        loc="lower right", fontsize=10, framealpha=0.88)
    fig.subplots_adjust(wspace=0.04)
    return save(fig, "fig_review_frame.png")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    fig_composition()
    fig_models()
    fig_tradeoff()
    fig_waterfall()
    fig_events()
    fig_frame()
