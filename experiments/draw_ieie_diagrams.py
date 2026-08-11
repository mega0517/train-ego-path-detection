"""IEIE 덱에 넣을 블록 다이어그램을 직접 그린다.

지금까지는 fig_arch_pipeline.png 한 장을 조각내 썼다. 조각은 상자 사이를 지나가야
하는데 글자를 관통하기 일쑤였고, 잘라 키운 자리는 원본 해상도에 묶여 거칠었다.
그려서 만들면 그 두 문제가 같이 사라진다 — 필요한 만큼만 담고, 어느 크기로 넣어도
선이 뭉개지지 않는다.

색은 KRRI 팔레트에서만 고른다(스킬 references/color-tokens.md). 슬라이드가 흰
바탕이므로 배경도 흰색으로 두고, 강조 블록에만 Blue/Primary를 채운다.

  python experiments/draw_ieie_diagrams.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = "/data3/bhkim/workspace/train-ego-path-detection/figures/ieie_dia"

NAVY = "#002060"        # 글자
BRAND = "#124378"       # 상자 테두리
FILL = "#F1F5F9"        # 상자 채움
BLUE = "#066DB6"        # 강조 블록
SKY = "#0F8DDB"         # 되먹임 화살표
GRAY = "#7A7A7A"        # 연결 화살표
FONT = "Malgun Gothic"

plt.rcParams["font.family"] = FONT
plt.rcParams["axes.unicode_minus"] = False


def canvas(w_in, h_in):
    fig = plt.figure(figsize=(w_in, h_in), dpi=300)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100 * h_in / w_in)
    ax.axis("off")
    fig.patch.set_facecolor("white")
    return fig, ax


def box(ax, x, y, w, h, lines, accent=False, fs=10.5):
    """모서리 둥근 상자 하나. accent면 파란 채움에 흰 글씨."""
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0,rounding_size=1.6",
        linewidth=1.1, edgecolor=BLUE if accent else BRAND,
        facecolor=BLUE if accent else FILL, zorder=2))
    ax.text(x + w / 2, y + h / 2, "\n".join(lines), ha="center", va="center",
            fontsize=fs, color="white" if accent else NAVY,
            linespacing=1.45, zorder=3)


def arrow(ax, x1, y1, x2, y2, dashed=False, color=None):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=11,
        linewidth=1.1, color=color or (SKY if dashed else GRAY),
        linestyle=(0, (3, 2)) if dashed else "solid",
        shrinkA=0, shrinkB=0, zorder=1))


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, name)
    fig.savefig(p, facecolor="white", edgecolor="none")
    plt.close(fig)
    print("  ", name)
    return p


def process_full():
    """전체 프로세스 — 위가 데이터 구축, 아래가 학습, 사이가 부트스트래핑."""
    fig, ax = canvas(7.6, 5.35)
    H = 100 * 5.35 / 7.6
    top, bot, h = H - 22, H - 58, 15
    xs = [3, 23, 43, 63, 82]
    w = 16
    up = [(["입력 영상", "$I_1 … I_N$"], False),
          (["SAM 2", "비디오 전파"], True),
          (["레일 변환", "행별 경계 · RDP"], False),
          (["레일 라벨", "초안"], False),
          (["검수 · 보정", "점 편집"], True)]
    for (lines, acc), x in zip(up, xs):
        box(ax, x, top, w, h, lines, acc, fs=10.5)
    for a, b in zip(xs, xs[1:]):
        arrow(ax, a + w, top + h / 2, b, top + h / 2)

    dn = [(["학습 데이터", "레일 라벨"], False),
          (["검출 모델", "학습"], True),
          (["자기 경로", "예측"], False)]
    dxs = [23, 48, 73]
    for (lines, acc), x in zip(dn, dxs):
        box(ax, x, bot, w, h, lines, acc, fs=10.5)
    for a, b in zip(dxs, dxs[1:]):
        arrow(ax, a + w, bot + h / 2, b, bot + h / 2)

    # 검수 결과가 학습으로 내려가고, 미세조정 모델이 초안으로 되돌아온다.
    arrow(ax, xs[4] + w / 2, top, xs[4] + w / 2, bot + h + 8)
    arrow(ax, xs[4] + w / 2, bot + h + 8, dxs[0] + w, bot + h + 8)
    arrow(ax, dxs[1] + w / 2, bot + h, xs[3] + w / 2, top, dashed=True)
    # 점선 위에 글자를 얹으면 선과 겹쳐 둘 다 읽기 어렵다. 왼쪽으로 빼고
    # 흰 배경을 깔아 되돌아오는 화살표와 분리한다.
    ax.text(26, (bot + h + top) / 2 + 3,
            "부트스트래핑\n미세조정 모델로 재라벨", ha="center", va="center",
            fontsize=9.0, color=SKY, linespacing=1.4,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="none"))
    return save(fig, "process_full.png")


def step_card(name, blocks, caption, accent_idx=None):
    """카드 한 칸에 들어갈 작은 도해. 블록 두셋을 화살표로 잇는다."""
    fig, ax = canvas(3.04, 1.70)
    H = 100 * 1.70 / 3.04
    n = len(blocks)
    w = (92 - 7 * (n - 1)) / n
    y, h = H / 2 - 10, 20
    xs = [4 + i * (w + 7) for i in range(n)]
    for i, (x, lines) in enumerate(zip(xs, blocks)):
        box(ax, x, y, w, h, lines, accent=(i == accent_idx), fs=11.0)
    for a, b in zip(xs, xs[1:]):
        arrow(ax, a + w, y + h / 2, b, y + h / 2)
    if caption:
        ax.text(50, y - 6, caption, ha="center", va="top",
                fontsize=9.0, color=GRAY)
    return save(fig, name)


def small_card(name, blocks, accent_idx=None):
    """슬라이드 5의 네 칸처럼 아주 작은 자리."""
    fig, ax = canvas(2.34, 1.30)
    H = 100 * 1.30 / 2.34
    n = len(blocks)
    w = (90 - 8 * (n - 1)) / n
    y, h = H / 2 - 13, 26
    xs = [5 + i * (w + 8) for i in range(n)]
    for i, (x, lines) in enumerate(zip(xs, blocks)):
        box(ax, x, y, w, h, lines, accent=(i == accent_idx), fs=11.5)
    for a, b in zip(xs, xs[1:]):
        arrow(ax, a + w, y + h / 2, b, y + h / 2)
    return save(fig, name)


if __name__ == "__main__":
    print("그리는 중")
    process_full()

    # 슬라이드 7 — 단계 3의 세 칸
    step_card("conv_grid.png", [["궤도 마스크"], ["48행 격자", "좌우 경계"]],
              "행마다 최좌우 화소를 취한다", accent_idx=1)
    step_card("conv_extend.png", [["경계 점열"], ["하단 연장"]],
              "최하단 점을 영상 아래까지", accent_idx=1)
    step_card("conv_rdp.png", [["48점"], ["RDP", "단순화"], ["6점"]],
              "형상을 보존하며 축약", accent_idx=1)

    # 슬라이드 5 — 네 단계 카드
    small_card("st_prompt.png", [["첫 프레임"], ["궤도 위", "한 점"]], accent_idx=1)
    small_card("st_prop.png", [["$I_1$"], ["SAM 2"], ["$I_N$"]], accent_idx=1)
    small_card("st_conv.png", [["마스크"], ["레일", "점열"]], accent_idx=1)
    small_card("st_review.png", [["초안"], ["점 편집"], ["확정"]], accent_idx=1)

    # 슬라이드 4 우측, 슬라이드 8 가운데 — 학습 되먹임
    step_card("loop_train.png",
              [["확정 라벨"], ["모델", "미세조정"], ["전체", "재라벨"]],
              "소량 보정본으로 전체를 다시 만든다", accent_idx=1)
    print("완료:", OUT)
