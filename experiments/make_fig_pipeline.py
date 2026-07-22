"""Regenerate figures/fig_arch_pipeline.png with larger, block-filling text.

The original PNG had no source script; this recreates the same two-row pipeline
(SAM 2 auto-labeling  ->  TEP-Net training/inference) with enlarged fonts.
"""
import os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

os.chdir("/data3/bhkim/workspace/train-ego-path-detection")
REG = "/usr/local/share/fonts/malgun.ttf"
fm.fontManager.addfont(REG)
KR = fm.FontProperties(fname=REG).get_name()
plt.rcParams["font.family"] = KR
plt.rcParams["axes.unicode_minus"] = False

W, H = 2657, 1337
GREEN, TEAL, ORANGE = "#2e7d32", "#00838f", "#e65100"
BOXFILL, BOXEDGE, GREY = "#eef2f4", "#8a9aa5", "#4a5560"

fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis("off")

# font sizes (pt) — enlarged so text fills the blocks
FS_SEC, FS_CONT, FS_BLK, FS_SUB, FS_NOTE = 35, 31, 28, 24, 26


def box(x, y, w, h, text, *, fill=BOXFILL, edge=BOXEDGE, lw=2.2, fs=FS_BLK,
        bold=False, tcolor="#1a1a1a", rad=18):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                 boxstyle=f"round,pad=0,rounding_size={rad}",
                 fc=fill, ec=edge, lw=lw, mutation_aspect=1))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, color=tcolor, fontweight="bold" if bold else "normal",
            linespacing=1.2)


def container(x, y, w, h, label, edge):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                 boxstyle="round,pad=0,rounding_size=22",
                 fc="none", ec=edge, lw=3))
    ax.text(x + w / 2, y + 34, label, ha="center", va="center",
            fontsize=FS_CONT, color=edge, fontweight="bold")


def arrow(p, q, *, color=GREY, lw=2.6, style="-", rad=0.0, dashed=False):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=26,
                 lw=lw, color=color, shrinkA=2, shrinkB=2,
                 linestyle="--" if dashed else "-",
                 connectionstyle=f"arc3,rad={rad}"))


# ── section titles ──────────────────────────────────────────────
ax.text(24, 40, "① 학습 데이터 자동 구축 (SAM 2 비디오 전파)",
        fontsize=FS_SEC, color=GREEN, fontweight="bold", va="center")
ax.text(24, 792, "② 자기 경로 검출 모델 학습·추론",
        fontsize=FS_SEC, color=TEAL, fontweight="bold", va="center")

# ── ROW 1 ───────────────────────────────────────────────────────
box(24, 150, 250, 220, "입력\n비디오 프레임\n$I_1 … I_N$")
# SAM2 container + inner blocks
container(320, 95, 960, 300, "SAM 2 비디오 예측기", GREEN)
box(348, 175, 268, 195, "이미지\n인코더\n(Hiera)")
box(636, 175, 246, 195, "메모리\n어텐션")
box(902, 175, 356, 195, "마스크\n디코더")
# prompt (orange) + memory bank below
box(300, 430, 568, 150, "프롬프트: 첫 프레임 궤도 클릭\n(미지정 시 하단 중앙 자동 시드)",
    edge=ORANGE, tcolor=ORANGE, fs=FS_SUB, lw=2.4)
box(880, 430, 400, 130, "메모리 뱅크 (과거 프레임)", fs=FS_SUB)
box(1330, 150, 250, 220, "프레임별\n궤도 마스크\n$M_1 … M_N$")
box(1636, 150, 380, 220, "레일 변환\n행별 좌·우 경계 추출\n→ 하단 연장\n→ RDP 단순화")
box(2072, 150, 270, 220, "레일 라벨\n초안", bold=True, edge=GREEN)
box(2398, 150, 235, 220, "검수·보정\n(점 편집)", fill=GREEN, edge=GREEN, tcolor="white", bold=True)

# ── ROW 2 ───────────────────────────────────────────────────────
box(24, 900, 250, 220, "학습 데이터\n(레일 라벨)")
container(320, 840, 960, 300, "자기 경로 검출기 (TEP-Net 계열)", TEAL)
box(348, 920, 300, 195, "백본\nResNet /\nEfficientNet")
box(668, 920, 300, 195, "검출 헤드\n세그멘테이션 ·\n회귀 · 분류")
box(988, 920, 268, 195, "RNN 정련부\n(시간적, 선택)")
box(2072, 900, 270, 220, "자기 경로\n예측\n(좌·우 레일)", bold=True, edge=TEAL)
box(2398, 900, 235, 220, "평가\n프레임별 IoU·\n지연시간\n시간적 안정성", fs=FS_SUB)

# ── arrows: row 1 ───────────────────────────────────────────────
arrow((274, 260), (320, 260))
arrow((616, 272), (636, 272))
arrow((882, 272), (902, 272))
arrow((1280, 260), (1330, 260))
arrow((1580, 260), (1636, 260))
arrow((2016, 260), (2072, 260))
arrow((2342, 260), (2398, 260))
arrow((595, 430), (735, 372), color=ORANGE, lw=2.6)          # prompt -> memory attn
arrow((1080, 370), (1080, 430), color=GREY, dashed=True)      # decoder -> mem bank
arrow((880, 480), (770, 372), color=GREY, rad=-0.3)           # mem bank -> attn (curved)

# ── arrows: bootstrap + row-1→row-2 handoff ─────────────────────
arrow((2515, 370), (2515, 760), color=GREEN, lw=3.0)          # 검수보정 down
arrow((2515, 760), (150, 760), color=GREEN, lw=3.0)           # across
arrow((150, 760), (150, 900), color=GREEN, lw=3.0)            # into 학습데이터
ax.text(1640, 690, "부트스트래핑:\n미세조정 모델로 재라벨",
        ha="center", va="center", fontsize=FS_NOTE, color=ORANGE, fontweight="bold")
arrow((2180, 900), (2200, 372), color=ORANGE, lw=2.8, rad=0.12, dashed=True)  # 예측 -> 레일 라벨 초안

# ── arrows: row 2 ───────────────────────────────────────────────
arrow((274, 1010), (320, 1010))
arrow((648, 1017), (668, 1017))
arrow((968, 1017), (988, 1017))
arrow((1280, 1010), (2072, 1010))
arrow((2342, 1010), (2398, 1010))

OUT = sys.argv[1] if len(sys.argv) > 1 else "figures/fig_arch_pipeline.png"
fig.savefig(OUT, dpi=100, facecolor="white")
print("saved", OUT, f"({W}x{H})")
