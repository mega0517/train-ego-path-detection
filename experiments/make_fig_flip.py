"""Figure: the branch flip is a discrete event, and averaging cannot repair it."""
import json
import numpy as np
from matplotlib import pyplot as plt, font_manager, rcParams

FP = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
font_manager.fontManager.addfont(FP)
rcParams["font.family"] = font_manager.FontProperties(fname=FP).get_name()
rcParams["axes.unicode_minus"] = False

S = json.load(open("/home/bhkim/.claude/jobs/7a82c532/tmp/series035.json"))
fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 3.6), width_ratios=[1.55, 1])

STYLE = {
    "seg (분할)":       dict(color="#c0392b", lw=2.0, zorder=3),
    "reg (회귀)":       dict(color="#2c3e50", lw=1.4, ls="--"),
    "reg + ema0.5":     dict(color="#2980b9", lw=1.6),
    "seg, 크롭 전 원본": dict(color="#7f8c8d", lw=1.4, ls=":"),
}
for tag, vals in S.items():
    y = [np.nan if v is None else v for v in vals]
    ax.plot(range(1, len(y) + 1), y, label=tag, **STYLE[tag])

ax.axvspan(39, 47, color="#f1c40f", alpha=0.18, lw=0)
ax.text(43, 0.345, "오분기 구간\n#39–#47 (약 2.3초)", ha="center", va="bottom",
        fontsize=8.5, color="#8a6d00")
ax.set_xlabel("프레임 (4 fps)")
ax.set_ylabel("원거리 경로 중심 (화면폭 대비)")
ax.set_title("(a) 분기 통과 시 원거리 진로 위치", fontsize=10.5)
ax.set_xlim(20, 60)
ax.legend(fontsize=8, loc="lower left", framealpha=0.9)
ax.grid(alpha=0.25)

seg = [np.nan if v is None else v for v in S["seg (분할)"]]
d = np.abs(np.diff(seg))
d = d[~np.isnan(d)]
ax2.hist(d, bins=np.linspace(0, 0.19, 40), color="#95a5a6", edgecolor="none")
med = np.median(d)
ax2.axvline(med, color="#2c3e50", lw=1.4)
ax2.text(med * 1.5, ax2.get_ylim()[1] * 0.82, f"중앙값\n{med:.4f}",
         fontsize=8.5, color="#2c3e50")
for x, lab in ((0.054, "#38→#39\n13배"), (0.178, "#47→#48\n44배")):
    ax2.axvline(x, color="#c0392b", lw=1.4)
    ax2.text(x, ax2.get_ylim()[1] * 0.55, lab, fontsize=8.5, color="#c0392b",
             ha="right" if x > 0.1 else "left")
ax2.set_xlabel("프레임 간 이동량 |Δ| (화면폭 대비)")
ax2.set_ylabel("전이 수")
ax2.set_title("(b) 이동량 분포: 실패는 이산 사건이다", fontsize=10.5)
ax2.set_yscale("log")

fig.tight_layout()
fig.savefig("figures/fig_flip.png", dpi=200)
print("wrote figures/fig_flip.png")
