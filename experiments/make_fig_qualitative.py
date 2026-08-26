"""그림 3: 정성 비교 — 대표 프레임 오버레이 + 예측 경로 위치의 시계열.

프레임 3장만으로는 '흔들림'이 눈에 잘 띄지 않는다. 예측 경로의 대표 위치를
이벤트 전체(81프레임)에 걸쳐 그리면 base가 진동하고 EMA가 매끄럽게 따라가는
차이가 한눈에 드러난다.
"""
import json, os, sys
import numpy as np, cv2
import matplotlib; matplotlib.use("Agg")
from matplotlib import pyplot as plt, font_manager, rcParams
FP="/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
font_manager.fontManager.addfont(FP); rcParams["font.family"]=font_manager.FontProperties(fname=FP).get_name()
rcParams["axes.unicode_minus"]=False

REPO="/data3/bhkim/workspace/train-ego-path-detection"
SW="/data3/bhkim/datasets/Rail_switch_crawling/switch_events"
raw=json.load(open(f"{REPO}/output/switch_stability/rails_raw.json"))
sample={m["event"]:m for m in json.load(open(f"{SW}/eval_sample.json"))}

EV   = sys.argv[1] if len(sys.argv)>1 else "evt003__n0UnTeYC9Us__D_gleiswechsel_0040"
CROP = tuple(float(v) for v in (sys.argv[2].split(",") if len(sys.argv)>2 else "0.30,0.74,0.26,0.74".split(",")))
METHODS=[("base","단일 프레임 (base)","#e53935"),
         ("rnn","학습형 정련 (rnn)","#2e9e2e"),
         ("base-ema0.5","EMA0.5","#1e88e5")]

fr=sorted(raw["results"]["base"][EV])
ci=fr.index(sample[EV]["center_frame"]) if sample[EV]["center_frame"] in fr else 40
ts=np.array([(k-ci)*0.25 for k in range(len(fr))])

def jitter(a,b):
    """연속 두 프레임 예측의 공통 높이 구간 평균 |Δx| — 표 1과 동일한 지표."""
    vals=[]
    for s_ in (0,1):
        p,c=np.array(a[s_],float),np.array(b[s_],float)
        if len(p)<2 or len(c)<2: return None
        lo,hi=max(p[:,1].min(),c[:,1].min()),min(p[:,1].max(),c[:,1].max())
        if hi<=lo: return None
        ys=np.linspace(lo,hi,32); p=p[np.argsort(p[:,1])]; c=c[np.argsort(c[:,1])]
        vals.append(np.abs(np.interp(ys,p[:,1],p[:,0])-np.interp(ys,c[:,1],c[:,0])).mean())
    return float(np.mean(vals))

JIT={mid:[None]+[jitter(raw["results"][mid][EV][fr[i-1]],raw["results"][mid][EV][fr[i]])
                 for i in range(1,len(fr))] for mid,_,_ in METHODS}

fig=plt.figure(figsize=(11.5,4.0))
gs=fig.add_gridspec(1,2,width_ratios=[1.05,1.5],wspace=0.18)

# (a) 대표 프레임 + 세 방법 오버레이
zone=[i for i in range(len(fr)) if 2.5 < (i-ci)*0.25 <= 6.0]
k=max(zone,key=lambda i: JIT["base"][i] or 0)   # 불확실 구간에서 base가 가장 크게 흔들린 프레임
img=cv2.imread(os.path.join(SW,EV,fr[k]))
W={"base":11,"rnn":5,"base-ema0.5":5}
for mid,_,col in METHODS:
    bgr=tuple(int(col[i:i+2],16) for i in (5,3,1))
    for side in (0,1):
        cv2.polylines(img,[np.array(raw["results"][mid][EV][fr[k]][side],np.int32)],
                      False,bgr,W[mid],cv2.LINE_AA)
h,w=img.shape[:2]; y0,y1,x0,x1=CROP
ax=fig.add_subplot(gs[0,0])
ax.imshow(cv2.cvtColor(img[int(y0*h):int(y1*h),int(x0*w):int(x1*w)],cv2.COLOR_BGR2RGB))
ax.set_xticks([]); ax.set_yticks([])
ax.set_title(f"(a) 불확실 구간의 한 프레임 (t = +{(k-ci)*0.25:.2f} s)",fontsize=10.5)

# (b) 프레임 간 흔들림 시계열
ax2=fig.add_subplot(gs[0,1])
ax2.axvspan(2.5,6.0,color="#ffcc80",alpha=.45,zorder=0)
ax2.text(4.25,0.97,"불확실 구간",ha="center",va="top",fontsize=9,color="#8a5a00",
         transform=ax2.get_xaxis_transform())
ax2.axvline(0,color="#c62828",lw=1.6,zorder=1)
STYLE={"base":dict(lw=2.6,ls="-",zorder=3),
       "rnn":dict(lw=1.5,ls="--",zorder=4),
       "base-ema0.5":dict(lw=2.0,ls="-",zorder=5)}
for mid,label,col in METHODS:
    ax2.plot(ts,[np.nan if v is None else v for v in JIT[mid]],color=col,label=label,**STYLE[mid])
ax2.axvline(ts[k],color="#555",lw=1.0,ls=":",zorder=2)
ax2.set_xlabel("분기기 통과 시각 기준 경과 시간 (초)")
ax2.set_ylabel("프레임 간 흔들림 (px)")
ax2.set_xlim(-10,10); ax2.set_ylim(bottom=0); ax2.grid(alpha=.25)
ax2.legend(fontsize=9,loc="upper left")
ax2.set_title("(b) 프레임 간 흔들림의 시간 변화",fontsize=10.5)
plt.tight_layout()
plt.savefig(f"{REPO}/figures/fig_qualitative.png",dpi=190,bbox_inches="tight")
print("wrote fig_qualitative.png", EV, f"t=+{(k-ci)*0.25:.2f}s")
