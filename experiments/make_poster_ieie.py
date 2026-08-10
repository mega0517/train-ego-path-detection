"""IEIE 제출본 포스터 (A0 세로 841 x 1189 mm) — KRRI 서식 승계.

포스터는 16:9 슬라이드와 읽는 방식이 다르다. 청중이 1~2 m 앞에서 서서 훑고,
발표자는 옆에 서서 짚어 준다. 그래서 (1) 본문 최소 28 pt, (2) 한 단을 위에서
아래로 읽으면 논리가 끝나도록 3단 구성, (3) 결론과 수치는 색 반전 박스로 멀리서도
찾히게 두었다.

레이아웃 토큰(색·서체)은 krri_template.pptx에서 실측한 값을 그대로 쓴다.
템플릿의 66개 레이아웃은 13.33x7.5in 전용이라 포스터에는 적용되지 않는다.

  python experiments/make_poster_ieie.py [out.pptx]
"""
import os
import sys

from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Emu, Inches, Pt

REPO = "/data3/bhkim/workspace/train-ego-path-detection"
FIG = os.path.join(REPO, "figures")
LOGO = os.path.join(REPO, "static", "krri_logo.png")
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(REPO, "poster_ieie.pptx")

MM = 1 / 25.4
W, H = Inches(841 * MM), Inches(1189 * MM)          # A0 세로
FONT = "맑은 고딕"

INK = RGBColor(0x1B, 0x26, 0x2C)
SUB = RGBColor(0x47, 0x50, 0x5C)
PANEL = RGBColor(0xF2, 0xF5, 0xF7)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
META = RGBColor(0xB0, 0xBE, 0xC5)
RED = RGBColor(0xC6, 0x28, 0x28)
GREEN = RGBColor(0x2E, 0x7D, 0x32)
TEAL = RGBColor(0x00, 0x83, 0x8F)
LIME = RGBColor(0x9C, 0xCC, 0x65)
SKY = RGBColor(0x4F, 0xC3, 0xF7)

MARGIN = Inches(1.15)
HEAD_H = Inches(5.6)
FOOT_H = Inches(1.7)
GAP = Inches(0.75)
COL_W = (W - 2 * MARGIN - 2 * GAP) / 3
COL_X = [MARGIN + i * (COL_W + GAP) for i in range(3)]
BODY_TOP = HEAD_H + Inches(0.75)


def rect(s, x, y, w, h, fill, line=None):
    sh = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, int(x), int(y), int(w), int(h))
    sh.fill.solid()
    sh.fill.fore_color.rgb = fill
    if line:
        sh.line.color.rgb = line
        sh.line.width = Pt(2)
    else:
        sh.line.fill.background()
    sh.shadow.inherit = False
    return sh


def text(s, x, y, w, h, paras, size=28, color=INK, bold=False,
         align=PP_ALIGN.LEFT, spacing=1.25, anchor=MSO_ANCHOR.TOP, space=8):
    tb = s.shapes.add_textbox(int(x), int(y), int(w), int(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    items = [paras] if isinstance(paras, str) else paras
    for i, p in enumerate(items):
        par = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        par.alignment = align
        par.line_spacing = spacing
        if i:
            par.space_before = Pt(space)
        for piece in ([(p, bold, color)] if isinstance(p, str) else p):
            t, b, c = piece
            r = par.add_run()
            r.text = t
            r.font.name = FONT
            r.font.size = Pt(size)
            r.font.bold = b
            r.font.color.rgb = c
    return tb


def wrapped_lines(txt, w_emu, size):
    """한글은 글자 폭이 글자 크기와 거의 같다. 이 근사로 줄 수를 세지 않으면
    본문이 상자를 넘어 아래 그림과 겹친다(첫 조판에서 실제로 겹쳤다)."""
    per_line = max(8, int((w_emu / 914400) / (size / 72.0) * 1.12))
    return max(1, -(-len(txt) // per_line))


def section(s, x, y, w, title, accent):
    """절 제목: 색 바 + 제목. 반환값은 다음 요소의 y."""
    rect(s, x, y, Inches(0.34), Inches(1.05), accent)
    text(s, x + Inches(0.6), y + Inches(0.05), w - Inches(0.6), Inches(1.0),
         title, 48, INK, True)
    return y + Inches(1.5)


def body(s, x, y, w, items, size=33):
    """불릿 문단. 반환값은 소비한 높이를 더한 y."""
    paras, n_lines = [], 0
    for lead, rest in items:
        run = [("• " + lead, True, INK)] if lead else []
        if rest:
            run.append((("  " if lead else "") + rest, False, SUB))
        paras.append(run)
        n_lines += wrapped_lines("• " + lead + "  " + rest, w, size)
    h = Inches(size / 72.0 * 1.30) * n_lines + Inches(0.22) * len(items)
    text(s, x, y, w, h, paras, size, spacing=1.30, space=16)
    return y + h + Inches(0.45)


def figure(s, x, y, w, name, caption):
    p = os.path.join(FIG, name)
    with Image.open(p) as im:
        iw, ih = im.size
    h = int(w * ih / iw)
    rect(s, x - Inches(0.12), y - Inches(0.12), w + Inches(0.24),
         h + Inches(0.24) + Inches(0.95), PANEL)
    s.shapes.add_picture(p, int(x), int(y), int(w), h)
    text(s, x, y + h + Inches(0.18), w, Inches(0.7), caption, 21, SUB,
         align=PP_ALIGN.LEFT, spacing=1.18)
    return y + h + Inches(1.15)


def dark_box(s, x, y, w, title, paras, size=29):
    n = sum(1 + len(str(p)) // 40 for p in paras)
    h = Inches(1.15) + Inches(0.55) * n + Inches(0.45)
    rect(s, x, y, w, h, INK)
    text(s, x + Inches(0.45), y + Inches(0.35), w - Inches(0.9), Inches(0.7),
         title, 32, WHITE, True)
    text(s, x + Inches(0.45), y + Inches(1.25), w - Inches(0.9), h - Inches(1.6),
         paras, size, WHITE, spacing=1.3, space=10)
    return y + h + Inches(0.5)


def table(s, x, y, w, rows, col_w, size=25):
    n_r, n_c = len(rows), len(rows[0])
    row_h = Inches(0.85)
    shp = s.shapes.add_table(n_r, n_c, int(x), int(y), int(w), int(row_h * n_r))
    tbl = shp.table
    tbl.first_row = False
    tbl.horz_banding = False
    tot = sum(col_w)
    for j, cw in enumerate(col_w):
        tbl.columns[j].width = Emu(int(w * cw / tot))
    for i, row in enumerate(rows):
        tbl.rows[i].height = row_h
        for j, v in enumerate(row):
            c = tbl.cell(i, j)
            c.margin_left = c.margin_right = Inches(0.15)
            c.margin_top = c.margin_bottom = 0
            c.vertical_anchor = MSO_ANCHOR.MIDDLE
            c.fill.solid()
            c.fill.fore_color.rgb = INK if i == 0 else (WHITE if i % 2 else PANEL)
            p = c.text_frame.paragraphs[0]
            p.alignment = PP_ALIGN.LEFT if j == 0 else PP_ALIGN.CENTER
            r = p.add_run()
            r.text = str(v)
            r.font.name = FONT
            r.font.size = Pt(size)
            r.font.bold = (i == 0)
            r.font.color.rgb = WHITE if i == 0 else INK
    return y + row_h * n_r + Inches(0.5)


def build():
    prs = Presentation()
    prs.slide_width, prs.slide_height = int(W), int(H)
    s = prs.slides.add_slide(prs.slide_layouts[6])

    # ---------------- 헤더
    rect(s, 0, 0, W, HEAD_H, INK)
    with Image.open(LOGO) as im:
        lw, lh = im.size
    logo_w = Inches(7.2)
    logo_h = logo_w * lh / lw
    rect(s, MARGIN, Inches(0.55), logo_w + Inches(0.34), logo_h + Inches(0.34), WHITE)
    s.shapes.add_picture(LOGO, int(MARGIN + Inches(0.17)), int(Inches(0.72)),
                         int(logo_w), int(logo_h))
    text(s, MARGIN, Inches(1.95), W - 2 * MARGIN, Inches(1.9),
         "SAM 2 비디오 마스크 전파를 활용한\n철도 자기 경로 학습 데이터 반자동 구축 기법",
         66, WHITE, True, spacing=1.16)
    text(s, MARGIN, Inches(4.05), W - 2 * MARGIN, Inches(0.6),
         "Semi-automatic Construction of Railway Ego-Path Training Data "
         "Using SAM 2 Video Mask Propagation", 30, META)
    text(s, MARGIN, Inches(4.78), W - 2 * MARGIN, Inches(0.6),
         [[("김백현", True, WHITE), ("  ·  ", False, META), ("황현철", True, WHITE),
           ("†", False, SKY), ("      한국철도기술연구원", False, META)]], 30)
    text(s, W - MARGIN - Inches(11), Inches(4.78), Inches(11), Inches(0.6),
         "대한전자공학회 2026 전자·반도체·인공지능 학술대회", 26, META,
         align=PP_ALIGN.RIGHT)

    # ---------------- 1단
    x, y = COL_X[0], BODY_TOP
    rect(s, x, y, COL_W, Inches(6.5), PANEL)
    text(s, x + Inches(0.5), y + Inches(0.4), COL_W - Inches(1.0), Inches(0.6),
         "요약", 32, TEAL, True)
    text(s, x + Inches(0.5), y + Inches(1.25), COL_W - Inches(1.0), Inches(5.0),
         "첫 프레임에서 궤도를 한 번만 지정하면 영상 전체의 레일 라벨 초안이 "
         "만들어지는 반자동 학습 데이터 구축 기법을 제안한다. 기존 검출 모델이 "
         "동작하지 않는 도심 노면 전차 영상에서도 단일 프롬프트만으로 사용 가능한 "
         "초안이 생성됨을 확인하였다.", 27, SUB, spacing=1.32)
    y += Inches(7.1)

    y = section(s, x, y, COL_W, "1. 서론", RED)
    y = body(s, x, y, COL_W, [
        ("라벨링 비용", "프레임마다 좌우 두 곡선을 사람이 그려야 해 비디오 단위 구축 비용이 크다"),
        ("도메인 이동", "공개 데이터셋은 일반 간선철도 위주여서 궤도가 노면에 매립된 환경에는 적용되지 않는다"),
        ("자동 라벨의 한계", "학습된 모델로 초안을 만드는 방식은 정작 모델이 실패하는 신규 도메인에서 성립하지 않는다"),
    ])
    y = dark_box(s, x, y, COL_W, "본 연구의 접근", [
        [("파운데이션 모델로 우회한다.", True, LIME)],
        [("SAM 2는 레일을 학습한 적이 없어도 사용자가 지정한 궤도 영역을 "
          "시퀀스 전반에서 추적한다.", False, WHITE)],
    ])
    figure(s, x, y, COL_W, "fig3_model_fail_tram.jpg",
           "그림 1. 도메인 이동 하에서 기존 검출 모델의 실패. 일반 철도로 학습된 "
           "검출기가 자기 궤도를 벗어나 인접 궤도와 주변 차량으로 이탈한다.")

    # ---------------- 2단
    x, y = COL_X[1], BODY_TOP
    y = section(s, x, y, COL_W, "2. 제안 기법", GREEN)
    y = body(s, x, y, COL_W, [
        ("궤도 프롬프트와 전파", "첫 프레임에서 자기 궤도 위 한 점을 클릭하면 SAM 2 비디오 예측기가 "
         "이후 모든 프레임으로 마스크를 전파한다"),
        ("자동 시드", "미지정 시 화면 하단 중앙을 시드로 쓴다. 자기 궤도는 차량 정면 하단에서 시작한다"),
        ("마스크에서 레일로", "48행 격자를 따라 행별 최좌우 경계 화소를 취해 좌우 레일 점열을 얻는다"),
        ("폴리라인 단순화", "RDP 단순화로 레일당 평균 48점이 6점 내외로 줄어 보정 부담이 감소한다"),
        ("검수와 학습 연계", "확정 라벨이 검출 모델의 학습 입력으로 바로 쓰이고 재라벨로 순환한다"),
    ])
    figure(s, x, y, COL_W, "fig_arch_pipeline.png",
           "그림 2. 제안 파이프라인의 전체 구성. 첫 프레임 프롬프트가 전 프레임으로 "
           "전파되어 궤도 마스크가 되고, 레일 변환과 검수를 거쳐 학습 데이터가 된다.")

    # ---------------- 3단
    x, y = COL_X[2], BODY_TOP
    y = section(s, x, y, COL_W, "3. 실험", TEAL)
    y = body(s, x, y, COL_W, [
        ("대상", "RailSem19로 학습된 검출 모델과 제안 기법을 도심 노면 전차 영상 290 프레임에 적용"),
        ("난점", "레일이 도로에 매립되어 침목과 자갈 등 학습된 문맥 단서가 없고 차량 폐색이 많다"),
        ("초안 생성", "첫 프레임 자동 시드만으로 290 프레임 전체에 전파되어 곡선 구간까지 궤도를 감쌌다"),
    ])
    y = figure(s, x, y, COL_W, "fig4_sam2_tram.jpg",
               "그림 3. 노면 전차 영상에 대한 SAM 2 라벨 초안. 처음, 중간, 끝 프레임에서 "
               "좌우 레일이 매립 궤도를 안정적으로 추종한다.")
    y = figure(s, x, y, COL_W, "fig3_bootstrap_relabel.jpg",
               "그림 4. 부트스트래핑 재라벨 비교. 좌측은 미세조정 이전 초안, 우측은 "
               "30 프레임 보정본으로 미세조정한 모델의 재라벨.")
    y = table(s, x, y, COL_W,
              [["구분", "대상 도메인", "원 도메인"],
               ["미세조정 전", "0.389", "0.975"],
               ["미세조정 후", "0.485", "0.893"]],
              [1.5, 1.0, 1.0])
    y = dark_box(s, x, y, COL_W, "4. 결론", [
        [("단일 프롬프트로 전 구간의 사용 가능한 초안이 생성되고, 부트스트래핑 1회로 "
          "대상 도메인 초안 품질이 개선되었다.", False, WHITE)],
        [("다만 기저망까지 미세조정한 대가로 원 도메인 IoU가 하락하는 "
          "파국적 망각이 관찰되었다.", True, SKY)],
        [("향후 망각을 억제하는 도메인 적응과 반복 부트스트래핑의 정량 평가로 확장한다.",
          False, WHITE)],
    ], 27)

    # ---------------- 푸터
    rect(s, 0, H - FOOT_H, W, FOOT_H, PANEL)
    text(s, MARGIN, H - FOOT_H + Inches(0.32), W - 2 * MARGIN, Inches(1.1),
         ["본 연구는 한국철도기술연구원 주요사업 「철도 특화 로봇 기반 선로점검 "
          "핵심기술 개발(PK26312A1)」의 지원으로 수행되었습니다.",
          "[1] T. Laurent, Train Ego-Path Detection on Railway Tracks Using "
          "End-to-End Deep Learning, 2024.   [2] O. Zendel et al., RailSem19, "
          "CVPRW 2019.   [4] N. Ravi et al., SAM 2, 2024.   "
          "[5] D. Douglas and T. Peucker, Cartographica 10(2), 1973."],
         20, SUB, spacing=1.3, space=6)

    prs.save(OUT)
    print("wrote", OUT, f"({W/914400:.2f} x {H/914400:.2f} in = A0 세로)")


if __name__ == "__main__":
    build()
