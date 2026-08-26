"""IEIE 포스터 세션 규격 산출물 — 제목 배너 1장 + A4 세로 12장.

대회 준비요령: 보드판 100 x 180 cm, 논문번호는 본부 준비, 그 아래 제목/저자/소속
박스, 그 아래 A4 12장(4열 x 3행). 글자 크기는 제목과 저자 25~30 mm, 각 장의 제목
15~20 mm, 발표 내용 7~10 mm.

mm를 pt로 옮기면(1 mm = 2.835 pt) 각 장 제목 약 46 pt, 본문 약 24 pt다. A4 폭
186 mm에 24 pt 한글은 한 줄에 스물너덧 자뿐이라, 문장을 짧게 끊지 않으면 규정
글자 크기를 지킬 수 없다. 그래서 장마다 요지 하나만 담았다.

  python experiments/make_poster_ieie_a4.py
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

MM = 1 / 25.4
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

A4W, A4H = Inches(210 * MM), Inches(297 * MM)
MG = Inches(12 * MM)
HEAD_H = Inches(26 * MM)
BODY_W = A4W - 2 * MG
BODY_TOP = HEAD_H + Inches(7 * MM)
BODY_BOT = A4H - Inches(10 * MM)

TITLE_PT = 46          # 각 장의 제목 16 mm
BODY_PT = 27           # 발표 내용 9.5 mm (규정 7~10 mm)
CAP_PT = 17


def rect(s, x, y, w, h, fill):
    sh = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, int(x), int(y), int(w), int(h))
    sh.fill.solid()
    sh.fill.fore_color.rgb = fill
    sh.line.fill.background()
    sh.shadow.inherit = False
    return sh


def text(s, x, y, w, h, paras, size=BODY_PT, color=INK, bold=False,
         align=PP_ALIGN.LEFT, spacing=1.3, anchor=MSO_ANCHOR.TOP, space=10):
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


def lines_of(txt, w_emu, size):
    """한글 폭은 글자 크기와 거의 같다. 추정이 한 줄이라도 모자라면 다음 요소가
    위로 올라붙어 겹치므로, 한 줄당 글자 수를 보수적으로 잡는다."""
    per = max(6, int((w_emu / 914400) / (size / 72.0) * 0.98))
    return max(1, -(-len(txt) // per))


def sheet(prs, no, title, accent):
    """A4 한 장의 머리: 어두운 띠에 장 번호와 제목."""
    s = prs.slides.add_slide(prs.slide_layouts[6])
    rect(s, 0, 0, A4W, HEAD_H, INK)
    rect(s, 0, HEAD_H - Inches(2.6 * MM), A4W, Inches(2.6 * MM), accent)
    text(s, MG, Inches(5 * MM), Inches(22 * MM), Inches(10 * MM),
         "%02d" % no, 30, META, True)
    text(s, MG + Inches(20 * MM), Inches(4 * MM), BODY_W - Inches(20 * MM),
         Inches(16 * MM), title, TITLE_PT, WHITE, True, anchor=MSO_ANCHOR.MIDDLE)
    return s


def bullets(s, y, items, size=BODY_PT):
    paras, n = [], 0
    for lead, rest in items:
        run = [("• " + lead, True, INK)] if lead else []
        if rest:
            run.append((("  " if lead else "") + rest, False, SUB))
        paras.append(run)
        n += lines_of("• " + lead + "  " + rest, BODY_W, size)
    h = Inches(size / 72.0 * 1.52) * n + Inches(0.14) * len(items)
    text(s, MG, y, BODY_W, h, paras, size, spacing=1.34, space=12)
    return y + h + Inches(6 * MM)


def figure(s, y, name, caption, max_h=None):
    p = os.path.join(FIG, name)
    with Image.open(p) as im:
        iw, ih = im.size
    w = BODY_W
    h = int(w * ih / iw)
    if max_h and h > max_h:
        h = int(max_h)
        w = int(h * iw / ih)
    x = MG + (BODY_W - w) / 2
    s.shapes.add_picture(p, int(x), int(y), int(w), int(h))
    ch = Inches(CAP_PT / 72.0 * 1.3) * lines_of(caption, BODY_W, CAP_PT)
    text(s, MG, y + h + Inches(3 * MM), BODY_W, ch, caption, CAP_PT, SUB,
         spacing=1.25)
    return y + h + Inches(3 * MM) + ch + Inches(5 * MM)


def dark(s, y, title, paras, size=BODY_PT):
    n = sum(lines_of(str(p if isinstance(p, str) else
                     "".join(t for t, _, _ in p)), BODY_W - Inches(14 * MM), size)
            for p in paras)
    h = Inches(15 * MM) + Inches(size / 72.0 * 1.52) * n + Inches(9 * MM)
    rect(s, MG, y, BODY_W, h, INK)
    text(s, MG + Inches(7 * MM), y + Inches(5 * MM), BODY_W - Inches(14 * MM),
         Inches(9 * MM), title, 28, WHITE, True)
    text(s, MG + Inches(7 * MM), y + Inches(16 * MM), BODY_W - Inches(14 * MM),
         h - Inches(20 * MM), paras, size, WHITE, spacing=1.36, space=8)
    return y + h + Inches(6 * MM)


def table(s, y, rows, col_w, size=22):
    n_r, n_c = len(rows), len(rows[0])
    rh = Inches(13 * MM)
    tbl = s.shapes.add_table(n_r, n_c, int(MG), int(y), int(BODY_W),
                             int(rh * n_r)).table
    tbl.first_row = False
    tbl.horz_banding = False
    tot = sum(col_w)
    for j, cw in enumerate(col_w):
        tbl.columns[j].width = Emu(int(BODY_W * cw / tot))
    for i, row in enumerate(rows):
        tbl.rows[i].height = rh
        for j, v in enumerate(row):
            c = tbl.cell(i, j)
            c.margin_left = c.margin_right = Inches(3 * MM)
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
    return y + rh * n_r + Inches(6 * MM)


# ------------------------------------------------------------------ 제목 배너
def build_banner():
    """보드판 상단 제목 박스. 판 너비에 맞춘 1000 x 210 mm."""
    prs = Presentation()
    W, H = Inches(1000 * MM), Inches(210 * MM)
    prs.slide_width, prs.slide_height = int(W), int(H)
    s = prs.slides.add_slide(prs.slide_layouts[6])
    rect(s, 0, 0, W, H, INK)
    rect(s, 0, H - Inches(6 * MM), W, Inches(6 * MM), TEAL)
    with Image.open(LOGO) as im:
        lw, lh = im.size
    logo_w = Inches(190 * MM)
    logo_h = logo_w * lh / lw
    rect(s, Inches(30 * MM), Inches(20 * MM), logo_w + Inches(10 * MM),
         logo_h + Inches(10 * MM), WHITE)
    s.shapes.add_picture(LOGO, int(Inches(35 * MM)), int(Inches(25 * MM)),
                         int(logo_w), int(logo_h))
    # 제목 25~30 mm -> 80 pt(28 mm)
    text(s, Inches(30 * MM), Inches(62 * MM), W - Inches(60 * MM), Inches(60 * MM),
         "SAM 2 비디오 마스크 전파를 활용한\n철도 자기 경로 학습 데이터 반자동 구축 기법",
         80, WHITE, True, spacing=1.18)
    text(s, Inches(30 * MM), Inches(142 * MM), W - Inches(60 * MM), Inches(16 * MM),
         "Semi-automatic Construction of Railway Ego-Path Training Data "
         "Using SAM 2 Video Mask Propagation", 34, META)
    # 저자 25~30 mm -> 72 pt
    text(s, Inches(30 * MM), Inches(163 * MM), W - Inches(60 * MM), Inches(26 * MM),
         [[("김백현", True, WHITE), ("  ·  ", False, META), ("황현철", True, WHITE),
           ("†", False, SKY), ("      한국철도기술연구원", False, META)]], 72)
    out = os.path.join(REPO, "poster_ieie_banner.pptx")
    prs.save(out)
    print("wrote", out, "(1000 x 210 mm 제목 배너)")


# ------------------------------------------------------------------ A4 12장
def build_sheets():
    prs = Presentation()
    prs.slide_width, prs.slide_height = int(A4W), int(A4H)

    # 01 연구 배경
    s = sheet(prs, 1, "연구 배경", RED)
    y = bullets(s, BODY_TOP, [
        ("라벨링 비용", "프레임마다 좌우 두 곡선을 사람이 그린다"),
        ("도메인 이동", "공개 데이터셋은 일반 간선철도 위주다"),
        ("반복 구축", "새 노선마다 처음부터 다시 라벨링한다"),
    ])
    dark(s, y, "문제", [
        [("자기 경로 검출의 병목은 모델이 아니라 학습 데이터 확보다.",
          True, LIME)]])

    # 02 기존 자동 라벨링의 한계
    s = sheet(prs, 2, "자동 라벨의 한계", RED)
    y = bullets(s, BODY_TOP, [
        ("전제", "학습된 모델이 초안을 만든다"),
        ("한계", "모델이 실패하는 신규 도메인에서는 초안이 성립하지 않는다"),
    ])
    figure(s, y, "fig3_model_fail_tram.jpg",
           "그림 1. 일반 철도로 학습된 검출기를 노면 전차 영상에 적용하면 "
           "자기 궤도를 벗어나 인접 궤도와 주변 차량으로 이탈한다.")

    # 03 제안 기법 개요
    s = sheet(prs, 3, "제안 기법", GREEN)
    y = bullets(s, BODY_TOP, [
        ("착안", "SAM 2는 레일을 학습한 적이 없어도 지정한 영역을 추적한다"),
        # 한 항목만 더한다. lines_of가 19자 단위로 끊어 짧게 써도 두 줄로 잡히고,
        # 두 항목을 더하면 아래 다크 박스가 면 밖으로 0.75 in 밀려난다.
        ("우회", "학습된 모델 대신 파운데이션 모델을 쓴다"),
        ("1단계", "궤도 프롬프트와 비디오 전파"),
        ("2단계", "마스크에서 레일 점열로 변환"),
        ("3단계", "검수 보정과 학습 재라벨 연계"),
    ])
    dark(s, y, "핵심", [
        [("첫 프레임에서 한 번만 지정하면 영상 전체의 라벨 초안이 만들어진다.",
          True, LIME)]])

    # 04 1단계
    s = sheet(prs, 4, "1단계 전파", GREEN)
    bullets(s, BODY_TOP, [
        ("프롬프트", "첫 프레임에서 자기 궤도 위 한 점을 클릭한다"),
        ("자동 시드", "미지정 시 화면 하단 중앙을 쓴다"),
        ("근거", "자기 궤도는 차량 정면 하단에서 시작한다"),
        ("전파", "SAM 2가 이후 모든 프레임으로 마스크를 옮긴다"),
        ("이점", "모델 내부 메모리로 추적해 모션 추정이 필요 없다"),
    ])

    # 05 2단계
    s = sheet(prs, 5, "2단계 변환", GREEN)
    y = bullets(s, BODY_TOP, [
        ("격자", "48행을 따라 행별 최좌우 경계 화소를 취한다"),
        ("연장", "최하단 점을 영상 하단 경계까지 늘린다"),
        ("단순화", "RDP로 형상을 보존하며 점을 줄인다"),
    ])
    dark(s, y, "효과", [
        [("레일당 평균 48점이 6점 내외로 줄어", False, WHITE)],
        [("수동 보정에서 다룰 점이 크게 감소한다.", True, LIME)]])

    # 06 3단계
    s = sheet(prs, 6, "3단계 연계", GREEN)
    bullets(s, BODY_TOP, [
        ("검수", "초안이 편집 화면에 즉시 로드된다"),
        ("보정", "점 단위 이동과 추가, 삭제로 다듬는다"),
        ("학습", "확정 라벨을 그대로 학습 입력으로 쓴다"),
        ("순환", "소량 보정본으로 미세조정한 모델이 나머지 초안을 다시 만든다"),
    ])

    # 07 파이프라인
    s = sheet(prs, 7, "파이프라인 구성", TEAL)
    figure(s, BODY_TOP, "fig_arch_pipeline.png",
           "그림 2. 첫 프레임 프롬프트가 전 프레임으로 전파되어 궤도 마스크가 되고, "
           "레일 변환과 검수를 거쳐 학습 데이터가 된다. 미세조정 모델로 초안을 "
           "재생성하는 부트스트래핑 루프를 지원한다.")

    # 08 실험 설정
    s = sheet(prs, 8, "실험 설정", TEAL)
    y = bullets(s, BODY_TOP, [
        ("대상", "도심 노면 전차 주행 영상 290 프레임"),
        ("해상도", "1920 x 1080"),
        ("비교", "RailSem19로 학습된 검출 모델과 제안 기법"),
    ])
    dark(s, y, "난점", [
        [("레일이 도로에 매립되어 침목과 자갈 같은 문맥 단서가 없다.",
          False, WHITE)],
        [("차선과 횡단보도 등 유사 선형 패턴과 차량 폐색이 많다.", False, WHITE)]])

    # 09 초안 생성 결과
    s = sheet(prs, 9, "초안 생성 결과", TEAL)
    y = bullets(s, BODY_TOP, [
        ("결과", "자동 시드만으로 290 프레임 전체에 전파되었다"),
        ("범위", "곡선 구간을 포함해 궤도 양측을 감쌌다"),
        ("한계", "마스크 경계의 국소적 흔들림으로 일부 프레임은 소폭 보정이 필요하다"),
        ("의의", "전 구간에서 초안이 성립해 기존 모델 기반 접근과 구별된다"),
    ])
    figure(s, y, "fig4_sam2_tram.jpg",
           "그림 3. 처음과 중간, 끝 프레임에서 좌우 레일이 매립 궤도를 "
           "안정적으로 추종한다.")

    # 10 부트스트래핑
    s = sheet(prs, 10, "부트스트래핑", TEAL)
    y = bullets(s, BODY_TOP, [
        ("흐름", "초안 30장 보정, 미세조정, 전체 재라벨"),
    ])
    figure(s, y, "fig3_bootstrap_relabel.jpg",
           "그림 4. 좌측은 미세조정 이전 초안, 우측은 30 프레임 보정본으로 "
           "미세조정한 모델의 재라벨. 지그재그와 이탈이 사라졌다.")

    # 11 정량 결과
    s = sheet(prs, 11, "정량 결과", RED)
    y = table(s, BODY_TOP,
              [["구분", "대상 도메인", "원 도메인"],
               ["미세조정 전", "0.389", "0.975"],
               ["미세조정 후", "0.485", "0.893"]],
              [1.4, 1.0, 1.0])
    y = bullets(s, y, [
        ("대상", "보정에 쓴 30장 기준 IoU가 상승했다"),
        ("원 도메인", "기저망까지 미세조정한 대가로 하락했다"),
    ])
    dark(s, y, "파국적 망각", [
        [("도메인별 모델 분리 또는 기저망 동결로 완화할 수 있다.", True, SKY)]])

    # 12 결론
    s = sheet(prs, 12, "결론", RED)
    y = bullets(s, BODY_TOP, [
        ("기여", "학습 이력이 없는 신규 궤도 환경에서도 동작한다"),
        ("확인", "단일 프롬프트로 전 구간 초안이 생성되었다"),
        ("확인", "부트스트래핑 1회로 초안 품질이 개선되었다"),
        ("향후", "망각을 억제하는 도메인 적응"),
        ("향후", "반복 부트스트래핑의 정량 평가"),
    ])
    dark(s, y, "사사", [
        [("한국철도기술연구원 주요사업 철도 특화 로봇 기반 선로점검 "
          "핵심기술 개발(PK26312A1)", False, META)]], 20)

    out = os.path.join(REPO, "poster_ieie_a4.pptx")
    prs.save(out)
    print("wrote", out, f"(A4 세로 {len(prs.slides._sldIdLst)}장)")


if __name__ == "__main__":
    build_banner()
    build_sheets()
