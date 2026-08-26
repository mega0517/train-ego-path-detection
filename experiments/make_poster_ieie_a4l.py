"""IEIE 포스터 — 제목 배너 1장 + A4 가로 12장, KRRI 색·서체로.

세로판(make_poster_ieie_a4.py)과 내용은 같고 판형·배색·구성이 다르다.

판형. 보드판 100 x 180 cm에 A4 12장을 세로로 붙이면 4열 x 3행(840 x 891 mm),
가로로 붙이면 3열 x 4행(891 x 840 mm)이다. 둘 다 들어간다. 가로판의 이점은 폭
273 mm로, 규정 본문 크기 27 pt에서 한 줄에 스물여덟 자쯤 들어간다는 것이다.
세로판은 스물너덧 자였다.

구성. 열두 장을 논문 목차가 아니라 작업 순서대로 놓았다. 도입 두 장, 전체
프로세스 한 장, 단계 다섯 장, 검증 세 장, 마무리 한 장이다. 장마다 그림을 하나씩
크게 올리고 글은 그 아래 두세 줄로 줄였다 — 서서 읽는 물건이라 그림이 먼저
눈에 들어와야 한다. 원본 그림이 여섯 개뿐이라 파이프라인 그림과 띠 그림을
부분으로 잘라 열두 개로 나눴다(figures/ieie_crops).

배색. KRRI 템플릿(krri_template.pptx)에서 서체와 색을 가져왔다. 템플릿은 16:9라
레이아웃을 그대로 쓸 수 없어 — A4 가로는 1.41:1이다 — 기하는 이 판형에 맞춰
새로 계산하고 색만 팔레트에서 정했다. 세로판의 어두운 박스와 라임 강조는 KRRI
팔레트에 없어, 본문 지배색을 흰색으로 두고 강조는 오프화이트 카드에 파란 악센트
바를 세우는 방식으로 바꿨다.

글자 크기는 대회 준비요령을 따른다. 제목과 저자 25~30 mm, 각 장 제목 15~20 mm,
발표 내용 7~10 mm. mm를 pt로 옮기면(1 mm = 2.835 pt) 장 제목 46 pt, 본문 27 pt다.

  python experiments/make_poster_ieie_a4l.py
"""
import os

from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

REPO = "/data3/bhkim/workspace/train-ego-path-detection"
FIG = os.path.join(REPO, "figures")

MM = 1 / 25.4
FONT = "맑은 고딕"                       # KRRI 템플릿 테마 서체

# 색은 스킬의 팔레트 문서에서만 고른다(references/color-tokens.md).
NAVY_DEEP = RGBColor(0x00, 0x20, 0x60)   # 제목, 표 헤더 텍스트, 카드 헤딩
NAVY_BRAND = RGBColor(0x12, 0x43, 0x78)  # 불릿 앞말, 카드 헤딩
NAVY_COVER = RGBColor(0x04, 0x3C, 0x6F)  # 상단 밴드
BLUE = RGBColor(0x06, 0x6D, 0xB6)        # 표 헤더 채움, 악센트 바
CYAN = RGBColor(0x29, 0xAA, 0xE1)        # 제목 언더라인 룰
TEAL = RGBColor(0x39, 0xBC, 0xBD)        # 희소 강조 — 한 장에만
SKY_PALE = RGBColor(0x8A, 0xD7, 0xE5)    # 표 행 구분선
OFFWHITE = RGBColor(0xF8, 0xF8, 0xF8)    # 카드 내부, 교차 행
BODY = RGBColor(0x1A, 0x1A, 0x1A)        # 본문
GRAY = RGBColor(0x7A, 0x7A, 0x7A)        # 캡션, 푸터
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
BAND_NO = RGBColor(0x8A, 0xD7, 0xE5)     # 밴드 안 장 번호

A4W, A4H = Inches(297 * MM), Inches(210 * MM)
MG = Inches(12 * MM)
HEAD_H = Inches(24 * MM)
RULE_H = Inches(2.6 * MM)
BODY_W = A4W - 2 * MG
BODY_TOP = HEAD_H + Inches(7 * MM)
BODY_BOT = A4H - Inches(9 * MM)

# 그림이 있는 장은 좌우로 나눈다. 왼쪽이 글, 오른쪽이 그림.
GAP = Inches(8 * MM)
COL_L = Inches(148 * MM)
COL_R = BODY_W - COL_L - GAP
X_R = MG + COL_L + GAP

TITLE_PT = 46
BODY_PT = 27
CAP_PT = 17


def rect(s, x, y, w, h, fill, line=None):
    sh = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, int(x), int(y), int(w), int(h))
    sh.fill.solid()
    sh.fill.fore_color.rgb = fill
    if line is None:
        sh.line.fill.background()
    else:
        sh.line.color.rgb = line
        sh.line.width = Pt(1)
    sh.shadow.inherit = False
    return sh


def text(s, x, y, w, h, paras, size=BODY_PT, color=BODY, bold=False,
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


def sheet(prs, no, title):
    """한 장의 머리: 네이비 밴드에 장 번호와 제목, 아래에 시안 룰."""
    s = prs.slides.add_slide(prs.slide_layouts[6])
    rect(s, 0, 0, A4W, HEAD_H, NAVY_COVER)
    rect(s, 0, HEAD_H - RULE_H, A4W, RULE_H, CYAN)
    text(s, MG, Inches(4.5 * MM), Inches(22 * MM), Inches(10 * MM),
         "%02d" % no, 30, BAND_NO, True)
    text(s, MG + Inches(20 * MM), Inches(3.5 * MM), BODY_W - Inches(20 * MM),
         Inches(15 * MM), title, TITLE_PT, WHITE, True, anchor=MSO_ANCHOR.MIDDLE)
    return s


def bullets(s, y, items, x=MG, w=BODY_W, size=BODY_PT):
    paras, n = [], 0
    for lead, rest in items:
        run = [("• " + lead, True, NAVY_BRAND)] if lead else []
        if rest:
            run.append((("  " if lead else "") + rest, False, BODY))
        paras.append(run)
        n += lines_of("• " + lead + "  " + rest, w, size)
    h = Inches(size / 72.0 * 1.52) * n + Inches(0.14) * len(items)
    text(s, x, y, w, h, paras, size, spacing=1.34, space=12)
    return y + h + Inches(6 * MM)


def card(s, y, title, paras, x=MG, w=BODY_W, size=BODY_PT, accent=BLUE):
    """강조 상자. 본문 지배색이 흰색이므로 어두운 판 대신 오프화이트 카드에
    악센트 바를 세운다(팔레트 운용 규칙: 슬라이드당 지배색 1개)."""
    n = sum(lines_of(str(p if isinstance(p, str) else
                     "".join(t for t, _, _ in p)), w - Inches(16 * MM), size)
            for p in paras)
    h = Inches(14 * MM) + Inches(size / 72.0 * 1.52) * n + Inches(8 * MM)
    rect(s, x, y, w, h, OFFWHITE)
    rect(s, x, y, Inches(3 * MM), h, accent)          # 왼쪽 악센트 바
    text(s, x + Inches(9 * MM), y + Inches(4.5 * MM), w - Inches(16 * MM),
         Inches(8 * MM), title, 26, NAVY_DEEP, True)
    text(s, x + Inches(9 * MM), y + Inches(14 * MM), w - Inches(16 * MM),
         h - Inches(18 * MM), paras, size, spacing=1.34, space=8)
    return y + h + Inches(6 * MM)


def figure(s, y, name, caption, x=MG, w=BODY_W, max_h=None, cap_w=None):
    p = os.path.join(FIG, name)
    if not os.path.exists(p):
        text(s, x, y, w, Inches(10 * MM), f"[{name} 없음]", CAP_PT, GRAY)
        return y + Inches(12 * MM)
    with Image.open(p) as im:
        ar = im.height / im.width
    h = int(w * ar)
    px = x
    if max_h and h > max_h:
        h = int(max_h)
        w = int(h / ar)
        px = x + (BODY_W - w) // 2 if x == MG else x   # 줄어든 그림은 가운데로
    s.shapes.add_picture(p, int(px), int(y), int(w), int(h))
    cy = y + h + Inches(2.5 * MM)
    # 캡션 높이도 줄 수에서 계산한다. 고정 14 mm로 두었더니 세 줄짜리 캡션이
    # 상자 밖으로 흘러 페이지 아래로 잘렸다 — 도형 좌표 검사로는 안 잡힌다.
    cw = cap_w or (BODY_W if x == MG else w)
    ch = Inches(CAP_PT / 72.0 * 1.35) * lines_of(caption, cw, CAP_PT) + Inches(2 * MM)
    text(s, x, cy, cw, ch, caption, CAP_PT, GRAY, spacing=1.25)
    return cy + ch


def table(s, y, rows, col_w, x=MG, size=22):
    """KRRI 표 표준: 헤더 파랑 채움에 흰 글씨, 교차 행 오프화이트,
    행 구분선 연하늘.

    행 높이는 글자를 담고도 남아야 한다. 12 mm에 24 pt를 넣었더니 헤더가 위아래로
    잘리고 다음 행 사각형이 그 위에 덮여, 표가 통째로 뭉개졌다. 한 줄 높이는
    size x 1.3이므로 22 pt면 10 mm, 위아래 여백까지 16 mm를 잡는다.
    """
    rh = Inches(16 * MM)
    xs, cx = [], x
    for cw in col_w:
        xs.append(cx)
        cx += Inches(cw)
    for r, row in enumerate(rows):
        ry = y + rh * r
        for c, cell in enumerate(row):
            fill = BLUE if r == 0 else (OFFWHITE if r % 2 == 0 else WHITE)
            rect(s, xs[c], ry, Inches(col_w[c]), rh, fill)
            if r:
                rect(s, xs[c], ry, Inches(col_w[c]), Inches(0.4 * MM), SKY_PALE)
            text(s, xs[c] + Inches(4 * MM), ry + Inches(2.5 * MM),
                 Inches(col_w[c]) - Inches(8 * MM), rh - Inches(5 * MM),
                 str(cell), size, WHITE if r == 0 else BODY, r == 0,
                 align=PP_ALIGN.LEFT if c == 0 else PP_ALIGN.CENTER)
    return y + rh * len(rows) + Inches(6 * MM)


def build_banner():
    """제목 밴드. 규정상 제목과 저자는 25~30 mm."""
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(1000 * MM), Inches(210 * MM)
    s = prs.slides.add_slide(prs.slide_layouts[6])
    rect(s, 0, 0, prs.slide_width, prs.slide_height, NAVY_COVER)
    rect(s, 0, prs.slide_height - Inches(4 * MM), prs.slide_width,
         Inches(4 * MM), CYAN)
    text(s, Inches(30 * MM), Inches(34 * MM),
         prs.slide_width - Inches(60 * MM), Inches(80 * MM),
         "SAM 2 비디오 마스크 전파를 활용한\n철도 자기 경로 학습 데이터 반자동 구축 기법",
         82, WHITE, True, align=PP_ALIGN.CENTER, spacing=1.18)
    text(s, Inches(30 * MM), Inches(150 * MM),
         prs.slide_width - Inches(60 * MM), Inches(22 * MM),
         "한국철도기술연구원 철도피지컬AI연구실", 52, SKY_PALE, False,
         align=PP_ALIGN.CENTER)
    out = os.path.join(REPO, "poster_ieie_a4l_banner.pptx")
    prs.save(out)
    print("wrote", out, "(1000 x 210 mm 제목 배너)")


def hero(s, name, caption, cap_w=None):
    """장의 주인공 그림. 위에 크게 놓고 캡션을 아래 붙인 뒤, 남은 y를 돌려준다.

    포스터는 서서 읽는 물건이라 문장보다 그림이 먼저 눈에 들어와야 한다. 그림에
    면의 절반쯤을 주고, 글은 그 아래 두세 줄로 줄인다.
    """
    return figure(s, BODY_TOP, name, caption,
                  max_h=Inches(96 * MM), cap_w=cap_w)


def build_sheets():
    prs = Presentation()
    prs.slide_width, prs.slide_height = int(A4W), int(A4H)

    # 열두 장을 논문 목차가 아니라 작업 순서대로 놓는다. 앞서는 파이프라인이
    # 03(①부분)과 07(전체)에 두 번 나오고 그 사이에 단계가 끼어, 어디서부터 읽어야
    # 하는지가 흐렸다. 이제 전체 그림은 03에 한 번만 두고, 04~08이 그 그림의 왼쪽
    # 끝에서 오른쪽 끝까지 순서대로 확대해 간다. 09~11이 그 결과를 검증한다.
    #
    # 도입 01-02 · 전체 03 · 단계 04-08 · 검증 09-11 · 마무리 12

    # 01 문제
    s = sheet(prs, 1, "문제")
    y = hero(s, "ieie_crops/label_canvas.png",
             "그림 1. 라벨링 화면. 프레임마다 좌우 두 곡선을 사람이 그린다.")
    bullets(s, y, [
        ("비용", "비디오 단위 구축 비용이 크다"),
        ("반복", "새 노선마다 처음부터 다시 라벨링한다"),
    ])

    # 02 기존 방식의 한계
    s = sheet(prs, 2, "기존 방식의 한계")
    y = hero(s, "ieie_crops/fail_a.png",
             "그림 2. 일반 철도로 학습된 검출기를 노면 전차에 적용한 결과. "
             "자기 궤도를 벗어나 인접 궤도와 주변 차량으로 이탈한다.")
    bullets(s, y, [
        ("한계", "모델이 실패하는 신규 도메인에서는 초안이 성립하지 않는다"),
    ])

    # 03 전체 프로세스 — 전체 그림은 여기 한 번만
    s = sheet(prs, 3, "전체 프로세스")
    y = hero(s, "fig_arch_pipeline.png",
             "그림 3. 첫 프레임 프롬프트가 전 프레임으로 전파되어 궤도 마스크가 "
             "되고, 레일 변환과 검수를 거쳐 학습 데이터가 된다. 미세조정 모델로 "
             "초안을 재생성하는 부트스트래핑 루프로 이어진다.")
    bullets(s, y, [
        ("착안", "SAM 2는 레일을 학습한 적이 없어도 지정한 영역을 추적한다"),
    ])

    # 04 단계 1 — 프롬프트
    s = sheet(prs, 4, "단계 1 · 프롬프트")
    y = hero(s, "ieie_crops/pipe_step1.png",
             "그림 4. 첫 프레임에서 자기 궤도 위 한 점을 찍는다.")
    bullets(s, y, [
        ("자동 시드", "미지정 시 화면 하단 중앙을 쓴다"),
        ("근거", "자기 궤도는 차량 정면 하단에서 시작한다"),
    ])

    # 05 단계 2 — 비디오 전파
    s = sheet(prs, 5, "단계 2 · 비디오 전파")
    y = hero(s, "fig4_sam2_tram.jpg",
             "그림 5. 처음과 중간, 끝 프레임. 한 번의 지정으로 290 프레임 전체에 "
             "마스크가 전파된다.")
    bullets(s, y, [
        ("이점", "모델 내부 메모리로 추적해 모션 추정이 필요 없다"),
    ])

    # 06 단계 3 — 레일 변환
    s = sheet(prs, 6, "단계 3 · 레일 변환")
    y = hero(s, "ieie_crops/pipe_step2.png",
             "그림 6. 궤도 마스크에서 행별 좌우 경계를 뽑아 레일 점열로 바꾼다.")
    bullets(s, y, [
        ("격자", "48행을 따라 행별 최좌우 경계 화소를 취한다"),
        ("효과", "레일당 평균 48점이 6점 내외로 줄어든다"),
    ])

    # 07 단계 4 — 검수·보정
    s = sheet(prs, 7, "단계 4 · 검수 보정")
    y = hero(s, "screen_labeling.png",
             "그림 7. 초안이 편집 화면에 즉시 로드되어 점 단위로 다듬을 수 있다.")
    bullets(s, y, [
        ("보정", "점 단위 이동과 추가, 삭제로 다듬는다"),
    ])

    # 08 단계 5 — 학습과 재라벨
    s = sheet(prs, 8, "단계 5 · 학습 재라벨")
    # ① 행의 오른쪽 끝을 쓰면 검수 상자가 나와 단계 4를 두 번 보여주게 된다.
    # 학습과 재라벨은 ② 행에 있다.
    y = hero(s, "ieie_crops/pipe_step5.png",
             "그림 8. 확정 라벨이 학습 입력이 되고, 미세조정 모델이 나머지 초안을 "
             "다시 만든다.")
    bullets(s, y, [
        ("순환", "소량 보정본으로 전체를 다시 라벨링한다"),
    ])

    # 09 실험 설정
    s = sheet(prs, 9, "실험 설정")
    y = hero(s, "ieie_crops/sam2_a.png",
             "그림 9. 대상 장면. 레일이 도로에 매립되어 침목과 자갈 같은 문맥 "
             "단서가 없다.")
    bullets(s, y, [
        ("대상", "도심 노면 전차 290 프레임, 1920 × 1080"),
        ("비교", "RailSem19 학습 모델과 제안 기법"),
    ])

    # 10 부트스트래핑 결과
    s = sheet(prs, 10, "부트스트래핑 결과")
    y = hero(s, "fig3_bootstrap_relabel.jpg",
             "그림 10. 좌측은 미세조정 이전 초안, 우측은 30 프레임 보정본으로 "
             "미세조정한 모델의 재라벨.")
    bullets(s, y, [
        ("결과", "지그재그와 이탈이 사라지고 궤도를 매끄럽게 따른다"),
    ])

    # 11 정량 결과
    s = sheet(prs, 11, "정량 결과")
    figure(s, BODY_TOP, "ieie_crops/boot_after.png",
           "그림 11. 미세조정 모델의 재라벨.",
           x=X_R, w=COL_R, max_h=Inches(96 * MM))
    y = table(s, BODY_TOP,
              [["구분", "대상 도메인", "원 도메인"],
               ["미세조정 전", "0.389", "0.975"],
               ["미세조정 후", "0.485", "0.893"]],
              [1.9, 1.9, 1.9])
    card(s, y, "파국적 망각",
         [[("기저망까지 미세조정한 대가로 원 도메인이 하락했다. 도메인별 모델 "
            "분리 또는 기저망 동결로 완화할 수 있다.", False, BODY)]],
         w=COL_L)

    # 12 결론
    s = sheet(prs, 12, "결론")
    y = hero(s, "ieie_crops/sam2_c.png",
             "그림 12. 학습 이력이 없는 신규 궤도 환경에서 생성된 초안.")
    bullets(s, y, [
        ("확인", "단일 프롬프트로 전 구간 초안이 생성되었다"),
        ("향후", "망각을 억제하는 도메인 적응, 반복 부트스트래핑의 정량 평가"),
    ])
    text(s, MG, BODY_BOT - Inches(9 * MM), BODY_W, Inches(9 * MM),
         "한국철도기술연구원 주요사업 철도 특화 로봇 기반 선로점검 "
         "핵심기술 개발(PK26312A1)", 16, GRAY)

    out = os.path.join(REPO, "poster_ieie_a4l.pptx")
    prs.save(out)
    print("wrote", out, f"(A4 가로 {len(prs.slides._sldIdLst)}장)")


if __name__ == "__main__":
    build_banner()
    build_sheets()
