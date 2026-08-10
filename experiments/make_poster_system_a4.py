"""TEP-Net 통합 개발 환경 포스터 — 제목 배너 1장 + A4 세로 12장.

deck_system.pptx(16장)의 본문을 IEIE 포스터 세션 규격으로 옮긴 것이다. 표지와
목차, Q&A는 보드판의 제목 박스가 대신하므로 본문 13장을 12장으로 합쳤다.

조판 헬퍼(줄 수 추정, 어두운 박스, 그림, 표)는 make_poster_ieie_a4에서 가져온다.
그 파일에서 겹침을 잡은 계수를 두 벌로 유지하면 한쪽만 고쳐지기 때문이다.

  python experiments/make_poster_system_a4.py
"""
import os
import sys

from PIL import Image
from pptx import Presentation
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_poster_ieie_a4 as P  # noqa: E402

MM, REPO, LOGO = P.MM, P.REPO, P.LOGO
INK, SUB, WHITE, META = P.INK, P.SUB, P.WHITE, P.META
RED, GREEN, TEAL, LIME, SKY = P.RED, P.GREEN, P.TEAL, P.LIME, P.SKY
MG, BODY_TOP = P.MG, P.BODY_TOP


def build_banner():
    prs = Presentation()
    W, H = Inches(1000 * MM), Inches(210 * MM)
    prs.slide_width, prs.slide_height = int(W), int(H)
    s = prs.slides.add_slide(prs.slide_layouts[6])
    P.rect(s, 0, 0, W, H, INK)
    P.rect(s, 0, H - Inches(6 * MM), W, Inches(6 * MM), TEAL)
    with Image.open(LOGO) as im:
        lw, lh = im.size
    logo_w = Inches(190 * MM)
    P.rect(s, Inches(30 * MM), Inches(20 * MM), logo_w + Inches(10 * MM),
           logo_w * lh / lw + Inches(10 * MM), WHITE)
    s.shapes.add_picture(LOGO, int(Inches(35 * MM)), int(Inches(25 * MM)),
                         int(logo_w), int(logo_w * lh / lw))
    P.text(s, Inches(30 * MM), Inches(64 * MM), W - Inches(60 * MM), Inches(60 * MM),
           "철도 자기 경로 검출을 위한\nTEP-Net 통합 개발 환경", 80, WHITE, True,
           spacing=1.18)
    P.text(s, Inches(30 * MM), Inches(142 * MM), W - Inches(60 * MM), Inches(16 * MM),
           "TEP-Net Integrated Development Environment for Railway Ego-Path Detection",
           34, META)
    P.text(s, Inches(30 * MM), Inches(163 * MM), W - Inches(60 * MM), Inches(26 * MM),
           [[("김백현", True, WHITE), ("  ·  ", False, META), ("황현철", True, WHITE),
             ("†", False, SKY), ("      한국철도기술연구원", False, META)]], 72)
    out = os.path.join(REPO, "poster_system_banner.pptx")
    prs.save(out)
    print("wrote", out, "(1000 x 210 mm 제목 배너)")


def build_sheets():
    prs = Presentation()
    prs.slide_width, prs.slide_height = int(P.A4W), int(P.A4H)

    # 01 통합 환경
    s = P.sheet(prs, 1, "통합 환경", RED)
    y = P.bullets(s, BODY_TOP, [
        ("과제", "라벨링 비용과 노선이 바뀌면 무너지는 정확도"),
        ("접근", "초안 생성과 검수, 학습을 한 브라우저에서 잇는다"),
    ])
    y = P.figure(s, y, "screen_laurent.png",
                 "화면 1. 원본 저장소 뷰어. TEP-Net 구조와 README를 앱 안에서 확인한다.")
    P.dark(s, y, "핵심", [
        [("도구를 오가며 생기는 형식 변환을 없앤다.", True, LIME)]])

    # 02 세 계층
    s = P.sheet(prs, 2, "세 계층", RED)
    y = P.bullets(s, BODY_TOP, [
        ("1계층 라벨링", "초안 생성과 점 단위 보정, 검수 큐"),
        ("2계층 학습", "베이스 위에 RNN을 전이학습"),
        ("3계층 추론", "단일 프레임과 시간 모델을 나란히 비교"),
    ])
    P.dark(s, y, "환경", [
        [("등록된 검출 모델 8종", False, WHITE)],
        [("NVIDIA A100 80GB 가속", False, WHITE)],
        [("서버 데이터셋 폴더 18개를 앱에서 직접 관리", True, SKY)]])

    # 03 라벨링 화면
    s = P.sheet(prs, 3, "라벨링 화면", GREEN)
    y = P.bullets(s, BODY_TOP, [
        ("전파", "첫 프레임에 프롬프트를 찍으면 폴더 전체로 퍼진다"),
    ])
    P.figure(s, y, "fig1_labeling_sam2.jpg",
             "화면 2. 좌우 레일을 점 단위로 보정한다. 곡선 보간과 굵기를 조절하고 "
             "확정 라벨을 다음 프레임 초안으로 전파한다.")

    # 04 자동 라벨 세 가지
    s = P.sheet(prs, 4, "자동 라벨", GREEN)
    y = P.bullets(s, BODY_TOP, [
        ("모델 자동 라벨", "도메인이 맞을 때 가장 깨끗한 초안을 낸다"),
        ("SAM 2 전파", "레일을 학습한 적이 없어도 지정 궤도를 추적한다"),
        ("곡선 보간", "점 세 개에서 레일 한 줄을 만든다"),
    ])
    P.dark(s, y, "선택 기준", [
        [("도메인이 맞으면 모델, 바뀌면 SAM 2 전파를 쓴다.", True, LIME)],
        [("어긋난 프레임만 사람이 손본다.", False, WHITE)]])

    # 05 도메인 이동
    s = P.sheet(prs, 5, "도메인 이동", GREEN)
    y = P.bullets(s, BODY_TOP, [
        ("실증", "일반 철도로 학습한 모델은 노면 전차에서 궤도를 놓친다"),
    ])
    y = P.figure(s, y, "fig3_model_fail_tram.jpg",
                 "화면 3. 학습 모델의 초안이 자기 궤도를 벗어나 인접 궤도와 "
                 "주변 차량으로 이탈한다.")
    P.dark(s, y, "대안", [
        [("SAM 2 전파는 첫 프레임 지정만으로 290 프레임을 추적했다.", True, LIME)]])

    # 06 학습 화면
    s = P.sheet(prs, 6, "학습 화면", TEAL)
    y = P.bullets(s, BODY_TOP, [
        ("전이학습", "베이스 모델을 골라 그 위에 RNN을 얹는다"),
        ("설정", "에폭과 배치, 학습률, 학습 GPU를 화면에서 지정한다"),
    ])
    P.figure(s, y, "screen_training.png",
             "화면 4. 베이스 동결과 미세조정, GPU 전처리, 다중 GPU를 선택한다. "
             "학습 곡선과 출력 모델명이 같은 화면에 남는다.")

    # 07 추론 화면
    s = P.sheet(prs, 7, "추론 화면", TEAL)
    y = P.bullets(s, BODY_TOP, [
        ("3분할", "원본과 단일 프레임, 시간 모델을 한 화면에 낸다"),
    ])
    # 이 그림은 세로가 길다. 높이를 묶지 않으면 아래 어두운 박스가 페이지 밖으로 밀린다.
    y = P.figure(s, y, "fig2_inference_compare.jpg",
                 "화면 5. 같은 입력에 두 모델을 동시에 적용하고 프레임당 지연과 "
                 "초당 처리량을 함께 표시한다.", max_h=Inches(118 * MM))
    P.dark(s, y, "측정된 지연", [
        [("30장 처리 5.8초", True, SKY)],
        [("단일 프레임 32.8 ms, 시간 모델 33.2 ms", False, WHITE)]])

    # 08 운영 화면
    s = P.sheet(prs, 8, "운영 화면", TEAL)
    y = P.bullets(s, BODY_TOP, [
        ("파일 이동", "로컬과 서버를 좌우 패널로 두고 복사한다"),
    ])
    P.figure(s, y, "screen_files.png",
             "화면 6. 업로드와 다운로드, 압축 해제를 앱 안에서 처리한다. "
             "서버 데이터셋 폴더 18개가 이미지 수와 함께 보인다.")

    # 09 분기기 갤러리
    s = P.sheet(prs, 9, "분기기 갤러리", TEAL)
    y = P.bullets(s, BODY_TOP, [
        ("열람", "국가와 노선, 영상 ID로 추려 대표 프레임을 확인한다"),
    ])
    P.figure(s, y, "screen_switch.png",
             "화면 7. 분기기 통과 이벤트 273건. 영국 76, 일본 34, 슬로베니아 25, "
             "독일 22 등 18개 지역에서 수집했다.")

    # 10 보유 데이터
    s = P.sheet(prs, 10, "보유 데이터", RED)
    y = P.table(s, BODY_TOP,
                [["구분", "규모", "용도"],
                 ["분기기 이벤트", "273", "이벤트당 81 프레임"],
                 ["OSDaR23", "1,535", "시간 모델 학습"],
                 ["크롤링 분기기", "613", "재추출로 화질 보강"],
                 ["노면 전차", "290", "도메인 이동 평가"]],
                [1.3, 0.8, 1.5])
    P.dark(s, y, "검수 진행", [
        [("검수 큐 417 프레임 중 9 완료", True, SKY)],
        [("RailSem19 rs19_val 원본 보유", False, WHITE)]])

    # 11 등록 모델
    s = P.sheet(prs, 11, "등록 모델", RED)
    y = P.table(s, BODY_TOP,
                [["모델", "헤드", "백본"],
                 ["twinkling-rocket-21", "seg", "effnet-b3"],
                 ["chromatic-laughter-5", "reg", "effnet-b3"],
                 ["brilliant-horse-15", "seg", "resnet18"],
                 ["logical-tree-1", "reg", "resnet18"],
                 ["fortuitous-goat-12", "cls", "effnet-b3"]],
                [1.7, 0.7, 1.1], size=20)
    P.dark(s, y, "활용", [
        [("헤드와 백본 조합을 골라 같은 입력에 바로 적용한다.", True, LIME)],
        [("세그멘테이션은 자동 라벨용, 회귀는 시간 모델 베이스", False, WHITE)]])

    # 12 검수 큐와 정리
    s = P.sheet(prs, 12, "검수 큐와 정리", RED)
    y = P.bullets(s, BODY_TOP, [
        ("선별", "모델들이 갈리는 프레임만 큐에 올린다"),
        ("규모", "전체의 12.8퍼센트인 417 프레임, 43개 이벤트"),
        ("정리", "구축 비용과 도메인 대응, 비교 가능성을 확보했다"),
    ])
    y = P.dark(s, y, "이어질 작업", [
        [("검수 완료 후 분기부 학습셋 재조립과 도메인 적응", True, LIME)]])
    P.dark(s, y, "사사", [
        [("한국철도기술연구원 주요사업 철도 특화 로봇 기반 선로점검 "
          "핵심기술 개발(PK26312A1)", False, META)]], 20)

    out = os.path.join(REPO, "poster_system_a4.pptx")
    prs.save(out)
    print("wrote", out, f"(A4 세로 {len(prs.slides._sldIdLst)}장)")


if __name__ == "__main__":
    build_banner()
    build_sheets()
