#!/usr/bin/env python3
"""
krri-ppt-builder 빌드 엔진 v1.4 (원작 개발: 이상덕, 변성준)

사용법:
  1) PLAN(아래 예시 구조)을 별도 파이썬 파일로 작성: plan.py 안에 ORDER, FILL, NOTES 정의
  2) python build_krri.py --template <KRRI템플릿.pptx> --plan plan.py --out out.pptx
  3) 렌더 QA 후 잔여 템플릿 슬라이드 삭제는 --finalize (사용자 승인 후에만!)

PLAN 구조:
  DUP   = {3: 1, 15: 3, 21: 5}          # 원본 레이아웃 번호 → 복제 수 (복제본은 67,68,... 순서로 생성)
  ORDER = [1, 2, 3, 13, ...]            # 최종 발표 순서의 슬라이드 파일 번호 (원본+복제본)
  FILL  = {1: {0: '텍스트', ...}, ...}   # 슬라이드 번호 → {문단인덱스: 새 텍스트}. ''는 비움
  NOTES = {4: '개조식 노트', ...}        # 발표 순서(1-base) → slide_notes (80~125자, 종결어미 없음)
  STRIP_PICS = [12]                      # 대형(cy>700000) 그림을 제거할 슬라이드 번호
  FOOTNOTES = {5: '※ FRA: 미국 연방철도청 | 출처: ...'}  # 슬라이드 파일 번호 → 하단 각주 (자기완결성)

문단 인덱스 의미는 references/layout-map.md 참조. 글자수 한도 초과 금지.
"""
import argparse, importlib.util, os, re, shutil, subprocess, sys, zipfile
from lxml import etree

A = 'http://schemas.openxmlformats.org/drawingml/2006/main'
P = 'http://schemas.openxmlformats.org/presentationml/2006/main'
R = 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'


def t21(title, headcopy, header, rows):
    """레이아웃 21(표) 채우기 헬퍼: 헤더 5칸 + 5행×5열."""
    # BSJ-107: 문단 인덱스 6~30이 본문 5×5, 31이 하단 카피 — 실측값이므로 변경 금지
    d = {0: title, 31: headcopy}
    for i, h in enumerate(header):
        d[1 + i] = h
    for r, row in enumerate(rows):
        for c, v in enumerate(row):
            d[6 + r * 5 + c] = v
    return d


def fill_slide(path, repl, keep=None, strict=True):
    """문단 인덱스 기반 치환 + default-deny(v3.12.0): repl·keep에 없는 문단은 자동 소거.
    템플릿 텍스트는 콘텐츠가 아니라 마커다 — opt-in 채움의 원문 잔존 실사고 방지."""
    keep = keep or set()
    tree = etree.parse(path)
    idx = 0
    blanked = 0
    for p in tree.getroot().iter(f'{{{A}}}p'):
        txt = ''.join(t.text or '' for t in p.findall(f'.//{{{A}}}t'))
        if not txt.strip():
            continue
        if strict and idx not in repl and idx not in keep:
            runs = p.findall(f'{{{A}}}r')
            if runs:
                t0 = runs[0].find(f'{{{A}}}t')
                if t0 is None:
                    t0 = etree.SubElement(runs[0], f'{{{A}}}t')
                t0.text = ''
                for r_el in runs[1:]:
                    p.remove(r_el)
                blanked += 1
            idx += 1
            continue
        if idx in repl:
            runs = p.findall(f'{{{A}}}r')
            if runs:
                t0 = runs[0].find(f'{{{A}}}t')
                if t0 is None:
                    t0 = etree.SubElement(runs[0], f'{{{A}}}t')
                t0.text = repl[idx]
                for r_el in runs[1:]:
                    p.remove(r_el)
                for br in p.findall(f'{{{A}}}br'):
                    p.remove(br)
        idx += 1
    tree.write(path, xml_declaration=True, encoding='UTF-8', standalone=True)
    return blanked


def strip_big_pics(path, min_cy=700000):
    tree = etree.parse(path)
    root = tree.getroot()
    n = 0
    for pic in list(root.iter(f'{{{P}}}pic')):
        xfrm = pic.find(f'.//{{{A}}}xfrm')
        ext = xfrm.find(f'{{{A}}}ext') if xfrm is not None else None
        if ext is not None and int(ext.get('cy', 0)) > min_cy:
            pic.getparent().remove(pic)
            n += 1
    tree.write(path, xml_declaration=True, encoding='UTF-8', standalone=True)
    return n


def reorder(unpacked, order):
    rels = etree.parse(f'{unpacked}/ppt/_rels/presentation.xml.rels')
    t2r = {}
    for rel in rels.getroot():
        m = re.match(r'slides/slide(\d+)\.xml', rel.get('Target') or '')
        if m:
            t2r[int(m.group(1))] = rel.get('Id')
    pres = f'{unpacked}/ppt/presentation.xml'
    tree = etree.parse(pres)
    lst = tree.getroot().find(f'{{{P}}}sldIdLst')
    # BSJ-112: 잔여 슬라이드는 반드시 후미 배치(G11) — rId 재발급 없이 순서만 변경
    entries = {e.get(f'{{{R}}}id'): e for e in lst}
    wanted = [t2r[n] for n in order]
    leftover = [e for e in lst if e.get(f'{{{R}}}id') not in wanted]
    for e in list(lst):
        lst.remove(e)
    for rid in wanted:
        lst.append(entries[rid])
    for e in leftover:
        lst.append(e)
    tree.write(pres, xml_declaration=True, encoding='UTF-8', standalone=True)
    return len(leftover)


def zip_dir(unpacked, out):
    if os.path.exists(out):
        os.remove(out)
    subprocess.run(['zip', '-Xrq', os.path.abspath(out), '.'], cwd=unpacked, check=True)


def load_plan(path):
    spec = importlib.util.spec_from_file_location('plan', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--template', required=True)
    ap.add_argument('--plan', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--add-slide-script', default=None,
                    help='pptx 스킬의 scripts/add_slide.py 경로 (복제 필요 시)')
    ap.add_argument('--finalize', action='store_true',
                    help='잔여 템플릿 슬라이드 삭제 (사용자 승인 후에만!)')
    args = ap.parse_args()
    plan = load_plan(args.plan)

    U = 'krri_unpacked'
    shutil.rmtree(U, ignore_errors=True)
    zipfile.ZipFile(args.template).extractall(U)

    # 1) 레이아웃 복제 (add_slide.py 사용 — 파일명은 실행 순서대로 slide67, 68, ...)
    dup = getattr(plan, 'DUP', {})
    if dup:
        assert args.add_slide_script, 'DUP 사용 시 --add-slide-script 필요'
        for src_n, cnt in dup.items():
            for _ in range(cnt):
                subprocess.run([sys.executable, args.add_slide_script, U, f'slide{src_n}.xml'],
                               check=True, capture_output=True)

    # 2) 텍스트 주입 (strict: 미지정 문단 자동 소거 — plan.KEEP={슬라이드: {유지 인덱스}}로 디자인 라벨 선언)
    KEEP = getattr(plan, 'KEEP', {})
    nb = 0
    for n, repl in plan.FILL.items():
        nb += fill_slide(f'{U}/ppt/slides/slide{n}.xml', repl, keep=KEEP.get(n)) or 0
    if nb:
        print(f'strict fill: 미지정 문단 {nb}개 소거 (템플릿 원문 잔존 방지)')

    # 3) 대형 그림 제거
    for n in getattr(plan, 'STRIP_PICS', []):
        strip_big_pics(f'{U}/ppt/slides/slide{n}.xml')

    # 3.5) 하단 각주 삽입 (슬라이드 자기완결성 — v1.4)
    footnotes = getattr(plan, 'FOOTNOTES', {})
    if footnotes:
        from add_footnote import add_footnote_slide_xml
        for n, txt in footnotes.items():
            add_footnote_slide_xml(f'{U}/ppt/slides/slide{n}.xml', txt)
        print(f'각주: {len(footnotes)}장 삽입')

    # 4) 임베디드 글꼴 제거 (Pretendard Variable 설치 실패 대화상자 방지 — v1.2)
    from strip_fonts import strip_embedded_fonts_dir
    nf = strip_embedded_fonts_dir(U)
    if nf:
        print(f'임베디드 글꼴 {nf}개 파트 제거 (설치 실패 대화상자 방지)')

    # 4.5) 작성자 메타데이터 설정 (원작 개발자 — v1.3)
    from set_author import set_author_dir, DEFAULT_AUTHOR
    if set_author_dir(U):
        print(f'작성자 메타데이터 설정: {DEFAULT_AUTHOR}')

    # 5) 재배열 (신규 앞, 원본 뒤 — G11: 선삭제 금지)
    leftover = reorder(U, plan.ORDER)
    zip_dir(U, args.out)
    print(f'built: {args.out} | 발표 {len(plan.ORDER)}장 + 잔여 {leftover}장 보존')

    # 5) 노트 주입 (python-pptx 저장 시 슬라이드 파일명이 발표 순서로 재번호화됨에 주의)
    notes = getattr(plan, 'NOTES', {})
    if notes:
        from pptx import Presentation
        prs = Presentation(args.out)
        for i, note in notes.items():
            prs.slides[i - 1].notes_slide.notes_text_frame.text = note
        prs.save(args.out)
        print(f'notes: {len(notes)}건 주입')

    # 6) finalize — 잔여 삭제 (G11: 반드시 사용자 승인 후)
    if args.finalize and leftover:
        shutil.rmtree(U, ignore_errors=True)
        zipfile.ZipFile(args.out).extractall(U)
        pres = f'{U}/ppt/presentation.xml'
        tree = etree.parse(pres)
        lst = tree.getroot().find(f'{{{P}}}sldIdLst')
        for e in list(lst)[len(plan.ORDER):]:
            lst.remove(e)
        tree.write(pres, xml_declaration=True, encoding='UTF-8', standalone=True)
        # pptx 스킬 clean.py가 있으면 고아 파트 정리 권장
        clean = os.path.join(os.path.dirname(args.add_slide_script or ''), 'clean.py')
        if os.path.exists(clean):
            subprocess.run([sys.executable, clean, U], check=True, capture_output=True)
        zip_dir(U, args.out)
        print(f'finalized: 잔여 {leftover}장 삭제')

    # 7) 렌더 안정화 (v1.6): 정적 글꼴 재임베드(Variable 제외) + 좁은 박스 자동 보정 + WIDEN
    from fix_boxes import postprocess
    postprocess(args.out, args.template, len(plan.ORDER), getattr(plan, 'WIDEN', None))

    print('다음 단계: validate.py + LibreOffice 렌더로 전 장 육안 QA (SKILL.md 참조)')


if __name__ == '__main__':
    main()
