"""Render a paper markdown to a viewing PDF via headless Chrome.

LibreOffice's HTML export collapses table columns and inflates the layout, so
the markdown is emitted as HTML here and printed by Chrome, which lays out the
tables and figures correctly. Block parsing is reused from make_hwpx so the PDF
and the HWPX are built from exactly the same interpretation of the source.

  python experiments/make_pdf.py [input.md] [out.pdf]
"""
import html
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO)

SRC = sys.argv[1] if len(sys.argv) > 1 else "paper_KCI_temporal_2026.md"
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.splitext(SRC)[0] + ".pdf"

import importlib.util
spec = importlib.util.spec_from_file_location(
    "mk", os.path.join(REPO, "experiments", "make_hwpx.py"))
# make_hwpx builds documents at import time; parse its helpers only.
import re
src = open(os.path.join(REPO, "experiments", "make_hwpx.py"), encoding="utf-8").read()
ns = {}
head = src[:src.index("def build(")]
head = head.replace('SRC = sys.argv[1] if len(sys.argv) > 1 else "paper_draft_KCI.md"', f'SRC = {SRC!r}')
head = head.replace('BASE = sys.argv[2] if len(sys.argv) > 2 else "paper_KCI"', 'BASE = "unused"')
exec(compile(head, "make_hwpx-head", "exec"), ns)
BLOCKS = ns["BLOCKS"]

CHROME = next((p for p in ("/usr/bin/google-chrome", "/usr/bin/chromium",
                           "/usr/bin/chromium-browser") if os.path.exists(p)), None)
if not CHROME:
    sys.exit("chrome/chromium not found")

CSS = """
@page { size: A4; margin: 18mm 16mm; }
body { font-family: 'Noto Sans CJK KR','Noto Sans KR',sans-serif; font-size: 10.5pt;
       line-height: 1.65; color: #111; }
h1 { font-size: 17pt; margin: 0 0 6pt; }
h2 { font-size: 13pt; margin: 16pt 0 6pt; border-bottom: 1px solid #ccc; padding-bottom: 3pt; }
h3 { font-size: 11.5pt; margin: 12pt 0 4pt; }
p { margin: 5pt 0; text-align: justify; }
table { border-collapse: collapse; width: 100%; margin: 8pt 0; font-size: 9pt;
        page-break-inside: avoid; }
th, td { border: 1px solid #999; padding: 3pt 5pt; text-align: center; }
th { background: #eef2f6; font-weight: 600; }
td:first-child, th:first-child { text-align: left; }
img { max-width: 100%; display: block; margin: 8pt auto 2pt; page-break-inside: avoid; }
blockquote { color: #555; font-size: 9pt; border-left: 3px solid #ccc;
             margin: 6pt 0; padding: 2pt 8pt; }
ul { margin: 4pt 0 4pt 16pt; } li { margin: 2pt 0; }
hr { border: none; border-top: 1px solid #ddd; margin: 10pt 0; }
.cap { text-align: center; font-size: 9pt; color: #333; margin: 0 0 8pt; }
"""

parts = ["<meta charset='utf-8'><style>" + CSS + "</style>"]
for kind, payload in BLOCKS:
    if kind == "table":
        rows = payload
        t = ["<table><thead><tr>" + "".join(f"<th>{html.escape(c)}</th>" for c in rows[0]) + "</tr></thead><tbody>"]
        for r in rows[1:]:
            t.append("<tr>" + "".join(f"<td>{html.escape(c)}</td>" for c in r) + "</tr>")
        t.append("</tbody></table>")
        parts.append("".join(t))
    elif kind == "img":
        parts.append(f"<img src='{html.escape(payload)}'>")
    elif kind == "hr":
        parts.append("<hr>")
    elif kind in ("h1", "h2", "h3"):
        parts.append(f"<{kind}>{html.escape(payload)}</{kind}>")
    elif kind == "quote":
        parts.append(f"<blockquote>{html.escape(payload)}</blockquote>")
    elif kind == "bullet":
        parts.append(f"<ul><li>{html.escape(payload)}</li></ul>")
    else:
        text = html.escape(payload)
        cls = " class='cap'" if text.startswith("그림") or text.startswith("표") else ""
        parts.append(f"<p{cls}>{text}</p>")

tmp = os.path.join(REPO, "_paper_view.html")
open(tmp, "w", encoding="utf-8").write("\n".join(parts))
r = subprocess.run([CHROME, "--headless", "--no-sandbox", "--disable-gpu",
                    "--no-pdf-header-footer", f"--print-to-pdf={OUT}",
                    "--virtual-time-budget=20000", f"file://{tmp}"],
                   capture_output=True, text=True, timeout=300)
os.remove(tmp)
if os.path.exists(OUT):
    print(f"wrote {OUT} ({os.path.getsize(OUT)/1024:.0f} KB)")
else:
    sys.exit(f"chrome failed: {r.stderr[-300:]}")
