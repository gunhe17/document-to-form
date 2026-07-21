"""_verify/raw/*.json(재사용 raw 그라운딩) → 절차별 고화질 HTML 리포트.

LLM 재호출 없음 — raw JSON의 elements 로 세그·carve 만 재계산(결정론·무료).
스테이지: 입력 → ①영역분리(SoM) → ②대구획 → ③raw 그라운딩 → ⑤carve(정밀배치).
탭=서식, 탭 안=스테이지 세로 스택. 실행: python _verify/render_raw.py
"""
import sys, json, base64, html
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core import pipeline                       # noqa: E402
from core.region_segment import segment         # noqa: E402
from core.ground_focus import focus_areas_from_groups, _split_big_groups  # noqa: E402

MAX_BLOCK = 10                                  # extract_form 기본값과 일치 (거대 대구획 소블록 분할 기준)

RAW = ROOT / "_verify" / "raw"
OUT = ROOT / "_verify" / "pipeline_raw.html"
MAXW = 1000                                     # 스테이지 이미지 폭(화질)

TCOL = {"text": (194, 113, 25), "textarea": (194, 113, 25), "number": (153, 133, 12),
        "email": (194, 113, 25), "phone": (194, 113, 25), "date": (181, 54, 156),
        "time": (181, 54, 156), "radio": (12, 89, 232), "checkbox_group": (12, 89, 232),
        "select": (12, 89, 232), "consent": (62, 138, 43), "signature": (68, 158, 47),
        "image": (217, 65, 103)}                # BGR
PAL = [(66, 135, 245), (75, 200, 60), (240, 160, 0), (200, 60, 200), (0, 165, 255),
       (0, 200, 200), (120, 90, 240), (90, 180, 90)]


def b64(im):
    h, w = im.shape[:2]
    if w > MAXW:
        im = cv2.resize(im, (MAXW, int(h * MAXW / w)), interpolation=cv2.INTER_AREA)
    return base64.b64encode(cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, 90])[1]).decode()


def st_groups(color, atoms, groups):
    """②대구획: 배경 어둡게, 각 구획 멤버 원자를 구획색으로, union 좌상단에 구획번호."""
    out = (color.astype(np.float32) * 0.28).astype(np.uint8)
    rectof = {j: r for j, r in atoms}
    for gi, g in enumerate(groups):
        c = PAL[gi % len(PAL)]
        for j in g:
            if j not in rectof:
                continue
            x, y, w, h = rectof[j]
            out[y:y + h, x:x + w] = color[y:y + h, x:x + w]
            cv2.rectangle(out, (x, y), (x + w, y + h), c, 3)
    for gi, a in enumerate(focus_areas_from_groups(atoms, groups)):
        x, y, w, h = a["rect"]; c = PAL[gi % len(PAL)]
        cv2.rectangle(out, (x, y), (x + w, y + h), c, 2)
        cv2.rectangle(out, (x, y), (x + 46, y + 26), c, -1)
        cv2.putText(out, f"G{gi}", (x + 4, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def st_boxes(clean, els, IW, IH, carved=False):
    """③raw(box 0~1000) 또는 ⑤carve(rect px) 박스 오버레이."""
    im = clean.copy()
    for e in els:
        if carved:
            x, y, w, h = e["rect"]; p1 = (int(x), int(y)); p2 = (int(x + w), int(y + h))
            c = TCOL.get(e["type"], (194, 113, 25))
        else:
            bx = e.get("box")
            if not bx or len(bx) != 4:
                continue
            ymin, xmin, ymax, xmax = bx
            p1 = (int(xmin / 1000 * IW), int(ymin / 1000 * IH)); p2 = (int(xmax / 1000 * IW), int(ymax / 1000 * IH))
            c = (0, 0, 230) if e.get("option") else (230, 120, 0)
        cv2.rectangle(im, p1, p2, c, 2)
    return im


STAGES = [("입력", "빈 서식 원본"),
          ("① 영역분리", "region_segment → atoms · SoM 번호 (CV·결정론)"),
          ("② 대구획 → 소블록", f"LLM이 논리 구획으로 묶은 뒤, {MAX_BLOCK}개 초과 구획은 소블록으로 분할(색=실제 그라운딩 단위)"),
          ("③ raw 그라운딩", "소블록별 하이라이트 → 입력칸 검출 + region-clamp (파랑=값 빨강=선택지)"),
          ("⑤ carve", "CV가 □·셀·글자에 정밀 스냅 (타입색)")]


def panel(rec, tab):
    path = rec["image"]
    color = cv2.imread(path)
    if color is None:
        return f'<div class=view id=v{tab} style="display:none"><p>이미지 없음: {html.escape(path)}</p></div>', 0
    gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
    IH, IW = gray.shape
    S = segment(gray); atoms = pipeline.atoms_of(S)
    els = rec["elements"]
    blocks_split = _split_big_groups(atoms, rec["groups"], cap=MAX_BLOCK)   # 실제 그라운딩 단위(소블록)
    carved = pipeline.place_elements(gray, S, els)
    imgs = [b64(color), b64(pipeline.som_mark(color, atoms)),
            b64(st_groups(color, atoms, blocks_split)),
            b64(st_boxes(color, els, IW, IH)),
            b64(st_boxes(color, carved, IW, IH, carved=True))]
    parts = ""
    for (title, desc), im in zip(STAGES, imgs):
        parts += (f'<div class=stage><div class=sh><b>{title}</b><span>{html.escape(desc)}</span></div>'
                  f'<img loading=lazy src="data:image/jpeg;base64,{im}"></div>')
    m = rec["_meta"]
    split_note = f'대구획 {len(rec["groups"])} → 소블록 {len(blocks_split)}' if len(blocks_split) != len(rec["groups"]) else f'구획 {len(rec["groups"])}'
    stat = (f'atoms {rec["n_atoms"]} · {split_note} · raw {len(els)} · '
            f'carve {len(carved)} · ${m.get("cost_sum", 0):.3f} · {rec.get("elapsed","?")}s')
    disp = "block" if tab == 0 else "none"
    return (f'<div class=view id=v{tab} style="display:{disp}"><div class=stat>{stat}</div>{parts}</div>',
            len(carved))


def main():
    recs = []
    for f in sorted(RAW.glob("[0-9]*.json"), key=lambda p: int(p.name[:2])):
        recs.append(json.loads(f.read_text()))
    tabs = panels = ""
    for tab, rec in enumerate(recs):
        p, ncarve = panel(rec, tab)
        on = " on" if tab == 0 else ""
        tabs += f'<button class="tab{on}" onclick="show({tab})">{html.escape(rec["name"])}</button>'
        panels += p
        print(f"{rec['name']}: carve {ncarve}")
    doc = _PAGE.replace("__TABS__", tabs).replace("__PANELS__", panels).replace("__N__", str(len(recs)))
    OUT.write_text(doc, encoding="utf-8")
    print(f"\n✓ {OUT} ({OUT.stat().st_size // 1024}KB, {len(recs)}서식)")


_PAGE = r"""<!doctype html><meta charset=utf-8><title>image-to-form · 절차별 결과(raw 재사용)</title>
<style>*{box-sizing:border-box}body{margin:0;font:14px/1.5 system-ui,-apple-system,sans-serif;background:#eef1f5;color:#1a1f28}
#bar{position:sticky;top:0;background:#12161d;padding:11px 16px;z-index:10}
#bar h1{font-size:15px;color:#fff;margin:0 0 3px}#bar .l{color:#8fa3bf;font-size:12px;margin-bottom:8px}
.tab{border:0;background:#2a3240;color:#c7d0dc;padding:5px 11px;border-radius:7px;cursor:pointer;font-size:12.5px;margin:0 4px 5px 0}
.tab.on{background:#4dabf7;color:#fff;font-weight:700}
.wrap{max-width:1040px;margin:0 auto;padding:18px 16px 80px}
.stat{background:#12161d;color:#7dd3a0;border-radius:10px;padding:9px 14px;font-size:12.5px;font-weight:600;margin-bottom:16px;font-family:ui-monospace,monospace}
.stage{background:#fff;border:1px solid #dbe1e9;border-radius:12px;padding:14px 16px;margin-bottom:16px;box-shadow:0 1px 4px rgba(20,30,50,.05)}
.sh{margin-bottom:10px}.sh b{font-size:16px}.sh span{color:#667;font-size:12.5px;margin-left:10px}
.stage img{width:100%;border:1px solid #cfd6de;border-radius:6px;display:block}</style>
<div id=bar><h1>image-to-form · 절차별 결과 — 확정 대구획 파이프라인 (raw 재사용, LLM 재호출 없음)</h1>
<div class=l>입력 → ①영역분리 → ②대구획 → ③raw 그라운딩(+clamp) → ⑤carve · 서식 __N__개</div>__TABS__</div>
<div class=wrap>__PANELS__</div>
<script>const V=s=>[...document.querySelectorAll(s)];
function show(i){V('.view').forEach((v,j)=>v.style.display=j==i?'block':'none');V('.tab').forEach((t,j)=>t.classList.toggle('on',j==i));window.scrollTo(0,0)}</script>"""


if __name__ == "__main__":
    main()
