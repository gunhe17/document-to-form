"""정답 검증 뷰어 — 일반화 carve 증분 결과를 폼별로 이미지 위에 올려 눈으로 확인.
증분1(radio): slot(초록=매칭) vs baseline(파랑 점선). 미매칭 radio는 빨강(커버리지 구멍).

실행: python _verify/carve_review.py  →  _verify/carve_review.html
"""
import json, sys, html, base64
from pathlib import Path
import cv2
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core import pipeline, carve, carve_slots
from core.region_segment import segment

CACHE = Path("/private/tmp/claude-501/-Users-gunhee-workspace-codespace-project-lab-voucher-image-to-form"
             "/06ad7bff-82ab-4edb-ba1f-8a8dda5599b0/scratchpad/focus_multidoc_x3.json")
OUT = ROOT / "_verify" / "carve_review.html"
LR = {"L", "R", "좌", "우"}


def flat(run):
    o = []
    for j in sorted(run, key=int):
        o += run[j]
    return o


def b64(im, q=78):
    return base64.b64encode(cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, q])[1]).decode()


def bx(rect, IW, IH, cls, lab):
    x, y, w, h = rect
    return (f'<div class="b {cls}" style="left:{x/IW*100:.2f}%;top:{y/IH*100:.2f}%;'
            f'width:{w/IW*100:.2f}%;height:{h/IH*100:.2f}%"><span>{html.escape(lab)}</span></div>')


def main():
    data = json.loads(CACHE.read_text())
    panels = ""
    tabs = ""
    for di, doc in enumerate(data["docs"]):
        img = cv2.imread(doc["path"]); gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        IH, IW = gray.shape; S = segment(gray); rectof = dict(pipeline.atoms_of(S))
        items0 = flat(doc["runs"][0])
        # px rect 부여
        for e in items0:
            b = e.get("box") or [0, 0, 0, 0]; y0, x0, y1, x1 = b
            e["rect"] = (x0/1000*IW, y0/1000*IH, (x1-x0)/1000*IW, (y1-y0)/1000*IH)
        smap = carve_slots.place_radios(gray, rectof, items0)
        boxes = ""
        n_match = n_miss = 0
        for i, e in enumerate(items0):
            if e.get("type") != "radio" or not e.get("option") or str(e.get("option")) in LR:
                continue
            opt = str(e.get("option"))
            # baseline (파랑 점선)
            rb, _, _ = carve.place(gray, e["rect"], "radio", opt, (IH, IW), bounds=rectof.get(e.get("region")))
            boxes += bx(tuple(int(v) for v in rb), IW, IH, "base", "")
            # slot (초록=매칭 / 빨강=미매칭은 baseline 위치에)
            if i in smap:
                boxes += bx(smap[i], IW, IH, "slot", opt); n_match += 1
            else:
                boxes += bx(tuple(int(v) for v in rb), IW, IH, "miss", opt + "?"); n_miss += 1
        on = " on" if di == 0 else ""
        tabs += (f'<button class="tab{on}" onclick="show({di})">{html.escape(doc["name"])} '
                 f'<span class=mm>✓{n_match} <b>✗{n_miss}</b></span></button>')
        disp = "block" if di == 0 else "none"
        panels += (f'<div class=panel id=p{di} style="display:{disp}">'
                   f'<div class=cap>{html.escape(doc["name"])} · radio: <span style="color:#5c5">매칭 {n_match}</span> '
                   f'<span style="color:#f66">미매칭 {n_miss}</span> · 초록=slot(정답검증) 파랑점선=baseline 빨강=미매칭</div>'
                   f'<div class=wrap><img src="data:image/jpeg;base64,{b64(img)}">{boxes}</div></div>')

    doc_html = f"""<!doctype html><meta charset=utf-8><title>carve 정답검증 · 증분1 radio</title>
<style>
*{{box-sizing:border-box}} body{{margin:0;font:13px system-ui,sans-serif;background:#0e0e0e;color:#e8e8e8}}
#top{{padding:9px 14px;border-bottom:1px solid #333;background:#161616;position:sticky;top:0;z-index:9}}
h1{{margin:0 0 6px;font-size:14px}}
.tab{{background:#222;color:#ccc;border:1px solid #333;border-radius:6px;padding:5px 10px;margin:0 5px 4px 0;cursor:pointer;font:inherit}}
.tab.on{{background:#2a3d2a;border-color:#5a8;color:#fff}} .mm{{color:#999;font-size:11px;margin-left:4px}} .mm b{{color:#f66}}
.cap{{padding:8px 14px;color:#aaa;font-size:12px}}
.wrap{{position:relative;margin:0 14px 20px;border:1px solid #333;background:#000}}
.wrap>img{{width:100%;display:block}}
.b{{position:absolute;box-sizing:border-box}}
.b>span{{position:absolute;left:0;top:-13px;font-size:10px;padding:0 3px;border-radius:2px;white-space:nowrap;background:rgba(0,0,0,.8)}}
.slot{{border:2px solid #4c4}} .slot>span{{color:#8f8}}
.base{{border:1.5px dashed #58f;opacity:.7}}
.miss{{border:2px solid #f55}} .miss>span{{color:#f88}}
</style>
<div id=top><h1>carve 정답검증 — 증분1: radio (영역OCR slot vs baseline)</h1>{tabs}</div>
{panels}
<script>
function show(i){{document.querySelectorAll('.panel').forEach((p,j)=>p.style.display=j===i?'block':'none');
  document.querySelectorAll('.tab').forEach((t,j)=>t.classList.toggle('on',j===i));}}
</script>"""
    OUT.write_text(doc_html)
    print(f"→ {OUT} ({OUT.stat().st_size/1e6:.1f}MB)")


if __name__ == "__main__":
    main()
