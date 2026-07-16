"""최종 결과 갤러리 — 5개 폼 전체 페이지에 모든 carve 결과(박스) 오버레이.
실행: python _verify/carve_final.py  →  _verify/carve_final.html
"""
import sys, json, html, base64
from pathlib import Path
import cv2
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core import pipeline, extract, ground_focus
from core.region_segment import segment

CACHE = Path("/private/tmp/claude-501/-Users-gunhee-workspace-codespace-project-lab-voucher-image-to-form"
             "/06ad7bff-82ab-4edb-ba1f-8a8dda5599b0/scratchpad/focus_multidoc_x3.json")
OUT = ROOT / "_verify" / "carve_final.html"
DATA = json.loads(CACHE.read_text())

# 타입 → (색 BGR, 범례색 CSS)
STY = {"text": "#3a86ff", "textarea": "#3a86ff", "email": "#3a86ff", "phone": "#3a86ff",
       "number": "#b14aed", "date": "#ff8c1a", "time": "#ff8c1a",
       "radio": "#2ec27e", "checkbox_group": "#e01b24", "consent": "#e01b24",
       "signature": "#e05a9c", "image": "#e5c100"}


def hexbgr(h):
    h = h.lstrip("#"); return (int(h[4:6], 16), int(h[2:4], 16), int(h[0:2], 16))


def flat(run):
    o = []
    for j in sorted(run, key=int):
        o += run[j]
    return o


def b64(im):
    return base64.b64encode(cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, 82])[1]).decode()


sections = ""
for doc in DATA["docs"]:
    img = cv2.imread(doc["path"]); gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY); S = segment(gray)
    raw = ground_focus.dedup_nested(flat(doc["runs"][0]), {j: r[2] * r[3] for j, r in pipeline.atoms_of(S)})
    placed = extract.assign_keys(pipeline.place_elements(gray, S, raw))
    o = img.copy()
    for e in placed:
        x, y, w, h = [int(v) for v in e["rect"]]
        cv2.rectangle(o, (x, y), (x + w, y + h), hexbgr(STY.get(e.get("type", "text"), "#3a86ff")), 2)
    sections += (f'<figure><figcaption>{html.escape(doc["name"])}</figcaption>'
                 f'<img src="data:image/jpeg;base64,{b64(o)}"></figure>')

doc = f"""<!doctype html><meta charset=utf-8><title>carved forms</title>
<style>
*{{box-sizing:border-box}} body{{margin:0;background:#f4f4f5;font:14px system-ui,sans-serif}}
.wrap{{max-width:900px;margin:0 auto;padding:28px 20px 60px}}
figure{{margin:0 0 34px}}
figcaption{{font-size:13px;color:#666;margin:0 0 8px;letter-spacing:.02em}}
img{{width:100%;display:block;border:1px solid #ddd;border-radius:8px;box-shadow:0 1px 6px rgba(0,0,0,.08)}}
</style>
<div class=wrap>{sections}</div>"""
OUT.write_text(doc)
print(f"→ {OUT} ({OUT.stat().st_size/1e6:.1f}MB)")


if __name__ == "__main__":
    pass
