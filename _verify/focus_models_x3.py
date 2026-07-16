"""영역별 focus 그라운딩을 모델별 ×3회 → 변동성 비교 HTML.

focus_by_region_x3.html 와 같은 레이아웃(사이드 영역탭 = n1/n2/n3 변동, 본문 = 하이라이트+3회 표),
상단에 모델 토글. temp=0.2 로 저온 안정성 비교(gemini-2.5-pro vs qwen3-vl-235b).

실행:  python _verify/focus_models_x3.py          # 캐시 있으면 렌더만
       FORCE=1 python _verify/focus_models_x3.py  # API 재호출
출력:  _verify/focus_models_x3.html
"""
from __future__ import annotations

import base64
import html
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core import pipeline, extract, ground_focus  # noqa: E402

IMG = ("/Users/gunhee/workspace/codespace/project/lab-voucher/.old/document/"
       "제1편_장애아가족_양육지원_서식/_extract/서식2호_장애아돌보미_지원서/pages/p-1.png")
OUT = ROOT / "_verify" / "focus_models_x3.html"
CACHE = Path("/private/tmp/claude-501/-Users-gunhee-workspace-codespace-project-lab-voucher-image-to-form"
             "/06ad7bff-82ab-4edb-ba1f-8a8dda5599b0/scratchpad/focus_models_x3.json")

# (tag, model_id, temp) — 이미지 '읽기' 가능 모델만. image-생성(-image)·중복변형 제외.
MODELS = [
    ("gemini-2.5-flash-lite", "google/gemini-2.5-flash-lite", 0.2),
    ("gemini-2.5-flash", "google/gemini-2.5-flash", 0.2),
    ("gemini-2.5-pro", "google/gemini-2.5-pro", 0.2),
    ("gemini-3.1-flash-lite", "google/gemini-3.1-flash-lite", 0.2),
    ("gemini-3-flash-preview", "google/gemini-3-flash-preview", 0.2),
    ("gemini-3.5-flash", "google/gemini-3.5-flash", 0.2),
    ("gemini-3.1-pro-preview", "google/gemini-3.1-pro-preview", 0.2),
    ("qwen3-vl-235b", "qwen/qwen3-vl-235b-a22b-instruct", 0.2),
]
N_RUNS = 3
EFFORT = "low"
WORKERS = 8

RUN_COLORS = ["#4dabf7", "#69db7c", "#ff922b"]
WCLS = {"text": "w-txt", "textarea": "w-txt", "number": "w-num", "email": "w-txt", "phone": "w-txt",
        "date": "w-dt", "time": "w-dt", "radio": "w-ch", "checkbox_group": "w-ch", "select": "w-ch",
        "consent": "w-cn", "signature": "w-sg", "image": "w-img"}


def b64(im, q=72):
    return base64.b64encode(cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, q])[1]).decode()


# ── 실행: 모델별 ×3회 ground_by_regions (없는 모델만, 끝나는 대로 캐시 저장) ──
def run_missing(img, atoms, by_tag):
    for tag, mid, temp in MODELS:
        if tag in by_tag and not os.environ.get("FORCE"):
            print(f"skip {tag} (cache)", flush=True)
            continue
        print(f"\n=== {tag} ({mid}) temp={temp} ===", flush=True)
        runs = []          # 3회, 각 {region_id: [elements]}
        cost = 0.0
        t0 = time.perf_counter()
        for r in range(N_RUNS):
            res = ground_focus.ground_by_regions(
                img, atoms, reasoning_effort=EFFORT, workers=WORKERS,
                model=mid, temperature=temp,
            )
            per = {p["region"]: p["elements"] for p in res["_per_region"]}
            runs.append(per)
            cost += float((res["_meta"] or {}).get("cost_sum") or 0)
            n = sum(len(v) for v in per.values())
            print(f"  run{r+1}: {n} elements · {res['_meta'].get('per_region_counts')}", flush=True)
        dt = time.perf_counter() - t0
        by_tag[tag] = {"tag": tag, "mid": mid, "temp": temp, "runs": runs,
                       "cost": round(cost, 4), "secs": round(dt, 1)}
        _save(by_tag)   # 모델 단위 증분 저장 — 중간 실패해도 보존
    return by_tag


def _save(by_tag):
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps({"models": list(by_tag.values())}, ensure_ascii=False))


# ── 렌더 ────────────────────────────────────────────────────────────
def overlay(elements, run_idx):
    color = RUN_COLORS[run_idx]
    ov = ""
    for n, e in enumerate(elements):
        box = e.get("box") or [0, 0, 0, 0]
        ymin, xmin, ymax, xmax = box
        t = e.get("type", "text")
        cls = WCLS.get(t, "w-txt")
        opt = e.get("option")
        lab = e.get("label") or ""
        inner = (f'<span class=oc>{html.escape(str(opt)[:10])}</span>' if opt
                 else f'<span class=fl>{html.escape(str(lab)[:14])}</span>')
        ov += (f'<div class="fw {cls}" style="left:{xmin/10:.2f}%;top:{ymin/10:.2f}%;'
               f'width:{(xmax-xmin)/10:.2f}%;height:{(ymax-ymin)/10:.2f}%;border-color:{color}" '
               f'data-run="{run_idx}" data-n="{n}" '
               f'title="run{run_idx+1} · {html.escape(lab)} ({t})">{inner}</div>')
    return ov


def run_table(elements, run_idx):
    color = RUN_COLORS[run_idx]
    tc = Counter(e.get("type", "?") for e in elements)
    summ = " · ".join(f"{k}={v}" for k, v in sorted(tc.items()))
    rows = ""
    for n, e in enumerate(elements):
        box = e.get("box") or [0, 0, 0, 0]
        rows += (f'<tr class=er data-run="{run_idx}" data-n="{n}">'
                 f'<td>{n+1}</td><td>{html.escape(str(e.get("type","")))}</td>'
                 f'<td>{html.escape(str(e.get("label") or ""))}</td>'
                 f'<td>{html.escape(str(e.get("option") or ""))}</td>'
                 f'<td>{html.escape(str(e.get("unit") or ""))}</td>'
                 f'<td>{"✓" if e.get("_clamped") else ""}</td>'
                 f'<td class=bx>{html.escape(str(box))}</td></tr>')
    return (f'<div class=runcard style="border-top:3px solid {color}">'
            f'<div class=runtitle>run{run_idx+1} · <b>{len(elements)}</b>개'
            f'{" · " + summ if summ else ""}</div>'
            f'<table><thead><tr><th>#</th><th>type</th><th>label</th><th>opt</th>'
            f'<th>unit</th><th>c</th><th>box</th></tr></thead><tbody>{rows}</tbody></table></div>')


def render_view(mi, mtag, runs, atoms, focus_imgs, IW, IH):
    """한 모델의 view = 사이드 영역탭 + 본문 rpanel들."""
    ids = [j for j, _ in atoms]
    counts = {j: [len(runs[r].get(j, [])) for r in range(N_RUNS)] for j in ids}
    # 사이드 탭
    tabs = ""
    for k, j in enumerate(ids):
        c = counts[j]
        empty = " empty" if sum(c) == 0 else ""
        on = " on" if k == 0 else ""
        tabs += (f'<button class="rtab{on}{empty}" onclick="show({mi},{k})">'
                 f'<span class=rid>#{j}</span><span class=cnt>{"/".join(map(str,c))}</span></button>')
    # 본문 패널
    rectof = dict(atoms)
    panels = ""
    for k, j in enumerate(ids):
        x, y, w, h = rectof[j]
        c = counts[j]
        disp = "block" if k == 0 else "none"
        vis = (f'<span class=tog><label><input type=radio name=vis{mi}_{k} value=all checked '
               f'onchange="setVis({mi},{k},\'all\')"> 전부</label> ')
        for r in range(N_RUNS):
            vis += (f'<label><input type=radio name=vis{mi}_{k} value={r} '
                    f'onchange="setVis({mi},{k},{r})"><span style="color:{RUN_COLORS[r]}">run{r+1}</span></label> ')
        vis += "</span>"
        layers = "".join(
            f'<div class="layer run{r}" data-run="{r}">{overlay(runs[r].get(j, []), r)}</div>'
            for r in range(N_RUNS))
        tables = "".join(run_table(runs[r].get(j, []), r) for r in range(N_RUNS))
        panels += (
            f'<div class=rpanel data-mi="{mi}" style="display:{disp}">'
            f'<div class=head><b>region {j}</b> · {w}×{h}px · 3회 = {"/".join(map(str,c))} {vis}</div>'
            f'<div class=pair><div class=imgcol><div class=cap>하이라이트 + 3회 라벨 (색=회차)</div>'
            f'<div class=wrap><img src="data:image/jpeg;base64,{focus_imgs[j]}">'
            f'<div class=rguide style="left:{x/IW*100:.2f}%;top:{y/IH*100:.2f}%;'
            f'width:{w/IW*100:.2f}%;height:{h/IH*100:.2f}%"></div>{layers}</div></div>'
            f'<div class=tblcol>{tables}</div></div></div>')
    active = " active" if mi == 0 else ""
    return (f'<div class="view{active}" id=view{mi} style="display:{"flex" if mi==0 else "none"}">'
            f'<div id=side>{tabs}</div><div id=main>{panels}</div></div>')


def stats(runs, atoms):
    ids = [j for j, _ in atoms]
    same = var = blank = 0
    for j in ids:
        c = [len(runs[r].get(j, [])) for r in range(N_RUNS)]
        if sum(c) == 0:
            blank += 1
        elif len(set(c)) == 1:
            same += 1
        else:
            var += 1
    return same, var, blank


def render(data, atoms, img):
    IH, IW = img.shape[:2]
    focus_imgs = {j: b64(ground_focus.mark_focus(img, atoms, j)) for j, _ in atoms}
    mtabs, views = "", ""
    for mi, m in enumerate(data["models"]):
        same, var, blank = stats(m["runs"], atoms)
        on = " on" if mi == 0 else ""
        mtabs += (f'<button class="mtab{on}" onclick="pick({mi})">{html.escape(m["tag"])}@{m.get("temp","?")} '
                  f'<span class=mmeta>동일 {same} · 변동 {var} · 빈칸 {blank} · ${m["cost"]} · {m["secs"]}s</span></button>')
        views += render_view(mi, m["tag"], m["runs"], atoms, focus_imgs, IW, IH)
    html_doc = f"""<!doctype html><meta charset=utf-8>
<title>focus 모델비교 ×3</title>
<style>
*{{box-sizing:border-box}}
body{{margin:0;font:13px/1.4 system-ui,sans-serif;background:#0e0e0e;color:#e8e8e8;display:flex;flex-direction:column;height:100vh}}
#top{{padding:10px 16px;border-bottom:1px solid #333;background:#161616}}
h1{{margin:0 0 4px;font-size:16px}} .sub{{color:#999;font-size:12px}}
.mtab{{background:#222;color:#ccc;border:1px solid #333;border-radius:6px;padding:6px 12px;margin:6px 8px 0 0;cursor:pointer;font:inherit}}
.mtab.on{{background:#2a3d2a;border-color:#5a8;color:#fff}} .mmeta{{color:#8b8;font-size:11px;margin-left:6px}}
.legend span{{display:inline-block;width:10px;height:10px;border-radius:2px;margin:0 4px 0 10px;vertical-align:middle}}
#body{{flex:1;display:flex;min-height:0}}
.view{{flex:1;display:flex;min-height:0}}
#side,.view>#side{{flex:0 0 180px;overflow:auto;border-right:1px solid #333;padding:8px;background:#141414}}
.rtab{{display:flex;justify-content:space-between;width:100%;background:#222;color:#ccc;border:1px solid #333;
  border-radius:6px;padding:7px 10px;margin:0 0 5px;cursor:pointer;font:inherit}}
.rtab.on{{background:#2a3d2a;border-color:#5a8;color:#fff}} .rtab.empty{{opacity:.4}}
.rid{{font-weight:700}} .cnt{{font-size:10px;background:#333;border-radius:8px;padding:1px 6px;font-variant-numeric:tabular-nums}}
.rtab.on .cnt{{background:#3a5}}
#main,.view>#main{{flex:1;overflow:auto;padding:12px 16px}}
.head{{margin:0 0 10px;color:#9cf}} .tog{{float:right;color:#aaa;font-weight:400;font-size:12px}}
.tog label{{margin-left:6px;cursor:pointer}}
.pair{{display:grid;grid-template-columns:1.1fr 1fr;gap:12px;align-items:start}}
@media(max-width:1100px){{.pair{{grid-template-columns:1fr}}}}
.cap{{font-size:11px;color:#888;margin-bottom:6px}}
.wrap{{position:relative;border:1px solid #333;border-radius:4px;overflow:hidden;background:#000}}
.wrap>img{{width:100%;display:block}}
.layer{{position:absolute;inset:0;pointer-events:none}}
.rguide{{position:absolute;border:2px dashed rgba(255,200,80,.85);box-sizing:border-box;pointer-events:none;z-index:1}}
.fw{{position:absolute;border:2px solid #0af;box-sizing:border-box;overflow:hidden;font-size:8px;color:#fff;
  background:rgba(0,0,0,.35);pointer-events:auto;cursor:pointer;z-index:2}}
.fw.hi{{border-width:3px;filter:brightness(1.3);z-index:4}}
.w-dt{{background:rgba(120,60,0,.35)}} .w-ch{{background:rgba(0,80,40,.35)}}
.fl,.oc{{padding:0 2px;background:rgba(0,0,0,.55);display:inline-block;max-width:100%;overflow:hidden;white-space:nowrap}}
.tblcol{{max-height:80vh;overflow:auto;display:flex;flex-direction:column;gap:10px}}
.runcard{{background:#151515;border:1px solid #333;border-radius:6px;padding:6px 8px}}
.runtitle{{margin-bottom:4px;font-size:12px}}
table{{border-collapse:collapse;width:100%;font-size:11px}}
th,td{{border:1px solid #333;padding:3px 5px;text-align:left}} th{{background:#1c1c1c}}
tr.er{{cursor:pointer}} tr.er:hover{{background:#1a2430}} tr.er.on{{background:#2a1a1a}}
.bx{{font-size:10px;color:#888}} .muted{{color:#666;text-align:center}}
</style>
<div id=top>
<h1>focus · 영역별 ×3 · 모델 비교</h1>
<div class=sub>서식2호 · 영역별 focus ×3 · effort={EFFORT} · 저온 안정성 (모델@temp) · 기준선 gemini-3.1-pro@1.0 = 동일22/변동3/빈칸25</div>
<div class=legend>
  <span style="background:{RUN_COLORS[0]}"></span>run1
  <span style="background:{RUN_COLORS[1]}"></span>run2
  <span style="background:{RUN_COLORS[2]}"></span>run3
</div>
<div style="margin-top:4px">{mtabs}</div>
</div>
<div id=body>{views}</div>
<script>
function pick(mi){{
  document.querySelectorAll('.view').forEach((v,j)=>{{v.style.display=j===mi?'flex':'none';v.classList.toggle('active',j===mi)}});
  document.querySelectorAll('.mtab').forEach((t,j)=>t.classList.toggle('on',j===mi));
}}
function show(mi,i){{
  const v=document.getElementById('view'+mi);
  v.querySelectorAll('.rpanel').forEach((p,j)=>p.style.display=j===i?'block':'none');
  v.querySelectorAll('.rtab').forEach((t,j)=>t.classList.toggle('on',j===i));
}}
function setVis(mi,i,mode){{
  const v=document.getElementById('view'+mi);
  const p=v.querySelectorAll('.rpanel')[i];
  p.querySelectorAll('.layer').forEach(l=>{{l.style.display=(mode==='all'||+l.dataset.run===+mode)?'block':'none'}});
}}
function hilite(p,run,n){{
  p.querySelectorAll('.fw').forEach(b=>b.classList.toggle('hi',+b.dataset.run===run&&+b.dataset.n===n));
  p.querySelectorAll('tr.er').forEach(tr=>tr.classList.toggle('on',+tr.dataset.run===run&&+tr.dataset.n===n));
}}
document.addEventListener('click',e=>{{
  const fw=e.target.closest('.fw'); const tr=e.target.closest('tr.er');
  const p=(fw||tr)?.closest('.rpanel'); if(!p) return;
  const src=fw||tr; hilite(p,+src.dataset.run,+src.dataset.n);
}});
document.addEventListener('keydown',e=>{{
  const v=document.querySelector('.view.active'); if(!v) return;
  const tabs=[...v.querySelectorAll('.rtab')];
  const cur=tabs.findIndex(t=>t.classList.contains('on'));
  const mi=+v.id.replace('view','');
  if(e.key==='ArrowDown'||e.key==='j'){{e.preventDefault();show(mi,Math.min(tabs.length-1,cur+1))}}
  if(e.key==='ArrowUp'||e.key==='k'){{e.preventDefault();show(mi,Math.max(0,cur-1))}}
}});
</script>"""
    OUT.write_text(html_doc)
    print(f"\n→ {OUT}  ({OUT.stat().st_size/1e6:.1f}MB)")


def main():
    img = cv2.imread(IMG)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    from core.region_segment import segment
    atoms = pipeline.atoms_of(segment(gray))
    print(f"atoms={len(atoms)}")

    by_tag = {}
    if CACHE.exists():
        for m in json.loads(CACHE.read_text())["models"]:
            m["runs"] = [{int(k): v for k, v in run.items()} for run in m["runs"]]  # region id int 복원
            by_tag[m["tag"]] = m

    by_tag = run_missing(img, atoms, by_tag)

    # MODELS 순서대로 렌더 (캐시에만 있고 목록에 없는 것도 뒤에)
    order = [t for t, _, _ in MODELS] + [t for t in by_tag if t not in {x[0] for x in MODELS}]
    data = {"models": [by_tag[t] for t in order if t in by_tag]}
    render(data, atoms, img)


if __name__ == "__main__":
    main()
