"""서식2호 LLM ground N회 → carve(place)까지 비교 HTML.

실행: python _verify/ground_runs.py
입력: /tmp/img2form_test/서식2호_5runs/run{1..N}_full.json
출력: _verify/ground_runs.html
"""
from __future__ import annotations

import base64
import html
import json
import sys
from collections import Counter
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core import pipeline  # noqa: E402

IMG = ("/Users/gunhee/workspace/codespace/project/lab-voucher/.old/document/"
       "제1편_장애아가족_양육지원_서식/_extract/서식2호_장애아돌보미_지원서/pages/p-1.png")
RUNS_DIR = Path("/tmp/img2form_test/서식2호_5runs_so")  # SO+assign_keys (이전: 서식2호_5runs)
OUT = ROOT / "_verify" / "ground_runs.html"

WCLS = {"text": "w-txt", "textarea": "w-txt", "number": "w-num", "email": "w-txt", "phone": "w-txt",
        "date": "w-dt", "time": "w-dt", "radio": "w-ch", "checkbox_group": "w-ch", "select": "w-ch",
        "consent": "w-cn", "signature": "w-sg", "image": "w-img"}

RULE_KO = {"_snap_mark": "□스냅", "_refine": "빈칸축소", "_ocr_anchor": "OCR앵커", "_ocr_word": "OCR문구",
           "_radio_pair": "radio쌍", "_lr_pair": "L/R쌍", "_date_inline": "날짜카브", "_snap_cell": "격자스냅",
           "_fit_bounded": "경계맞춤", "_fit_word": "글자맞춤", "_fit_ink": "잉크맞춤", "_after_label": "라벨뒤",
           "_placeholder": "플레이스홀더", "_fill_cell": "셀채움", "_cb_assign": "□배정", "_keep": "유지"}


def b64(im, q=72):
    return base64.b64encode(cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, q])[1]).decode()


def overlay_placed(items, IW, IH, tab):
    ov = ""
    for i, it in enumerate(items):
        rx, ry, rw, rh = it["rect"]
        t = it["type"]
        opt = it.get("option")
        lab = it.get("label") or it.get("key") or ""
        cls = WCLS.get(t, "w-txt")
        inner = (f'<span class=oc>{html.escape(str(opt)[:10])}</span>' if opt
                 else f'<span class=fl>{html.escape(str(lab)[:14])}</span>')
        ymin, xmin, ymax, xmax = it.get("box") or (0, 0, 0, 0)
        da = (f'data-n="{i}" data-key="{html.escape(str(it.get("key") or ""))}" '
              f'data-label="{html.escape(lab)}" data-type="{t}" '
              f'data-opt="{html.escape(str(opt or ""))}" data-unit="{html.escape(str(it.get("unit") or ""))}" '
              f'data-region="{it.get("region", "")}" data-rule="{html.escape(it.get("rule") or "")}" '
              f'data-corr="{1 if it.get("corrected") else 0}" '
              f'data-rect="{rx},{ry},{rw},{rh}" data-box="{ymin},{xmin},{ymax},{xmax}"')
        ov += (f'<div class="fw {cls}" style="left:{rx/IW*100:.2f}%;top:{ry/IH*100:.2f}%;'
               f'width:{rw/IW*100:.2f}%;height:{rh/IH*100:.2f}%" {da} '
               f'title="{html.escape(lab)} ({t}) · {html.escape(it.get("rule") or "")}">{inner}</div>')
    return ov


def overlay_llm_ghost(raw, IW, IH):
    """carve 위에 LLM 원본 box를 점선으로 겹쳐 비교."""
    ov = ""
    for e in raw:
        bx = e.get("box")
        if not bx or len(bx) != 4:
            continue
        ymin, xmin, ymax, xmax = bx
        rx = xmin / 1000 * IW; ry = ymin / 1000 * IH
        rw = (xmax - xmin) / 1000 * IW; rh = (ymax - ymin) / 1000 * IH
        ov += (f'<div class=ghost style="left:{rx/IW*100:.2f}%;top:{ry/IH*100:.2f}%;'
               f'width:{rw/IW*100:.2f}%;height:{rh/IH*100:.2f}%"></div>')
    return ov


def rows_placed(items):
    rows = ""
    for i, it in enumerate(items, 1):
        t = it["type"]
        opt = it.get("option") or ""
        unit = it.get("unit") or ""
        rect = it["rect"]
        rule = RULE_KO.get(it.get("rule") or "", it.get("rule") or "")
        corr = "✓" if it.get("corrected") else ""
        rows += (
            f'<tr class=er data-n="{i-1}">'
            f'<td>{i}</td><td>{it.get("region", "")}</td>'
            f'<td><span class="tag t-{html.escape(t)}">{html.escape(t)}</span></td>'
            f'<td><code>{html.escape(str(it.get("key") or ""))}</code></td>'
            f'<td>{html.escape(str(it.get("label") or ""))}</td>'
            f'<td>{html.escape(str(opt))}</td>'
            f'<td class=rule>{html.escape(rule)}</td>'
            f'<td class=corr>{corr}</td>'
            f'<td class=bx>{html.escape(str(rect))}</td></tr>'
        )
    return rows


def type_table(runs_data, key="types"):
    all_types = sorted({t for r in runs_data for t in r[key]})
    hdr = "<tr><th>type</th>" + "".join(f"<th>run{r['n']}</th>" for r in runs_data) + "<th>min~max</th></tr>"
    body = ""
    for t in all_types:
        counts = [r[key].get(t, 0) for r in runs_data]
        mn, mx = min(counts), max(counts)
        cls = "diff" if mn != mx else ""
        body += f'<tr class="{cls}"><td>{html.escape(t)}</td>'
        body += "".join(f"<td>{c}</td>" for c in counts)
        body += f"<td>{mn}~{mx}</td></tr>"
    total_row = "<tr class=tot><td><b>합계</b></td>"
    totals = [r["total"] for r in runs_data]
    total_row += "".join(f"<td><b>{t}</b></td>" for t in totals)
    total_row += f"<td>{min(totals)}~{max(totals)}</td></tr>"
    return hdr + body + total_row


def grid_carved(img_b64, runs_data, IW, IH):
    cells = ""
    for r in runs_data:
        tab_id = f"g{r['n']}"
        cells += (
            f'<div class=gc><div class=gl>run{r["n"]} · {r["total"]}요소 · 수정{r["ncorr"]}</div>'
            f'<div class=canvas><img src="data:image/jpeg;base64,{img_b64}">'
            f'{overlay_placed(r["placed"], IW, IH, tab_id)}</div></div>'
        )
    return cells


def main():
    global RUNS_DIR
    if len(sys.argv) > 1:
        RUNS_DIR = Path(sys.argv[1])
    run_files = sorted(RUNS_DIR.glob("run*_full.json"))
    if not run_files:
        raise SystemExit(f"없음: {RUNS_DIR}/run*_full.json — 먼저 5회 호출 필요")

    print(f"입력: {RUNS_DIR}")
    img = cv2.imread(IMG)
    if img is None:
        raise SystemExit(f"이미지 없음: {IMG}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    S = pipeline.segment(gray)
    IH, IW = gray.shape
    img_b64 = b64(img)

    runs_data = []
    for fp in run_files:
        n = int(fp.stem.replace("run", "").replace("_full", ""))
        raw = json.loads(fp.read_text(encoding="utf-8")).get("elements") or []
        placed = pipeline.place_elements(gray, S, raw)
        ncorr = sum(1 for it in placed if it.get("corrected"))
        runs_data.append({
            "n": n, "raw": raw, "placed": placed, "total": len(placed), "ncorr": ncorr,
            "types": dict(Counter(e.get("type") for e in raw)),
            "rules": dict(Counter(it.get("rule") for it in placed if it.get("corrected"))),
        })
    runs_data.sort(key=lambda r: r["n"])

    tabs = panels = ""
    for i, r in enumerate(runs_data):
        on = i == 0
        tabs += (f'<button class="tab{" on" if on else ""}" onclick="show({i})">'
                 f'run{r["n"]} ({r["total"]})</button>')
        type_s = " · ".join(f"{k}={v}" for k, v in sorted(r["types"].items()))
        rules_s = " · ".join(f"{RULE_KO.get(k, k)}={v}" for k, v in sorted(r["rules"].items()))
        panels += (
            f'<div class=view id=v{i} style="display:{"flex" if on else "none"}">'
            f'<div class=main>'
            f'<div class=mh>run{r["n"]} · ④ carve 배치 {r["total"]}개 (수정 {r["ncorr"]}) — 점선=LLM원본</div>'
            f'<div class=typebar>LLM타입: {html.escape(type_s)}</div>'
            f'<div class=typebar>배치규칙: {html.escape(rules_s)}</div>'
            f'<div class=canvas><img src="data:image/jpeg;base64,{img_b64}">'
            f'{overlay_llm_ghost(r["raw"], IW, IH)}'
            f'{overlay_placed(r["placed"], IW, IH, i)}</div></div>'
            f'<div class=list><div class=lh>배치 목록 ({r["total"]})</div>'
            f'<div class=lt><table><thead><tr>'
            f'<th>#</th><th>rgn</th><th>type</th><th>key</th><th>label</th><th>opt</th>'
            f'<th>rule</th><th>수정</th><th>rect</th>'
            f'</tr></thead><tbody id=tb{i}>{rows_placed(r["placed"])}</tbody></table></div></div>'
            f'</div>'
        )

    cmp_i = len(runs_data)
    tabs += f'<button class="tab" onclick="show({cmp_i})">비교요약</button>'
    panels += (
        f'<div class=view id=v{cmp_i} style="display:none">'
        f'<div class=cmp>'
        f'<h2>타입별 개수 (LLM)</h2><table class=ct>{type_table(runs_data)}</table>'
        f'<h2>5회 carve 썸네일</h2><div class=grid>{grid_carved(img_b64, runs_data, IW, IH)}</div>'
        f'</div></div>'
    )

    page = _PAGE.replace("__TABS__", tabs).replace("__PANELS__", panels)
    OUT.write_text(page, encoding="utf-8")
    print(f"✓ {OUT}  ({len(runs_data)} runs, carve 포함)")


_PAGE = r"""<!doctype html><meta charset=utf-8><title>서식2호 LLM 5회 · carve 비교</title>
<style>*{box-sizing:border-box} body{margin:0;font:13px/1.45 system-ui,sans-serif;background:#f1f3f5}
#bar{position:sticky;top:0;background:#212529;padding:9px 14px;z-index:10}
#bar h1{font-size:14px;color:#fff;margin:0 0 3px}#bar .l{color:#adb5bd;font-size:11.5px;margin-bottom:7px}
.tab{border:0;background:#495057;color:#ced4da;padding:5px 10px;border-radius:6px;cursor:pointer;font-size:12px;margin:0 3px 4px 0}
.tab.on{background:#4dabf7;color:#fff;font-weight:700}
.view{display:flex;gap:10px;height:calc(100vh - 58px);padding:10px}
.main{flex:1;overflow:auto;background:#fff;border:1px solid #dee2e6;border-radius:8px;min-width:0}
.mh{padding:8px 12px;font-size:12px;font-weight:700;border-bottom:1px solid #eef1f4;position:sticky;top:0;background:#fff;z-index:2}
.typebar{padding:4px 12px;font-size:10.5px;color:#868e96;border-bottom:1px solid #f1f3f5;word-break:break-all}
.canvas{position:relative;display:inline-block} .canvas>img{display:block;max-width:100%}
.ghost{position:absolute;border:1px dashed rgba(120,120,120,.55);background:rgba(180,180,180,.06);pointer-events:none;z-index:1}
.fw{position:absolute;border:2px solid;border-radius:2px;overflow:hidden;font-size:8px;cursor:pointer;z-index:2}
.fl{font-size:7.5px;font-weight:700;color:#111;background:rgba(255,255,255,.75);padding:0 2px}
.oc{font-size:7px;background:rgba(255,255,255,.85);border:1px solid rgba(0,0,0,.12);border-radius:4px;padding:0 2px}
.fw.sel,.er.sel{outline:2px solid #f03e3e;background:rgba(240,62,62,.08)!important}
.w-txt{border-color:#1971c2;background:rgba(25,113,194,.12)} .w-num{border-color:#0c8599;background:rgba(12,133,153,.12)}
.w-dt{border-color:#9c36b5;background:rgba(156,54,181,.12)} .w-ch{border-color:#e8590c;background:rgba(232,89,12,.13)}
.w-cn{border-color:#2b8a3e;background:rgba(43,138,62,.12)} .w-sg{border-color:#2f9e44;background:rgba(47,158,68,.14)} .w-img{border-color:#6741d9;background:rgba(103,65,217,.11)}
.list{flex:0 0 44%;overflow:hidden;display:flex;flex-direction:column;background:#fff;border:1px solid #dee2e6;border-radius:8px}
.lh{padding:8px 12px;font-size:12px;font-weight:700;border-bottom:1px solid #eef1f4}
.lt{overflow:auto;flex:1} table{width:100%;border-collapse:collapse;font-size:11px}
th,td{padding:4px 6px;border-bottom:1px solid #f1f3f5;text-align:left;vertical-align:top}
th{position:sticky;top:0;background:#f8f9fa;font-size:10.5px;z-index:1}
.er{cursor:pointer} .er:hover{background:#f8f9fa}
td.bx{font-size:9.5px;color:#868e96;max-width:100px;word-break:break-all}
td.rule{font-size:10px;color:#495057} td.corr{color:#2b8a3e;font-weight:700;text-align:center}
.tag{display:inline-block;font-size:9.5px;font-weight:700;padding:1px 5px;border-radius:4px;color:#fff}
.t-text,.t-textarea,.t-email,.t-phone{background:#1971c2} .t-number{background:#0c8599} .t-date,.t-time{background:#9c36b5}
.t-checkbox_group,.t-radio{background:#e8590c} .t-consent{background:#2b8a3e} .t-signature{background:#2f9e44} .t-image{background:#6741d9}
.cmp{flex:1;overflow:auto;padding:14px 18px;background:#fff;border:1px solid #dee2e6;border-radius:8px}
.cmp h2{font-size:13px;margin:18px 0 8px} .cmp h2:first-child{margin-top:0}
.ct td,.ct th{padding:6px 10px;border:1px solid #e9ecef} .ct th{background:#f8f9fa}
.ct tr.diff td{background:#fff3bf} .ct tr.tot td{background:#e7f5ff;font-weight:600}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:12px}
.gc{border:1px solid #dee2e6;border-radius:8px;overflow:hidden} .gl{background:#f8f9fa;padding:5px 9px;font-size:11px;font-weight:700}
#info{position:fixed;right:12px;bottom:12px;background:#fff;border:1px solid #dee2e6;border-radius:8px;padding:10px 12px;
  font-size:11px;max-width:280px;box-shadow:0 4px 12px rgba(0,0,0,.12);display:none;z-index:20}
#info b{display:block;margin-bottom:4px;font-size:12px}</style>
<div id=bar><h1>서식2호 · Structured Output + key코드 · 5회 → carve</h1>
<div class=l>실선=carve · 점선=LLM box · key=r&#123;region&#125;_&#123;type&#125;_… · 비교요약 탭</div>__TABS__</div>__PANELS__
<div id=info></div>
<script>
const $=s=>[...document.querySelectorAll(s)];
function show(i){$('.view').forEach((v,j)=>v.style.display=j==i?(i<$('.view').length-1?'flex':'block'):'none');
  $('.tab').forEach((t,j)=>t.classList.toggle('on',j==i)); document.getElementById('info').style.display='none'}
function esc(s){return (s||'').replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]))}
function highlight(view, n){
  view.querySelectorAll('.fw.sel,.er.sel').forEach(x=>x.classList.remove('sel'));
  const fw=view.querySelector(`.fw[data-n="${n}"]`), tr=view.querySelector(`#tb${view.id.slice(1)} tr[data-n="${n}"]`);
  if(fw){
    fw.classList.add('sel');
    const d=fw.dataset, inf=document.getElementById('info');
    inf.style.display='block';
    inf.innerHTML=`<b>${esc(d.label)}</b>타입: ${esc(d.type)}<br>key: <code>${esc(d.key)}</code><br>`
      +(d.opt?`option: ${esc(d.opt)}<br>`:'')+(d.unit?`unit: ${esc(d.unit)}<br>`:'')
      +`rule: ${esc(d.rule)} ${d.corr==='1'?'· <b style=color:#2b8a3e>수정됨</b>':''}<br>`
      +`rect: ${esc(d.rect)}<br>LLM box: ${esc(d.box)}`;
  }
  if(tr){tr.classList.add('sel');tr.scrollIntoView({block:'nearest'})}
}
$('.view').forEach(v=>{
  if(!v.querySelector('.canvas .fw')) return;
  v.querySelectorAll('.fw').forEach(b=>b.addEventListener('click',e=>{e.stopPropagation();highlight(v,b.dataset.n)}));
  v.querySelectorAll('.er').forEach(tr=>tr.addEventListener('click',()=>highlight(v,tr.dataset.n)));
});
</script>"""


if __name__ == "__main__":
    main()
