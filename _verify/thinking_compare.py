"""low / medium / high thinking 각 5회 → carve 비교 HTML.

실행: python _verify/thinking_compare.py
입력:
  /tmp/img2form_test/서식2호_5runs_so      (low)
  /tmp/img2form_test/서식2호_5runs_medium
  /tmp/img2form_test/서식2호_5runs_high
출력: _verify/thinking_compare.html
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

# reuse overlays from ground_runs
from _verify.ground_runs import (  # noqa: E402
    RULE_KO, WCLS, b64, overlay_placed, overlay_llm_ghost, rows_placed,
)

IMG = ("/Users/gunhee/workspace/codespace/project/lab-voucher/.old/document/"
       "제1편_장애아가족_양육지원_서식/_extract/서식2호_장애아돌보미_지원서/pages/p-1.png")
DIRS = [
    ("low", Path("/tmp/img2form_test/서식2호_5runs_so")),
    ("medium", Path("/tmp/img2form_test/서식2호_5runs_medium")),
    ("high", Path("/tmp/img2form_test/서식2호_5runs_high")),
]
OUT = ROOT / "_verify" / "thinking_compare.html"


def load_group(effort, d):
    rows = []
    for i in range(1, 6):
        fp = d / f"run{i}_full.json"
        if not fp.exists():
            raise SystemExit(f"없음: {fp}")
        data = json.loads(fp.read_text(encoding="utf-8"))
        raw = data.get("elements") or []
        meta = data.get("_meta") or {}
        usage = meta.get("usage") or {}
        rows.append({
            "effort": effort, "n": i, "raw": raw,
            "meta": meta, "usage": usage,
            "total_llm": len(raw),
            "types": dict(Counter(e.get("type") for e in raw)),
        })
    return rows


def metric_row(label, groups, fn):
    cells = "".join(f"<td>{html.escape(str(fn(g)))}</td>" for g in groups)
    return f"<tr><td><b>{html.escape(label)}</b></td>{cells}</tr>"


def type_matrix(groups):
    """effort × type 평균/범위."""
    all_types = sorted({t for g in groups for r in g for t in r["types"]})
    hdr = "<tr><th>type</th>" + "".join(
        f"<th>{html.escape(g[0]['effort'])}</th>" for g in groups
    ) + "</tr>"
    body = ""
    for t in all_types:
        cols = []
        vals_all = []
        for g in groups:
            vals = [r["types"].get(t, 0) for r in g]
            vals_all.append(vals)
            mn, mx = min(vals), max(vals)
            mean = sum(vals) / len(vals)
            s = f"{mean:.1f} ({mn}~{mx})"
            cls = "diff" if mn != mx else "ok"
            cols.append(f'<td class="{cls}">{s}</td>')
        # highlight if efforts differ a lot
        means = [sum(v) / len(v) for v in vals_all]
        row_cls = "diff" if max(means) - min(means) >= 1 else ""
        body += f'<tr class="{row_cls}"><td>{html.escape(t)}</td>{"".join(cols)}</tr>'
    # totals
    tot_cols = []
    for g in groups:
        vals = [r["total_llm"] for r in g]
        tot_cols.append(
            f'<td class="tot"><b>{sum(vals)/len(vals):.1f}</b> ({min(vals)}~{max(vals)})</td>'
        )
    body += f'<tr class=tot><td><b>합계</b></td>{"".join(tot_cols)}</tr>'
    return hdr + body


def per_run_totals_table(groups):
    hdr = "<tr><th>effort</th>" + "".join(f"<th>run{i}</th>" for i in range(1, 6)) + "<th>mean</th></tr>"
    body = ""
    for g in groups:
        vals = [r["total_llm"] for r in g]
        body += (
            f'<tr><td><b>{html.escape(g[0]["effort"])}</b></td>'
            + "".join(f"<td>{v}</td>" for v in vals)
            + f'<td><b>{sum(vals)/len(vals):.1f}</b></td></tr>'
        )
    return hdr + body


def cost_time_table(groups):
    hdr = ("<tr><th>effort</th><th>지연(초)</th><th>비용($)</th>"
           "<th>completion tok</th><th>reasoning tok</th></tr>")
    body = ""
    for g in groups:
        secs, costs, comps, reasons = [], [], [], []
        for r in g:
            meta, u = r["meta"], r["usage"]
            if meta.get("sec") is not None:
                secs.append(meta["sec"])
            if u.get("cost") is not None:
                costs.append(u["cost"])
            if u.get("completion_tokens") is not None:
                comps.append(u["completion_tokens"])
            det = u.get("completion_tokens_details") if isinstance(u.get("completion_tokens_details"), dict) else {}
            rt = det.get("reasoning_tokens") if det else u.get("reasoning_tokens")
            if rt is not None:
                reasons.append(rt)

        def fmt(xs, money=False):
            if not xs:
                return "—"
            if money:
                return f"{sum(xs)/len(xs):.3f} (Σ{sum(xs):.3f})"
            return f"{sum(xs)/len(xs):.0f} ({min(xs)}~{max(xs)})"

        body += (
            f'<tr><td><b>{html.escape(g[0]["effort"])}</b></td>'
            f'<td>{fmt(secs)}</td><td>{fmt(costs, True)}</td>'
            f'<td>{fmt(comps)}</td><td>{fmt(reasons)}</td></tr>'
        )
    return hdr + body


def thumb_grid(group, img_ph, IW, IH):
    """img_ph = placeholder class; JS fills src from PAGE_IMG."""
    cells = ""
    for r in group:
        uid = f'{r["effort"]}{r["n"]}'
        meta = r["meta"]
        sec = meta.get("sec")
        cost = (r["usage"] or {}).get("cost")
        extra = ""
        if sec is not None:
            extra += f" · {sec}s"
        if cost is not None:
            extra += f" · ${cost:.3f}"
        cells += (
            f'<div class=gc data-goto="{uid}">'
            f'<div class=gl>{r["effort"]}·run{r["n"]} · {r["total"]}요소 · 수정{r["ncorr"]}{extra}</div>'
            f'<div class=canvas>'
            f'<img class=pageimg alt="">'
            f'{overlay_placed(r["placed"], IW, IH, uid)}'
            f'</div></div>'
        )
    return f'<div class=grid>{cells}</div>'


def main():
    for _, d in DIRS:
        if not d.is_dir():
            raise SystemExit(f"디렉터리 없음: {d}")

    img = cv2.imread(IMG)
    if img is None:
        raise SystemExit(f"이미지 없음: {IMG}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    S = pipeline.segment(gray)
    IH, IW = gray.shape
    img_b64 = b64(img)

    groups = []
    flat = []  # detail panels order
    for effort, d in DIRS:
        print(f"load {effort} ← {d}")
        g = load_group(effort, d)
        for r in g:
            print(f"  carve {effort} run{r['n']} ...", flush=True)
            placed = pipeline.place_elements(gray, S, r["raw"])
            r["placed"] = placed
            r["total"] = len(placed)
            r["ncorr"] = sum(1 for it in placed if it.get("corrected"))
            flat.append(r)
        groups.append(g)

    # --- summary panel (index 0) ---
    tabs = '<button class="tab on" onclick="show(0)">비교요약</button>'
    summary = (
        '<div class=view id=v0 style="display:block">'
        '<div class=cmp>'
        '<h2>요소 수 (LLM) · run별</h2>'
        f'<table class=ct>{per_run_totals_table(groups)}</table>'
        '<h2>타입별 평균 (min~max)</h2>'
        f'<table class=ct>{type_matrix(groups)}</table>'
        '<h2>지연 · 비용 · 토큰</h2>'
        f'<table class=ct>{cost_time_table(groups)}</table>'
        '<p class=note>노란 칸=effort 내 분산. low는 SO+assign_keys·reasoning low. '
        'medium/high는 동일 SO에 effort만 변경. 썸네일 클릭→상세 탭.</p>'
    )
    for g in groups:
        effort = g[0]["effort"]
        summary += f'<h2>{html.escape(effort)} · carve 5회</h2>{thumb_grid(g, img_b64, IW, IH)}'
    summary += '</div></div>'

    panels = summary
    # --- detail panels ---
    id_map = {}  # "low1" -> panel index
    for i, r in enumerate(flat):
        idx = i + 1
        uid = f'{r["effort"]}{r["n"]}'
        id_map[uid] = idx
        label = f'{r["effort"]}·{r["n"]} ({r["total"]})'
        tabs += f'<button class="tab" onclick="show({idx})">{html.escape(label)}</button>'
        type_s = " · ".join(f"{k}={v}" for k, v in sorted(r["types"].items()))
        meta = r["meta"]
        u = r["usage"]
        meta_s = f'effort={r["effort"]}'
        if meta.get("sec") is not None:
            meta_s += f' · {meta["sec"]}s'
        if u.get("cost") is not None:
            meta_s += f' · ${u["cost"]:.4f}'
        if u.get("completion_tokens") is not None:
            meta_s += f' · out={u["completion_tokens"]}'
        det = u.get("completion_tokens_details") if isinstance(u.get("completion_tokens_details"), dict) else {}
        rt = det.get("reasoning_tokens") if det else None
        if rt:
            meta_s += f' · reason={rt}'

        panels += (
            f'<div class=view id=v{idx} style="display:none">'
            f'<div class=main>'
            f'<div class=mh>{html.escape(r["effort"])} run{r["n"]} · carve {r["total"]} '
            f'(수정 {r["ncorr"]}) — 점선=LLM box</div>'
            f'<div class=typebar>{html.escape(meta_s)}</div>'
            f'<div class=typebar>타입: {html.escape(type_s)}</div>'
            f'<div class=canvas><img class=pageimg alt="">'
            f'{overlay_llm_ghost(r["raw"], IW, IH)}'
            f'{overlay_placed(r["placed"], IW, IH, idx)}</div></div>'
            f'<div class=list><div class=lh>배치 목록 ({r["total"]})</div>'
            f'<div class=lt><table><thead><tr>'
            f'<th>#</th><th>rgn</th><th>type</th><th>key</th><th>label</th><th>opt</th>'
            f'<th>rule</th><th>수정</th><th>rect</th>'
            f'</tr></thead><tbody id=tb{idx}>{rows_placed(r["placed"])}</tbody></table></div></div>'
            f'</div>'
        )

    id_map_js = json.dumps(id_map)
    page = (_PAGE
            .replace("__TABS__", tabs)
            .replace("__PANELS__", panels)
            .replace("__IMG__", img_b64)
            .replace("__IDMAP__", id_map_js))
    OUT.write_text(page, encoding="utf-8")
    print(f"✓ {OUT}  (15 runs: low/medium/high ×5)")


_PAGE = r"""<!doctype html><meta charset=utf-8><title>thinking low·medium·high ×5 비교</title>
<style>*{box-sizing:border-box} body{margin:0;font:13px/1.45 system-ui,sans-serif;background:#f1f3f5}
#bar{position:sticky;top:0;background:#212529;padding:9px 14px;z-index:10}
#bar h1{font-size:14px;color:#fff;margin:0 0 3px}#bar .l{color:#adb5bd;font-size:11.5px;margin-bottom:7px}
.tab{border:0;background:#495057;color:#ced4da;padding:5px 9px;border-radius:6px;cursor:pointer;font-size:11.5px;margin:0 3px 4px 0}
.tab.on{background:#4dabf7;color:#fff;font-weight:700}
.view{display:flex;gap:10px;height:calc(100vh - 62px);padding:10px}
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
.list{flex:0 0 42%;overflow:hidden;display:flex;flex-direction:column;background:#fff;border:1px solid #dee2e6;border-radius:8px}
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
.ct td.diff,.ct tr.diff td{background:#fff3bf} .ct td.ok{background:#ebfbee} .ct tr.tot td,.ct td.tot{background:#e7f5ff;font-weight:600}
.note{color:#868e96;font-size:11.5px;margin:8px 0 0}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(240px,1fr));gap:10px}
.gc{border:1px solid #dee2e6;border-radius:8px;overflow:hidden;cursor:pointer} .gc:hover{outline:2px solid #4dabf7}
.gl{background:#f8f9fa;padding:5px 9px;font-size:10.5px;font-weight:700}
#info{position:fixed;right:12px;bottom:12px;background:#fff;border:1px solid #dee2e6;border-radius:8px;padding:10px 12px;
  font-size:11px;max-width:280px;box-shadow:0 4px 12px rgba(0,0,0,.12);display:none;z-index:20}
#info b{display:block;margin-bottom:4px;font-size:12px}</style>
<div id=bar><h1>서식2호 · thinking low / medium / high × 각 5회 (SO+key코드 → carve)</h1>
<div class=l>비교요약: 표+썸네일 · 탭: effort·run 상세 · 썸네일 클릭=해당 상세</div>__TABS__</div>__PANELS__
<div id=info></div>
<script>
const PAGE_IMG="data:image/jpeg;base64,__IMG__";
const IDMAP=__IDMAP__;
const $=s=>[...document.querySelectorAll(s)];
document.querySelectorAll('img.pageimg').forEach(im=>{im.src=PAGE_IMG});
function show(i){
  $('.view').forEach((v,j)=>{
    v.style.display = j===i ? (i===0 ? 'block' : 'flex') : 'none';
  });
  $('.tab').forEach((t,j)=>t.classList.toggle('on',j===i));
  document.getElementById('info').style.display='none';
}
function esc(s){return (s||'').replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]))}
function highlight(view, n){
  view.querySelectorAll('.fw.sel,.er.sel').forEach(x=>x.classList.remove('sel'));
  const fw=view.querySelector(`.fw[data-n="${n}"]`);
  const tb=view.querySelector(`[id^="tb"]`);
  const tr=tb?tb.querySelector(`tr[data-n="${n}"]`):null;
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
$('.gc').forEach(g=>g.addEventListener('click',()=>{
  const id=g.dataset.goto; if(id!=null && IDMAP[id]!=null) show(IDMAP[id]);
}));
</script>"""


if __name__ == "__main__":
    main()
