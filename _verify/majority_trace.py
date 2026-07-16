"""majority 병합 과정 시각화.

실행: python _verify/majority_trace.py
입력: /tmp/img2form_test/서식2호_majority_low5/run{1..5}_llm.json
출력: _verify/majority_trace.html
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
from core import extract  # noqa: E402

IMG = ("/Users/gunhee/workspace/codespace/project/lab-voucher/.old/document/"
       "제1편_장애아가족_양육지원_서식/_extract/서식2호_장애아돌보미_지원서/pages/p-1.png")
RUN_DIR = Path("/tmp/img2form_test/서식2호_majority_low5")
OUT = ROOT / "_verify" / "majority_trace.html"

RUN_COLORS = ["#e03131", "#f08c00", "#2f9e44", "#1971c2", "#9c36b5"]  # run0..4


def b64(im, q=72):
    return base64.b64encode(cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, q])[1]).decode()


def box_pct(box, IW, IH):
    ymin, xmin, ymax, xmax = box
    return (xmin / 1000 * 100, ymin / 1000 * 100,
            (xmax - xmin) / 1000 * 100, (ymax - ymin) / 1000 * 100)


def rect_div(box, IW, IH, cls, label="", title="", data=""):
    if not box or len(box) != 4:
        return ""
    l, t, w, h = box_pct(box, IW, IH)
    lab = f'<span class=fl>{html.escape(label[:16])}</span>' if label else ""
    return (f'<div class="{cls}" style="left:{l:.2f}%;top:{t:.2f}%;width:{w:.2f}%;height:{h:.2f}%" '
            f'title="{html.escape(title)}" {data}>{lab}</div>')


def main():
    runs = []
    for i in range(1, 6):
        fp = RUN_DIR / f"run{i}_llm.json"
        if not fp.exists():
            raise SystemExit(f"없음: {fp} — 먼저 majority low5 실행 필요")
        runs.append(json.loads(fp.read_text())["elements"])

    trace = extract.majority_merge_trace(runs, min_votes=3, iou_thresh=0.25)
    clusters = trace["clusters"]
    kept_n = sum(1 for c in clusters if c["kept"])
    drop_n = len(clusters) - kept_n

    img = cv2.imread(IMG)
    IH, IW = img.shape[:2]
    img_b64 = b64(img)

    # --- overlays ---
    # overview: kept green median + dropped red median
    ov_keep = ov_drop = ""
    for c in clusters:
        box = c["median_box"]
        sig = c["sig"]
        title = (f'#{c["id"]} votes={c["votes"]} {sig["type"]} r{sig["region"]} '
                 f'opt={sig["option"]} unit={sig["unit"]} · {c["label"]}')
        data = f'data-cid="{c["id"]}"'
        if c["kept"]:
            ov_keep += rect_div(box, IW, IH, "bx keep", f'{c["votes"]} {c["label"]}', title, data)
        else:
            ov_drop += rect_div(box, IW, IH, "bx drop", f'{c["votes"]} {c["label"]}', title, data)

    # members per run colors on one canvas (all clusters faint)
    ov_members = ""
    for c in clusters:
        for m in c["members"]:
            col = RUN_COLORS[m["run"]]
            ov_members += rect_div(
                m["box"], IW, IH, "bx mem", "",
                f'run{m["run"]+1} · cluster#{c["id"]} · votes={c["votes"]} · {m.get("label")}',
                f'data-cid="{c["id"]}" style="border-color:{col};background:{col}22"',
            )

    # fix: rect_div already has style= - can't pass style in data easily. rewrite mem overlay.
    ov_members = ""
    for c in clusters:
        for m in c["members"]:
            if not m.get("box") or len(m["box"]) != 4:
                continue
            col = RUN_COLORS[m["run"]]
            l, t, w, h = box_pct(m["box"], IW, IH)
            title = (f'run{m["run"]+1} cluster#{c["id"]} votes={c["votes"]} '
                     f'{m.get("type")} · {m.get("label")}')
            ov_members += (
                f'<div class="bx mem" data-cid="{c["id"]}" '
                f'style="left:{l:.2f}%;top:{t:.2f}%;width:{w:.2f}%;height:{h:.2f}%;'
                f'border-color:{col};background:{col}33" title="{html.escape(title)}"></div>'
            )

    # cluster list rows
    rows = ""
    for c in clusters:
        sig = c["sig"]
        status = "채택" if c["kept"] else "탈락"
        st_cls = "ok" if c["kept"] else "no"
        runs_s = ",".join(str(r + 1) for r in c["run_ids"])
        labels = " · ".join(
            f'r{m["run"]+1}:{html.escape(str(m.get("label") or "")[:20])}' for m in c["members"]
        )
        rows += (
            f'<tr class="cr {st_cls}" data-cid="{c["id"]}">'
            f'<td>{c["id"]}</td>'
            f'<td class=v{c["votes"]}>{c["votes"]}</td>'
            f'<td><span class="st {st_cls}">{status}</span></td>'
            f'<td>{sig["region"]}</td>'
            f'<td>{html.escape(str(sig["type"] or ""))}</td>'
            f'<td>{html.escape(str(sig["option"] or ""))}</td>'
            f'<td>{html.escape(str(sig["unit"] or ""))}</td>'
            f'<td>{html.escape(c["label"][:24])}</td>'
            f'<td>{runs_s}</td>'
            f'<td class=labs>{labels}</td>'
            f'</tr>'
        )

    # vote histogram
    vc = Counter(c["votes"] for c in clusters)
    hist = "".join(
        f'<div class=hbar><span class=hk>{k}표</span>'
        f'<span class=hb style="width:{v * 12}px"></span>'
        f'<span class=hn>{v}클러스터 '
        f'({"채택" if k >= trace["min_votes"] else "탈락"})</span></div>'
        for k, v in sorted(vc.items(), reverse=True)
    )

    # type kept vs drop
    type_rows = ""
    types = sorted({c["sig"]["type"] for c in clusters if c["sig"]["type"]})
    for t in types:
        k = sum(1 for c in clusters if c["sig"]["type"] == t and c["kept"])
        d = sum(1 for c in clusters if c["sig"]["type"] == t and not c["kept"])
        type_rows += f'<tr><td>{html.escape(t)}</td><td class=ok>{k}</td><td class=no>{d}</td><td>{k+d}</td></tr>'

    # 5 run small grids (LLM only)
    run_grid = ""
    for i, els in enumerate(runs):
        ov = ""
        for e in els:
            ov += rect_div(e.get("box"), IW, IH, "bx run", "",
                           f'{e.get("type")} {e.get("label")}', "")
        run_grid += (
            f'<div class=gc><div class=gl style="border-left:4px solid {RUN_COLORS[i]}">'
            f'LLM run{i+1} · {len(els)}요소</div>'
            f'<div class=canvas><img class=pageimg>{ov}</div></div>'
        )

    # detail panel for selected cluster (JS filled)
    legend = "".join(
        f'<span class=lg><i style="background:{RUN_COLORS[i]}"></i>run{i+1}</span>'
        for i in range(5)
    )

    clusters_json = json.dumps(clusters, ensure_ascii=False)

    page = f"""<!doctype html><meta charset=utf-8><title>majority 병합 과정</title>
<style>
*{{box-sizing:border-box}} body{{margin:0;font:13px/1.45 system-ui,sans-serif;background:#f1f3f5}}
#bar{{position:sticky;top:0;background:#212529;padding:9px 14px;z-index:10}}
#bar h1{{font-size:14px;color:#fff;margin:0 0 3px}}#bar .l{{color:#adb5bd;font-size:11.5px;margin-bottom:7px}}
.tab{{border:0;background:#495057;color:#ced4da;padding:5px 10px;border-radius:6px;cursor:pointer;font-size:12px;margin:0 3px 4px 0}}
.tab.on{{background:#4dabf7;color:#fff;font-weight:700}}
.view{{display:none;gap:10px;height:calc(100vh - 62px);padding:10px}}
.view.on{{display:flex}}
.main{{flex:1;overflow:auto;background:#fff;border:1px solid #dee2e6;border-radius:8px;min-width:0}}
.mh{{padding:8px 12px;font-size:12px;font-weight:700;border-bottom:1px solid #eef1f4;position:sticky;top:0;background:#fff;z-index:2}}
.note{{padding:6px 12px;font-size:11px;color:#868e96;border-bottom:1px solid #f1f3f5}}
.canvas{{position:relative;display:inline-block}} .canvas>img{{display:block;max-width:100%}}
.bx{{position:absolute;border:2px solid;border-radius:2px;font-size:8px;cursor:pointer;z-index:2}}
.bx.keep{{border-color:#2f9e44;background:rgba(47,158,68,.2);z-index:3}}
.bx.drop{{border-color:#e03131;background:rgba(224,49,49,.12);border-style:dashed;z-index:2}}
.bx.mem{{border-width:2px;z-index:2}}
.bx.sel{{outline:3px solid #228be6;outline-offset:1px;z-index:5}}
.fl{{background:rgba(255,255,255,.8);font-weight:700;padding:0 2px;font-size:7.5px}}
.side{{flex:0 0 46%;overflow:hidden;display:flex;flex-direction:column;background:#fff;border:1px solid #dee2e6;border-radius:8px}}
.lh{{padding:8px 12px;font-weight:700;border-bottom:1px solid #eef1f4}}
.lt{{overflow:auto;flex:1}}
table{{width:100%;border-collapse:collapse;font-size:11px}}
th,td{{padding:4px 6px;border-bottom:1px solid #f1f3f5;text-align:left;vertical-align:top}}
th{{position:sticky;top:0;background:#f8f9fa;z-index:1;font-size:10.5px}}
.cr{{cursor:pointer}} .cr:hover{{background:#f8f9fa}} .cr.hl{{background:#d0ebff}}
.st{{font-size:10px;font-weight:700;padding:1px 6px;border-radius:4px;color:#fff}}
.st.ok{{background:#2f9e44}} .st.no{{background:#e03131}}
td.ok{{color:#2f9e44;font-weight:700}} td.no{{color:#e03131;font-weight:700}}
td.v5,td.v4,td.v3{{color:#2f9e44;font-weight:700}} td.v2,td.v1{{color:#e03131;font-weight:700}}
.labs{{font-size:10px;color:#495057;max-width:220px;word-break:break-all}}
.cmp{{flex:1;overflow:auto;padding:14px 18px;background:#fff;border:1px solid #dee2e6;border-radius:8px}}
.cmp h2{{font-size:13px;margin:16px 0 8px}} .cmp h2:first-child{{margin-top:0}}
.rule{{background:#f8f9fa;border:1px solid #e9ecef;border-radius:8px;padding:10px 12px;font-size:12px}}
.rule code{{background:#fff;padding:1px 5px;border-radius:3px;border:1px solid #dee2e6}}
.hbar{{display:flex;align-items:center;gap:8px;margin:4px 0}}
.hk{{width:36px;font-weight:700}} .hb{{height:14px;background:#4dabf7;border-radius:3px}} .hn{{font-size:11.5px;color:#495057}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:10px}}
.gc{{border:1px solid #dee2e6;border-radius:8px;overflow:hidden}} .gl{{padding:5px 8px;font-size:11px;font-weight:700;background:#f8f9fa}}
.lg{{display:inline-flex;align-items:center;gap:4px;margin-right:10px;font-size:11px}}
.lg i{{width:10px;height:10px;border-radius:2px;display:inline-block}}
#detail{{padding:10px 12px;border-top:1px solid #eef1f4;font-size:11.5px;max-height:28%;overflow:auto;background:#f8f9fa}}
#detail:empty{{display:none}}
.ct td,.ct th{{border:1px solid #e9ecef;padding:5px 8px}}
</style>
<div id=bar>
<h1>majority 병합 과정 · low×5 · IoU&gt;0.25 · ≥3표 채택</h1>
<div class=l>초록=채택 median · 빨강 점선=탈락 median · 멤버탭=run별 색 · 목록 클릭=클러스터 하이라이트</div>
<button class="tab on" onclick="show(0)">① 기준·요약</button>
<button class="tab" onclick="show(1)">② 채택 vs 탈락</button>
<button class="tab" onclick="show(2)">③ run 멤버 겹침</button>
<button class="tab" onclick="show(3)">④ LLM 5회 원본</button>
</div>

<div class="view on" id=v0>
<div class=cmp>
<h2>병합 기준</h2>
<div class=rule>
<ol style="margin:0;padding-left:18px">
<li><b>같은 슬롯</b> = <code>(region, type, option, unit)</code> 동일 <b>그리고</b> box IoU &gt; <code>{trace["iou_thresh"]}</code></li>
<li>한 run은 같은 클러스터에 <b>1표만</b> (이미 들어간 run은 재매칭 안 함)</li>
<li><b>채택</b> = 서로 다른 run 수 ≥ <code>{trace["min_votes"]}</code> / {trace["n_runs"]}</li>
<li>채택 시 box=<b>median</b>, label=<b>최빈</b>, key=재생성</li>
</ol>
</div>
<h2>결과</h2>
<p>클러스터 <b>{len(clusters)}</b> · 채택 <b style="color:#2f9e44">{kept_n}</b> · 탈락 <b style="color:#e03131">{drop_n}</b>
· LLM 입력 {[len(r) for r in runs]}</p>
<h2>표 수 분포</h2>
{hist}
<h2>타입별 채택/탈락</h2>
<table class=ct><tr><th>type</th><th>채택</th><th>탈락</th><th>합</th></tr>{type_rows}</table>
</div></div>

<div class=view id=v1>
<div class=main>
<div class=mh>② median box · 초록=채택({kept_n}) · 빨강점선=탈락({drop_n})</div>
<div class=note>박스/목록 클릭 → 해당 클러스터 하이라이트 · run 멤버는 ③번 탭</div>
<div class=canvas><img class=pageimg>{ov_keep}{ov_drop}</div>
</div>
<div class=side>
<div class=lh>클러스터 목록 (표↓ · 채택 먼저 정렬은 표↓)</div>
<div class=lt><table>
<thead><tr><th>#</th><th>표</th><th>결과</th><th>rgn</th><th>type</th><th>opt</th><th>unit</th><th>label</th><th>runs</th><th>멤버 label</th></tr></thead>
<tbody id=clist>{rows}</tbody>
</table></div>
<div id=detail></div>
</div></div>

<div class=view id=v2>
<div class=main>
<div class=mh>③ 클러스터 멤버 box (run별 색) · {legend}</div>
<div class=note>같은 클러스터=같은 슬롯 후보 · 클릭으로 목록 연동</div>
<div class=canvas><img class=pageimg>{ov_members}</div>
</div>
<div class=side>
<div class=lh>클러스터 목록</div>
<div class=lt><table>
<thead><tr><th>#</th><th>표</th><th>결과</th><th>rgn</th><th>type</th><th>opt</th><th>unit</th><th>label</th><th>runs</th><th>멤버 label</th></tr></thead>
<tbody id=clist2>{rows}</tbody>
</table></div>
<div id=detail2></div>
</div></div>

<div class=view id=v3>
<div class=cmp>
<h2>④ 병합 전 LLM 5회 (원본 box)</h2>
<div class=grid>{run_grid}</div>
</div></div>

<script>
const PAGE_IMG="data:image/jpeg;base64,{img_b64}";
const CLUSTERS={clusters_json};
const RUN_COLORS={json.dumps(RUN_COLORS)};
document.querySelectorAll('img.pageimg').forEach(im=>im.src=PAGE_IMG);
const $=s=>[...document.querySelectorAll(s)];
function show(i){{
  $('.view').forEach((v,j)=>v.classList.toggle('on',j===i));
  $('.tab').forEach((t,j)=>t.classList.toggle('on',j===i));
}}
function pick(cid){{
  $('.bx.sel,.cr.hl').forEach(x=>{{x.classList.remove('sel');x.classList.remove('hl')}});
  $(`[data-cid="${{cid}}"]`).forEach(x=>{{
    if(x.classList.contains('bx')) x.classList.add('sel');
    if(x.classList.contains('cr')) x.classList.add('hl');
  }});
  const c=CLUSTERS.find(x=>x.id===cid);
  if(!c) return;
  const sig=c.sig;
  let h=`<b>cluster #${{c.id}}</b> · ${{c.votes}}표 · ${{c.kept?'채택':'탈락'}}<br>`;
  h+=`sig: region=${{sig.region}} type=${{sig.type}} opt=${{sig.option||'—'}} unit=${{sig.unit||'—'}}<br>`;
  h+=`median label: ${{c.label}} · box: ${{JSON.stringify(c.median_box)}}<br><br>`;
  h+='<b>멤버</b><br>';
  c.members.forEach(m=>{{
    const col=RUN_COLORS[m.run];
    h+=`<span style="color:${{col}};font-weight:700">run${{m.run+1}}</span> `;
    h+=`${{m.type}} · ${{m.label||''}} · box=${{JSON.stringify(m.box)}}<br>`;
  }});
  ['detail','detail2'].forEach(id=>{{
    const el=document.getElementById(id);
    if(el) el.innerHTML=h;
  }});
  const tr=document.querySelector(`#clist tr[data-cid="${{cid}}"], #clist2 tr[data-cid="${{cid}}"]`);
  if(tr) tr.scrollIntoView({{block:'nearest'}});
}}
$('.bx[data-cid], .cr[data-cid]').forEach(el=>{{
  el.addEventListener('click',e=>{{e.stopPropagation();pick(+el.dataset.cid)}});
}});
</script>"""

    # Fix duplicate style attribute issue - already handled for members
    OUT.write_text(page, encoding="utf-8")
    # also dump trace json
    (RUN_DIR / "merge_trace.json").write_text(
        json.dumps({**{k: v for k, v in trace.items() if k != "kept"},
                    "kept_count": kept_n, "dropped_count": drop_n},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"✓ {OUT}")
    print(f"clusters={len(clusters)} kept={kept_n} dropped={drop_n}")
    print("votes:", dict(vc))


if __name__ == "__main__":
    main()
