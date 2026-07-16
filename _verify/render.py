"""파이프라인 검증 리포트 — pipeline.build 결과를 절차별(①분리 ②SoM ③LLM원본 ④배치) HTML로.

실행: python _verify/render.py [문서번호...]   (인자 없으면 서식2·서식15)
LLM 응답은 /tmp/img2form_test/gcache_{i}.json 에 캐시 (재실행 무료).
"""
import sys, os, base64, html
from pathlib import Path
import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core import pipeline  # noqa: E402

CACHE = Path("/tmp/img2form_test"); CACHE.mkdir(parents=True, exist_ok=True)
OUT = ROOT / "_verify" / "step5_grounding.html"

WCLS = {"text": "w-txt", "textarea": "w-txt", "number": "w-num", "email": "w-txt", "phone": "w-txt",
        "date": "w-dt", "time": "w-dt", "radio": "w-ch", "checkbox_group": "w-ch", "select": "w-ch",
        "consent": "w-cn", "signature": "w-sg", "image": "w-img"}

D = "/Users/gunhee/workspace/codespace/project/lab-voucher/.old/document/"
B = "제1편_장애아가족_양육지원_서식/_extract/서식24호_장애아가족_양육지원사업_(_)월_실적_보고서/pages/"
DOCS = [("서식2호", D + "제1편_장애아가족_양육지원_서식/_extract/서식2호_장애아돌보미_지원서/pages/p-1.png"),
        ("서식15호", D + "제1편_장애아가족_양육지원_서식/_extract/서식15호_사고보고서/pages/p-1.png"),
        ("서식4-1호", D + "제2편_발달재활서비스_서식/_extract/서식4_1호_발달재활서비스_의뢰서/pages/p-1.png"),
        ("서식8호", D + "제3편_언어발달지원_서식/_extract/서식8호_언어발달지원_서비스_제공(이용)_계획서/pages/p-1.png"),
        ("서식24호", D + B + "p-01.png"),
        ("서식3호", D + "제1편_장애아가족_양육지원_서식/_extract/서식3호_장애아돌보미_개인정보_수집･이용_및_제3자_제공_동의서/pages/p-1.png"),
        ("서식6호", D + "제2편_발달재활서비스_서식/_extract/서식6호_장애아동_발달재활_서비스_제공(이용)계약서/pages/p-1.png"),
        ("booklet p05", D + B + "p-05.png"), ("booklet p02", D + B + "p-02.png"), ("booklet p07", D + B + "p-07.png"),
        ("booklet p12", D + B + "p-12.png"), ("booklet p17", D + B + "p-17.png"), ("booklet p22", D + B + "p-22.png"),
        # --- 추가 10장 ---
        ("서식2호 p2", D + "제1편_장애아가족_양육지원_서식/_extract/서식2호_장애아돌보미_지원서/pages/p-2.png"),
        ("서식15호 p2", D + "제1편_장애아가족_양육지원_서식/_extract/서식15호_사고보고서/pages/p-2.png"),
        ("서식6호 p2", D + "제2편_발달재활서비스_서식/_extract/서식6호_장애아동_발달재활_서비스_제공(이용)계약서/pages/p-2.png"),
        ("서식6호 p3", D + "제2편_발달재활서비스_서식/_extract/서식6호_장애아동_발달재활_서비스_제공(이용)계약서/pages/p-3.png"),
        ("booklet p03", D + B + "p-03.png"), ("booklet p06", D + B + "p-06.png"), ("booklet p09", D + B + "p-09.png"),
        ("booklet p11", D + B + "p-11.png"), ("booklet p14", D + B + "p-14.png"), ("booklet p19", D + B + "p-19.png"),
        # --- 아동정서발달지원서비스.pdf 마지막 두 페이지 (서식) ---
        ("아동정서발달 추천서", str(ROOT / "_verify" / "아동정서발달" / "p-09.png")),
        ("아동정서발달 소견서", str(ROOT / "_verify" / "아동정서발달" / "p-10.png"))]


def b64(im):
    return base64.b64encode(cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, 72])[1]).decode()


def thumb_split(img, S):
    im = img.copy()
    for x0, y0, x1, y1 in S["frames"]: cv2.rectangle(im, (x0, y0), (x1, y1), (180, 60, 200), 3)
    for x0, y0, x1, y1 in S["tables"]: cv2.rectangle(im, (x0, y0), (x1, y1), (0, 140, 240), 2)
    for x, y, w, h in S["cells"]: cv2.rectangle(im, (x, y), (x + w, y + h), (40, 170, 40), 1)
    for x0, b0, x1, b1 in S["bands"]: cv2.rectangle(im, (x0, b0), (x1, b1), (230, 120, 30), 2)
    return im


def thumb_raw(img, raw, IW, IH):
    im = img.copy()
    for e in raw:
        bx = e.get("box")
        if not bx or len(bx) != 4: continue
        ymin, xmin, ymax, xmax = bx
        p1 = (int(xmin / 1000 * IW), int(ymin / 1000 * IH)); p2 = (int(xmax / 1000 * IW), int(ymax / 1000 * IH))
        cv2.rectangle(im, p1, p2, (0, 0, 230) if e.get("option") else (230, 120, 0), 2)
    return im


def overlay(elements, IW, IH, i):
    ov = ""
    for it in elements:
        rx, ry, rw, rh = it["rect"]; t = it["type"]; opt = it["option"]; labf = it["label"]; cls = WCLS.get(t, "w-txt")
        ymin, xmin, ymax, xmax = it["box"]
        inner = f'<span class=oc>{html.escape(str(opt)[:10])}</span>' if opt else f'<span class=fl>{html.escape(labf[:14])}</span>'
        da = (f'data-i="{i}" data-key="{html.escape(str(it.get("key") or ""))}" data-label="{html.escape(labf)}" data-type="{t}" '
              f'data-opt="{html.escape(str(opt or ""))}" data-unit="{html.escape(str(it.get("unit") or ""))}" data-region="{it.get("region", "")}" '
              f'data-box="{ymin}, {xmin}, {ymax}, {xmax}" data-rule="{it["rule"]}" data-corr="{1 if it["corrected"] else 0}"')
        ov += (f'<div class="fw {cls}" style="left:{rx / IW * 100:.2f}%;top:{ry / IH * 100:.2f}%;width:{rw / IW * 100:.2f}%;height:{rh / IH * 100:.2f}%" '
               f'{da} title="{html.escape(labf)} ({t})">{inner}</div>')
    return ov


def main(idxs):
    tabs = panels = ""
    for tab, i in enumerate(idxs):
        name, path = DOCS[i]
        built = pipeline.build(path, cache_path=str(CACHE / f"gcache_{i}.json"))
        S = built["segmentation"]; img = built["img"]; IW, IH = built["page"]["w"], built["page"]["h"]
        els = built["elements"]; raw = built["raw"]
        ncorr = sum(1 for it in els if it["corrected"])
        th1 = b64(thumb_split(img, S)); th2 = b64(built["marked"]); th3 = b64(thumb_raw(img, raw, IW, IH)); clean = b64(img)
        on = (tab == 0)
        tabs += f'<button class="tab{" on" if on else ""}" onclick="show({tab})">{html.escape(name)}</button>'
        panels += (f'<div class=view id=v{tab} style="display:{"flex" if on else "none"}">'
                   f'<div class=side><div class=stat>atom {len(built["atoms"])} · LLM요소 {len(raw)} · 렌더 {len(els)} · 수정 {ncorr}</div>'
                   f'<div class=th><div class=thh>① 영역분리 (LLM 없음)</div><img onclick="zoom(this.src)" src="data:image/jpeg;base64,{th1}"></div>'
                   f'<div class=th><div class=thh>② SoM 번호마킹 (LLM 입력)</div><img onclick="zoom(this.src)" src="data:image/jpeg;base64,{th2}"></div>'
                   f'<div class=th><div class=thh>③ LLM 원본 box (place 전)</div><img onclick="zoom(this.src)" src="data:image/jpeg;base64,{th3}"></div></div>'
                   f'<div class=main><div class=mh>④ 타입규칙 배치 — 위젯 클릭 → 우측 정보 · {len(els)}요소</div>'
                   f'<div class=canvas><img src="data:image/jpeg;base64,{clean}">{overlay(els, IW, IH, tab)}</div></div>'
                   f'<div class=info><div class=ih>선택 요소 정보</div><div class=ibody id=info{tab}b><div class=hint>← 위젯 클릭</div></div></div></div>')
        print(f"{name}: atom{len(built['atoms'])} LLM요소{len(raw)} 렌더{len(els)} 수정{ncorr}")
    OUT.write_text(_PAGE.replace("__TABS__", tabs).replace("__PANELS__", panels), encoding="utf-8")
    print("✓", OUT)


_PAGE = r"""<!doctype html><meta charset=utf-8><title>image-to-form 파이프라인</title>
<style>*{box-sizing:border-box} body{margin:0;font:13px/1.5 system-ui,sans-serif;background:#f1f3f5}
#bar{position:sticky;top:0;background:#212529;padding:9px 14px;z-index:10}#bar h1{font-size:14px;color:#fff;margin:0 0 3px}#bar .l{color:#adb5bd;font-size:11.5px;margin-bottom:7px}
.tab{border:0;background:#495057;color:#ced4da;padding:5px 10px;border-radius:6px;cursor:pointer;font-size:12px;margin:0 3px 4px 0}.tab.on{background:#4dabf7;color:#fff;font-weight:700}
.view{display:flex;gap:12px;height:calc(100vh - 58px);padding:12px}
.side{flex:0 0 236px;overflow:auto} .stat{background:#fff;border:1px solid #dee2e6;border-radius:8px;padding:8px 11px;margin-bottom:10px;font-size:11.5px;font-weight:600}
.th{background:#fff;border:1px solid #e9ecef;border-radius:8px;overflow:hidden;margin-bottom:10px} .thh{background:#f8f9fa;padding:5px 9px;font-size:11px;font-weight:700;border-bottom:1px solid #eef1f4} .th img{width:100%;display:block;cursor:zoom-in}
.main{flex:1;overflow:auto;background:#fff;border:1px solid #dee2e6;border-radius:8px} .mh{padding:8px 12px;font-size:12px;font-weight:700;border-bottom:1px solid #eef1f4;color:#495057;position:sticky;top:0;background:#fff;z-index:2}
#lb{display:none;position:fixed;inset:0;background:rgba(0,0,0,.85);z-index:100;align-items:flex-start;justify-content:center;overflow:auto;padding:20px;cursor:zoom-out} #lb img{max-width:96%;height:auto}
.canvas{position:relative;display:inline-block} .canvas>img{display:block;max-width:100%}
.fw{position:absolute;border:2px solid;border-radius:3px;overflow:hidden;font-size:9px;display:flex;flex-wrap:wrap;gap:1px;padding:0 1px}
.fl{font-size:8px;font-weight:700;line-height:1.1;color:#111;background:rgba(255,255,255,.72);padding:0 2px;border-radius:2px}
.oc{font-size:7.5px;background:rgba(255,255,255,.85);border:1px solid rgba(0,0,0,.15);border-radius:5px;padding:0 2px}
.fw.sel{outline:3px solid #f03e3e;outline-offset:1px;box-shadow:0 0 0 3px rgba(240,62,62,.35);z-index:5}
.w-txt{border-color:#1971c2;background:rgba(25,113,194,.13)} .w-num{border-color:#0c8599;background:rgba(12,133,153,.13)}
.w-dt{border-color:#9c36b5;background:rgba(156,54,181,.13)} .w-ch{border-color:#e8590c;background:rgba(232,89,12,.14)}
.w-cn{border-color:#2b8a3e;background:rgba(43,138,62,.13)} .w-sg{border-color:#2f9e44;background:rgba(47,158,68,.15)} .w-img{border-color:#6741d9;background:rgba(103,65,217,.12)}
.info{flex:0 0 268px;overflow:auto;background:#fff;border:1px solid #dee2e6;border-radius:8px}
.ih{padding:8px 12px;font-size:12px;font-weight:700;border-bottom:1px solid #eef1f4;color:#495057} .ibody{padding:10px 12px} .hint{color:#adb5bd;font-size:11.5px}
.row{display:flex;gap:8px;padding:4px 0;border-bottom:1px solid #f1f3f5;font-size:12px} .rk{flex:0 0 62px;color:#868e96;font-size:11px} .rv{flex:1;word-break:break-all}
.tag{display:inline-block;color:#fff;font-size:10.5px;font-weight:700;padding:1px 7px;border-radius:5px}</style>
<div id=bar><h1>image-to-form 파이프라인 · 절차별 (①분리 ②SoM ③LLM원본 ④배치)</h1>
<div class=l>좌: 절차 썸네일(클릭=확대) · 중: 최종배치(위젯 클릭) · 우: 요소정보. 파랑=텍스트 청록=숫자 보라=날짜 주황=선택 초록=서명 남보라=사진</div>__TABS__</div>__PANELS__
<div id=lb onclick="this.style.display='none'"><img></div>
<script>const $=s=>[...document.querySelectorAll(s)];
function zoom(src){const z=document.getElementById('lb');z.querySelector('img').src=src;z.style.display='flex'}
function show(i){$('.view').forEach((v,j)=>v.style.display=j==i?'flex':'none');$('.tab').forEach((t,j)=>t.classList.toggle('on',j==i))}
const TC={'w-txt':'#1971c2','w-num':'#0c8599','w-dt':'#9c36b5','w-ch':'#e8590c','w-cn':'#2b8a3e','w-sg':'#2f9e44','w-img':'#6741d9'};
function esc(s){return (s||'').replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]))}
function row(k,v){return v?`<div class=row><span class=rk>${k}</span><span class=rv>${v}</span></div>`:''}
function pick(el){
  const v=el.closest('.view'); v.querySelectorAll('.fw.sel').forEach(x=>x.classList.remove('sel')); el.classList.add('sel');
  const d=el.dataset, col=TC[el.classList[1]]||'#868e96';
  let h=`<div class=row><span class=rk>라벨</span><span class=rv><b>${esc(d.label)}</b></span></div>`;
  h+=`<div class=row><span class=rk>타입</span><span class=rv><span class=tag style="background:${col}">${d.type}</span></span></div>`;
  h+=row('필드키',`<code>${esc(d.key)}</code>`); if(d.opt) h+=row('선택값',`<b style=color:#e8590c>${esc(d.opt)}</b>`);
  if(d.unit) h+=row('단위',esc(d.unit));
  const RN={'_snap_mark':'□ 스냅','_refine':'빈칸 축소','_ocr_anchor':'OCR 글자앵커','_ocr_word':'OCR 보기fit','_radio_pair':'radio 쌍 글자','_lr_pair':'L/R 쌍','_date_inline':'날짜행 카브','_snap_cell':'격자 스냅','_fit_bounded':'상하좌우 맞춤','_fit_word':'글자 맞춤','_fit_ink':'잉크 맞춤','_keep':'유지'};
  h+=row('배치규칙',`${RN[d.rule]||d.rule} ${d.corr==='1'?'<b style=color:#2b8a3e>· 수정됨</b>':'<span style=color:#adb5bd>· 원본유지</span>'}`);
  h+=row('영역#',esc(d.region)); h+=row('box(0~1e3)',esc(d.box));
  document.getElementById('info'+d.i+'b').innerHTML=h;
}
$('.fw').forEach(b=>b.addEventListener('click',e=>{e.stopPropagation();pick(b)}));</script>"""


if __name__ == "__main__":
    args = [int(a) for a in sys.argv[1:] if a.isdigit()]
    main(args or [0, 1])
