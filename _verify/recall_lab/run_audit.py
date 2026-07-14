"""완전성 감사(recall audit): 정답 요소 대비 4가지 추출법의 커버리지/누락.
   목적 = 위치파악 '전단계'에서 위치가 필요한 요소를 빠짐없이 가져오는가 (좌표 아님, recall).
   사용: python3 run_audit.py   (결과 없으면 LLM 호출·캐시, 있으면 재사용)"""
import sys, json, re, os, html, urllib.request, base64
LAB=os.path.dirname(os.path.abspath(__file__))
SCR="/private/tmp/claude-501/-Users-gunhee-workspace-codespace-project-lab-voucher/34205743-8819-4388-b090-e5d4c5f56b9c/scratchpad"
sys.path.insert(0,SCR); sys.path.insert(0,"/Users/gunhee/workspace/codespace/project/lab-voucher/image-to-md")
sys.argv=['x']
import cellid, multimodal
from core.backend import load_key

D="/Users/gunhee/workspace/codespace/project/lab-voucher/.old/document/"
FORMS=[("서식2호","combined2",None),
       ("서식15호","form15",D+"제1편_장애아가족_양육지원_서식/_extract/서식15호_사고보고서/pages/p-1.recon.md"),
       ("서식4-1호","form41",None)]

def norm(s): return re.sub(r"[\s·･‧∙•・=(){}\[\]／/,.'‘’“”\"]","",str(s))

# ── 텍스트-only 예측(HWPX) ──
PSYS=("한국 정부 서식 표 셀 구조(행별 셀, [빈]=빈칸, (가로N)=병합)가 주어진다. 좌표 없음. "
 "작성 시 존재할 입력 필드를 모두 예측하라. 출력 JSON만: {\"fields\":[{\"key\":\"영문\",\"label\":\"한글\","
 "\"type\":\"타입\",\"options\":[]?}]}. type∈[text,textarea,number,date,time,email,phone,radio,checkbox_group,consent,signature,select]. "
 "단위(년월일→date,원→number)·콜론→text·□나열→checkbox_group·서술칸→textarea·개인정보/동의 문구(수집·이용·제3자제공)는 consent 1개. JSON 외 금지.")
def predict_hwpx(txt):
    body=json.dumps({"model":"google/gemini-3.1-pro-preview","temperature":1.0,"max_tokens":8000,
      "reasoning":{"effort":"low"},"messages":[{"role":"system","content":PSYS},
      {"role":"user","content":"셀 구조:\n"+txt}]}).encode()
    for _ in range(3):
        req=urllib.request.Request("https://openrouter.ai/api/v1/chat/completions",data=body,
          headers={"Authorization":f"Bearer {load_key()}","Content-Type":"application/json"})
        out=json.load(urllib.request.urlopen(req,timeout=240))["choices"][0]["message"]["content"]
        try: return json.loads(re.search(r"\{.*\}",out.replace("```",""),re.S).group(0))["fields"]
        except Exception: pass
    return []

# ── CV(트리거) = 결정론 파이프라인 요소 → 필드 그룹 ──
def cv_fields(mod):
    F=__import__(mod); g={}
    for e in F.elements:
        base=e[1].split("=")[0].split("·")[0].strip()
        g.setdefault(base,{"label":base,"type":e[0],"options":[]})
        if "=" in e[1]: g[base]["options"].append(e[1].split("=")[-1])
    return list(g.values())

def cache(form,method,gen):
    p=f"{LAB}/cache/{form}_{method}.json"
    if os.path.exists(p): return json.load(open(p))
    v=gen(); json.dump(v,open(p,"w"),ensure_ascii=False,indent=1); return v

def get(form,mod,mdp,method):
    if method=="cv":       return cache(form,method,lambda: cv_fields(mod))
    if method=="hwpx":     return cache(form,method,lambda: predict_hwpx(multimodal.table_text(__import__(mod).cells)))
    if method=="cellid":   return cache(form,method,lambda: cellid.run(mod)[1]["fields"])
    if method=="multimodal":return cache(form,method,lambda: multimodal.run(mod,mdp)[1]["fields"])

def match(gt,fields):
    labs=[gt["label"]]+gt.get("alts",[]); gopt={norm(o) for o in gt.get("options",[])}
    if gt.get("type")=="consent":                       # 동의는 개념 1개 → 타입으로 매칭
        for f in fields:
            if f.get("type")=="consent": return f
    for f in fields:
        ml=norm(f.get("label","")); mopt={norm(o) for o in (f.get("options") or [])}
        for gl in labs:
            if norm(gl) and (norm(gl) in ml or ml in norm(gl)) and ml: return f
        if gopt and mopt and len(gopt&mopt)>=max(1,len(gopt)//3): return f
    return None

METHODS=[("cv","CV 트리거","#868e96"),("hwpx","HWPX 텍스트","#1971c2"),
         ("cellid","셀 ID","#7048e8"),("multimodal","멀티모달","#e8590c")]
tabs=""; panels=""
for i,(form,mod,mdp) in enumerate(FORMS):
    gt=json.load(open(f"{LAB}/groundtruth/{form}.json"))
    outs={m:get(form,mod,mdp,m) for m,_,_ in METHODS}
    # 매트릭스
    matched={m:set() for m,_,_ in METHODS}
    # 이미지 + 멀티모달 bbox (정답 위치 시각화용). 각 박스에 매칭된 정답 id 태깅
    import cv2
    F=__import__(mod); IH,IW=F.img.shape[:2]
    png=base64.b64encode(cv2.imencode(".jpg",F.img,[cv2.IMWRITE_JPEG_QUALITY,80])[1]).decode()
    mm=outs["multimodal"]; tag={}
    for el in gt["elements"]:
        hit=match(el,mm)
        if hit is not None:
            try: tag[mm.index(hit)]=el["id"]
            except ValueError: pass
    boxes=""
    for idx,f in enumerate(mm):
        b=f.get("bbox") or []
        if len(b)!=4: continue
        x,y,w,h=[max(0,min(1,v)) for v in b]
        boxes+=(f'<div class=bx data-gt="{tag.get(idx,"")}" title="{html.escape(f.get("label",""))}" '
                f'style="left:{x*100:.2f}%;top:{y*100:.2f}%;width:{w*100:.2f}%;height:{h*100:.2f}%"></div>')
    head="".join(f'<th style="color:{c}">{lab}</th>' for _,lab,c in METHODS)
    rows=""
    for el in gt["elements"]:
        cells=""
        for m,_,c in METHODS:
            hit=match(el,outs[m])
            if hit: matched[m].add(id(hit))
            cells+=(f'<td class=hit>✓</td>' if hit else '<td class=miss>✗</td>')
        opt=f' <span class=go>{len(el.get("options",[]))}지</span>' if el.get("options") else ''
        loc='' if el["id"] in tag.values() else ' <span class=noloc>위치?</span>'
        rows+=f'<tr class=drow data-gt="{el["id"]}"><td class=gl><b>{html.escape(el["label"])}</b> <span class=gt>{el["type"]}</span>{opt}{loc}</td>{cells}</tr>'
    # 요약
    N=len(gt["elements"]); summ=""
    for m,lab,c in METHODS:
        cov=sum(1 for el in gt["elements"] if match(el,outs[m])); extra=len(outs[m])-len(matched[m])
        pct=round(cov/N*100)
        summ+=(f'<div class=mrow><span class=mlab style="color:{c}">{lab}</span>'
               f'<span class=bar><span class=fill style="width:{pct}%;background:{c}"></span></span>'
               f'<b>{pct}%</b> <span class=det>{cov}/{N} 검출 · 추가 {extra}</span></div>')
    a=(i==0)
    tabs+=f'<button class="tab{" on" if a else ""}" onclick="show({i})">{form}</button>'
    panels+=(f'<div class=view id=v{i} style="display:{"flex" if a else "none"}">'
             f'<div class=imgcol><div class=wrap><img src="data:image/jpeg;base64,{png}">{boxes}</div>'
             f'<div class=imgnote>박스 = 멀티모달 검출 위치 · 정답 행 클릭 시 하이라이트</div></div>'
             f'<div class=matcol><h2>{html.escape(gt["form"])}</h2>'
             f'<div class=note>정답 {N}개 대비 <b>recall</b> (좌표 무관, 존재만). 오른쪽 매트릭스 행 클릭→왼쪽 위치</div>'
             f'<div class=summ>{summ}</div>'
             f'<table class=mx><thead><tr><th>정답 요소 (사람이 채우는 칸)</th>{head}</tr></thead><tbody>{rows}</tbody></table></div></div>')

DOC=f"""<!doctype html><meta charset=utf-8><title>완전성 감사 랩</title>
<style>*{{box-sizing:border-box}} body{{margin:0;font:13px/1.55 system-ui,sans-serif;background:#f1f3f5;color:#212529}}
#bar{{position:sticky;top:0;background:#212529;padding:10px 16px;z-index:5}}
#bar h1{{font-size:15px;color:#fff;margin:0 0 4px}} #bar .l{{color:#adb5bd;font-size:12px;margin-bottom:8px}}
.tab{{border:0;background:#495057;color:#ced4da;padding:6px 13px;border-radius:6px;cursor:pointer;font-size:12px;margin-right:5px}}
.tab.on{{background:#4dabf7;color:#fff;font-weight:700}}
.view{{display:flex;gap:16px;height:calc(100vh - 70px);padding:14px}}
.imgcol{{flex:0 0 46%;overflow:auto}} .wrap{{position:relative;display:inline-block}} .wrap img{{display:block;max-width:100%}}
.bx{{position:absolute;border:2px solid #e8590c;background:rgba(232,89,12,.08);cursor:pointer}}
.bx:hover,.bx.hi{{background:rgba(255,212,59,.5);border-color:#fab005;box-shadow:0 0 0 2px #fab005;z-index:2}}
.imgnote{{color:#868e96;font-size:11px;margin-top:6px}}
.matcol{{flex:1;overflow:auto}}
h2{{font-size:17px;margin:0 0 4px}} .note{{color:#868e96;font-size:12px;margin-bottom:14px}}
.noloc{{color:#e8850c;font-size:10px}} .drow{{cursor:pointer}} .drow:hover td{{background:#fff9db}} .drow.hi td{{background:#fff3bf}}
.summ{{background:#fff;border:1px solid #dee2e6;border-radius:8px;padding:12px 16px;margin-bottom:16px}}
.mrow{{display:flex;align-items:center;gap:10px;padding:4px 0}}
.mlab{{flex:0 0 90px;font-weight:700;font-size:12px}}
.bar{{flex:1;height:14px;background:#f1f3f5;border-radius:7px;overflow:hidden}} .fill{{display:block;height:100%}}
.mrow b{{flex:0 0 42px;text-align:right}} .det{{color:#868e96;font-size:11px;flex:0 0 150px}}
table.mx{{width:100%;border-collapse:collapse;font-size:12px;background:#fff;border-radius:8px;overflow:hidden}}
.mx th{{background:#f8f9fa;padding:7px 8px;text-align:center;border-bottom:2px solid #e9ecef;font-size:11.5px}}
.mx th:first-child{{text-align:left}}
.mx td{{padding:5px 8px;border-bottom:1px solid #f1f3f5;text-align:center}}
.gl{{text-align:left!important}} .gt{{color:#adb5bd;font-size:10px}} .go{{color:#e8850c;font-size:10px}}
.hit{{color:#2f9e44;font-weight:700}} .miss{{color:#e03131;background:#fff5f5;font-weight:700}}
</style>
<div id=bar><h1>완전성 감사 랩 (recall audit)</h1>
<div class=l>위치파악 <b>전단계</b>에서 "위치가 필요한 모든 요소"를 빠짐없이 가져오는가. 정답 대비 4가지 추출법 커버리지·누락. (좌표 정확도 아님)</div>{tabs}</div>
{panels}
<script>const $=s=>[...document.querySelectorAll(s)];
function show(i){{$('.view').forEach((v,j)=>v.style.display=j==i?'flex':'none');$('.tab').forEach((t,j)=>t.classList.toggle('on',j==i))}}
function clr(){{$('.hi').forEach(e=>e.classList.remove('hi'))}}
document.addEventListener('click',e=>{{let r=e.target.closest('.drow'),b=e.target.closest('.bx');
 if(r){{clr();r.classList.add('hi');let v=r.closest('.view');
   v.querySelectorAll('.bx').forEach(x=>{{if(x.dataset.gt==r.dataset.gt&&r.dataset.gt){{x.classList.add('hi');x.scrollIntoView({{block:'center'}})}}}});}}
 else if(b){{clr();b.classList.add('hi');let v=b.closest('.view');
   v.querySelectorAll('.drow').forEach(x=>{{if(x.dataset.gt==b.dataset.gt&&b.dataset.gt){{x.classList.add('hi');x.scrollIntoView({{block:'center'}})}}}});}}}});</script>"""
open(f"{LAB}/report.html","w").write(DOC)
print("✓ report.html")
