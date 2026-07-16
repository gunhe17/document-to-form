"""carve 정답 라벨링 서버 — stdlib http.server + sqlite3 (새 의존성 0).

UI에서 각 요소의 carve box가 정답인지(정답/오답/메모) 클릭 저장 → verify.db.
carve.py를 고친 뒤 서버 재시작하면 새 box가 뜨고, 기존 정답셋과 대조된다.

실행: python _verify/verify_server.py          # http://localhost:8765
     PORT=9000 python _verify/verify_server.py
"""
import json, os, sqlite3, sys, threading
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core import pipeline, extract, ground_focus
from core.region_segment import segment

CACHE = Path("/private/tmp/claude-501/-Users-gunhee-workspace-codespace-project-lab-voucher-image-to-form"
             "/06ad7bff-82ab-4edb-ba1f-8a8dda5599b0/scratchpad/focus_multidoc_x3.json")
DB = ROOT / "_verify" / "verify.db"
PORT = int(os.environ.get("PORT", "8765"))
_lock = threading.Lock()


def db():
    con = sqlite3.connect(DB)
    con.execute("""CREATE TABLE IF NOT EXISTS judgments(
        doc TEXT, key TEXT, verdict TEXT, note TEXT, rect TEXT, updated_at TEXT,
        PRIMARY KEY(doc,key))""")
    return con


def flat(run):
    o = []
    for j in sorted(run, key=int):
        o += run[j]
    return o


def build_docs():
    """코퍼스 → 폼별 carve 결과(run1) + 이미지 바이트. 서버 시작 시 1회."""
    data = json.loads(CACHE.read_text())
    docs = []
    for doc in data["docs"]:
        img = cv2.imread(doc["path"]); gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        IH, IW = gray.shape
        S = segment(gray)
        raw = ground_focus.dedup_nested(flat(doc["runs"][0]),
                                        {j: r[2] * r[3] for j, r in pipeline.atoms_of(S)})  # 중첩 이중검출 제거
        placed = pipeline.place_elements(gray, S, raw)
        keyed = extract.assign_keys(placed)                    # 안정 key 부여
        els = [{"key": e.get("key"), "region": e.get("region"), "type": e.get("type"),
                "option": e.get("option"), "label": e.get("label"), "unit": e.get("unit"),
                "rule": e.get("rule"), "rect": [int(v) for v in e["rect"]]} for e in keyed]
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 82])
        docs.append({"name": doc["name"], "w": IW, "h": IH, "elements": els, "_img": buf.tobytes()})
        print(f"  {doc['name']}: {len(els)} elements", flush=True)
    return docs


PAGE = """<!doctype html><meta charset=utf-8><title>carve 정답 라벨링</title>
<style>
*{box-sizing:border-box}html,body{margin:0;height:100%}
body{font:13px system-ui,sans-serif;background:#0e0e0e;color:#e8e8e8;display:flex;flex-direction:column}
#top{padding:8px 14px;border-bottom:1px solid #333;background:#161616;display:flex;gap:8px;align-items:center;flex-wrap:wrap}
h1{margin:0;font-size:14px} .tab{background:#222;color:#ccc;border:1px solid #333;border-radius:6px;padding:4px 9px;cursor:pointer;font:inherit}
.tab.on{background:#2a3d2a;border-color:#5a8;color:#fff} .prog{margin-left:auto;color:#9c9;font-size:12px}
#body{flex:1;display:flex;min-height:0}
#imgwrap{flex:1;overflow:auto;position:relative;background:#000;padding:10px}
.canvas{position:relative;display:inline-block}
.canvas>img{display:block;max-width:none}
.b{position:absolute;border:2px solid #888;box-sizing:border-box;cursor:pointer}
.b.ok{border-color:#4c4;background:rgba(60,200,60,.12)}
.b.bad{border-color:#f55;background:rgba(240,70,70,.14)}
.b.stale{border-color:#f90;border-style:dashed;background:rgba(255,150,0,.14)}
.b.cur{border-color:#fd0;border-width:3px;box-shadow:0 0 0 2px #fd0,0 0 10px #fd0;z-index:5}
.b>span{position:absolute;left:0;top:-13px;font-size:10px;background:rgba(0,0,0,.8);padding:0 3px;white-space:nowrap}
#side{flex:0 0 300px;border-left:1px solid #333;background:#141414;padding:12px;overflow:auto}
.row{margin:6px 0} .k{color:#8ab} b{color:#fff}
button.j{font-size:15px;padding:8px 14px;border-radius:7px;border:1px solid #444;cursor:pointer;margin-right:8px}
button.okb{background:#1c3a1c;color:#7f7} button.badb{background:#3a1c1c;color:#f88}
button.j.sel{outline:2px solid #fd0}
#note{width:100%;background:#111;color:#eee;border:1px solid #333;border-radius:5px;padding:6px;margin-top:6px}
.nav{margin-top:10px;color:#999} kbd{background:#222;border:1px solid #444;border-radius:3px;padding:1px 5px;font:inherit}
.list{margin-top:12px;max-height:38vh;overflow:auto;border-top:1px solid #333;padding-top:8px}
.li{padding:3px 5px;border-radius:4px;cursor:pointer;display:flex;justify-content:space-between}
.li:hover{background:#1a2430} .li.cur{background:#2a2a12}
.li .v.ok{color:#5c5} .li .v.bad{color:#f66} .li .v.stale{color:#f90} .li .v{color:#666}
</style>
<div id=top><h1>carve 정답 라벨링</h1><span id=tabs></span><span class=prog id=prog></span></div>
<div id=body>
  <div id=imgwrap><div class=canvas id=canvas><img id=img></div></div>
  <div id=side>
    <div class=row><b id=selttl>요소 선택</b></div>
    <div class=row><span class=k>type</span> <span id=styp></span> · <span class=k>rule</span> <span id=srule></span></div>
    <div class=row><span class=k>값</span> <span id=sval></span></div>
    <div class=row><span class=k>region</span> <span id=sreg></span> · <span class=k>rect</span> <span id=srect></span></div>
    <div class=row style="margin-top:10px">
      <button class="j okb" id=bok onclick="judge('ok')">✓ 정답 (O)</button>
      <button class="j badb" id=bbad onclick="judge('bad')">✗ 오답 (X)</button>
    </div>
    <textarea id=note rows=2 placeholder="메모 (오답 사유 등) — 자동 저장"></textarea>
    <div class=nav><kbd>O</kbd> 정답 <kbd>X</kbd> 오답 <kbd>↑↓</kbd>/<kbd>J K</kbd> 이동 · 판정하면 자동 다음</div>
    <div class=list id=list></div>
  </div>
</div>
<script>
let DOCS=[], di=0, ci=0, V={};   // V[doc][key]={verdict,note}
async function load(){
  const d=await (await fetch('/api/data')).json();
  DOCS=d.docs; V=d.judgments;
  document.getElementById('tabs').innerHTML=DOCS.map((x,i)=>
    `<button class="tab${i==0?' on':''}" onclick="pick(${i})">${x.name}</button>`).join('');
  pick(0);
}
function pick(i){di=i;ci=0;
  document.querySelectorAll('.tab').forEach((t,j)=>t.classList.toggle('on',j==i));
  const doc=DOCS[i], img=document.getElementById('img');
  img.src='/img/'+i; img.onload=()=>{render();};
  if(img.complete) render();
}
function vof(key){const doc=DOCS[di]; return (V[doc.name]&&V[doc.name][key])||{};}
function stale(e){const j=vof(e.key);return j.verdict&&j.rect&&JSON.stringify(j.rect)!==JSON.stringify(e.rect);}
function render(){
  const doc=DOCS[di], img=document.getElementById('img'), cv=document.getElementById('canvas');
  const sx=img.clientWidth/doc.w, sy=img.clientHeight/doc.h;
  cv.querySelectorAll('.b').forEach(e=>e.remove());
  doc.elements.forEach((e,k)=>{
    const [x,y,w,h]=e.rect, v=vof(e.key).verdict||'', st=stale(e);
    const b=document.createElement('div');
    b.className='b '+(st?'stale':v)+(k==ci?' cur':'');
    b.style.cssText=`left:${x*sx}px;top:${y*sy}px;width:${w*sx}px;height:${h*sy}px`;
    b.onclick=()=>{ci=k;render();};
    b.innerHTML=`<span>${(e.option||e.label||'').slice(0,10)}</span>`;
    cv.appendChild(b);
  });
  // list
  document.getElementById('list').innerHTML=doc.elements.map((e,k)=>{
    const v=vof(e.key).verdict||'', st=stale(e);
    return `<div class="li${k==ci?' cur':''}" onclick="ci=${k};render()">
      <span>${e.type} · ${(e.option||e.label||'').slice(0,14)}</span>
      <span class="v ${st?'stale':v}">${st?'⟳':v=='ok'?'✓':v=='bad'?'✗':'·'}</span></div>`;}).join('');
  const e=doc.elements[ci]||{};
  document.getElementById('selttl').textContent=`#${ci+1}/${doc.elements.length}  ${e.type||''}`;
  document.getElementById('styp').textContent=e.type||'';
  document.getElementById('srule').textContent=e.rule||'';
  document.getElementById('sval').textContent=e.option||e.label||'';
  document.getElementById('sreg').textContent=e.region;
  document.getElementById('srect').textContent=JSON.stringify(e.rect);
  const cur=vof(e.key);
  document.getElementById('note').value=cur.note||'';
  document.getElementById('bok').classList.toggle('sel',cur.verdict=='ok');
  document.getElementById('bbad').classList.toggle('sel',cur.verdict=='bad');
  const doc2=DOCS[di], done=doc2.elements.filter(x=>vof(x.key).verdict).length;
  const nst=doc2.elements.filter(x=>stale(x)).length;
  document.getElementById('prog').textContent=`${doc2.name}: ${done}/${doc2.elements.length} 판정`+(nst?` · ⟳재검증 ${nst}`:'');
  const cb=document.querySelector('.b.cur'); if(cb) cb.scrollIntoView({block:'center',inline:'center'});
}
async function judge(v){
  const doc=DOCS[di], e=doc.elements[ci];
  const note=document.getElementById('note').value;
  V[doc.name]=V[doc.name]||{}; V[doc.name][e.key]={verdict:v,note};
  await fetch('/api/judge',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({doc:doc.name,key:e.key,verdict:v,note,rect:e.rect})});
  if(ci<doc.elements.length-1) ci++;   // 자동 다음
  render();
}
async function saveNote(){
  const doc=DOCS[di], e=doc.elements[ci], v=vof(e.key).verdict||null;
  const note=document.getElementById('note').value;
  V[doc.name]=V[doc.name]||{}; V[doc.name][e.key]={verdict:v,note};
  await fetch('/api/judge',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({doc:doc.name,key:e.key,verdict:v,note,rect:e.rect})});
}
document.getElementById('note').addEventListener('change',saveNote);
document.addEventListener('keydown',ev=>{
  if(ev.target.tagName=='TEXTAREA') return;
  const doc=DOCS[di]; const n=doc.elements.length;
  if(ev.key=='o'||ev.key=='O'){judge('ok')}
  else if(ev.key=='x'||ev.key=='X'){judge('bad')}
  else if(ev.key=='ArrowDown'||ev.key=='j'){ci=Math.min(n-1,ci+1);render()}
  else if(ev.key=='ArrowUp'||ev.key=='k'){ci=Math.max(0,ci-1);render()}
});
window.addEventListener('resize',render);
load();
</script>"""


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, ctype, body):
        self.send_response(code); self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body))); self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/":
            self._send(200, "text/html; charset=utf-8", PAGE.encode())
        elif self.path.startswith("/img/"):
            i = int(self.path.split("/")[-1])
            self._send(200, "image/jpeg", DOCS[i]["_img"])
        elif self.path == "/api/data":
            with _lock:
                con = db()
                jud = {}
                for doc, key, verdict, note, rect in con.execute(
                        "SELECT doc,key,verdict,note,rect FROM judgments"):
                    jud.setdefault(doc, {})[key] = {
                        "verdict": verdict, "note": note,
                        "rect": json.loads(rect) if rect else None}
                con.close()
            payload = {"docs": [{k: v for k, v in d.items() if k != "_img"} for d in DOCS],
                       "judgments": jud}
            self._send(200, "application/json; charset=utf-8",
                       json.dumps(payload, ensure_ascii=False).encode())
        else:
            self._send(404, "text/plain", b"nope")

    def do_POST(self):
        if self.path == "/api/judge":
            n = int(self.headers.get("Content-Length", 0))
            j = json.loads(self.rfile.read(n))
            with _lock:
                con = db()
                con.execute("""INSERT INTO judgments(doc,key,verdict,note,rect,updated_at)
                    VALUES(?,?,?,?,?,?)
                    ON CONFLICT(doc,key) DO UPDATE SET
                      verdict=excluded.verdict, note=excluded.note,
                      rect=excluded.rect, updated_at=excluded.updated_at""",
                    (j["doc"], j["key"], j.get("verdict"), j.get("note") or "",
                     json.dumps(j.get("rect")), datetime.now().isoformat(timespec="seconds")))
                con.commit(); con.close()
            self._send(200, "application/json", b'{"ok":true}')
        else:
            self._send(404, "text/plain", b"nope")


DOCS = []
if __name__ == "__main__":
    print("carve 계산 중(EasyOCR 로딩 포함)...", flush=True)
    DOCS = build_docs()
    print(f"\n준비 완료 · {len(DOCS)}폼 · DB={DB}\n→ http://localhost:{PORT}\n", flush=True)
    ThreadingHTTPServer(("127.0.0.1", PORT), H).serve_forever()
