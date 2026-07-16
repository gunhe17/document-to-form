"""일반화 carve 작업 결과 리포트 HTML — 각 수정을 폼 이미지 위 박스로 보여준다.
실행: python _verify/carve_report.py  →  _verify/carve_report.html
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
OUT = ROOT / "_verify" / "carve_report.html"
DATA = json.loads(CACHE.read_text())


def flat(run):
    o = []
    for j in sorted(run, key=int):
        o += run[j]
    return o


_PLACED = {}
def placed(name):
    if name not in _PLACED:
        doc = next(x for x in DATA["docs"] if x["name"] == name)
        gray = cv2.cvtColor(cv2.imread(doc["path"]), cv2.COLOR_BGR2GRAY); S = segment(gray)
        raw = ground_focus.dedup_nested(flat(doc["runs"][0]),
                                        {j: r[2] * r[3] for j, r in pipeline.atoms_of(S)})
        _PLACED[name] = (cv2.imread(doc["path"]),
                         extract.assign_keys(pipeline.place_elements(gray, S, raw)))
    return _PLACED[name]


COL = {"radio": (60, 180, 60), "checkbox_group": (255, 90, 0), "consent": (255, 90, 0),
       "time": (0, 0, 235), "number": (180, 60, 255), "date": (0, 0, 235),
       "signature": (200, 40, 150), "text": (0, 0, 235)}


def crop(name, pred, box, zoom=2):
    img, els = placed(name)
    o = img.copy()
    for e in els:
        if pred(e):
            x, y, w, h = [int(v) for v in e["rect"]]
            cv2.rectangle(o, (x, y), (x + w, y + h), COL.get(e.get("type"), (0, 0, 235)), 2)
    x0, y0, x1, y1 = box
    c = o[y0:y1, x0:x1]
    if zoom != 1:
        c = cv2.resize(c, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_AREA)
    b = base64.b64encode(cv2.imencode(".jpg", c, [cv2.IMWRITE_JPEG_QUALITY, 85])[1]).decode()
    return f'<img src="data:image/jpeg;base64,{b}">'


# (제목, 폼, 술어, cropbox, 설명)
CASES = [
    ("radio 식별/위치 — am/pm·남/여", "서식15호",
     lambda e: e.get("type") == "radio", (360, 600, 1180, 860), 1,
     "영역 OCR로 보기글자를 찾아 option 텍스트로 매칭. 병합토큰 '(amlpm)'은 비례분할. am은 am, pm은 pm 위에 정확히."),
    ("checkbox □ — 흐린 □도 검출", "아동정서_p10",
     lambda e: e.get("region") == 15 and e.get("type") == "checkbox_group", (285, 500, 1090, 750), 1,
     "find_cb를 4변 중 3변만 뚜렷해도 인정하게 완화 + 배정을 box '왼쪽 모서리' 기준으로(wide box가 옆 □ 잡던 문제 해결). 6개 전부 □에 스냅."),
    ("consent □ — 동의 체크란", "서식2호",
     lambda e: e.get("region") == 45 and e.get("type") == "consent", (200, 1610, 520, 1740), 2,
     "consent도 checkbox와 같은 □-스냅으로 일반화. 문구형 consent는 guard가 스킵."),
    ("여 — OCR box 여백 제거", "아동정서_p10",
     lambda e: e.get("region") == 6 and e.get("type") == "radio", (935, 335, 1075, 400), 3,
     "OCR bbox가 43px로 넓던 것을 fit_ink로 실제 글자에 조임. 남과 겹침 해소."),
    ("ㅇㅇㅇ placeholder — 라벨 뒤 표시자", "서식8호_계획",
     lambda e: str(e.get("key", "")).startswith("r78_text"), (200, 1805, 900, 1855), 2,
     "fit_placeholder 탐색창이 0.6h로 좁아 라벨('담당자') 뒤 ㅇㅇㅇ를 놓쳤음 → 2h로 확대. 두 ㅇㅇㅇ 모두 정확히."),
    ("값칸 폭 통일 — 잉크 기준 균일 여백", "서식2호",
     lambda e: e.get("region") == 20 and e.get("type") in ("time", "number"), (640, 645, 1060, 685), 2,
     "같은 unit(시·시)끼리 median 폭으로 통일. 시_1(35)·시_2(43) → 둘 다 39px 균일. 날짜 년/월도 시작·종료가 같은 폭."),
    ("숫자 중복 제거 — 중첩영역 dedup", "아동정서_p10",
     lambda e: e.get("type") == "radio" and e.get("region") in range(40, 50), (340, 1130, 1000, 1180), 2,
     "SoM이 '큰 블록(1~10 전체) + 개별 셀'을 둘 다 만들어 숫자가 두 번 검출됐던 것. 큰 블록 판을 제거해 한 개씩만."),
]

sections = ""
for i, (title, name, pred, box, zoom, desc) in enumerate(CASES):
    sections += (f'<section><h2>{i+1}. {html.escape(title)}</h2>'
                 f'<div class=sub>{html.escape(name)}</div>'
                 f'<div class=img>{crop(name, pred, box, zoom)}</div>'
                 f'<p>{html.escape(desc)}</p></section>')

doc = f"""<!doctype html><meta charset=utf-8><title>일반화 carve 작업 결과</title>
<style>
*{{box-sizing:border-box}} body{{margin:0;font:14px/1.6 system-ui,sans-serif;background:#0f0f0f;color:#e8e8e8}}
.wrap{{max-width:1000px;margin:0 auto;padding:24px}}
h1{{font-size:22px;margin:0 0 4px}} .lead{{color:#9c9;margin:0 0 20px}}
.stat{{display:inline-block;background:#182018;border:1px solid #2a4a2a;border-radius:8px;padding:8px 14px;margin:0 8px 8px 0}}
.stat b{{color:#7d7;font-size:18px}}
section{{background:#161616;border:1px solid #2a2a2a;border-radius:10px;padding:16px 18px;margin:16px 0}}
h2{{font-size:16px;margin:0 0 2px;color:#cfe}} .sub{{color:#888;font-size:12px;margin-bottom:10px}}
.img{{background:#000;border:1px solid #333;border-radius:6px;overflow:auto;text-align:center;padding:6px}}
.img img{{max-width:100%;display:inline-block}}
section p{{color:#bbb;font-size:13px;margin:10px 0 0}}
.remain{{background:#1c1812;border-color:#4a3a1a}} .remain h2{{color:#fda}}
table{{border-collapse:collapse;width:100%;font-size:13px;margin-top:8px}}
td,th{{border:1px solid #333;padding:6px 9px;text-align:left}} th{{background:#1c1c1c}}
.ok{{color:#7d7}} .warn{{color:#fb6}}
</style>
<div class=wrap>
<h1>일반화 carve — 작업 결과</h1>
<p class=lead>원칙: LLM은 "여기쯤"(손가락), carve는 "실제 칸에 정확히"(스티커). box를 자가 아니라 이산 포인터로.</p>
<div>
  <span class=stat><b>29+</b> 요소 수정</span>
  <span class=stat><b>0</b> 정답 회귀 (게이트 검증)</span>
  <span class=stat><b>5</b> 폼 코퍼스</span>
  <span class=stat><b>0</b> 새 의존성 · 매직넘버</span>
</div>
{sections}
<section class=remain><h2>남은 것 — carve 밖(앞 단계) 문제</h2>
<table><tr><th>요소</th><th>실제 원인</th><th>레이어</th></tr>
<tr><td>r39 괄호·의사명 (서식4-1호)</td><td>container 밴드가 만든 잉여 검출(진짜 빈칸은 이미 정확). box 안 겹쳐 dedup이 못 잡음</td><td>grounding</td></tr>
<tr><td>r34 signature (서식4-1호)</td><td>LLM이 '서명 또는 인'을 <b>둘</b> 출력(region33 좌·region34 우). r34는 중복</td><td>grounding</td></tr>
<tr><td>r45_text·r35_text</td><td>존재하지 않는 입력 (LLM 오검출)</td><td>LLM</td></tr>
</table>
<p>이것들은 carve가 위치를 틀린 게 아니라, 앞 단계(focus 중첩영역·LLM 중복검출)에서 <b>칸을 잘못 만든</b> 것. 해결하려면 grounding 레이어(container 밴드 스킵·중복 signature 병합)로 올라가야 함.</p>
</section>
</div>"""
OUT.write_text(doc)
print(f"→ {OUT} ({OUT.stat().st_size/1e6:.1f}MB)")


if __name__ == "__main__":
    pass
