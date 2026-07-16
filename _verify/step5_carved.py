"""step5_grounding.html 형식 그대로, 내 focus 결과(dedup+수정 반영) 5폼으로 렌더.
render.py의 _PAGE·overlay·thumb 재사용. 새 LLM 호출 없음(캐시된 focus 결과 사용).
실행: python _verify/step5_carved.py  →  _verify/step5_carved.html
"""
import sys, json, base64, html
from pathlib import Path
import cv2
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core import pipeline, extract, ground_focus
from core.region_segment import segment
import render as R   # _PAGE·overlay·thumb_split·thumb_raw·b64 재사용 (import 부작용 없음)

CACHE = Path("/private/tmp/claude-501/-Users-gunhee-workspace-codespace-project-lab-voucher-image-to-form"
             "/06ad7bff-82ab-4edb-ba1f-8a8dda5599b0/scratchpad/focus_multidoc_x3.json")
OUT = ROOT / "_verify" / "step5_carved.html"


def flat(run):
    o = []
    for j in sorted(run, key=int):
        o += run[j]
    return o


def main():
    data = json.loads(CACHE.read_text())
    tabs = panels = ""
    for tab, doc in enumerate(data["docs"]):
        img = cv2.imread(doc["path"]); gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        IH, IW = gray.shape; S = segment(gray); atoms = pipeline.atoms_of(S)
        rawll = flat(doc["runs"][0])                                    # LLM 원본(중복 포함)
        raw = ground_focus.dedup_nested(rawll, {j: r[2] * r[3] for j, r in atoms})
        els = extract.assign_keys(pipeline.place_elements(gray, S, raw))
        ncorr = sum(1 for it in els if it.get("corrected"))
        th1 = R.b64(R.thumb_split(img, S)); th2 = R.b64(pipeline.som_mark(img, atoms))
        th3 = R.b64(R.thumb_raw(img, rawll, IW, IH)); clean = R.b64(img)
        on = (tab == 0)
        tabs += f'<button class="tab{" on" if on else ""}" onclick="show({tab})">{html.escape(doc["name"])}</button>'
        panels += (f'<div class=view id=v{tab} style="display:{"flex" if on else "none"}">'
                   f'<div class=side><div class=stat>atom {len(atoms)} · LLM요소 {len(rawll)} · 렌더 {len(els)} · 수정 {ncorr}</div>'
                   f'<div class=th><div class=thh>① 영역분리 (LLM 없음)</div><img onclick="zoom(this.src)" src="data:image/jpeg;base64,{th1}"></div>'
                   f'<div class=th><div class=thh>② SoM 번호마킹 (LLM 입력)</div><img onclick="zoom(this.src)" src="data:image/jpeg;base64,{th2}"></div>'
                   f'<div class=th><div class=thh>③ LLM 원본 box (place 전)</div><img onclick="zoom(this.src)" src="data:image/jpeg;base64,{th3}"></div></div>'
                   f'<div class=main><div class=mh>④ 타입규칙 배치 — 위젯 클릭 → 우측 정보 · {len(els)}요소</div>'
                   f'<div class=canvas><img src="data:image/jpeg;base64,{clean}">{R.overlay(els, IW, IH, tab)}</div></div>'
                   f'<div class=info><div class=ih>선택 요소 정보</div><div class=ibody id=info{tab}b><div class=hint>← 위젯 클릭</div></div></div></div>')
        print(f"{doc['name']}: atom{len(atoms)} LLM{len(rawll)} 렌더{len(els)} 수정{ncorr}")
    OUT.write_text(R._PAGE.replace("__TABS__", tabs).replace("__PANELS__", panels), encoding="utf-8")
    print("✓", OUT, f"({OUT.stat().st_size/1e6:.1f}MB)")


if __name__ == "__main__":
    main()
