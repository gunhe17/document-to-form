"""carve 씨앗불변성 오프라인 검증 — 캐시된 3회 LLM box → place_elements ×3 → rect 수렴 측정.
API 0원. carve.py 고치고 이거 돌려서 45/59 → 59/59 가는지 반복 확인.

실행: python _verify/carve_converge.py [--diff]   # --diff: 현재값 대비 이동한 요소도 출력
"""
import json, sys
from pathlib import Path
import cv2
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core import pipeline
from core.region_segment import segment

CACHE = Path("/private/tmp/claude-501/-Users-gunhee-workspace-codespace-project-lab-voucher-image-to-form"
             "/06ad7bff-82ab-4edb-ba1f-8a8dda5599b0/scratchpad/focus_models_x3.json")
IMG = ("/Users/gunhee/workspace/codespace/project/lab-voucher/.old/document/"
       "제1편_장애아가족_양육지원_서식/_extract/서식2호_장애아돌보미_지원서/pages/p-1.png")


def load_runs(tag="gemini-3.1-pro-preview"):
    d = json.loads(CACHE.read_text())
    m = next(x for x in d["models"] if x["tag"] == tag)
    return [{int(k): v for k, v in r.items()} for r in m["runs"]]


def flat(run):
    out = []
    for j in sorted(run):
        out += run[j]
    return out


def run():
    gray = cv2.cvtColor(cv2.imread(IMG), cv2.COLOR_BGR2GRAY)
    S = segment(gray)
    runs = load_runs()
    placed = [pipeline.place_elements(gray, S, flat(r)) for r in runs]
    n = min(len(p) for p in placed)
    ident = 0
    jitters = []
    for i in range(n):
        cv = [placed[r][i]["rect"] for r in range(3)]
        jc = max(max(c[k] for c in cv) - min(c[k] for c in cv) for k in range(4))
        if jc == 0:
            ident += 1
        else:
            e0 = placed[0][i]
            jitters.append((jc, i, e0.get("region"), e0.get("type"),
                            str(e0.get("option") or e0.get("label"))[:14],
                            [placed[r][i].get("rule") for r in range(3)], cv))
    jitters.sort(reverse=True)
    avg = sum(j[0] for j in jitters) / n
    print(f"place_elements 결과 {n}개 · 3회 rect 완전동일 {ident}/{n} ({100*ident/n:.0f}%) "
          f"· 평균 {avg:.2f}px · 최대 {jitters[0][0] if jitters else 0}px\n")
    print(f"{'jc':>3} {'reg':>3} {'type':13} {'값':15} rule(3회)")
    for jc, i, reg, t, val, rules, cv in jitters:
        rr = rules[0] if len(set(rules)) == 1 else "/".join(map(str, rules))
        print(f"{jc:>3} {reg:>3} {t:13} {val:15} {rr}")
    return ident, n, jitters


if __name__ == "__main__":
    run()
