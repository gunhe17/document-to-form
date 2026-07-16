"""carve 씨앗불변성 — 여러 폼 전체에서 측정·집계 (API 0원).
focus_multidoc_x3.json 의 3회 box → 폼마다 place_elements ×3 → rect 수렴.
rule별로 '몇 개 폼에서, 몇 요소가' 새는지 집계 → 일반적으로 새는 규칙을 찾는다.

실행: python _verify/carve_converge_multi.py [--list]   # --list: 새는 요소 전부 나열
"""
import json, sys
from collections import defaultdict
from pathlib import Path
import cv2
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core import pipeline
from core.region_segment import segment

CACHE = Path("/private/tmp/claude-501/-Users-gunhee-workspace-codespace-project-lab-voucher-image-to-form"
             "/06ad7bff-82ab-4edb-ba1f-8a8dda5599b0/scratchpad/focus_multidoc_x3.json")


def flat(run):  # run: {region(str/int): [elements]}
    out = []
    for j in sorted(run, key=int):
        out += run[j]
    return out


def doc_converge(doc):
    gray = cv2.cvtColor(cv2.imread(doc["path"]), cv2.COLOR_BGR2GRAY)
    S = segment(gray)
    placed = [pipeline.place_elements(gray, S, flat(r)) for r in doc["runs"]]
    n = min(len(p) for p in placed)
    rows = []
    for i in range(n):
        cv = [placed[r][i]["rect"] for r in range(3)]
        jc = max(max(c[k] for c in cv) - min(c[k] for c in cv) for k in range(4))
        e0 = placed[0][i]
        rules = tuple(placed[r][i].get("rule") for r in range(3))
        rows.append({"jc": jc, "region": e0.get("region"), "type": e0.get("type"),
                     "val": str(e0.get("option") or e0.get("label"))[:14], "rules": rules})
    return n, rows


def main(list_all=False):
    data = json.loads(CACHE.read_text())
    print(f"model={data['model']} temp={data['temp']} · docs={len(data['docs'])}\n")
    tot_n = tot_id = 0
    rule_leak = defaultdict(lambda: [0, 0, set()])   # rule -> [leak요소수, 지터합, {폼}]
    all_bad = []
    print(f"{'doc':16} {'요소':>4} {'동일':>5} {'수렴%':>6} {'평균px':>7} {'최대px':>7}")
    for doc in data["docs"]:
        n, rows = doc_converge(doc)
        bad = [r for r in rows if r["jc"] > 0]
        ident = n - len(bad)
        tot_n += n; tot_id += ident
        avg = sum(r["jc"] for r in rows) / n if n else 0
        mx = max((r["jc"] for r in rows), default=0)
        print(f"{doc['name']:16} {n:>4} {ident:>5} {100*ident/n:>5.0f}% {avg:>6.2f} {mx:>7}")
        for r in bad:
            # 새는 rule: 3회 중 등장한 rule 집합
            for ru in set(r["rules"]):
                rule_leak[ru][0] += 1
                rule_leak[ru][1] += r["jc"]
                rule_leak[ru][2].add(doc["name"])
            r["doc"] = doc["name"]
            all_bad.append(r)
    print(f"\n{'='*54}\n전체: {tot_id}/{tot_n} 수렴 ({100*tot_id/tot_n:.0f}%) · 미수렴 {tot_n-tot_id}개")
    print(f"\n[일반적으로 새는 rule]  (여러 폼에 걸치면 = 일반 문제)")
    print(f"{'rule':16} {'새는요소':>7} {'폼수':>5} {'지터합':>7}")
    for ru, (cnt, jsum, docs) in sorted(rule_leak.items(), key=lambda x: -x[1][0]):
        print(f"{ru:16} {cnt:>7} {len(docs):>5} {jsum:>7}   {sorted(docs)}")
    if list_all:
        print(f"\n[미수렴 요소 전체]")
        for r in sorted(all_bad, key=lambda r: -r["jc"]):
            rr = r["rules"][0] if len(set(r["rules"])) == 1 else "/".join(map(str, r["rules"]))
            print(f"  {r['jc']:>3}px {r['doc']:14} reg{r['region']:>3} {r['type']:12} {r['val']:14} {rr}")


if __name__ == "__main__":
    main(list_all="--list" in sys.argv)
