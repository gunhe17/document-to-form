"""여러 폼을 gemini-3.1-pro @0.2 focus ×3 호출 → 코퍼스 캐시.
carve 씨앗불변성을 '전체 폼에서' 일반화·검증하기 위한 데이터.

실행: python _verify/focus_multidoc.py          # 없는 폼만 호출
     FORCE=1 python _verify/focus_multidoc.py  # 전부 재호출
캐시: scratchpad/focus_multidoc_x3.json
"""
from __future__ import annotations
import json, os, sys, time
from pathlib import Path
import cv2
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core import pipeline, ground_focus
from core.region_segment import segment

CACHE = Path("/private/tmp/claude-501/-Users-gunhee-workspace-codespace-project-lab-voucher-image-to-form"
             "/06ad7bff-82ab-4edb-ba1f-8a8dda5599b0/scratchpad/focus_multidoc_x3.json")
OLD = ("/Users/gunhee/workspace/codespace/project/lab-voucher/.old/document")
DOCS = [
    ("서식2호", f"{OLD}/제1편_장애아가족_양육지원_서식/_extract/서식2호_장애아돌보미_지원서/pages/p-1.png"),
    # 깨끗한 .old 원본 사용 (_verify 로컬본은 주석/색상 오버레이가 있어 오염됨)
    ("서식15호", f"{OLD}/제1편_장애아가족_양육지원_서식/_extract/서식15호_사고보고서/pages/p-1.png"),
    ("서식4-1호", f"{OLD}/제2편_발달재활서비스_서식/_extract/서식4_1호_발달재활서비스_의뢰서/pages/p-1.png"),
    ("아동정서_p10", "_verify/아동정서발달/p-10.png"),
    ("서식8호_계획", f"{OLD}/제3편_언어발달지원_서식/_extract/서식8호_언어발달지원_서비스_제공(이용)_계획서/pages/p-1.png"),
]
MODEL = "google/gemini-3.1-pro-preview"
TEMP = 0.2
N_RUNS = 3
WORKERS = 8


def _save(by_name):
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps({"model": MODEL, "temp": TEMP, "docs": list(by_name.values())},
                                ensure_ascii=False))


def main():
    by_name = {}
    if CACHE.exists():
        for d in json.loads(CACHE.read_text())["docs"]:
            by_name[d["name"]] = d

    for name, path in DOCS:
        if name in by_name and not os.environ.get("FORCE"):
            print(f"skip {name} (cache)", flush=True); continue
        img = cv2.imread(path)
        if img is None:
            print(f"LOAD FAIL {name} {path}", flush=True); continue
        atoms = pipeline.atoms_of(segment(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)))
        print(f"\n=== {name} · atoms={len(atoms)} ===", flush=True)
        runs, cost = [], 0.0
        t0 = time.perf_counter()
        for r in range(N_RUNS):
            res = ground_focus.ground_by_regions(img, atoms, reasoning_effort="low",
                                                 workers=WORKERS, model=MODEL, temperature=TEMP)
            per = {p["region"]: p["elements"] for p in res["_per_region"]}
            runs.append(per)
            cost += float((res["_meta"] or {}).get("cost_sum") or 0)
            print(f"  run{r+1}: {sum(len(v) for v in per.values())} elements", flush=True)
        by_name[name] = {"name": name, "path": path, "atoms": len(atoms),
                         "runs": runs, "cost": round(cost, 4), "secs": round(time.perf_counter()-t0, 1)}
        _save(by_name)
        print(f"  saved {name} · ${by_name[name]['cost']} · {by_name[name]['secs']}s", flush=True)
    print(f"\n총 {len(by_name)} 폼 · ${sum(d['cost'] for d in by_name.values()):.2f}")


if __name__ == "__main__":
    main()
