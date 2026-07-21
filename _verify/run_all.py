"""확정 대구획 파이프라인을 모든 서식에 실행 → carve 전 raw 그라운딩을 _verify/raw/ 에 저장.

raw JSON은 재사용용(LLM 재호출 무료): image + elements 로 carve/렌더 언제든 재생성.
  ground_focus.extract_form = ①region_segment → ②detect_focus_regions → ③ground_blocks(+region-clamp)

실행: python _verify/run_all.py [인덱스...]   (없으면 SKIP 제외 전체 18개)
이미 있는 raw 는 건너뜀(resumable). --force 로 재생성.
"""
import sys, json, time
from pathlib import Path
import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core import ground_focus, budget   # noqa: E402
from render import DOCS, SKIP           # noqa: E402

RAW = ROOT / "_verify" / "raw"
RAW.mkdir(parents=True, exist_ok=True)


def slug(i, name):
    return f"{i:02d}_{name.replace(' ', '_').replace('/', '_')}.json"


def run(idxs, force=False):
    total = 0.0
    for i in idxs:
        name, path = DOCS[i]
        out = RAW / slug(i, name)
        if out.exists() and not force:
            print(f"· skip {name} (이미 있음)")
            continue
        img = cv2.imread(path)
        if img is None:
            print(f"!! {name}: 이미지 없음 {path}")
            continue
        budget.check(0.3)                       # 폼당 여유 예산 — 초과 방지
        t0 = time.time()
        r = ground_focus.extract_form(img)
        cost = float(r["_meta"].get("cost_sum") or 0)  # 대구획 검출 콜(~$0.01)은 미포함
        budget.add(cost)
        total += cost
        # 불완전 결과는 저장 안 함 → 다음 실행에 재시도(재사용 캐시 오염 방지)
        nf = r["_meta"].get("n_failed", 0)
        if nf:
            print(f"⚠ {name}: {nf}/{len(r['groups'])}블록 콜 실패 — 저장 안 함(재시도) ${cost:.4f}")
            continue
        if len(r["groups"]) > 12:               # PARTITION_SYS 최대 5~6 → 12↑ = 대구획 검출 실패(폴백)
            print(f"⚠ {name}: 대구획 {len(r['groups'])}개(검출 실패 의심) — 저장 안 함(재시도)")
            continue
        rec = {
            "name": name, "index": i, "image": path,
            "n_atoms": len(r["atoms"]), "groups": r["groups"],
            "elements": r["elements"], "_meta": r["_meta"],
            "elapsed": round(time.time() - t0, 1),
        }
        out.write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"✓ {name}: {len(r['elements'])}요소 {len(r['groups'])}구획 "
              f"{rec['elapsed']}s ${cost:.4f}  → {out.name}")
    print(f"\n합계 ${total:.4f} · 오늘 누적 ${budget.spent():.3f}/${budget.CAP:.0f}")


if __name__ == "__main__":
    force = "--force" in sys.argv
    nums = [int(a) for a in sys.argv[1:] if a.isdigit()]
    idxs = nums or [i for i in range(len(DOCS)) if i not in SKIP]
    run(idxs, force)
