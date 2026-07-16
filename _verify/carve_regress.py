"""회귀 게이트 — 현재 carve 결과를 사용자 정답 DB와 대조.
정답(ok)으로 찍힌 요소의 box가 움직였으면 = 회귀 위험. 오답(bad)이 바뀌었으면 = 수정 시도.

실행: python _verify/carve_regress.py [--moved]   # --moved: 움직인 ok 요소 전부 나열
"""
import json, sqlite3, sys
from pathlib import Path
from collections import Counter
import cv2
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core import pipeline, extract, ground_focus
from core.region_segment import segment

CACHE = Path("/private/tmp/claude-501/-Users-gunhee-workspace-codespace-project-lab-voucher-image-to-form"
             "/06ad7bff-82ab-4edb-ba1f-8a8dda5599b0/scratchpad/focus_multidoc_x3.json")
DB = ROOT / "_verify" / "verify.db"


def flat(run):
    o = []
    for j in sorted(run, key=int):
        o += run[j]
    return o


def current_rects():
    data = json.loads(CACHE.read_text())
    out = {}  # (doc,key) -> rect
    for doc in data["docs"]:
        gray = cv2.cvtColor(cv2.imread(doc["path"]), cv2.COLOR_BGR2GRAY)
        S = segment(gray)
        raw = ground_focus.dedup_nested(flat(doc["runs"][0]),
                                        {j: r[2] * r[3] for j, r in pipeline.atoms_of(S)})
        placed = extract.assign_keys(pipeline.place_elements(gray, S, raw))
        for e in placed:
            out[(doc["name"], e.get("key"))] = [int(v) for v in e["rect"]]
    return out


def main(list_moved=False):
    cur = current_rects()
    con = sqlite3.connect(DB)
    jud = {(d, k): (v, json.loads(r) if r else None)
           for d, k, v, r in con.execute("SELECT doc,key,verdict,rect FROM judgments")}
    tally = Counter()
    ok_moved, bad_changed, bad_same = [], [], []
    for (d, k), (verdict, jrect) in jud.items():
        nrect = cur.get((d, k))
        if nrect is None:
            tally["요소사라짐"] += 1; continue
        moved = jrect != nrect
        if verdict == "ok":
            if moved: tally["ok_이동(회귀위험)"] += 1; ok_moved.append((d, k, jrect, nrect))
            else: tally["ok_유지(안전)"] += 1
        elif verdict == "bad":
            if moved: tally["bad_변경(수정시도)"] += 1; bad_changed.append((d, k, jrect, nrect))
            else: tally["bad_그대로(미수정)"] += 1; bad_same.append((d, k))
    print("=== 회귀 게이트 (현재 carve vs 정답 DB) ===")
    for kk in ["ok_유지(안전)", "ok_이동(회귀위험)", "bad_변경(수정시도)", "bad_그대로(미수정)", "요소사라짐"]:
        print(f"  {kk:20} {tally.get(kk,0)}")
    if ok_moved:
        print(f"\n⚠️  정답인데 움직인 것 {len(ok_moved)}개 (회귀 확인 필요):")
        for d, k, jr, nr in (ok_moved if list_moved else ok_moved[:20]):
            print(f"   {d:12} {k[:40]:40} {jr} → {nr}")
    if bad_changed:
        print(f"\n✎ 오답인데 바뀐 것 {len(bad_changed)}개 (재검증 대상):")
        for d, k, jr, nr in bad_changed:
            print(f"   {d:12} {k[:40]:40} {jr} → {nr}")
    if bad_same:
        print(f"\n· 오답 그대로 {len(bad_same)}개 (아직 미수정):")
        for d, k in bad_same:
            print(f"   {d:12} {k[:44]}")


if __name__ == "__main__":
    main(list_moved="--moved" in sys.argv)
