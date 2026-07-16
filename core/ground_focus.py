"""실험용 그라운딩: SoM 영역마다 하이라이트 이미지 → LLM 1콜 → 합치기.

시스템 프롬프트(SYS_V2)는 콜마다 동일 → OpenRouter/Gemini prefix 캐시 이점 기대.
페이지 통짜 ground() 대체용. pipeline은 IMG2FORM_GROUND=focus 로 교체.
"""
from __future__ import annotations

import base64
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np

from . import extract


def mark_focus(img, atoms, focus_id, dim=0.18):
    """전체 페이지: focus만 원본 밝기·두꺼운 테두리, 나머지 강하게 어둡게.
    box 좌표는 페이지 기준(0~1000) 유지."""
    out = (img.astype(np.float32) * dim).astype(np.uint8)
    for j, (x, y, w, h) in atoms:
        if j != focus_id:
            continue
        out[y:y + h, x:x + w] = img[y:y + h, x:x + w]
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 165, 255), 4)
        cv2.rectangle(out, (x, y), (x + 30, y + 18), (0, 140, 255), -1)
        cv2.putText(out, str(j), (x + 3, y + 14), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1, cv2.LINE_AA)
        break
    # 나머지 번호는 아주 약하게만 (맥락)
    for j, (x, y, w, h) in atoms:
        if j == focus_id:
            continue
        cv2.rectangle(out, (x, y), (x + w, y + h), (60, 60, 60), 1)
        cv2.putText(out, str(j), (x + 1, y + 11), cv2.FONT_HERSHEY_SIMPLEX,
                    0.3, (140, 140, 140), 1, cv2.LINE_AA)
    return out


def _b64(im, q=72):
    return base64.b64encode(cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, q])[1]).decode()


def _atom_box_1000(rect, hw):
    """atom (x,y,w,h) px → [ymin,xmin,ymax,xmax] in 0~1000."""
    x, y, w, h = (int(v) for v in rect)
    IH, IW = hw
    return [
        int(y / IH * 1000),
        int(x / IW * 1000),
        int((y + h) / IH * 1000),
        int((x + w) / IW * 1000),
    ]


def _clamp_box_to_region(box, region_box):
    """box를 region_box 안으로 clamp. 면적 0이면 None."""
    by0, bx0, by1, bx1 = (int(v) for v in box)
    ry0, rx0, ry1, rx1 = region_box
    y0 = max(by0, ry0); x0 = max(bx0, rx0)
    y1 = min(by1, ry1); x1 = min(bx1, rx1)
    if y1 - y0 < 1 or x1 - x0 < 1:
        return None
    return [y0, x0, y1, x1]


def _clamp_region(els, focus_id, region_box=None):
    """region id 강제 + (있으면) box를 하이라이트 영역 안으로 clamp. 밖·붕괴 box는 제거."""
    out = []
    for e in els or []:
        if not e.get("box") or len(e["box"]) != 4:
            continue
        e = dict(e)
        e["region"] = focus_id
        if region_box is not None:
            clamped = _clamp_box_to_region(e["box"], region_box)
            if not clamped:
                continue
            if clamped != [int(v) for v in e["box"]]:
                e["_clamped"] = True
            e["box"] = clamped
        out.append(e)
    return out


def ground_one(img, atoms, focus_id, reasoning_effort="low", **kwargs):
    """단일 영역 하이라이트 → ground(focus). assign 없이 raw elements.
    반환 box는 하이라이트 atom 안으로 clamp."""
    IH, IW = img.shape[:2]
    rect = next(r for j, r in atoms if j == focus_id)
    region_box = _atom_box_1000(rect, (IH, IW))
    marked = mark_focus(img, atoms, focus_id)
    res = extract.ground(
        _b64(marked), len(atoms),
        focus_region=focus_id, assign=False,
        focus_style="highlight", image_first=False,
        reasoning_effort=reasoning_effort, **kwargs,
    )
    els = _clamp_region(res.get("elements"), focus_id, region_box)
    return {
        "region": focus_id,
        "elements": els,
        "_meta": res.get("_meta"),
        "n": len(els),
        "region_box": region_box,
    }


def ground_one_fixed(som_b64, atoms, focus_id, hw, reasoning_effort="low", **kwargs):
    """고정 SoM 이미지 + 텍스트로만 영역 N 지정. image_first로 prefix 캐시 유도.
    box는 atom 안으로 clamp."""
    rect = next(r for j, r in atoms if j == focus_id)
    region_box = _atom_box_1000(rect, hw)
    res = extract.ground(
        som_b64, len(atoms),
        focus_region=focus_id, assign=False,
        focus_style="text", image_first=True,
        reasoning_effort=reasoning_effort, **kwargs,
    )
    els = _clamp_region(res.get("elements"), focus_id, region_box)
    return {
        "region": focus_id,
        "elements": els,
        "_meta": res.get("_meta"),
        "n": len(els),
        "region_box": region_box,
        "mode": "fixed_image",
    }


_CIRC = "①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮"


def _norm_opt(o):
    """옵션 정규화 — 원문자 ①②→1 2, 공백·괄호 제거 (중복 판정용)."""
    o = str(o or "")
    for i, c in enumerate(_CIRC):
        o = o.replace(c, str(i + 1))
    return o.replace(" ", "").translate(str.maketrans("", "", "()（）"))


def _overlap_min(a, b):
    """교집합 / 작은 box 면적. box=[ymin,xmin,ymax,xmax]. 크기 다른 box도 겹침 판정."""
    ay0, ax0, ay1, ax1 = a
    by0, bx0, by1, bx1 = b
    inter = max(0, min(ax1, bx1) - max(ax0, bx0)) * max(0, min(ay1, by1) - max(ay0, by0))
    if inter <= 0:
        return 0.0
    m = min((ax1 - ax0) * (ay1 - ay0), (bx1 - bx0) * (by1 - by0))
    return inter / m if m else 0.0


def dedup_nested(elements, region_area, thr=0.5):
    """중첩 영역 이중검출 제거 — SoM이 '큰 블록 + 개별 셀'을 다 만들면 focus가 같은 입력을 두 번 뽑는다.
    같은 type + 정규화 옵션 + box 겹침(작은box 기준 ≥thr)이면 큰 영역(container) 판을 버리고
    작은 영역(leaf 셀) 판을 남긴다. 겹침 없으면(진짜 다른 입력) 안 건드림."""
    drop = set()
    for i, e in enumerate(elements):
        if i in drop or not e.get("box") or len(e["box"]) != 4:
            continue
        for j in range(i + 1, len(elements)):
            f = elements[j]
            if j in drop or not f.get("box") or len(f["box"]) != 4:
                continue
            if e.get("type") != f.get("type") or _norm_opt(e.get("option")) != _norm_opt(f.get("option")):
                continue
            if _overlap_min(e["box"], f["box"]) < thr:
                continue
            ai = region_area.get(e.get("region"), float("inf"))
            aj = region_area.get(f.get("region"), float("inf"))
            drop.add(i if ai >= aj else j)                     # 큰 영역 판 제거
    return [e for k, e in enumerate(elements) if k not in drop]


def ground_by_regions(img, atoms, reasoning_effort="low", workers=4, region_ids=None, **kwargs):
    """모든(또는 지정) 영역 병렬 호출 → concat + assign_keys.
    → {"elements":[...], "_meta":{...}, "_per_region":[...]}"""
    ids = list(region_ids) if region_ids is not None else [j for j, _ in atoms]
    per = [None] * len(ids)

    def _job(idx, rid):
        return idx, ground_one(img, atoms, rid, reasoning_effort=reasoning_effort, **kwargs)

    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        futs = [ex.submit(_job, i, rid) for i, rid in enumerate(ids)]
        for fut in as_completed(futs):
            i, one = fut.result()
            per[i] = one

    all_els = []
    usages = []
    cached = 0
    cost = 0.0
    for one in per:
        all_els.extend(one["elements"])
        u = (one.get("_meta") or {}).get("usage") or {}
        usages.append(u)
        cached += (u.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
        cost += float(u.get("cost") or 0)

    all_els = dedup_nested(all_els, {j: r[2] * r[3] for j, r in atoms})   # 중첩 영역 이중검출 제거
    els = extract.assign_keys(all_els)
    return {
        "elements": els,
        "_meta": {
            "mode": "focus_regions",
            "n_regions": len(ids),
            "per_region_counts": [p["n"] for p in per],
            "merged_count": len(els),
            "cached_tokens_sum": cached,
            "cost_sum": round(cost, 6),
            "effort": reasoning_effort,
            "workers": workers,
            "usages": usages,
        },
        "_per_region": per,
    }
