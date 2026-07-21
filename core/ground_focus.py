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


def mark_focus_multi(img, atoms, focus_ids, dim=0.18):
    """블록 하이라이트: focus_ids에 속한 원자 전부 원본 밝기·테두리, 나머지는 어둡게(맥락).
    box 좌표는 페이지 기준(0~1000) 유지 — crop 아님."""
    fs = set(focus_ids)
    out = (img.astype(np.float32) * dim).astype(np.uint8)
    for j, (x, y, w, h) in atoms:
        if j in fs:
            out[y:y + h, x:x + w] = img[y:y + h, x:x + w]
            cv2.rectangle(out, (x, y), (x + w, y + h), (0, 165, 255), 4)
            cv2.rectangle(out, (x, y), (x + 30, y + 18), (0, 140, 255), -1)
            cv2.putText(out, str(j), (x + 3, y + 14), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 1, cv2.LINE_AA)
    for j, (x, y, w, h) in atoms:                              # 나머지 번호는 아주 약하게만
        if j not in fs:
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


GROUP_SYS = (
    "번호(0~N-1) 박스로 영역이 표시된 빈 정부서식 이미지다. 작성자가 채우는 관점에서, "
    "'하나의 입력 단위로 함께 봐야 하는' 영역 번호들을 그룹으로 묶어라.\n"
    "묶는 예: 라벨 영역 + 바로 그 라벨의 입력칸(예 '이름'칸+오른쪽 빈칸), "
    "'1~10' 같은 택1 눈금의 낱칸들, 한 줄 '__년 __월 __일'의 단위+칸.\n"
    "독립적인 영역은 자기 혼자 그룹. 표의 서로 다른 셀은 각각 별개.\n"
    "규칙: 모든 번호는 정확히 한 그룹에 한 번만. JSON만."
)
GROUP_SCHEMA = {
    "type": "object",
    "properties": {"groups": {"type": "array", "items": {
        "type": "array", "items": {"type": "integer"}}}},
    "required": ["groups"], "additionalProperties": False,
}


# Gemini 3은 간결·직접 지시에 가장 잘 반응(gemini3devguide: 장황·복잡한 프롬프트는 과잉분석).
PARTITION_SYS = (
    "번호(0~N-1) 박스로 영역이 표시된 빈 정부서식이다. 이 번호들을 큰 논리 구획으로 묶어라.\n"
    "규칙:\n"
    "1. 하나의 표는 [열 제목 행 + 데이터 본문 전체]를 반드시 한 구획으로. 표의 머리행을 본문과 절대 분리하지 마라.\n"
    "2. 표 위의 큰 제목·안내 문구 밴드는 표와 별개 구획.\n"
    "3. 같은 형식이 반복되는 행들(명단 등)은 그 표 전체를 한 구획으로.\n"
    "4. 구획은 최대 5~6개. 잘게 쪼개지 말 것.\n"
    "5. 모든 번호는 정확히 한 구획에 한 번만. 위치상 인접한 번호끼리만.\n"
    "JSON만: {\"groups\":[[번호,...],...]}"
)


def llm_group_atoms(marked_b64, n_atoms, model=None, temperature=1.0,
                    reasoning_effort="low", retries=3, system=None):
    """SoM 마킹 이미지 → LLM이 원자 번호를 그룹/블록으로 묶음. → [[id,...], ...].
    system=None이면 GROUP_SYS(논리 그룹); PARTITION_SYS 주입 시 거친 위치 블록.
    실패/누락은 코드가 단독 그룹으로 보정. temperature=1.0 : Gemini 3 권장(gemini3devguide)."""
    import json, re, urllib.request
    model = model or extract.MODEL
    us = [{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + marked_b64}},
          {"type": "text", "text": f"위 서식의 번호 0~{n_atoms-1}을 규칙대로 구획으로 묶어라."}]
    body = json.dumps({
        "model": model, "temperature": temperature, "max_tokens": 8000,
        "reasoning": {"effort": reasoning_effort},
        "messages": [{"role": "system", "content": system or GROUP_SYS}, {"role": "user", "content": us}],
        "response_format": {"type": "json_schema", "json_schema": {
            "name": "atom_groups", "strict": True, "schema": GROUP_SCHEMA}},
    }).encode()
    groups = None
    for _ in range(retries):
        req = urllib.request.Request(extract.API, data=body, headers={
            "Authorization": f"Bearer {extract.load_key()}", "Content-Type": "application/json"})
        try:
            out = json.load(urllib.request.urlopen(req, timeout=200))["choices"][0]["message"]["content"]
            if isinstance(out, list):
                out = "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in out)
            groups = json.loads(re.search(r"\{.*\}", out.replace("```", ""), re.S).group(0)).get("groups") or []
            break
        except Exception:
            continue
    if groups is None:
        groups = [[j] for j in range(n_atoms)]                # 폴백: 전부 단독
    return _normalize_groups(groups, n_atoms)


def _normalize_groups(groups, n_atoms):
    """중복 제거 + 누락 원자 단독 그룹 보정 → 각 원자가 정확히 한 그룹에."""
    seen = set(); clean = []
    for g in groups:
        gg = [i for i in g if isinstance(i, int) and 0 <= i < n_atoms and i not in seen]
        if gg:
            seen.update(gg); clean.append(gg)
    for j in range(n_atoms):
        if j not in seen:
            clean.append([j])
    return clean


def focus_areas_from_groups(atoms, groups):
    """그룹 → focus 영역(멤버 rect들의 union). → [{'rect','members','kind'}]."""
    rects = {j: tuple(int(v) for v in r) for j, r in atoms}
    areas = []
    for g in groups:
        rs = [rects[j] for j in g if j in rects]
        if not rs:
            continue
        x0 = min(r[0] for r in rs); y0 = min(r[1] for r in rs)
        x1 = max(r[0] + r[2] for r in rs); y1 = max(r[1] + r[3] for r in rs)
        areas.append({"rect": (x0, y0, x1 - x0, y1 - y0), "members": g,
                      "kind": "atom" if len(g) == 1 else "group"})
    return areas


def detect_focus_regions(img, model=None, temperature=0.5, reasoning_effort="low"):
    """LLM으로 '큰 논리 구획(focus 영역)'을 검출 — 함께 확정한 과정:
    region_segment(atoms) → SoM 마킹 → LLM 대구획 판단(PARTITION_SYS) → 번호 목록.
    확정 최적 기본값(temp 0.5·reasoning low). → {'atoms','groups','areas'}.
    groups = [[atom_id,...], ...] (각 focus 영역의 번호 묶음), areas = union rect 정보."""
    import cv2
    from . import pipeline
    from .region_segment import segment
    color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    atoms = pipeline.atoms_of(segment(gray))
    groups = llm_group_atoms(_b64(pipeline.som_mark(color, atoms)), len(atoms),
                             model=model, temperature=temperature,
                             reasoning_effort=reasoning_effort, system=PARTITION_SYS)
    return {"atoms": atoms, "groups": groups, "areas": focus_areas_from_groups(atoms, groups)}


def _is_repeating_table(sub, min_rows=3, min_cols=3, frac=0.6):
    """반복 입력 테이블(명단 등) 판정 — 원자들이 [≥min_cols 열 × ≥min_rows 행]의 규칙 격자인가.
    각 행 밴드의 원자 수(=열 수)가 대부분(frac) 같고 ≥min_cols면 반복 테이블 → 자르지 않음.
    라벨|값 2열 폼(서식15호: 행당 2원자)이나 다-□ 밀집 셀은 격자가 아니라 False → 정상 분할."""
    from collections import Counter, defaultdict
    if len(sub) < min_rows * min_cols:
        return False
    hs = sorted(r[3] for _, r in sub); band = max(8, hs[len(hs) // 2])
    rows = defaultdict(int)
    for _, r in sub:
        rows[int((r[1] + r[3] / 2) / band)] += 1        # 행 밴드별 원자 수
    counts = list(rows.values())
    if len(counts) < min_rows:
        return False
    mode_n, freq = Counter(counts).most_common(1)[0]
    return mode_n >= min_cols and freq >= len(counts) * frac


def _split_big_groups(atoms, groups, cap=10):
    """너무 큰 대구획을 geo_partition으로 소블록 분할 → 그라운딩 붕괴(bimodal) 방지.
    거대 영역 한 콜은 모델이 draw마다 '완전열거↔요약'으로 진동(서식15호 57↔16 실측).
    작은 블록은 매번 안정적으로 완전 열거(cap8: 54/55/55). 작은 대구획은 그대로 둔다.
    단 반복 입력 테이블(명단 등 다열 규칙 격자)은 헤더·맥락 유지 위해 자르지 않고 통째로 둔다."""
    rectof = {j: r for j, r in atoms}
    blocks = []
    for g in groups:
        sub = [(j, rectof[j]) for j in g if j in rectof]
        if len(g) > cap and not _is_repeating_table(sub):
            blocks.extend(geo_partition(sub, cap=cap))
        else:
            blocks.append(g)
    return blocks


def extract_form(img, temperature=0.5, reasoning_effort="low", workers=4, model=None, max_block=10):
    """확정 대구획 파이프라인 end-to-end (LLM 그라운딩까지, carve 전 raw).
    ① region_segment → ② detect_focus_regions(대구획) → (거대 구획 소블록 분할) → ③ ground_blocks(하이라이트 그라운딩+clamp).
    max_block: 이보다 큰 대구획은 소블록으로 쪼갬 — 거대 영역의 stochastic 검출 붕괴 방지(실측: 안정 55).
    반환 elements = carve 전 raw. carve는 이 결과로 무료 재실행 가능.
    → {'atoms','groups','blocks','areas','elements','_meta'}. temp 0.5·low = 실측 확정 최적."""
    color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img
    d = detect_focus_regions(img, model=model, temperature=temperature,
                             reasoning_effort=reasoning_effort)
    blocks = _split_big_groups(d["atoms"], d["groups"], cap=max_block)
    res = ground_blocks(color, d["atoms"], blocks, reasoning_effort=reasoning_effort,
                        workers=workers, temperature=temperature, model=model)
    return {"atoms": d["atoms"], "groups": d["groups"], "blocks": blocks, "areas": d["areas"],
            "elements": res["elements"], "_meta": res["_meta"]}


def _reading_order(atoms):
    """행 밴딩 후 (행, x) 정렬 — 같은 줄 원자가 배치 안에 함께 남도록."""
    if not atoms:
        return []
    hs = sorted(r[3] for _, r in atoms)
    band = max(8, hs[len(hs) // 2])                       # 중앙 높이 = 행 밴드
    return sorted(atoms, key=lambda a: (int((a[1][1] + a[1][3] / 2) / band), a[1][0]))


def ground_batched(img, atoms, k=12, reasoning_effort="low", workers=4, pad=0.015, **kwargs):
    """의미 그룹핑 없이 읽기순 k개씩 배치 → 배치 crop을 page식 1콜로 그라운딩.
    각 배치: bbox crop → 로컬 SoM(0~k-1) 재번호 → ground(page) → box를 원본 0~1000 역매핑 + atom clamp.
    콜 수 = ceil(N/k). page(붕괴)와 focus(콜폭발) 사이 sweet spot."""
    from . import pipeline
    IH, IW = img.shape[:2]
    ordered = _reading_order(atoms)
    batches = [ordered[i:i + k] for i in range(0, len(ordered), k)]
    per = [None] * len(batches)

    def _job(bi, batch):
        xs = [r[0] for _, r in batch]; ys = [r[1] for _, r in batch]
        x1s = [r[0] + r[2] for _, r in batch]; y1s = [r[1] + r[3] for _, r in batch]
        mx = int(pad * IW); my = int(pad * IH)
        cx0 = max(0, min(xs) - mx); cy0 = max(0, min(ys) - my)
        cx1 = min(IW, max(x1s) + mx); cy1 = min(IH, max(y1s) + my)
        crop = img[cy0:cy1, cx0:cx1]
        ch, cw = crop.shape[:2]
        local = [(li, (r[0] - cx0, r[1] - cy0, r[2], r[3])) for li, (_, r) in enumerate(batch)]
        marked = pipeline.som_mark(crop, local)
        res = extract.ground(_b64(marked), len(batch), focus_region=None, assign=False,
                             reasoning_effort=reasoning_effort, **kwargs)
        out = []
        for e in res.get("elements") or []:
            if not e.get("box") or len(e["box"]) != 4:
                continue
            li = e.get("region")
            if not isinstance(li, int) or not (0 <= li < len(batch)):
                continue
            oid, arect = batch[li]
            by0, bx0, by1, bx1 = e["box"]                 # crop 0~1000
            fb = [int((cy0 + by0 / 1000 * ch) / IH * 1000), int((cx0 + bx0 / 1000 * cw) / IW * 1000),
                  int((cy0 + by1 / 1000 * ch) / IH * 1000), int((cx0 + bx1 / 1000 * cw) / IW * 1000)]
            clamped = _clamp_box_to_region(fb, _atom_box_1000(arect, (IH, IW)))
            if not clamped:
                continue
            e = dict(e); e["region"] = oid; e["box"] = clamped
            out.append(e)
        return bi, {"elements": out, "_meta": res.get("_meta"), "n": len(out)}

    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        for fut in as_completed([ex.submit(_job, bi, b) for bi, b in enumerate(batches)]):
            bi, one = fut.result()
            per[bi] = one

    all_els, cost, cached = [], 0.0, 0
    for one in per:
        all_els.extend(one["elements"])
        u = (one.get("_meta") or {}).get("usage") or {}
        cost += float(u.get("cost") or 0)
        cached += (u.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
    all_els = dedup_nested(all_els, {j: r[2] * r[3] for j, r in atoms})
    els = extract.assign_keys(all_els)
    return {"elements": els, "_meta": {
        "mode": "batched", "k": k, "n_calls": len(batches), "merged_count": len(els),
        "cost_sum": round(cost, 6), "cached_tokens_sum": cached, "effort": reasoning_effort}}


def geo_partition(atoms, cap=12, gap_mult=1.8):
    """행 밴딩 기하 분할(LLM 없음, 결정적): 읽기순으로 행을 쌓다가 cap 초과 또는 큰 세로 간격에서 끊음.
    행 중간은 절대 안 자름 → 라벨+입력칸이 다른 블록으로 안 쪼개짐. → 블록 리스트 [[id,...], ...]."""
    if not atoms:
        return []
    hs = sorted(r[3] for _, r in atoms)
    band = max(8, hs[len(hs) // 2])
    ordered = sorted(atoms, key=lambda a: (int((a[1][1] + a[1][3] / 2) / band), a[1][0]))
    rows, cur, ri = [], [], None                          # 같은 행 밴드끼리 행으로 묶기
    for j, r in ordered:
        k = int((r[1] + r[3] / 2) / band)
        if ri is None or k == ri:
            cur.append((j, r)); ri = k
        else:
            rows.append(cur); cur = [(j, r)]; ri = k
    if cur:
        rows.append(cur)
    blocks, blk, prev_bot = [], [], None                  # 행을 cap/간격 기준으로 블록에 담기
    for row in rows:
        top = min(r[1] for _, r in row); bot = max(r[1] + r[3] for _, r in row)
        gap = (top - prev_bot) if prev_bot is not None else 0
        if blk and (len(blk) + len(row) > cap or gap > band * gap_mult):
            blocks.append([j for j, _ in blk]); blk = []
        blk.extend(row); prev_bot = bot
    if blk:
        blocks.append([j for j, _ in blk])
    return blocks


def ground_blocks(img, atoms, blocks, reasoning_effort="low", workers=4, **kwargs):
    """LLM이 나눈 '큰 위치 블록'별로: 블록 atom 전부 하이라이트(전체 페이지) → ground(focus_ids).
    콜 수 = 블록 수(원자 개수 아님). box는 페이지 좌표, 각 요소를 자기 region atom 안으로 clamp."""
    IH, IW = img.shape[:2]
    box1000 = {j: _atom_box_1000(r, (IH, IW)) for j, r in atoms}
    per = [None] * len(blocks)

    def _job(bi, block):
        marked = mark_focus_multi(img, atoms, set(block))
        res = extract.ground(_b64(marked), len(atoms), focus_ids=list(block), assign=False,
                             reasoning_effort=reasoning_effort, **kwargs)
        out = []
        for e in res.get("elements") or []:
            if not e.get("box") or len(e["box"]) != 4:
                continue
            rb = box1000.get(e.get("region"))
            if rb is None:                                     # 블록 밖(어두운 영역)에 매긴 요소 버림
                continue
            cl = _clamp_box_to_region(e["box"], rb)
            if not cl:
                continue
            e = dict(e); e["box"] = cl
            out.append(e)
        return bi, {"elements": out, "_meta": res.get("_meta"), "n": len(out)}

    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        for fut in as_completed([ex.submit(_job, bi, b) for bi, b in enumerate(blocks)]):
            bi, one = fut.result()
            per[bi] = one

    all_els, cost, cached, n_failed = [], 0.0, 0, 0
    for one in per:
        all_els.extend(one["elements"])
        if one is None or one.get("_meta") is None:      # ground() 최종 실패경로(_meta 없음) = 콜 실패
            n_failed += 1
            continue
        u = (one.get("_meta") or {}).get("usage") or {}
        cost += float(u.get("cost") or 0)
        cached += (u.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
    all_els = dedup_nested(all_els, {j: r[2] * r[3] for j, r in atoms})
    els = extract.assign_keys(all_els)
    return {"elements": els, "_meta": {
        "mode": "blocks", "n_blocks": len(blocks), "n_calls": len(blocks),
        "n_failed": n_failed, "merged_count": len(els), "cost_sum": round(cost, 6),
        "cached_tokens_sum": cached, "effort": reasoning_effort}}


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
