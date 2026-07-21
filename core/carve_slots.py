"""일반화 carve (실험) — '영역에서 슬롯을 이미지로 먼저 검출 → 요소를 배정, box는 포인터'.

체크박스(_cb_assign)에서 이미 검증된 detect-in-region → assign-nearest 패턴을 다른 타입으로 확장.
씨앗(LLM box)은 '어느 슬롯이냐' 이산 선택에만 쓴다 → 회차마다 box가 흔들려도 같은 슬롯 → 결과 불변.

carve.py는 검증 전까지 미변경. 각 증분은 offline 하네스(수렴)+뷰어(정답)로 검증 후 통합.

증분1: radio. 영역 OCR 1회로 보기글자 위치를 잡고 option 텍스트로 매칭 → mark_pad.
       (기존 fit_radio는 요소마다 box-ROI로 OCR → 픽셀이 회차마다 달라 _ocr_word↔_fit_ink flip)
"""
from __future__ import annotations
import difflib
from collections import defaultdict

from . import carve


def _region_tokens(gray, rect):
    """영역 rect(x,y,w,h)를 OCR 1회 → 토큰 [(x0,y0,x1,y1,text)]. 인접 최대 3개 결합 후보도 포함
    (OCR이 '오전'/'오후'를 쪼개거나 붙일 수 있어서). box와 무관하니 회차마다 동일."""
    x, y, w, h = (int(v) for v in rect)
    res = carve._easyread(gray, x, y, x + w, y + h, detail=1, paragraph=False, width_ths=0.15)
    toks = []
    for bb, t, conf in res:
        t = t.strip()
        if not t:
            continue
        xs = [p[0] for p in bb]; ys = [p[1] for p in bb]
        toks.append((min(xs), min(ys), max(xs), max(ys), t))
    toks.sort()
    cands = list(toks)
    n = len(toks)
    for i in range(n):                              # 인접 토큰 결합 (한 줄에서만)
        x0, y0, x1, y1, txt = toks[i]
        cy = (y0 + y1) / 2; hh = y1 - y0
        for j in range(i + 1, min(i + 3, n)):
            nx0, ny0, nx1, ny1, nt = toks[j]
            if abs((ny0 + ny1) / 2 - cy) > hh:      # 다른 줄이면 중단
                break
            x1 = nx1; y0 = min(y0, ny0); y1 = max(y1, ny1); txt = txt + nt
            cands.append((x0, y0, x1, y1, txt))
    return cands


def place_radios(gray, rectof, items, min_ratio=0.5):
    """items 중 radio를 영역 OCR로 재배치 → {index: (x,y,w,h)}.
    각 radio의 option 텍스트를 영역 토큰과 퍼지매칭 → 글자box + mark_pad.
    같은 option이 여러 개면 box 중심에 가까운 후보 선택(이 때만 box 사용)."""
    LR = {"L", "R", "좌", "우"}                               # L/R쌍은 pipeline lr_pair가 처리
    tok_cache = {}
    out = {}
    byreg = defaultdict(list)
    for i, it in enumerate(items):
        if (it.get("type") == "radio" and it.get("option")
                and str(it["option"]) not in LR and it.get("rect")):
            byreg[it["region"]].append(i)
    for region, idxs in byreg.items():
        rect = rectof.get(region)
        if not rect:
            continue
        if region not in tok_cache:
            tok_cache[region] = _region_tokens(gray, rect)
        toks = tok_cache[region]
        for i in idxs:
            opt = carve._ocr_core(str(items[i]["option"]).replace(" ", ""))
            if not opt:
                continue
            bcx = items[i]["rect"][0] + items[i]["rect"][2] / 2
            # 후보 점수: 유사도 높은 것 → 가장 타이트한 토큰(길이차 작음) → box에 가까운 것.
            #  (병합 토큰 '(amlpm)'보다 'am' 개별 토큰을 선호. 개별이 없으면 병합을 뒤에서 분할)
            best = None; bestscore = None
            for x0, y0, x1, y1, t in toks:
                core = carve._ocr_core(t)
                r = difflib.SequenceMatcher(None, opt, core).ratio()
                if r < min_ratio:
                    continue
                score = (round(r, 3), -abs(len(core) - len(opt)), -abs((x0 + x1) / 2 - bcx))
                if bestscore is None or score > bestscore:
                    bestscore = score; best = (x0, y0, x1, y1, t, core)
            if not best:
                continue
            x0, y0, x1, y1, t, core = best
            if len(core) > len(opt):                             # 병합 토큰(남여·amlpm 등) → opt 위치로 비례 분할
                sm = difflib.SequenceMatcher(None, opt, t.replace(" ", ""))
                blocks = [b for b in sm.get_matching_blocks() if b.size > 0]
                if blocks:
                    raw = t.replace(" ", ""); cw = (x1 - x0) / max(1, len(raw))
                    j0 = blocks[0].b; j1 = blocks[-1].b + blocks[-1].size
                    x0, x1 = x0 + cw * j0, x0 + cw * j1
            tight = carve.fit_ink(gray, (int(x0), int(y0), int(x1 - x0), int(y1 - y0)), pad=1, thr=1)
            out[i] = carve.mark_pad(tight)                        # OCR bbox 여백 제거 후 여백 통일
    return out
