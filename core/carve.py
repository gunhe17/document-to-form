"""carve — CV/OCR 위치 도구함. LLM이 '무엇'(타입)을 주면, '어디'(정확한 좌표)를 여기서 잡는다.

진입점: place(gray, box, type, ..., bounds) — 그라운딩 파이프라인(pipeline.py)용. LLM box를 타입 규칙으로 정밀화.
  · checkbox → □스냅(_snap_mark→pipeline _cb_assign)
  · radio → fit_radio(OCR→fit_ink→mark_pad); L/R만 pipeline _radio_pair(letter_runs→mark_pad)
  · 단위입력(년/월/시…) → OCR앵커(ocr_anchor) · 그외 → refine_blank
  · pipeline 후처리: fit_bounded·fill_cell·fit_placeholder(○○○)·carve_inline(날짜행)·merge_unit_fields·행높이통일
  · bounds(SoM region)로 OCR 탐색을 배정 영역 안으로 제한(이웃 오앵커 방지).

OCR: 단위글자·체크박스 라벨은 EasyOCR(작은 한글에 강함, 자동 업스케일). 실패 시 CV 폴백.
좌표계: 모든 rect = 픽셀 (x, y, w, h). INK=회색 임계.
"""
import numpy as np
INK = 110

# ── LLM box 정밀화: 순수 빈칸으로 축소 ──
def refine_blank(gray, rect, pad=2, floor=0.14):
    """LLM 박스 안에서 테두리 잉크(괄호·단위 '급/년/월')를 제외한 최대 빈칸 span으로 축소.
    잉크가 거의 없으면(빈 셀) 원본 유지 → 안전. 가로만 정밀화, 세로는 유지."""
    x, y, w, h = (int(v) for v in rect)
    if w < 8 or h < 4: return (x, y, w, h)
    col = (gray[y:y+h, x:x+w] < INK).sum(0)          # 열별 잉크량
    ink = col > max(1, h*0.12)                        # 잉크 열(단위/괄호 획)
    if ink.sum() < 2: return (x, y, w, h)             # 거의 빈칸 → 전체가 입력
    gaps = []; s = None                               # 빈칸(gap) run
    for i, on in enumerate(list(ink)+[True]):
        if not on: s = i if s is None else s
        else:
            if s is not None: gaps.append((s, i)); s = None
    if not gaps: return (x, y, w, h)
    g0, g1 = max(gaps, key=lambda g: g[1]-g[0])       # 최대 빈칸
    if g1-g0 < w*floor: return (x, y, w, h)           # 유의미 빈칸 없음 → 원본
    return (x+g0+pad, y, max(4, g1-g0-2*pad), h)

# ── CV 프리미티브 (combined2에서 추출) ──
def text_bands(gray, x, y, w, h):
    """글자 줄 밴드 (b0,b1) — 가로 꽉 찬 테두리행 제외. y 상대좌표.
    gap 임계 3행: 폭 전체 합산이라 인접 줄 사이 진짜 여백만 3행 연속 0에 가까워도 뚜렷이 갈리고,
    글자 내부 획 사이 틈은 폭 전체를 다 비우지 않아 오분리 안 됨(줄간격 촘촘한 서식에서 확인)."""
    rs = (gray[y:y+h, x:x+w] < INK).sum(1)
    txt = (rs > 3) & (rs < 0.7*w)
    out = []; s = None; g = 0
    for i, t in enumerate(list(txt)+[False]):
        if t: s = i if s is None else s; g = 0
        else:
            if s is not None:
                g += 1
                if g >= 3:
                    if i-g-s >= 6: out.append((s, i-g))
                    s = None; g = 0
    if s is not None and len(txt)-g-s >= 6:   # 잉크가 rect 끝까지 이어져 트레일링 gap이 안 생기는 경우도 반영
        out.append((s, len(txt)-g))
    return out

def merged_runs(gray, x, by0, by1, w, mgap=8, cbs=()):
    """글자 런을 mgap 미만 간격으로 병합 → (rx0,rx1). 세로 테두리선·□ 제외. x 상대."""
    H = by1-by0; cnt = (gray[by0:by1, x:x+w] < INK).sum(0)
    glyph = (cnt > 0) & (cnt <= 0.7*H)
    for (bx, by, bw, bh) in cbs:
        if by < by1 and by+bh > by0: glyph[max(0, bx-x-2):bx-x+bw+2] = False
    gi = np.where(glyph)[0]
    if len(gi) == 0: return []
    runs = [[gi[0], gi[0]]]
    for c in gi[1:]:
        if c-runs[-1][1] <= mgap: runs[-1][1] = c
        else: runs.append([c, c])
    return [(a, b+1) for a, b in runs]

def find_cb(gray, x, y, w, h):
    """□ = 상·하 수평선 + 좌·우 중 1변 + 속 빔. 절대좌표 리스트 (읽기순 정렬)."""
    import cv2
    sub = (gray[y:y+h, x:x+w] < INK).astype(np.uint8)*255
    cnts, _ = cv2.findContours(sub, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE); out = []
    for cnt in cnts:
        bx, by, bw, bh = cv2.boundingRect(cnt)
        if not (12 <= bw <= 40 and 12 <= bh <= 40 and 0.6 <= bw/bh <= 1.6): continue
        reg = (gray[y+by:y+by+bh, x+bx:x+bx+bw] < INK); t = max(2, bw//8)
        top = (reg[:t, :].sum(0) > 0).mean(); bot = (reg[-t:, :].sum(0) > 0).mean()
        lft = (reg[:, :t].sum(1) > 0).mean(); rgt = (reg[:, -t:].sum(1) > 0).mean()
        inner = reg[int(bh*0.28):int(bh*0.72), int(bw*0.28):int(bw*0.72)]
        # □ = 속 비고(핵심) + 4변 모두 잉크 존재 + 테두리 총량 충분 + 2변 이상 뚜렷.
        # 흐린 인쇄 □(한두 변만 진함)도 잡는다. 오검출(비-□)은 pipeline이 'box 안'에서만 스냅해 걸러냄.
        edges = (top, bot, lft, rgt)
        if (inner.mean() < 0.12 and min(edges) > 0.1
                and sum(edges) > 2.2 and sum(e > 0.75 for e in edges) >= 2):
            out.append((x+bx, y+by, bw, bh))
    out = sorted(set(out), key=lambda b: (b[1]//20, b[0])); dd = []
    for b in out:
        if not any(abs(b[0]-d[0]) < 8 and abs(b[1]-d[1]) < 8 for d in dd): dd.append(b)
    return dd

# ── 인라인 빈칸 카브 (단위/괄호/콜론 앞뒤, 날짜행) ──
def _even(rect, n, vert=False):
    x, y, w, h = rect
    if vert: return [(x, y+i*h//n, w, h//n) for i in range(n)]
    return [(x+i*w//n, y, w//n, h) for i in range(n)]

def band_gaps(gray, x, by0, by1, w, allow_leading=False):
    """밴드 내 넓은 빈칸(gap) 후보 (g0,g1) x상대. 세로 테두리선·트레일링 제외.
    allow_leading=선행 빈칸 포함(단위칸 '___세'). 접수번호·괄호·콜론·단위 전부 '넓은 빈칸'으로 수렴."""
    H = by1-by0; cnt = (gray[by0:by1, x:x+w] < INK).sum(0)
    border = cnt > 0.7*H; glyph = (cnt > 0) & (~border)
    gi = np.where(glyph)[0]
    if len(gi) == 0: return []
    lo, hi = gi[0], gi[-1]
    lb = np.where(border[:lo])[0]
    start = (lb.max()+1 if len(lb) else 0) if allow_leading else lo
    thr = int(H*0.6); gaps = []; s = None
    for i in range(start, hi+1):        # hi까지 — 트레일링 제외
        if not glyph[i]: s = i if s is None else s
        else:
            if s is not None and i-s >= thr: gaps.append((s, i))
            s = None
    return gaps

def _enclosed(runs, g):
    """빈칸 g 오른쪽에 좁은 런(닫는괄호 ')'·단위 '세/급/년')이 바로 붙나 = 진짜 입력 신호."""
    return any(0 <= a-g[1] <= 4 and (b-a) <= 30 for a, b in runs)

def carve_inline(gray, rect, keys):
    """인라인 입력칸(단위/괄호/콜론 앞뒤 빈칸): 넓은 빈칸 N개 좌→우 배정.
    다줄 영역이면 입력이 있는 줄(unit-flanked 빈칸 최다 밴드=날짜/서명 줄)만 골라 카브."""
    x, y, w, h = rect; bands = text_bands(gray, x, y, w, h); n = len(keys)
    if not bands: return list(zip(keys, _even(rect, n)))       # 잉크 없음(순수 빈칸) → 등분
    best = None                                                # (enc수, b0, b1, runs, gaps)
    for bb0, bb1 in bands:
        runs = merged_runs(gray, x, y+bb0, y+bb1, w)
        gaps = band_gaps(gray, x, y+bb0, y+bb1, w, allow_leading=True)
        ne = sum(_enclosed(runs, g) for g in gaps)
        if best is None or ne >= best[0]: best = (ne, bb0, bb1, runs, gaps)   # 동수면 마지막(서명줄)
    _, b0, b1, runs, gaps = best
    ranked = sorted(gaps, key=lambda g: (_enclosed(runs, g), g[1]-g[0]), reverse=True)[:n]
    pick = sorted([(x+g0, g1-g0) for g0, g1 in ranked], key=lambda p: p[0])
    if len(pick) == n:
        return [(keys[i], (pick[i][0], y+b0, pick[i][1], b1-b0)) for i in range(n)]
    ev = _even((x, y+b0, w, b1-b0), n)                         # 부족 → 있는 것 + 등분
    return [(keys[i], (pick[i][0], y+b0, pick[i][1], b1-b0)) if i < len(pick) else (keys[i], ev[i]) for i in range(n)]


# ── OCR (EasyOCR, 자동 업스케일) + 타입별 place 규칙 ──
def ocr_label_left(gray, box, text, hw, min_ratio=0.5):
    """옵션 라벨 text를 EasyOCR로 찾아 그 라벨의 왼쪽 x 반환. 못 찾으면 None.
    쪼개진 글자(외출중→외/출/중)는 '시작 조각/포함' 매칭으로 흡수. 라벨 위치 = □ 앵커 기준."""
    try:
        import difflib
    except Exception:
        return None
    if not text: return None
    x, y, w, h = (int(v) for v in box); key = text.replace(" ", "")
    results = _easyread(gray, x-int(h*3), y-int(h*0.7), x+max(w, int(h*7)), y+h+int(h*0.7), detail=1, paragraph=False)
    cy = y+h/2; best = None; br = min_ratio
    for bb, t, conf in results:
        t = t.strip().replace(" ", "")
        if not t: continue
        wx = min(pt[0] for pt in bb); wcy = (min(pt[1] for pt in bb)+max(pt[1] for pt in bb))/2
        if abs(wcy-cy) > h*0.8: continue                       # 같은 줄만
        r = difflib.SequenceMatcher(None, key, t).ratio()
        if key.startswith(t) or t in key or key in t or t.startswith(key[:2]): r = max(r, 0.6)  # 시작조각/포함
        if r >= br: br = r; best = int(wx)
    return best

def _ocr_core(s):
    """OCR/라벨 비교용 — 공백·괄호 제거."""
    return s.replace(" ", "").translate(str.maketrans("", "", "()（）"))

def ocr_word_box(gray, box, text, hw, bounds=None, min_ratio=0.5, expand=3.0, allow_wrapped=False):
    """텍스트가 있는 LLM box를 넓게 OCR 재검색해 실제 글자 전체의 bbox로 스냅. 못 찾으면 None(폴백).
    radio·signature 공통. 인접 토큰은 최대 3개까지 이어붙여 후보로 본다(서명 문구 등).
    allow_wrapped=True(signature): OCR이 `(서명 또는 인)`처럼 괄호만 더 붙여도 라벨과 같으면 전체 bbox.
    radio는 allow_wrapped=False — amlpm 같은 긴 결합 토큰은 스킵."""
    import difflib
    if not text: return None
    x, y, w, h = (int(v) for v in box); key = text.replace(" ", "")
    ex = int(h * expand)
    X0, X1 = x - ex, x + max(w, ex) + ex; Y0, Y1 = y - int(h * 0.8), y + h + int(h * 0.8)
    X0, Y0, X1, Y1 = _clamp_roi(X0, Y0, X1, Y1, bounds)
    results = _easyread(gray, X0, Y0, X1, Y1, detail=1, paragraph=False, width_ths=0.15)
    cy = y + h / 2; toks = []
    for bb, t, conf in results:
        t = t.strip().replace(" ", "")
        if not t: continue
        wcy = (min(pt[1] for pt in bb) + max(pt[1] for pt in bb)) / 2
        if abs(wcy - cy) > h * 1.2: continue
        xs = [pt[0] for pt in bb]; ys = [pt[1] for pt in bb]
        toks.append((min(xs), max(xs), min(ys), max(ys), t))
    toks.sort()
    cands = list(toks)
    n = len(toks)
    for i in range(n):
        x0, x1, y0, y1, txt = toks[i]
        for j in range(i + 1, min(i + 3, n)):
            if toks[j][0] - x1 >= h * 1.5: break
            x1 = toks[j][1]; y0 = min(y0, toks[j][2]); y1 = max(y1, toks[j][3]); txt = txt + toks[j][4]
            cands.append((x0, x1, y0, y1, txt))
    if allow_wrapped:
        core_key = _ocr_core(key)
        wrapped = [(x0, y0, x1 - x0, y1 - y0) for x0, x1, y0, y1, t in cands
                   if core_key and _ocr_core(t) == core_key]
        if wrapped:
            return max(wrapped, key=lambda b: b[2])  # 가장 넓은 = 전체 인쇄 문구
    best = None; br = min_ratio
    for x0, x1, y0, y1, t in cands:
        if len(t) > len(key):
            continue  # amlpm 같은 결합 토큰 — 균일분할 안 함
        r = difflib.SequenceMatcher(None, key, t).ratio()
        if t == key or t in key or key.startswith(t):
            r = max(r, 0.6)
        if r >= br:
            br = r; best = (x0, y0, x1 - x0, y1 - y0)
    return best


_EASY_READER = None
def _easyocr():
    """EasyOCR 리더 (한글+영문) 지연 초기화 싱글턴 — 작은 한글 글자에 tesseract보다 정확."""
    global _EASY_READER
    if _EASY_READER is None:
        import easyocr
        _EASY_READER = easyocr.Reader(["ko", "en"], gpu=False, verbose=False)
    return _EASY_READER

def _clamp_roi(X0, Y0, X1, Y1, bounds, pad=3):
    """OCR 탐색 ROI를 배정 region(bounds=px rect) 안(+pad)으로 제한 — 이웃 영역 글자 오앵커 방지."""
    if not bounds: return X0, Y0, X1, Y1
    bx, by, bw, bh = (int(v) for v in bounds)
    return max(X0, bx-pad), max(Y0, by-pad), min(X1, bx+bw+pad), min(Y1, by+bh+pad)

def _easyread(gray, X0, Y0, X1, Y1, **kw):
    """ROI를 잘라 작으면 업스케일(INTER_CUBIC) 후 EasyOCR → bbox를 '원본 이미지 좌표'로 되돌려 반환.
    작은 글자(단위·단일문자·라벨)를 확대해 인식률↑ (년→녀 오독·단일글자 미검출 감소). 파이프라인 전 OCR 공통."""
    import cv2
    X0, Y0 = max(0, int(X0)), max(0, int(Y0)); X1, Y1 = min(gray.shape[1], int(X1)), min(gray.shape[0], int(Y1))
    roi = gray[Y0:Y1, X0:X1]
    if roi.size == 0 or min(roi.shape[:2]) < 3: return []
    m = min(roi.shape[:2]); k = 4 if m < 45 else (2 if m < 110 else 1)   # 작을수록 크게 확대
    up = cv2.resize(roi, (roi.shape[1]*k, roi.shape[0]*k), interpolation=cv2.INTER_CUBIC) if k > 1 else roi
    try:
        res = _easyocr().readtext(up, **kw)
    except Exception:
        return []
    return [([(X0+p[0]/k, Y0+p[1]/k) for p in bb], t, c) for bb, t, c in res]   # 원본 좌표

def _snap_mark(gray, box, option, hw):
    """checkbox_group/radio: OCR로 옵션 라벨 찾아 그 왼쪽 □에 앵커(충돌 없음). 실패 시 기하 폴백."""
    x, y, w, h = (int(v) for v in box); cy = y+h/2
    lx = ocr_label_left(gray, (x, y, w, h), option, hw)         # 라벨 텍스트 시작 x (OCR)
    if lx is not None:
        rx0 = max(0, lx-int(h*3)); ry0 = max(0, y-int(h*0.6))
        bs = find_cb(gray, rx0, ry0, min(hw[1]-rx0, int(h*4)), min(hw[0]-ry0, h+2*int(h*0.6)))
        if bs: return min(bs, key=lambda b: (b[0]+b[2]/2-lx)**2 + (b[1]+b[3]/2-cy)**2)   # 라벨에 가장 가까운 □
    vy = int(h*0.7); ex = int(h*1.6)                           # 폴백: 기하(같은행·좌측)
    rx0, ry0 = max(0, x-ex), max(0, y-vy)
    boxes = find_cb(gray, rx0, ry0, min(hw[1]-rx0, w+2*ex), min(hw[0]-ry0, h+2*vy))
    if not boxes: return (x, y, w, h)
    b = min(boxes, key=lambda b: (b[0]-x)**2 + ((b[1]+b[3]/2-cy)**2)*4)
    return b if abs((b[1]+b[3]/2)-cy) <= h*1.3 else (x, y, w, h)

def _refine(gray, box, option, hw):
    """text/number/date/time/email/phone: 테두리 잉크(괄호·단위) 제외한 순수 빈칸으로 축소."""
    return refine_blank(gray, box)

def ink_frac(gray, rect):
    """rect 안 잉크 픽셀 비율. 빈칸=낮음(~0), 인쇄 글자 위=높음. 단위글자-오배치 판별용."""
    x, y, w, h = (int(v) for v in rect)
    if w < 3 or h < 3: return 0.0
    return float((gray[y:y+h, x:x+w] < INK).mean())

def fit_ink(gray, rect, pad=1, thr=2):
    """박스 안 글자 잉크의 bbox로 축소. 세로/가로 테두리선 제외. thr=1이면 얇은 획 가장자리 유지(라디오 글자용).
    잉크 없으면 원본."""
    x, y, w, h = (int(v) for v in rect)
    ink = gray[y:y+h, x:x+w] < INK
    if ink.sum() < 3: return (x, y, w, h)
    ink[:, ink.sum(0) > h*0.75] = False        # 세로 테두리선 제거
    ink[ink.sum(1) > w*0.75, :] = False        # 가로 테두리선 제거
    cx = np.where(ink.sum(0) >= thr)[0]; ry = np.where(ink.sum(1) >= thr)[0]   # thr=2 노이즈제외 / thr=1 획 유지
    if len(cx) < 1 or len(ry) < 1:
        ys, xs = np.where(ink)
        if len(xs) < 3: return (x, y, w, h)
        cx, ry = xs, ys
    x0, x1, y0, y1 = int(cx.min()), int(cx.max()), int(ry.min()), int(ry.max())
    return (x+x0-pad, y+y0-pad, x1-x0+2*pad, y1-y0+2*pad)

def _fit_word(gray, box, option, hw):
    """radio: 보기 텍스트에 fit (순수 CV). 텍스트밴드(한 줄) → 박스 중심에 가까운 잉크 런 →
    런이 넓으면(am/pm 결합) 박스 위치로 절반 분할 → 그 글자에 fit. 표 테두리·다른 행·옆 보기 배제."""
    x, y, w, h = (int(v) for v in box)
    bands = text_bands(gray, x, y, w, h)
    b0, b1 = min(bands, key=lambda bb: abs((bb[0]+bb[1])/2 - h/2)) if bands else (0, h); bh = b1-b0
    runs = merged_runs(gray, x, y+b0, y+b1, w)      # 테두리(꽉 찬 열) 제외한 글자 런
    if not runs: return fit_ink(gray, (x, y+b0, w, bh))
    cx = w/2
    rx0, rx1 = min(runs, key=lambda r: abs((r[0]+r[1])/2 - cx))       # 박스 중심에 가까운 런
    if (rx1-rx0) > bh*2.2:                                            # 넓은 런(여러 보기 결합) → 박스 쪽 절반
        mid = (rx0+rx1)//2
        if cx < mid: rx1 = mid
        else: rx0 = mid
    return fit_ink(gray, (x+rx0, y+b0, max(4, rx1-rx0), bh))

def letter_runs(gray, x0, y0, x1, y1):
    """[x0,x1]×[y0,y1]에서 '글자 연결성분'의 x-구간들(좌→우). 얇은 괄호(폭↓)·낮은 콤마(세로↓)는 제외.
    '( L , R )'의 L·R처럼 문장부호에 붙은 단일글자를 CC로 깨끗이 분리 (merged_runs로는 못 가름)."""
    import cv2
    hh = max(1, y1-y0); sub = (gray[y0:y1, x0:x1] < INK).astype(np.uint8)
    if sub.size == 0: return []
    n, _, st, _ = cv2.connectedComponentsWithStats(sub, connectivity=8)
    out = [(x0+int(st[i][0]), x0+int(st[i][0])+int(st[i][2])) for i in range(1, n)
           if st[i][2] >= 0.22*hh and st[i][3] >= 0.45*hh]   # 폭 충분(괄호 아님)+세로 충분(콤마 아님)
    return sorted(out)

def _keep(gray, box, option, hw):
    return tuple(int(v) for v in box)

PLACE_RULES = {                                          # form type → 수정기(corrector)
    # carve 동작 타입: □ 스냅(checkbox), 빈칸 축소, 단위 앵커 (radio는 place에서 _fit_word로 별도 처리)
    "checkbox_group": _snap_mark,
    "text": _refine, "textarea": _refine, "number": _refine, "date": _refine, "time": _refine,
    # carve 불필요(값 전체 셀·문구·사진란): LLM box 그대로
    "email": _keep, "phone": _keep, "consent": _keep, "signature": _keep, "select": _keep, "image": _keep,
}

# 단독 입력필드가 될 수 없는 '순수 분류단위'만 병합대상 — 시/세/년/월/일은 진짜 값 입력이라 제외.
DROPPABLE_UNITS = {"급"}  # 실사용 검증상 텍스트 병합에 필요한 분류단위는 급뿐 (호/종/류… speculative 제거)
def snap_to_cell(rect, cells, min_iou=0.55, pad=2):
    """입력 박스가 표 셀 하나를 대략 덮으면(IoU≥임계) 그 셀 격자에 스냅(안쪽 pad). 아니면 원본.
    LLM 좌표의 셀 단위 어긋남을 CV 격자로 정렬. 서브셀 빈칸(나이·날짜슬롯)은 IoU 낮아 스냅 안 됨."""
    x, y, w, h = (int(v) for v in rect); a = w*h
    best = None; bi = min_iou
    for cx, cy, cw, ch in cells:
        ix0, iy0 = max(x, cx), max(y, cy); ix1, iy1 = min(x+w, cx+cw), min(y+h, cy+ch)
        inter = max(0, ix1-ix0)*max(0, iy1-iy0)
        if inter == 0: continue
        iou = inter/(a + cw*ch - inter)
        if iou > bi: bi = iou; best = (int(cx), int(cy), int(cw), int(ch))
    if best is None: return (x, y, w, h)
    cx, cy, cw, ch = best
    return (cx+pad, cy+pad, max(4, cw-2*pad), max(4, ch-2*pad))

def fit_bounded(gray, rect, cells, atom_rect=None, pad=3):
    """상하좌우 자동 맞춤: 포함 셀의 '박스가 놓인 텍스트 라인'(상·하)과 그 라인의 좌우 인접 잉크
    (라벨·단위·괄호) 사이 빈칸에 일정 여백으로 박스를 맞춘다. 단위 입력(___세·(_급))처럼 둘러싸인 값칸용.
    atom_rect: 괘선 셀이 안 걸리면(band) 이걸 폴백 컨테이너로 — band는 진짜 테두리가 없어 단일행이라도
    셀처럼 '행 전체'를 쓰지 않고 라벨 글자밴드 높이 그대로 씀(밑줄 위 여백 과다 방지)."""
    x, y, w, h = (int(v) for v in rect); ccx, ccy = x+w/2, y+h/2
    cell = None                                              # 가장 작은 포함 셀
    for cx, cy, cw, ch in cells:
        if cx-2 <= ccx <= cx+cw+2 and cy-2 <= ccy <= cy+ch+2:
            if cell is None or cw*ch < cell[2]*cell[3]: cell = (cx, cy, cw, ch)
    is_band = cell is None and atom_rect is not None
    if cell is None and not is_band: return (x, y, w, h)
    clx, cty, cw, ch = cell if cell else atom_rect
    bands = text_bands(gray, clx, cty, cw, ch)              # 세로: 박스가 놓인 '행' 높이
    if not bands and is_band: return (x, y, w, h)           # band인데 라벨조차 없음 → 기준 없이 원본 유지
    # 단일행 셀이라도, 이 요소의 (refine 후) box 높이가 이미 셀 높이 대부분을 차지하면 그 칸 자체가
    # 큰 작성란(예 지원동기)이라는 뜻 → 셀 전체 사용. box가 셀 높이의 절반도 안 되면(예: 서명 넣을 자리 때문에
    # 유독 큰 셀에 한 줄짜리 날짜·시각 값이 얹힌 경우) 셀 전체로 늘리지 않고 그 텍스트 줄에만 맞춘다.
    if not bands and not is_band: top, bot = cty, cty+ch; vpad = pad
    elif len(bands) <= 1 and not is_band and h >= ch*0.5:   # 단일행 + 박스가 이미 셀 대부분 → 큰 작성란
        top, bot = cty, cty+ch; vpad = pad
    else:                                                   # band, 다중행, 또는 셀 대비 유독 작은 단일값 → 그 텍스트 줄에 딱 맞춤
        ba = [(cty+b0, cty+b1) for b0, b1 in bands]          # (중점X: 마지막 줄이 셀 바닥까지 늘어나는 것 방지)
        top, bot = min(ba, key=lambda b: abs((b[0]+b[1])/2 - ccy)); vpad = 0   # 밴드=텍스트, 세로 깎지 않음
    # 세로 구분선(진짜 칼럼 경계·사이드바 등) 감지 — band가 옆 칼럼까지 포함하는 컨테이너라도
    # 그 선 밖은 아예 스캔하지 않는다(옆 칼럼 글자를 이 줄의 값으로 오인하는 것 방지).
    colcnt = (gray[top:bot, clx:clx+cw] < INK).sum(0)
    vb = np.where(colcnt > 0.7*(bot-top))[0]
    lb = max((clx+int(b) for b in vb if clx+b < ccx), default=clx)
    rb = min((clx+int(b) for b in vb if clx+b > ccx), default=clx+cw)
    left, right = lb, rb                                    # 가로: 라인 잉크 런의 좌우 인접
    min_w = max(4, (bot-top)*0.3)                           # 원본 박스 '안'에서만: 점(.) 같은 잡음은 경계로 안 침
    ix0, ix1 = x, x+w                                       # (밖에 있는 얇은 표시는 옆 칸과의 진짜 경계일 수 있어 그대로 존중)
    found_l = found_r = False
    for r0, r1 in merged_runs(gray, lb, top, bot, rb-lb):
        a0, a1 = lb+r0, lb+r1
        if r1-r0 < min_w and ix0 <= a0 and a1 <= ix1: continue   # 예: '날짜: __.__.__.'의 내부 마침표만 무시
        mid = lb+(r0+r1)/2
        if mid < ccx: left = max(left, lb+r1); found_l = True   # 왼쪽 라벨/괄호 뒤
        elif mid > ccx: right = min(right, lb+r0); found_r = True; break   # 오른쪽 단위 앞
    # band는 atom_rect(한 줄 전체 폭, 때로 옆 칸·사이드바까지 포함)가 컨테이너라 이 방향에 인접 잉크가
    # 아예 없으면(예: '주소 : ' 뒤가 그 줄 끝까지 진짜 빈칸) 컨테이너 끝까지 뻗어버린다 — 그 끝이 실제
    # 옆 칼럼(구분선 밖)일 수 있다. LLM 원본 box 자체가 이미 그렇게 크게 와서 ccx도 못 믿을 수 있으니,
    # 못 찾은 쪽은 ccx가 아니라 반대쪽에서 실제로 찾은 경계 기준으로 상식적 상한을 둔다.
    if is_band:
        # 오른쪽은 found_r이어도 항상 상한 적용 — 사이드바 세로캡션처럼 진짜 글자처럼 보이는 잡음이
        # 옆 칼럼에서 '경계'로 잡히는 경우가 있어(구분선이 이 행엔 안 지나가 위 검출로도 못 거름),
        # found_r=True여도 안심할 수 없다. 왼쪽은 found_l이면(라벨을 신뢰할 수 있으면) 그대로 둔다.
        cap = int(12*(bot-top))
        right = min(right, (left if found_l else ccx)+cap)
        if not found_l: left = max(left, (right if found_r else ccx)-cap)
    if right-left < 8:                                      # 가로 인접 잉크 없음 → 가로는 원본 유지, 세로만 반영
        left, right = x-pad, x+w+pad
    return (left+pad, top+vpad, right-left-2*pad, max(6, bot-top-2*vpad))

def fill_cell(rect, cells, pad=4):
    """여러 줄 쓰기칸(textarea)을 포함 셀 전체에 채운다(안쪽 pad). 한 줄로 축소하지 않음. 셀 없으면 원본."""
    x, y, w, h = (int(v) for v in rect); cx, cy = x+w/2, y+h/2
    cont = [c for c in cells if c[0]-2 <= cx <= c[0]+c[2]+2 and c[1]-2 <= cy <= c[1]+c[3]+2]
    if not cont: return (x, y, w, h)
    cX, cY, cW, cH = min(cont, key=lambda c: c[2]*c[3])
    return (int(cX+pad), int(cY+pad), int(max(6, cW-2*pad)), int(max(6, cH-2*pad)))

def fit_after_label(gray, rect, cells, pad=3):
    """개방형 빈칸('바닥의 형태 : ___'처럼 밑줄 없이 콜론 뒤 여백)용 재배치. box가 라벨 글자 위에
    얹혔을 때: 같은 줄 잉크런을 단어간격으로 묶어 '라벨 덩어리'를 만들고, 그 오른쪽 끝(콜론 뒤)~
    다음 잉크(또는 셀 우측)를 빈칸으로 잡아 이동. 빈칸이 충분치 않으면 None(→ fit_bounded 폴백)."""
    x, y, w, h = (int(v) for v in rect); ccx, ccy = x+w/2, y+h/2
    cell = None
    for cx, cy, cw, ch in cells:
        if cx-2 <= ccx <= cx+cw+2 and cy-2 <= ccy <= cy+ch+2:
            if cell is None or cw*ch < cell[2]*cell[3]: cell = (cx, cy, cw, ch)
    if cell is None: return None
    clx, cty, cw, ch = cell
    bands = text_bands(gray, clx, cty, cw, ch)
    if not bands: return None
    ba = [(cty+b0, cty+b1) for b0, b1 in bands]
    top, bot = min(ba, key=lambda b: abs((b[0]+b[1])/2 - ccy)); bh = bot-top
    runs = [(clx+r0, clx+r1) for r0, r1 in merged_runs(gray, clx, top, bot, cw)]
    if not runs: return None
    join = max(8, int(bh*1.2)); groups = [list(runs[0])]     # 단어간격 이내 런은 한 라벨로 결합
    for a, b in runs[1:]:
        if a-groups[-1][1] <= join: groups[-1][1] = b
        else: groups.append([a, b])
    left = [gp for gp in groups if gp[0] <= ccx+bh]           # box 중심 근처/왼쪽의 라벨 덩어리
    if not left: return None
    lab = max(left, key=lambda gp: gp[1])                     # 가장 오른쪽까지 간 라벨(콜론 포함)
    bl = lab[1]+pad
    rt = [gp[0] for gp in groups if gp[0] > bl]               # 그 오른쪽의 다음 잉크
    br = (min(rt) if rt else clx+cw) - pad
    if br-bl < max(20, bh): return None                       # 빈칸 폭 부족 → 폴백
    return (bl, top+1, br-bl, max(6, bh-2))

# 자리표시자 문자셋 — OCR가 도형(○□△)을 이렇게 읽기도 함
_PH = set("Oo0○〇°ㅇ□▢ㅁ△▲∆▽◇")
def _ph_run(s):
    """문자열 s에서 자리표시자 런 (i,j) 반환: _PH문자 2연속↑ 또는 '동일문자' 3연속↑(○○○·□□□·△△△·___ 등,
    OCR가 도형을 뭐로 읽든 반복이면 포착). 없으면 None."""
    best = None
    ci = 0                                                             # (a) _PH 문자 런 (≥2)
    while ci < len(s):
        if s[ci] in _PH:
            cj = ci
            while cj+1 < len(s) and s[cj+1] in _PH: cj += 1
            if cj-ci >= 1 and (best is None or cj-ci > best[1]-best[0]): best = (ci, cj)
            ci = cj+1
        else: ci += 1
    ci = 0                                                             # (b) 동일문자 런 (≥3)
    while ci < len(s):
        cj = ci
        while cj+1 < len(s) and s[cj+1] == s[ci]: cj += 1
        if cj-ci >= 2 and (best is None or cj-ci > best[1]-best[0]): best = (ci, cj)
        ci = cj+1
    return best

def fit_placeholder(gray, box, hw):
    """'○○○'·'□□□'·'△△△'·'000' 자리표시자 위에 덮어쓰는 입력칸(예 '○○○기관'·'담당자○○○'). box+문맥을
    EasyOCR해 자리표시자 런을 찾고, 그 x-구간(옆 한글 라벨 제외)을 실제 도형 윤곽에 스냅. 없으면 None."""
    x, y, w, h = (int(v) for v in box)
    # 가로 창을 넓게(±2h): box가 라벨('담당자') 위에 있고 표시자('○○○')가 라벨 바로 옆에 붙은 경우도
    # 잡는다(좁으면 라벨만 읽고 표시자를 놓침). 표시자 런은 _ph_run이 걸러 위치는 비례로 잡으므로 과확장 안전.
    results = _easyread(gray, x-int(h*2), y-int(h*0.4), x+w+int(h*2), y+h+int(h*0.4), detail=1, paragraph=False)
    best = None
    for bb, t, conf in results:
        s = t.strip()
        run = _ph_run(s)
        if run is None: continue
        bi, bj = run
        x0 = min(p[0] for p in bb); x1 = max(p[0] for p in bb)
        yy0 = min(p[1] for p in bb); yy1 = max(p[1] for p in bb)
        ws = [1.6 if "가" <= c <= "힣" else 1.0 for c in s]; tot = sum(ws) or 1   # 한글은 넓음(위치 보정)
        acc = [0]
        for wv in ws: acc.append(acc[-1]+wv)
        rx0 = x0+(x1-x0)*acc[bi]/tot; rx1 = x0+(x1-x0)*acc[bj+1]/tot
        cand = (int(rx0), int(yy0), int(rx1-rx0), int(yy1-yy0))
        if best is None or cand[2] > best[2]: best = cand
    if best:                                                           # 근사 구간을 실제 도형(○□△) 윤곽에 정확히 스냅
        bx, by, bw, bh = best                                          # 밴드 넉넉히(비례추정이 좁아도 모든 도형 포함) — 글자는 _shape_run이 걸러냄
        ref = _shape_run(gray, bx-int(bh*1.2), bx+bw+int(bh*1.2), by, bh, hw)
        if ref: best = (ref[0], by, ref[1]-ref[0], bh)
    return best if best and best[2] >= 8 else None

def _shape_run(gray, x0, x1, y, h, hw):
    """[x0,x1]×[y..y+h]에서 '반복 균일 도형'(○·□·△ 등, 속 빈/외곽선·정사각 근접) 윤곽들의 좌우 끝.
    라벨 글자(꽉 참·불균일)는 제외 — 도형 후보 중 크기 균일한 것만 남긴다."""
    import cv2
    X0, X1 = max(0, x0), min(hw[1], x1); Y0, Y1 = max(0, y-2), min(hw[0], y+h+2)
    sub = (gray[Y0:Y1, X0:X1] < INK).astype(np.uint8)
    if sub.size == 0: return None
    cnts, hier = cv2.findContours(sub, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hier is None: return None
    shp = []
    for i, c in enumerate(cnts):
        if hier[0][i][3] != -1: continue                               # 구멍(자식)은 건너뜀 — 바깥선만
        bx, by, bw, bh = cv2.boundingRect(c)
        if bh < 0.45*h or not (0.55 <= bw/max(1, bh) <= 1.9): continue  # 세로 충분 + 정사각~삼각 근접
        hollow = hier[0][i][2] != -1 or sub[by:by+bh, bx:bx+bw].mean() < 0.62   # 구멍 있거나 속 빔(○□), △=외곽/저밀도
        if hollow: shp.append((X0+bx, X0+bx+bw, bh))
    if not shp: return None
    mh = sorted(s[2] for s in shp)[len(shp)//2]                        # 크기 균일한 도형만(글자획·기호 배제)
    uni = sorted([s for s in shp if abs(s[2]-mh) <= 0.25*mh], key=lambda s: s[0])
    if not uni: return None
    if len(uni) <= 2: return (uni[0][0], uni[-1][1])
    cen = [(a+b)/2 for a, b, _ in uni]; gaps = [cen[i+1]-cen[i] for i in range(len(cen)-1)]
    mg = sorted(gaps)[len(gaps)//2]                                    # 등간격 연속 런만(자·☎ 등 가짜 원 배제)
    runs = [[uni[0]]]
    for i, gp in enumerate(gaps):
        if 0.5*mg <= gp <= 1.7*mg: runs[-1].append(uni[i+1])          # 간격 균일 → 같은 런
        else: runs.append([uni[i+1]])
    run = max(runs, key=len)
    return (run[0][0], run[-1][1])

def merge_unit_fields(items):
    """같은 region·행·인접 text 쌍에서 한쪽이 분류단위(급 등)이고 다른쪽이 그렇지 않으면
    같은 입력의 중복분할 → 분류단위 요소 제거, 그 단위는 살아남는 요소의 unit으로 이관
    (예: 자격증명+급 → 자격증명 요소 하나, unit='급' — '이름만' 대 '이름+급수 요구' 구분은
    unit 유무로 남긴다, 새 타입을 만들지 않음). → 제거할 index 집합.
    '시+총시간'처럼 둘 다 실제 값 입력인 경우는 시가 DROPPABLE이 아니라 안 건드림.
    items=[{region,label,type,option,unit,rect(x,y,w,h)}]."""
    from collections import defaultdict
    TEXT = {"text", "textarea", "number", "date", "time", "email", "phone"}
    by = defaultdict(list)
    for i, e in enumerate(items):
        if e["type"] in TEXT and not e.get("option") and e.get("rect"): by[e["region"]].append(i)
    drop = set()
    for r, idxs in by.items():
        idxs = sorted((i for i in idxs if i not in drop), key=lambda i: items[i]["rect"][0])
        for a, b in zip(idxs, idxs[1:]):
            ra, rb = items[a]["rect"], items[b]["rect"]
            same_row = abs((ra[1]+ra[3]/2) - (rb[1]+rb[3]/2)) < max(ra[3], rb[3])*0.7
            gap = rb[0] - (ra[0]+ra[2])
            adjacent = gap <= max(ra[3], rb[3])*2.0        # 겹침(음수 gap)도 같은 입력의 분할로 병합
            if not (same_row and adjacent): continue
            da = items[a]["label"].replace(" ", "") in DROPPABLE_UNITS or (items[a].get("unit") or "").strip() in DROPPABLE_UNITS
            db = items[b]["label"].replace(" ", "") in DROPPABLE_UNITS or (items[b].get("unit") or "").strip() in DROPPABLE_UNITS
            if db and not da:                   # 분류단위(오른쪽)를 왼쪽 빈칸으로 흡수
                items[a]["unit"] = items[a].get("unit") or items[b].get("unit") or items[b]["label"]
                drop.add(b)
            elif da and not db:
                items[b]["unit"] = items[b].get("unit") or items[a].get("unit") or items[a]["label"]
                drop.add(a)
    return drop

# 작은 폼 글자 OCR 오독 허용셋 (년→녀 등)
_UNIT_CONFUSE = {"년": {"년", "녀", "넌"}, "월": {"월"}, "일": {"일", "읽", "잌", "입"},
                 "세": {"세"}, "급": {"급"}, "시": {"시"}, "분": {"분"}, "회": {"회"}, "원": {"원"}}
VAL_MARGIN = 4  # 단위 값칸이 좌우 잉크(라벨·단위)에서 떨어지는 고정 여백 — 필드별 글자높이 대신 고정이라 모든 값칸 동일
def ocr_anchor(gray, rect, unit, hw, min_conf=0.25, bounds=None):
    """단위글자(unit)를 EasyOCR로 찾아 그 '왼쪽 빈칸' rect 반환. 못 찾으면 None → carve 폴백.
    tesseract가 못 읽던 년/월 등도 EasyOCR로 위치 확보 = '어느 글자에 붙었는지' 검증됨."""
    if not unit: return None
    x, y, w, h = (int(v) for v in rect); ex = int(h*1.6)
    X0, X1 = max(0, x-ex), min(hw[1], x+w+ex); Y0, Y1 = max(0, y-4), min(hw[0], y+h+4)
    X0, Y0, X1, Y1 = _clamp_roi(X0, Y0, X1, Y1, bounds)         # 배정 region 안으로 제한(이웃 단위글자 오앵커 방지)
    results = _easyread(gray, X0, Y0, X1, Y1, detail=1, width_ths=0.0)   # 업스케일 OCR(가로병합끔: ')월'·'시(총' 등 이웃과 안 뭉침), bbox 원본좌표
    ok = _UNIT_CONFUSE.get(unit, {unit}); gx = gcw = gyc = None
    for bb, t, conf in results:
        t = t.strip().replace(" ", "")
        if not t or conf < min_conf or not any(c in t for c in ok): continue
        # 결합 토큰(20년·'시(총' 등)이면 단위글자 위치·폭 보정, 단순 매칭이면 토큰 왼쪽
        xl = min(pt[0] for pt in bb); xr = max(pt[0] for pt in bb)
        ys0 = min(pt[1] for pt in bb); ys1 = max(pt[1] for pt in bb)
        cw = (xr-xl)/max(1, len(t))                                    # 한 글자 폭
        idx = t.find(next(c for c in ok if c in t))
        cand = xl + cw*idx if len(t) > 1 else xl                       # 단위글자 왼쪽 x
        if gx is None or abs(cand-(x+w/2)) < abs(gx-(x+w/2)):
            gx = int(cand); gcw = max(6, int(cw)); gyc = (ys0+ys1)/2   # 토큰 세로중심(ROI 좌표)
    if gx is None: return None
    # 세로: 단위글자 열의 잉크 중 '토큰 중심에 걸린 연속 런'만 (임계1로 얇은 획 유지 + 2px 간격허용, 위아래 이웃줄만 배제)
    idxr = np.where((gray[Y0:Y1, gx:min(X1, gx+gcw)] < INK).sum(1) >= 1)[0]
    if len(idxr):
        c = gyc-Y0; grp = min(np.split(idxr, np.where(np.diff(idxr) > 2)[0]+1),   # gyc는 원본좌표 → ROI 상대로
                           key=lambda gp: 0 if gp[0] <= c <= gp[-1] else min(abs(gp[0]-c), abs(gp[-1]-c)))
        gy, gh = Y0+int(grp[0]), int(grp[-1]-grp[0])+1
    else:
        gy, gh = y, h
    # 가로: 단위글자 왼쪽 빈칸. merged_runs를 '단위글자 줄'(gy~gy+gh)에만 한정 — 큰 LLM box가 아래 줄까지 덮어도 그 줄 잉크만.
    mruns = [(X0+a, X0+b) for a, b in merged_runs(gray, X0, gy, gy+max(6, gh), X1-X0)]
    ur = [r for r in mruns if r[1] > gx-2]                    # 단위가 든 런(오른쪽)
    uleft = min(r[0] for r in ur) if ur else gx
    lruns = sorted([r for r in mruns if r[1] <= uleft+2], key=lambda r: r[1])   # 단위 왼쪽 런들
    b1 = uleft; b0 = lruns[-1][1] if lruns else X0            # 단위 바로 왼쪽 빈칸(= 괄호 안·라벨 뒤)
    if b1-b0 < 6 and len(lruns) >= 2 and (lruns[-1][1]-lruns[-1][0]) < 0.6*gh:   # 단위가 얇은 런(괄호 ')')에 붙음 → 그 앞
        b1 = lruns[-1][0]; b0 = lruns[-2][1]
    b0 = max(b0, b1-int(6*gh))                                # 빈칸 상한: 중앙배치 '년'처럼 왼쪽 라벨이 멀면 과확장 방지
    bw = b1-b0
    vp = max(1, gh//8)                                        # 세로: 단위글자(년/월/일) 높이에 맞춤
    return (b0+VAL_MARGIN, gy-vp, bw-2*VAL_MARGIN, gh+2*vp) if bw >= 2*VAL_MARGIN+6 else None

MARK_PAD_PX = 4  # radio/signature 여백 — 글자 높이 비례 대신 고정 px (사방 동일)

def mark_pad(rect, fh=None, pad=MARK_PAD_PX):
    """타이트 텍스트 bbox에 고정 px 여백 — radio(원)·signature(도장) 공통.
    fh는 하위호환용(무시). 가로·세로 모두 pad px로 통일."""
    x, y, w, h = (int(v) for v in rect)
    return (x - pad, y - pad, w + 2 * pad, h + 2 * pad)

def fit_radio(gray, box, option, hw, bounds=None):
    """radio 공통: OCR(짧은 토큰만) → fit_ink → mark_pad(고정 px).
    OCR 실패 시 LLM box fit_ink → mark_pad. → (rect, rule)."""
    ox, oy, ow, oh = (int(v) for v in box)
    wb = ocr_word_box(gray, (ox, oy, ow, oh), option, hw, bounds=bounds) if option else None
    if wb:
        rule = "_ocr_word"
    else:
        wb = (ox, oy, ow, oh); rule = "_fit_ink"
    wb = fit_ink(gray, wb, pad=1, thr=1)
    return mark_pad(wb), rule

def place(gray, box, ftype, option=None, page_hw=None, mark=None, unit=None, bounds=None):
    """LLM box를 타입 규칙으로 검증→수정. → (rect, corrected, rule). box=픽셀(x,y,w,h).
    mark: radio 'box'/'circle'. unit: 단위글자(년/급 등) — date/number면 OCR로 글자 앵커 먼저 시도.
    bounds: 이 요소가 배정된 SoM region rect(px). 있으면 OCR 탐색을 그 안(+pad)으로 제한 — 이웃 글자 오앵커 방지."""
    hw = page_hw or gray.shape[:2]; ox, oy, ow, oh = (int(v) for v in box)
    res, rule = None, None
    if ftype in ("date", "number", "time") and unit:            # OCR 글자 앵커 우선
        res = ocr_anchor(gray, (ox, oy, ow, oh), unit, hw, bounds=bounds)
        if res: rule = "_ocr_anchor"
    if res is None and ftype == "radio" and option:
        res, rule = fit_radio(gray, (ox, oy, ow, oh), option, hw, bounds=bounds)
    if res is None:                                             # 폴백: PLACE_RULES (radio는 위에서 처리)
        fn = PLACE_RULES.get(ftype, _keep)
        res = fn(gray, (ox, oy, ow, oh), option, hw); rule = fn.__name__
    x, y, w, h = res
    x = max(0, min(x, hw[1]-1)); y = max(0, min(y, hw[0]-1))     # 경계 클램프
    w = max(2, min(w, hw[1]-x)); h = max(2, min(h, hw[0]-y))
    rect = (int(x), int(y), int(w), int(h))
    return rect, rect != (ox, oy, ow, oh), rule


def _demo():
    import cv2
    # find_cb: □ 3개 검출
    g = np.full((40, 300), 255, np.uint8)
    for i in range(3): cv2.rectangle(g, (10+i*90, 10), (30+i*90, 30), 0, 2)
    assert len(find_cb(g, 0, 0, 300, 40)) == 3, "find_cb 3 boxes"
    # carve_inline: 단위 '___세 ___회' 앞 빈칸 2개 좌→우 배정
    g4 = np.full((30, 220), 255, np.uint8)
    cv2.putText(g4, "SE", (70, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 0, 2)
    cv2.putText(g4, "HO", (190, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 0, 2)
    ci = carve_inline(g4, (0, 0, 220, 30), ["a", "b"])
    assert len(ci) == 2 and ci[0][1][0] < ci[1][1][0], "inline 2 blanks L→R"
    # letter_runs: 'L R' 글자 2개 분리 (문장부호 제외 · L/R 배정용)
    g5 = np.full((40, 120), 255, np.uint8)
    cv2.putText(g5, "L", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 0, 2)
    cv2.putText(g5, "R", (80, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 0, 2)
    assert len(letter_runs(g5, 0, 6, 120, 40)) == 2, "letter_runs L,R"
    # refine_blank: '(  급)' 박스 → 괄호/급 제외한 가운데 빈칸
    g6 = np.full((30, 200), 255, np.uint8)
    cv2.putText(g6, "(", (2, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 0, 2)
    cv2.putText(g6, "KUP)", (150, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 0, 2)
    rx, ry, rw, rh = refine_blank(g6, (0, 0, 200, 30))
    assert rx > 10 and rx+rw <= 152, f"refine strips paren+unit, got x={rx} w={rw}"
    # 빈 셀은 원본 유지
    assert refine_blank(np.full((30, 200), 255, np.uint8), (0, 0, 200, 30)) == (0, 0, 200, 30), "empty keeps"
    # place: checkbox_group 박스가 근처 □로 스냅
    g7 = np.full((60, 200), 255, np.uint8); cv2.rectangle(g7, (100, 20), (122, 42), 0, 2)
    r7, ch7, rule7 = place(g7, (95, 15, 40, 40), "checkbox_group", "a")
    assert rule7 == "_snap_mark" and ch7 and abs(r7[0]-100) < 6, f"cb snap to □, got {r7}"
    # place: signature 는 문구 크기 그대로 유지(세로 안 키움)
    r8, ch8, rule8 = place(np.full((200, 200), 255, np.uint8), (50, 100, 80, 20), "signature")
    assert rule8 == "_keep" and r8[1] == 100 and r8[3] == 20, "sig keeps phrase size"
    # place: text는 refine, 빈 박스면 원본 유지
    r9, ch9, _ = place(np.full((30, 200), 255, np.uint8), (0, 0, 200, 30), "text")
    assert not ch9, "empty text keeps"
    # place: radio → LLM box를 잉크에 조임(_fit_ink) 또는 L/R 폴백(_fit_word), 폭 타이트
    g10 = np.full((60, 200), 255, np.uint8); cv2.putText(g10, "M", (80, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 0, 2)
    r10, ch10, rule10 = place(g10, (40, 5, 100, 50), "radio", "m")
    assert rule10 in ("_ocr_word", "_fit_ink") and r10[2] < 100, f"radio fit_radio, got rule={rule10} w={r10[2]}"
    # mark_pad: 고정 px 사방 여백
    rp = mark_pad((100, 100, 40, 20))
    assert rp == (100 - MARK_PAD_PX, 100 - MARK_PAD_PX, 40 + 2 * MARK_PAD_PX, 20 + 2 * MARK_PAD_PX), f"mark_pad 고정px, got {rp}"
    # fit_bounded: 괘선 없는 band(밑줄만 있는 줄)는 라벨 글자밴드 높이로 세로를 좁힘(atom_rect 폴백).
    # 괘선 셀(cells에 걸림)은 기존처럼 셀 전체 높이 유지 — band만 달라짐, 회귀 없음.
    gB = np.full((60, 200), 255, np.uint8)
    cv2.putText(gB, "AB", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 0, 2)   # 왼쪽 라벨(예 "추천인:")
    cv2.putText(gB, "CD", (150, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 0, 2)  # 오른쪽 문구(예 "(서명 또는 인)")
    tall_box = (40, 2, 120, 54)                                            # LLM이 준 지나치게 큰 박스(atom 세로 거의 전체)
    rB = fit_bounded(gB, tall_box, [], atom_rect=(0, 0, 200, 60))
    assert rB[3] <= 15, f"band 세로를 라벨 줄 높이로 좁힘(원본54), got h={rB[3]}"
    assert 25 < rB[0] < 45, f"가로 왼쪽은 AB 라벨 뒤로, got x={rB[0]}"
    rCell = fit_bounded(gB, tall_box, [(0, 0, 200, 60)])                   # 진짜 괘선 셀이면 기존처럼 셀 전체 유지
    assert rCell[3] > 45, f"괘선 셀은 회귀 없이 전체 높이 유지, got h={rCell[3]}"
    # fit_bounded: band 오른쪽에 인접 잉크가 전혀 없으면(라벨 뒤가 그 줄 끝까지 진짜 빈칸) atom_rect
    # 끝까지 뻗지 않고 상식적 상한을 둔다 — band가 옆 칼럼(예 사이드바)까지 포함하는 문서에서
    # '주소:' 류 필드가 그 옆 칼럼까지 침범하는 걸 방지(서식6호 실측 버그).
    gD = np.full((30, 800), 255, np.uint8)
    cv2.putText(gD, "LBL", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 0, 2)   # 라벨만 있고 그 뒤는 끝까지 빈칸
    rD = fit_bounded(gD, (60, 2, 100, 26), [], atom_rect=(0, 0, 800, 30))
    assert rD[2] < 300, f"오른쪽 잉크 없으면 atom(800) 끝까지 안 뻗고 상한 적용, got w={rD[2]}"
    # fit_bounded: 단일행 괘선 셀이라도 box 높이가 셀의 절반에 못 미치면(서명란과 한 셀을 공유해 셀만 큰 경우)
    # 셀 전체가 아니라 그 텍스트 줄에만 맞춘다 — box가 이미 셀 대부분이면(지원동기류) 기존처럼 셀 전체 유지.
    gA = np.full((100, 200), 255, np.uint8)
    cv2.putText(gA, "LBL", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 0, 2)
    cellsA = [(0, 0, 200, 100)]
    rSmall = fit_bounded(gA, (60, 10, 100, 25), cellsA)     # h=25, cellH=100 → ratio .25
    assert rSmall[3] <= 20, f"작은 box는 줄 높이로, got h={rSmall[3]}"
    rBig = fit_bounded(gA, (60, 5, 100, 85), cellsA)        # h=85, cellH=100 → ratio .85
    assert rBig[3] > 80, f"큰 box(예 지원동기)는 셀 전체 유지, got h={rBig[3]}"
    # fit_bounded: 값 '안'의 마침표(예 날짜 __.__.__.)는 경계로 안 치지만, 옆 필드와의 진짜 경계(라벨)는 유지
    # — 그래서 인접한 두 필드가 같은 박스로 뭉개지지 않는다.
    gC = np.full((40, 300), 255, np.uint8)
    cv2.putText(gC, "TAG", (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 0, 2)
    cv2.rectangle(gC, (95, 20), (98, 23), 0, -1)                          # 값1 내부의 점(마침표)
    cv2.putText(gC, "NEXT", (150, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 0, 2)  # 옆 필드 라벨(진짜 경계)
    cellsC = [(0, 0, 300, 40)]
    r1 = fit_bounded(gC, (60, 5, 130, 30), cellsC)          # TAG 뒤 ~ NEXT 앞(내부에 점 포함)
    r2 = fit_bounded(gC, (200, 5, 90, 30), cellsC)          # NEXT 뒤 별개 필드
    assert r1[0]+r1[2] <= 150, f"내부 점 무시하되 NEXT 앞에서 멈춤, got right={r1[0]+r1[2]}"
    assert r2[0] >= 150, f"옆 필드는 NEXT 뒤에서 시작(뭉개짐 없음), got x={r2[0]}"
    # merged_runs: 기본 gap(8)로는 쉼표처럼 살짝 떨어진 문장부호가 글자에 뭉치지만, 좁힌 gap(4)이면
    # 별도 런으로 갈라진다 — ocr_word_box가 "(인),"에서 쉼표만 떼어낼 때 기대는 바로 그 성질.
    gE = np.full((30, 120), 255, np.uint8)
    cv2.putText(gE, "AB", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 0, 2)
    cv2.circle(gE, (42, 24), 1, 0, -1)                      # 쉼표 흉내: AB 뒤에 살짝(gap≈5px) 떨어진 점
    assert len(merged_runs(gE, 0, 0, 30, 120)) == 1, "기본 gap(8)이면 쉼표가 글자에 뭉침"
    assert len(merged_runs(gE, 0, 0, 30, 120, mgap=4)) == 2, "gap 좁히면(4) 쉼표가 별도 런으로 갈림"
    # ink_frac: 채워진 영역=높음, 빈칸=낮음
    gg = np.full((40, 60), 255, np.uint8); cv2.rectangle(gg, (15, 10), (45, 30), 0, -1)
    assert ink_frac(gg, (0, 0, 60, 40)) > 0.15 and ink_frac(np.full((40, 60), 255, np.uint8), (0, 0, 60, 40)) < 0.02, "ink_frac block vs blank"
    # merge_unit_fields: 자격증명+급 병합(급 제거, unit='급'로 이관 — 이름만 vs 이름+급수 구분은 unit 유무), 년|월은 유지
    mit = [{"region": 1, "label": "자격증명", "type": "text", "rect": (10, 10, 50, 20)},
           {"region": 1, "label": "급", "type": "text", "rect": (62, 10, 20, 20)},
           {"region": 2, "label": "년", "type": "date", "rect": (10, 10, 30, 20)},
           {"region": 2, "label": "월", "type": "date", "rect": (80, 10, 30, 20)}]
    d = merge_unit_fields(mit)
    assert d == {1}, f"drop 급(idx1) only, got {d}"
    assert mit[0]["unit"] == "급", f"unit propagated to survivor, got {mit[0].get('unit')}"
    assert "unit" not in mit[2] and "unit" not in mit[3], "년/월 진짜 별도 입력 — unit 이관 없음"
    # place: checkbox_group → □ 스냅, image → 유지(carve X)
    _, _, rule11 = place(np.full((60, 200), 255, np.uint8), (40, 5, 40, 40), "checkbox_group", "a")
    assert rule11 == "_snap_mark", "checkbox uses snap"
    assert place(np.full((60, 200), 255, np.uint8), (40, 5, 40, 40), "image")[2] == "_keep", "image=carve X"
    print("carve self-check OK · find_cb·inline·letters·refine·place(cb/sig/text/radio)·ink_frac·merge·placeholder")

if __name__ == "__main__":
    _demo()
