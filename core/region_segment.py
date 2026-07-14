"""영역 분리 (이미지 전용, LLM 없음): 표 → 셀, 표 밖 → 밴드.

빈틈없이 · 글자 안 자르고 · 프레임/컬럼 스코프. HWPX 불필요(이미지만).

파이프라인:
  ① 잉크 마스크        gray < INK
  ② 괘선 검출          가로/세로 morphology
  ③ 표 후보            가로·세로 괘선 공존(교차) 연결덩어리
  ④ 셀                 괘선 반전(흰 영역) 중 표 안 적당 크기
  ⑤ 프레임/표 판단     own-cells: 프레임 ⟺ 자기 소유 셀 0개(임계없음)
  ⑥ 블롭 줄            연결요소 y-band 묶기(성긴 줄·한글 상하획 ±7 병합)
  ⑦ 밴드               셀제외 strip 안, 줄마다, 경계는 빈칸, x범위=프레임/컬럼 폭

segment(gray) -> {frames, tables, cells, bands, ink, hor, ver, ccol}
"""
import cv2
import numpy as np

INK = 110  # 잉크(어두운 픽셀) 임계 — 글자 블롭(bands) 검출용


def _ink(gray):
    return (gray < INK).astype(np.uint8)


def _line_mask(gray):
    """괘선 검출 전용 마스크. 임계값은 이미지별 자동(Otsu, 최소 INK).

    글자 블롭(INK=110)과 달리, 괘선은 연회색(gray≈127)으로 인쇄된 서식도 있어
    고정 임계로는 놓친다. 서식 배경은 항상 흰색이라 Otsu가 배경/내용 경계를
    문서마다 자동으로 잡아준다. MORPH_OPEN(긴 커널)이 긴 직선만 남기므로,
    임계값을 배경 아래 넓은 안전대(≈135~220) 어디로 잡아도 뽑히는 선은 같다.
    max(INK, otsu)로 어두운 괘선 문서에서 현재보다 나빠지지 않게 가드한다.
    """
    t, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return (gray < max(INK, int(t))).astype(np.uint8)


def _ruled(lmask, H, W):
    """② 가로/세로로 긴 잉크만 추출 → 괘선."""
    hor = cv2.morphologyEx(lmask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (max(18, W // 18), 1)))
    ver = cv2.morphologyEx(lmask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(12, H // 60))))
    grid = cv2.dilate(cv2.bitwise_or(hor, ver), cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    return hor, ver, grid


def _table_candidates(grid, hor, ver, H, W):
    """③ 표 후보 = 가로·세로 괘선이 공존하는 연결덩어리(작은 표도 포함)."""
    n, lab, st, _ = cv2.connectedComponentsWithStats(grid, 8)
    out = []
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if w < 0.12 * W or h < 18:
            continue
        m = (lab == i)
        if hor[m].sum() > 40 and ver[m].sum() > 40:
            out.append((x, y, x + w, y + h))
    return out


def _cells(grid, cand, H, W):
    """④ 셀 = 괘선 반전(흰 영역) 중 표 후보 안 + 적당 크기 + 경계 안 닿음."""
    inv = (1 - grid).astype(np.uint8)
    n, _, st, _ = cv2.connectedComponentsWithStats(inv, 4)
    out = []
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if x <= 1 or y <= 1 or x + w >= W - 1 or y + h >= H - 1:
            continue
        if not (12 <= w and 10 <= h and a > 250 and w < 0.9 * W and h < 0.45 * H):
            continue
        cx, cy = x + w // 2, y + h // 2
        if any(bx0 - 4 <= cx <= bx1 + 4 and by0 - 4 <= cy <= by1 + 4 for bx0, by0, bx1, by1 in cand):
            out.append((x, y, w, h))
    return out


def _classify(cand, cells):
    """⑤ own-cells 판단: 프레임 ⟺ 자기 소유 셀 0개(정의직결·임계없음). 표 ⟺ own≥1.

    프레임은 다른 표를 담기만 하는 테두리(자기 격자 없음). 표는 자기를 셀로 분할.
    """
    own = [0] * len(cand)
    for x, y, w, h in cells:
        cx, cy = x + w // 2, y + h // 2
        ins = [(i, (t[2] - t[0]) * (t[3] - t[1])) for i, t in enumerate(cand) if t[0] <= cx <= t[2] and t[1] <= cy <= t[3]]
        if ins:
            own[min(ins, key=lambda z: z[1])[0]] += 1  # 최소포함 표가 소유
    frames = [t for i, t in enumerate(cand) if own[i] == 0]
    tables = [t for i, t in enumerate(cand) if own[i] >= 1]
    return frames, tables


def _text_lines(ink, cells, H, W):
    """⑥ 블롭 줄: 셀 밖 연결요소를 y-band로 묶음.

    성긴 줄(주소·년월일)도 글자마다 블롭이라 잡히고, ±7 허용으로 한글 상하획을
    한 줄로 병합(줄 간격 >10px 은 분리) → 경계가 글자를 관통하지 않음.
    """
    n, _, st, _ = cv2.connectedComponentsWithStats(ink, 8)

    def in_cell(bx, by, bw, bh):
        cx, cy = bx + bw // 2, by + bh // 2
        return any(x <= cx <= x + w and y <= cy <= y + h for x, y, w, h in cells)

    blobs = sorted([(x, y, w, h) for i in range(1, n) for x, y, w, h, a in [st[i]]
                    if 5 <= h <= 60 and w >= 3 and a >= 18 and not in_cell(x, y, w, h)], key=lambda b: b[1])
    lns = []
    for x, y, w, h in blobs:
        for L in lns:
            if not (y > L[3] + 7 or y + h < L[1] - 7):  # ±7: 상하획 병합, 줄간격 분리
                L[0] = min(L[0], x); L[1] = min(L[1], y); L[2] = max(L[2], x + w); L[3] = max(L[3], y + h)
                break
        else:
            lns.append([x, y, x + w, y + h])
    return sorted([l for l in lns if l[2] - l[0] >= 15 and l[3] - l[1] >= 8], key=lambda l: l[1])


def _content_column(gray, H, W):
    """세로 거터(넓은 빈 컬럼)로 분할 → 잉크 최다 컬럼(사이드바 크롬 제외)."""
    colink = (gray < INK).sum(0)
    guts = []
    s = None
    for x in range(W):
        if colink[x] < H * 0.004:
            if s is None:
                s = x
        elif s is not None:
            if x - s >= 25:
                guts.append((s, x))
            s = None
    if s is not None and W - s >= 25:
        guts.append((s, W))
    bd = [0] + [(a + b) // 2 for a, b in guts] + [W]
    cols = [(bd[i], bd[i + 1]) for i in range(len(bd) - 1) if bd[i + 1] - bd[i] > W * 0.18]
    return max(cols, key=lambda c: int(colink[c[0]:c[1]].sum())) if cols else (0, W)


def _bands(lines, cells, frames, ccol, H, W):
    """⑦ 밴드: 셀제외 strip 안에서 줄마다, 경계는 빈칸, x범위는 프레임→컬럼 폭.

    반환: (x0, y0, x1, y1) 리스트. 셀 영역은 strip 이 끊어 표 관통을 막고,
    x범위는 프레임(있으면)/컨텐츠컬럼(없으면) 으로 한정해 밖으로 안 삐진다.
    """
    cellcov = np.zeros(H, bool)
    for x, y, w, h in cells:
        cellcov[y:y + h] = True
    strips = []
    s = None
    for y in range(H):
        if not cellcov[y]:
            if s is None:
                s = y
        elif s is not None:
            strips.append((s, y)); s = None
    if s is not None:
        strips.append((s, H))

    def xr(cy):
        for fx0, fy0, fx1, fy1 in frames:
            if fy0 <= cy <= fy1:
                return fx0 + 4, fx1 - 4
        return ccol[0] + 4, ccol[1] - 4

    out = []
    for ys, ye in strips:
        sl = sorted([l for l in lines if ys <= (l[1] + l[3]) // 2 < ye], key=lambda l: l[1])
        for i, l in enumerate(sl):
            top = (sl[i - 1][3] + l[1]) // 2 if i > 0 else ys
            bot = (l[3] + sl[i + 1][1]) // 2 if i < len(sl) - 1 else ye
            x0, x1 = xr((l[1] + l[3]) // 2)
            # 국소 보정: 그 줄의 실제 잉크(l[0]·l[2])가 컬럼/프레임 밖으로 나가면 그만큼 확장.
            # 하단 서명 '(서명 또는 인)'처럼 세로로 성겨 _content_column 이 거터로 잘라낸
            # 우측 콘텐츠를 회복한다. 프레임 안 줄은 잉크가 내부라 변화 없음(안전).
            x0 = max(0, min(x0, l[0] - 4)); x1 = min(W, max(x1, l[2] + 4))
            out.append((x0, max(ys, top), x1, min(ye, bot)))
    return out


def segment(gray):
    """그레이 이미지 → 영역 분리 결과 dict."""
    H, W = gray.shape
    ink = _ink(gray)
    hor, ver, grid = _ruled(_line_mask(gray), H, W)
    cand = _table_candidates(grid, hor, ver, H, W)
    cells = _cells(grid, cand, H, W)
    frames, tables = _classify(cand, cells)
    lines = _text_lines(ink, cells, H, W)
    ccol = _content_column(gray, H, W)
    bands = _bands(lines, cells, frames, ccol, H, W)
    return dict(ink=ink, hor=hor, ver=ver, frames=frames, tables=tables, cells=cells, bands=bands, ccol=ccol)
