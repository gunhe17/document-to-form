"""image-to-form 파이프라인 — 서식 이미지 → 배치된 입력 요소 → FormSchema.

  ① 영역분리   region_segment.segment      (CV, LLM 없음)
  ② SoM 마킹   원자(cells+bands)에 번호     (LLM 입력 이미지)
  ③ 그라운딩   extract.ground              (LLM: 무엇·타입·option·unit·대략위치)
  ④ 배치       carve.place + 후처리         (CV/OCR: 정확한 위치)
  ⑤ 병합/조립  merge_unit_fields → FormSchema

원칙: LLM=의미(무엇이 입력), CV/OCR=위치(어디에).
"""
import os, json, base64
from collections import defaultdict
import cv2

from . import carve, extract
from .region_segment import segment

# ④ 배치 후처리 대상 (타입별)
_BOUND = {"text", "email", "phone", "number", "date", "time"}              # fit_bounded: 상하좌우 경계 맞춤
_SNAP = {"image"}                                                          # snap_to_cell: 표 셀 격자 스냅
_FILL = {"textarea"}                                                       # fill_cell: 여러 줄 쓰기칸 → 셀 전체 채움

# 타입 → FormSchema 위젯
WIDGET = {"text": "text_box", "textarea": "textarea", "number": "text_box", "date": "date_box",
          "time": "time_box", "email": "text_box", "phone": "text_box", "radio": "radio",
          "checkbox_group": "checkbox", "consent": "consent", "signature": "sign",
          "select": "select", "image": "image"}


def _b64(img):
    return base64.b64encode(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 72])[1]).decode()


def atoms_of(S):
    """분리 결과 → 번호 붙은 원자 [(id, (x,y,w,h))] — cells 먼저, 그 다음 bands."""
    cells = [tuple(c) for c in S["cells"]]
    bands = [(a, b, c - a, d - b) for a, b, c, d in S["bands"]]
    return list(enumerate(cells + bands))


def som_mark(img, atoms):
    """원자마다 번호 박스를 그린 SoM 이미지 (LLM 입력용)."""
    im = img.copy()
    for j, (x, y, w, h) in atoms:
        cv2.rectangle(im, (x, y), (x + 22, y + 13), (0, 150, 255), -1)
        cv2.rectangle(im, (x, y), (x + w, y + h), (0, 150, 255), 1)
        cv2.putText(im, str(j), (x + 1, y + 11), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (255, 255, 255), 1, cv2.LINE_AA)
    return im


def _unit_of(e):
    """number: LLM이 준 unit. date/time: 라벨에서 단위글자 추출(년/월/일/시/분)."""
    if e.get("unit"):
        return e["unit"]
    if e.get("type") in ("date", "time"):
        return next((c for c in "년월일시분" if c in (e.get("label") or "")), None)
    return None


def place_elements(gray, S, raw_elements):
    """③ LLM 요소 → ④ 배치된 items. 타입별 carve.place + 후처리 + 병합."""
    IH, IW = gray.shape
    cells = S["cells"]
    rectof = {j: r for j, r in atoms_of(S)}
    items = []
    for e in raw_elements:
        bx = e.get("box")
        if not bx or len(bx) != 4:
            continue
        ymin, xmin, ymax, xmax = bx
        x = xmin / 1000 * IW; y = ymin / 1000 * IH
        w = (xmax - xmin) / 1000 * IW; h = (ymax - ymin) / 1000 * IH
        if w <= 1 or h <= 1:
            continue
        t = e.get("type", "text"); opt = e.get("option")
        bounds = rectof.get(e.get("region"))                 # 배정 SoM region rect → OCR 탐색을 그 안으로 제한
        (rx, ry, rw, rh), corrected, rule = carve.place(gray, (x, y, w, h), t, opt, (IH, IW), None, _unit_of(e), bounds)
        if t in _BOUND and not opt and rule != "_ocr_anchor":  # 단위 값칸·인라인 → 상하좌우 여백 맞춤 (OCR앵커는 이미 글자높이 맞춤)
            moved = None
            obox = (int(x), int(y), int(w), int(h))            # LLM 원본 box(_refine 붕괴 전) 기준으로 재배치 판단
            if t in ("text", "textarea") and carve.ink_frac(gray, obox) > 0.06:  # box가 잉크 위 → 자리표시자? 라벨?
                moved = carve.fit_placeholder(gray, obox, (IH, IW))               # '○○○·000' 위 → 자리표시자에 배치
                if moved: rx, ry, rw, rh = moved; corrected = True; rule = "_placeholder"
                else:                                                            # 라벨 글자 위 → 콜론 뒤 빈칸으로 이동
                    moved = carve.fit_after_label(gray, obox, cells)
                    if moved: rx, ry, rw, rh = moved; corrected = True; rule = "_after_label"
            if moved is None:
                r2 = carve.fit_bounded(gray, (rx, ry, rw, rh), cells)
                if r2 != (rx, ry, rw, rh): rx, ry, rw, rh = r2; corrected = True; rule = "_fit_bounded"
        elif t in _SNAP and not opt:                         # 사진란 → 표 셀 격자 스냅
            r2 = carve.snap_to_cell((rx, ry, rw, rh), cells)
            if r2 != (rx, ry, rw, rh): rx, ry, rw, rh = r2; corrected = True; rule = "_snap_cell"
        elif t in _FILL and not opt:                         # 여러 줄 쓰기칸 → 포함 셀 전체 채움(한 줄 축소 X)
            r2 = carve.fill_cell((rx, ry, rw, rh), cells)
            if r2 != (rx, ry, rw, rh): rx, ry, rw, rh = r2; corrected = True; rule = "_fill_cell"
        elif t == "signature":                               # '(서명 또는 인)' 인쇄문구 잉크에 맞춤
            r2 = carve.fit_ink(gray, (rx, ry, rw, rh))
            if r2 != (rx, ry, rw, rh): rx, ry, rw, rh = r2; corrected = True; rule = "_fit_ink"
        items.append({"region": e.get("region"), "key": e.get("key"),
                      "label": e.get("label") or e.get("key") or "", "type": t, "option": opt,
                      "unit": e.get("unit"), "rect": tuple(int(v) for v in (rx, ry, rw, rh)),
                      "box": tuple(bx), "corrected": corrected, "rule": rule})
    # ⑤a0 checkbox_group: region별 □ 검출 → LLM box 위치로 1:1 배정(중복 방지). OCR 라벨 우회 대체.
    cbyr = defaultdict(list)
    for i, it in enumerate(items):
        if it["type"] == "checkbox_group":
            cbyr[it["region"]].append(i)
    for region, idxs in cbyr.items():
        if region not in rectof:
            continue
        rx, ry, rw, rh = rectof[region]
        marks = carve.find_cb(gray, rx, ry, rw, rh)          # 그 영역 모든 □
        if len(marks) < len(idxs):                           # □가 옵션보다 적으면 개별 결과 유지
            continue
        used = set()
        for i in sorted(idxs, key=lambda k: (items[k]["box"][0], items[k]["box"][1])):  # 읽기순(ymin,xmin)
            ymin, xmin, ymax, xmax = items[i]["box"]
            bcx = (xmin+xmax)/2/1000*IW; bcy = (ymin+ymax)/2/1000*IH
            j, m = min(((j, m) for j, m in enumerate(marks) if j not in used),
                       key=lambda jm: (jm[1][0]+jm[1][2]/2-bcx)**2 + (jm[1][1]+jm[1][3]/2-bcy)**2)
            used.add(j); items[i]["rect"] = tuple(int(v) for v in m); items[i]["corrected"] = True; items[i]["rule"] = "_cb_assign"
    # ⑤a0b 단일글자 radio(L/R) 쌍: 부위별 '( L , R )'에서 글자런을 좌→우로 L·R 배정 (개별 OCR 부정확 보정)
    lrp = defaultdict(dict)
    for i, it in enumerate(items):
        if it["type"] == "radio" and it.get("option") in ("L", "R", "좌", "우"):
            lrp[(it["region"], it["label"].rsplit(" ", 1)[0])][it["option"]] = i
    for d in lrp.values():
        iL = d.get("L", d.get("좌")); iR = d.get("R", d.get("우"))
        if iL is None or iR is None:
            continue
        rL, rR = items[iL]["rect"], items[iR]["rect"]
        hh = max(rL[3], rR[3]); yy = min(rL[1], rR[1])
        x0 = max(0, min(rL[0], rR[0])-int(hh*0.6)); x1 = min(IW, max(rL[0]+rL[2], rR[0]+rR[2])+int(hh*0.6))
        letters = carve.letter_runs(gray, x0, yy, x1, yy+hh)   # CC로 글자만 분리(괄호·콤마 제외)
        if len(letters) >= 2:
            (lx0, lx1), (rx0, rx1) = letters[-2], letters[-1]  # L·R = 오른쪽 2개(부위명·'('는 왼쪽이라 배제)
            items[iL]["rect"] = (lx0-1, yy, lx1-lx0+2, hh); items[iL]["rule"] = "_lr_pair"; items[iL]["corrected"] = True
            items[iR]["rect"] = (rx0-1, yy, rx1-rx0+2, hh); items[iR]["rule"] = "_lr_pair"; items[iR]["corrected"] = True
    # ⑤a 촘촘한 날짜행 재카브 (표 셀에 date 2개↑ → 셀에서 빈칸 N개 좌→우)
    dbyr = defaultdict(list)
    for i, it in enumerate(items):
        if it["type"] == "date":
            dbyr[it["region"]].append(i)
    for region, idxs in dbyr.items():
        if len(idxs) < 2 or region not in rectof or rectof[region][3] > 90:   # 큰 블록(밴드)은 OCR 앵커 유지
            continue
        idxs.sort(key=lambda k: items[k]["rect"][0])
        for (_, rc), k in zip(carve.carve_inline(gray, rectof[region], [str(k) for k in idxs]), idxs):
            items[k]["rect"] = tuple(int(v) for v in rc); items[k]["corrected"] = True; items[k]["rule"] = "_date_inline"
    # ⑤b 단위 중복분할 병합 (급 등)
    drop = carve.merge_unit_fields(items)
    final = [it for i, it in enumerate(items) if i not in drop]
    _unify_row_heights(final)                            # ⑤c 같은 행·같은 타입 세로 크기 통일(균형)
    return final


_ALIGN = {"text", "number", "date", "time", "email", "phone", "radio", "checkbox_group", "signature"}
def _unify_row_heights(items):
    """같은 타입끼리 y-중심으로 행을 묶어(근접 클러스터), 각 행의 높이·세로중심을 median으로 통일.
    가로(x·너비)는 유지. textarea·image·consent(각자 셀/블록 채움)는 제외."""
    from statistics import median
    by = defaultdict(list)
    for it in items:
        if it["type"] in _ALIGN: by[it["type"]].append(it)
    for els in by.values():
        els.sort(key=lambda it: it["rect"][1]+it["rect"][3]/2)
        rows = [[els[0]]]
        for it in els[1:]:
            prev = rows[-1][-1]["rect"]; c = it["rect"][1]+it["rect"][3]/2
            if abs(c - (prev[1]+prev[3]/2)) <= max(10, 0.7*prev[3]):     # y-중심 근접 → 같은 행
                rows[-1].append(it)
            else:
                rows.append([it])
        for row in rows:                                                # 같은 행이라도 수평으로 가까운 것끼리만 통일
            row.sort(key=lambda it: it["rect"][0])
            H0 = median(it["rect"][3] for it in row)
            groups = [[row[0]]]
            for it in row[1:]:
                pr = groups[-1][-1]["rect"]; gap = it["rect"][0] - (pr[0]+pr[2])   # 오른끝→왼끝 간격
                if gap <= max(120, 6*H0):                                          # 수평으로 가까움 → 같은 그룹
                    groups[-1].append(it)
                else:
                    groups.append([it])
            for cl in groups:
                if len(cl) < 2: continue
                H = int(median(it["rect"][3] for it in cl)); CY = median(it["rect"][1]+it["rect"][3]/2 for it in cl)
                for it in cl:
                    x, y, w, h = it["rect"]; it["rect"] = (x, int(CY-H/2), w, H)


def build(image_path, cache_path=None):
    """서식 이미지 → 파이프라인 산출물.
    cache_path: LLM 응답(JSON) 캐시. 있으면 재사용, 없으면 호출 후 저장(그라운딩 비용 절감).
    → {page, segmentation, atoms, img, marked, raw, elements}."""
    img = cv2.imread(str(image_path))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    IH, IW = gray.shape
    S = segment(gray)
    atoms = atoms_of(S)
    marked = som_mark(img, atoms)
    if cache_path and os.path.exists(cache_path):
        res = json.load(open(cache_path))
    else:
        res = extract.ground(_b64(marked), len(atoms))
        if cache_path:
            json.dump(res, open(cache_path, "w"), ensure_ascii=False)
    elements = place_elements(gray, S, res.get("elements", []))
    return {"page": {"w": IW, "h": IH}, "segmentation": S, "atoms": atoms,
            "img": img, "marked": marked, "raw": res.get("elements", []), "elements": elements}


def to_form_schema(built, image_name="form.png"):
    """배치 결과 → FormSchema dict (core.schema.validate_form_schema 로 검증 가능).
    fields=의미(key→타입/라벨/options/unit), elements=좌표(정규화 0..1 + widget + field_refs)."""
    IW, IH = built["page"]["w"], built["page"]["h"]
    fields, elements = {}, []
    for it in built["elements"]:
        k = it["key"] or it["label"]
        if not k:
            continue
        t = it["type"]; opt = it.get("option")
        if k not in fields:
            fd = {"type": t, "label": it["label"]}
            if it.get("unit"):
                fd["validation"] = {"unit": it["unit"]}
            fields[k] = fd
        if opt:
            fields[k].setdefault("options", [])
            if not any(o["value"] == str(opt) for o in fields[k]["options"]):
                fields[k]["options"].append({"value": str(opt), "label": str(opt)})
        x, y, w, h = it["rect"]
        el = {"id": f"e{len(elements)}", "page": 1,
              "rect": [round(x / IW, 4), round(y / IH, 4), round(w / IW, 4), round(h / IH, 4)],
              "widget": WIDGET.get(t, "box"), "field_refs": [k]}
        if opt is not None:
            el["option"] = str(opt)
        elements.append(el)
    return {"pages": [{"no": 1, "image": image_name, "w": IW, "h": IH}], "fields": fields, "elements": elements}
