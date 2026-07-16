"""radio(L/R·am/pm) carve 단계 시각화 — 영역 → OCR → 자르기 → 여백.

실행: python _verify/radio_steps.py
출력: _verify/radio_steps.html
"""
from __future__ import annotations

import base64
import html
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core import carve, pipeline  # noqa: E402

IMG = ("/Users/gunhee/workspace/codespace/project/lab-voucher/.old/document/"
       "제1편_장애아가족_양육지원_서식/_extract/서식15호_사고보고서/pages/p-2.png")
CACHE = Path("/tmp/img2form_test/gcache_14.json")
OUT = ROOT / "_verify" / "radio_steps.html"

OPTS = {"L", "R", "am", "pm"}
PAIR = (("L", "R"), ("좌", "우"))


def b64(im, q=85):
    return base64.b64encode(cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, q])[1]).decode()


def box_px(e, IW, IH):
    ymin, xmin, ymax, xmax = e["box"]
    x = int(xmin / 1000 * IW); y = int(ymin / 1000 * IH)
    w = max(1, int((xmax - xmin) / 1000 * IW)); h = max(1, int((ymax - ymin) / 1000 * IH))
    return x, y, w, h


def ocr_roi(box, hw, bounds, expand=3.0):
    x, y, w, h = (int(v) for v in box)
    ex = int(h * expand)
    X0, X1 = x - ex, x + max(w, ex) + ex
    Y0, Y1 = y - int(h * 0.8), y + h + int(h * 0.8)
    return carve._clamp_roi(X0, Y0, X1, Y1, bounds)


def crop_ctx(img, rect, pad=40):
    """rect 주변 컨텍스트 crop + 원본 좌표 offset."""
    H, W = img.shape[:2]
    x, y, w, h = (int(v) for v in rect)
    x0 = max(0, x - pad); y0 = max(0, y - pad)
    x1 = min(W, x + w + pad); y1 = min(H, y + h + pad)
    return img[y0:y1, x0:x1].copy(), x0, y0


def draw_rect(im, rect, color, thick=2, label=None, off=(0, 0)):
    x, y, w, h = (int(v) for v in rect)
    ox, oy = off
    p1 = (x - ox, y - oy); p2 = (x + w - ox, y + h - oy)
    cv2.rectangle(im, p1, p2, color, thick)
    if label:
        cv2.putText(im, label, (p1[0], max(12, p1[1] - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)


def zoom(im, scale=3):
    if im.size == 0:
        return im
    return cv2.resize(im, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)


def step_card(title, note, im, meta=""):
    return (
        f'<div class=step><div class=sh>{html.escape(title)}</div>'
        f'<div class=note>{html.escape(note)}</div>'
        f'{f"<div class=meta>{html.escape(meta)}</div>" if meta else ""}'
        f'<img src="data:image/jpeg;base64,{b64(im)}"></div>'
    )


def fit_radio_trace(gray, box, option, hw, bounds):
    """fit_radio와 동일 경로를 단계별로 반환."""
    ox, oy, ow, oh = (int(v) for v in box)
    roi = ocr_roi((ox, oy, ow, oh), hw, bounds)
    ocr_raw = carve.ocr_word_box(gray, (ox, oy, ow, oh), option, hw, bounds=bounds)
    X0, Y0, X1, Y1 = roi
    toks = carve._easyread(gray, X0, Y0, X1, Y1, detail=1, paragraph=False, width_ths=0.15)
    tok_boxes = []
    for bb, t, conf in toks:
        t = (t or "").strip().replace(" ", "")
        if not t:
            continue
        xs = [p[0] for p in bb]; ys = [p[1] for p in bb]
        tok_boxes.append((min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys), t, conf))

    if ocr_raw:
        src = ocr_raw; rule = "_ocr_word"
    else:
        src = (ox, oy, ow, oh); rule = "_fit_ink"
    cut = carve.fit_ink(gray, src, pad=1, thr=1)
    padded = carve.mark_pad(cut)
    return {
        "llm": (ox, oy, ow, oh), "roi": roi, "toks": tok_boxes,
        "ocr": ocr_raw, "ink_src": src, "cut": cut, "pad": padded, "rule": rule,
    }


def render_option(img, gray, e, bounds, IW, IH, pair_info=None):
    """한 option의 4단계(+쌍 보정) 카드 HTML."""
    opt = e.get("option"); lab = e.get("label") or ""
    hw = (IH, IW)
    box = box_px(e, IW, IH)
    tr = fit_radio_trace(gray, box, opt, hw, bounds)
    X0, Y0, X1, Y1 = tr["roi"]
    cx0 = min(tr["llm"][0], X0, tr["pad"][0])
    cy0 = min(tr["llm"][1], Y0, tr["pad"][1])
    cx1 = max(tr["llm"][0] + tr["llm"][2], X1, tr["pad"][0] + tr["pad"][2])
    cy1 = max(tr["llm"][1] + tr["llm"][3], Y1, tr["pad"][1] + tr["pad"][3])
    ctx_rect = (cx0, cy0, cx1 - cx0, cy1 - cy0)

    cards = []

    # ① 영역
    im1, ox, oy = crop_ctx(img, ctx_rect, pad=50)
    if bounds:
        draw_rect(im1, bounds, (180, 60, 200), 2, "region", (ox, oy))
    draw_rect(im1, (X0, Y0, X1 - X0, Y1 - Y0), (0, 180, 255), 1, "OCR ROI", (ox, oy))
    draw_rect(im1, tr["llm"], (0, 0, 230), 2, "LLM box", (ox, oy))
    cards.append(step_card(
        "① 영역 파악",
        "보라=SoM region · 하늘=OCR 탐색 ROI · 빨강=LLM box",
        zoom(im1),
        f"LLM={tr['llm']}  ROI=({X0},{Y0})-({X1},{Y1})",
    ))

    # ② OCR
    im2, ox, oy = crop_ctx(img, ctx_rect, pad=50)
    draw_rect(im2, (X0, Y0, X1 - X0, Y1 - Y0), (0, 180, 255), 1, None, (ox, oy))
    for tx, ty, tw, th, tt, conf in tr["toks"]:
        draw_rect(im2, (tx, ty, tw, th), (80, 80, 80), 1, f"{tt}", (ox, oy))
    if tr["ocr"]:
        draw_rect(im2, tr["ocr"], (0, 200, 0), 2, f"match:{opt}", (ox, oy))
        note = f"OCR 매칭 성공 → {opt}"
    else:
        draw_rect(im2, tr["llm"], (0, 0, 230), 2, "OCR실패→LLM", (ox, oy))
        note = "OCR 매칭 실패 → LLM box 폴백"
    tok_s = ", ".join(f"{t}({c:.2f})" for *_, t, c in tr["toks"][:8]) or "(없음)"
    cards.append(step_card("② OCR", note, zoom(im2), f"tokens: {tok_s}  rule후보={tr['rule']}"))

    # ③ 자르기 (fit_ink)
    im3, ox, oy = crop_ctx(img, ctx_rect, pad=50)
    if tr["ocr"]:
        draw_rect(im3, tr["ocr"], (0, 200, 0), 1, "OCR bbox", (ox, oy))
    draw_rect(im3, tr["cut"], (255, 120, 0), 2, "fit_ink", (ox, oy))
    cards.append(step_card(
        "③ 자르기 (fit_ink)",
        "OCR/LLM bbox 안 잉크만 남겨 글자 경계로 축소",
        zoom(im3),
        f"cut={tr['cut']}",
    ))

    # ④ 여백
    im4, ox, oy = crop_ctx(img, ctx_rect, pad=50)
    draw_rect(im4, tr["cut"], (255, 120, 0), 1, "cut", (ox, oy))
    draw_rect(im4, tr["pad"], (0, 180, 80), 2, "mark_pad", (ox, oy))
    cards.append(step_card(
        "④ 여백 (mark_pad)",
        "동그라미 칠 자리 — 글자 bbox에 가로·세로 여백",
        zoom(im4),
        f"pad={tr['pad']}  place_rule={tr['rule']}",
    ))

    # ⑤ L/R 쌍 보정
    if pair_info is not None:
        win, letters, tight, final, other = pair_info
        im5, ox, oy = crop_ctx(img, win, pad=40)
        draw_rect(im5, win, (120, 120, 120), 1, "pair window", (ox, oy))
        for i, (a, b) in enumerate(letters):
            draw_rect(im5, (a, win[1], b - a, win[3]), (200, 200, 0), 1, f"L{i}", (ox, oy))
        draw_rect(im5, tight, (255, 120, 0), 2, "letter", (ox, oy))
        draw_rect(im5, final, (0, 180, 80), 2, "pair+pad", (ox, oy))
        cards.append(step_card(
            "⑤ 쌍 보정 (_radio_pair)",
            "L/R 원본 box로 창 → letter_runs → 최근접 글자 → mark_pad (최종)",
            zoom(im5, 4),
            f"letters={len(letters)}  tight={tight}  final={final}",
        ))

    head = (f'<div class=opt><div class=oh>'
            f'<span class=tag>{html.escape(str(opt))}</span> '
            f'{html.escape(lab)} · key=<code>{html.escape(str(e.get("key") or ""))}</code> · '
            f'region={e.get("region")}</div>'
            f'<div class=steps>{"".join(cards)}</div></div>')
    return head, tr


def main():
    built = pipeline.build(IMG, cache_path=str(CACHE))
    img = built["img"]; gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    IW, IH = built["page"]["w"], built["page"]["h"]
    rectof = {j: r for j, r in pipeline.atoms_of(built["segmentation"])}

    radios = [e for e in built["raw"] if e.get("type") == "radio" and e.get("option") in OPTS]
    # 쌍 정보 미리 계산 (L/R)
    items_proxy = []
    for e in radios:
        items_proxy.append({**e, "px": box_px(e, IW, IH)})

    pair_map = {}  # index in radios → (win, letters, tight, final, other_opt)
    lrp = defaultdict(dict)
    for i, e in enumerate(radios):
        opt = e.get("option")
        pk = next(((a, b) for a, b in PAIR if opt in (a, b)), None)
        if not pk:
            continue
        lab = (e.get("label") or "").strip()
        site = "" if lab.replace(" ", "") in pk or lab == opt else lab.rsplit(" ", 1)[0]
        lrp[(e.get("region"), site, pk)][opt] = i
    for (_r, _s, (oa, ob)), d in lrp.items():
        if oa not in d or ob not in d:
            continue
        iA, iB = d[oa], d[ob]
        rA, rB = box_px(radios[iA], IW, IH), box_px(radios[iB], IW, IH)
        hh = max(rA[3], rB[3]); yy = min(rA[1], rB[1])
        x0 = max(0, min(rA[0], rB[0]) - int(hh * 0.6))
        x1 = min(IW, max(rA[0] + rA[2], rB[0] + rB[2]) + int(hh * 0.6))
        win = (x0, int(yy), x1 - x0, int(hh))
        letters = carve.letter_runs(gray, int(x0), int(yy), int(x1), int(yy + hh))
        if len(letters) < 2:
            continue
        for idx, opt, r in ((iA, oa, rA), (iB, ob, rB)):
            cx = r[0] + r[2] / 2
            lx0, lx1 = pipeline._nearest_letter(letters, cx)
            tight = (lx0 - 1, int(yy), lx1 - lx0 + 2, int(hh))
            final = carve.mark_pad(tight, tight[3])
            pair_map[idx] = (win, letters, tight, final, "R" if opt == "L" else "L")

    # 그룹: L/R 부위별, am/pm region별
    sections = []
    # overview
    ov = img.copy()
    for e in radios:
        draw_rect(ov, box_px(e, IW, IH), (0, 140, 255) if e["option"] in ("am", "pm") else (0, 0, 220), 2, e["option"])
    sections.append(
        f'<div class=sec><h2>전체 overview — 빨강=L/R · 주황=am/pm (LLM box)</h2>'
        f'<img class=full src="data:image/jpeg;base64,{b64(ov, 70)}"></div>'
    )

    # L/R by site
    lr = [e for e in radios if e["option"] in ("L", "R")]
    sites = defaultdict(list)
    for e in lr:
        lab = (e.get("label") or "").strip()
        site = lab.rsplit(" ", 1)[0] if " " in lab else lab
        sites[site].append(e)
    for site, group in sites.items():
        body = []
        for e in sorted(group, key=lambda x: x["option"]):
            idx = radios.index(e)
            bounds = rectof.get(e.get("region"))
            h, _ = render_option(img, gray, e, bounds, IW, IH, pair_map.get(idx))
            body.append(h)
        sections.append(f'<div class=sec><h2>L/R · {html.escape(site)}</h2>{"".join(body)}</div>')

    # am/pm by region
    ap = [e for e in radios if e["option"] in ("am", "pm")]
    byr = defaultdict(list)
    for e in ap:
        byr[e.get("region")].append(e)
    for region, group in byr.items():
        body = []
        for e in sorted(group, key=lambda x: x["option"]):
            bounds = rectof.get(e.get("region"))
            h, _ = render_option(img, gray, e, bounds, IW, IH, None)
            body.append(h)
        keys = ", ".join(sorted({e.get("key") or "" for e in group}))
        sections.append(
            f'<div class=sec><h2>am/pm · region {region} · {html.escape(keys)}</h2>{"".join(body)}</div>'
        )

    # 최하단: 파이프라인 최종 배치 overview
    final_ov = img.copy()
    n_final = 0
    for it in built["elements"]:
        if it.get("type") != "radio" or it.get("option") not in OPTS:
            continue
        n_final += 1
        col = (0, 140, 255) if it["option"] in ("am", "pm") else (0, 0, 220)
        draw_rect(final_ov, it["rect"], col, 2, f'{it["option"]}')
    sections.append(
        f'<div class=sec><h2>최종 결과 overview — 빨강=L/R · 주황=am/pm '
        f'(place→fit_radio→mark_pad, L/R는 _radio_pair 포함) · {n_final}개</h2>'
        f'<img class=full src="data:image/jpeg;base64,{b64(final_ov, 70)}"></div>'
    )

    page = f"""<!doctype html><meta charset=utf-8>
<title>radio carve steps · 서식15호 p2</title>
<style>
*{{box-sizing:border-box}}
body{{margin:0;font:13px/1.45 system-ui,sans-serif;background:#f1f3f5;color:#212529}}
header{{position:sticky;top:0;z-index:5;background:#212529;color:#fff;padding:12px 18px}}
header h1{{margin:0;font-size:15px}} header p{{margin:4px 0 0;color:#adb5bd;font-size:12px}}
.sec{{padding:16px 18px;border-bottom:1px solid #dee2e6}}
.sec h2{{margin:0 0 12px;font-size:14px}}
.full{{max-width:100%;border:1px solid #dee2e6;border-radius:8px}}
.opt{{background:#fff;border:1px solid #dee2e6;border-radius:10px;margin:0 0 14px;overflow:hidden}}
.oh{{padding:8px 12px;background:#f8f9fa;border-bottom:1px solid #e9ecef;font-weight:600}}
.tag{{display:inline-block;background:#e8590c;color:#fff;padding:1px 8px;border-radius:5px;font-size:11px}}
.steps{{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:10px;padding:10px}}
.step{{border:1px solid #e9ecef;border-radius:8px;overflow:hidden;background:#fff}}
.sh{{background:#212529;color:#fff;padding:5px 8px;font-size:11.5px;font-weight:700}}
.note{{padding:5px 8px;font-size:11px;color:#495057;background:#f8f9fa;border-bottom:1px solid #eef1f4}}
.meta{{padding:3px 8px;font:10px/1.3 ui-monospace,monospace;color:#868e96;word-break:break-all}}
.step img{{display:block;width:100%;background:#111}}
code{{font-size:11px}}
</style>
<header>
  <h1>radio carve 단계 시각화 · 서식15호 p2</h1>
  <p>공통: ①영역(LLM+ROI+region) → ②OCR → ③fit_ink 자르기 → ④mark_pad 여백 · L/R만 ⑤letter_runs 쌍 보정</p>
</header>
{"".join(sections)}
"""
    OUT.write_text(page, encoding="utf-8")
    print(f"radios={len(radios)}  L/R={len(lr)}  am/pm={len(ap)}")
    print("✓", OUT)


if __name__ == "__main__":
    main()
