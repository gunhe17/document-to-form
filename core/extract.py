"""③ LLM 그라운딩 — SoM 번호 이미지 → 입력 요소 상세 + 위치(box).

Gemini 3.1 Pro 비전. box=[ymin,xmin,ymax,xmax] 이미지 정규화 0~1000 (Gemini 그라운딩 규약).
temperature=1.0 (Gemini 3 권장; 낮추면 검출 저하·looping). reasoning effort=low.

LLM = 의미(무엇이 입력·타입·option·unit) + 대략 위치. 정확한 위치는 carve 가 잡는다.
"""
import os, json, re, time, urllib.request
from pathlib import Path

MODEL = "google/gemini-3.1-pro-preview"
API = "https://openrouter.ai/api/v1/chat/completions"

SYS = ("# 역할\n"
 "정부 서식(빈 양식) 이미지에서 서식을 제출하려는 작성자가 제출을 위해 채워 넣어야 하는 입력 요소를 추출한다(접수번호·기관명·수신처처럼 접수기관이 채우거나 인쇄된 문구는 제외). 이미지엔 번호 박스로 영역이 표시돼 있다.\n\n"
 "# 판단 절차\n"
 "1. 전체 맥락: 무슨 문서이고 어떤 정보를 수집하는지 파악.\n"
 "2. 영역 맥락: 각 번호 영역이 인쇄된 고정 텍스트인지, 사용자가 채우는 빈칸/선택지인지 판단.\n"
 "3. 요소·타입·위치: 채우는 부분마다 요소를 만들고 타입과 box를 정한다.\n\n"
 "# 타입\n"
 "- text/textarea/email/phone: 자유 입력칸(여러 줄이면 textarea).\n"
 "- number: 수량 단위(급·원·회·점 등)가 붙은 값. 단위는 unit 필드에 담고 box는 단위 앞 빈칸. 단위글자는 별도 요소로 만들지 않는다.\n"
 "- date: 날짜(년·월·일 각각 별도 요소, '생년월일' 단일칸은 합침). 'A~B' 기간/범위(예 '__년 __월 ~ __년 __월')는 시작년·시작월·종료년·종료월을 각각 별도 date로 — 이 예시는 4개. 하단 '20 __년 __월 __일'도 년·월·일 3개.\n"
 "- time: 시각(시·분).\n"
 "- checkbox_group: 네모 □ 체크칸이 있는 선택지(□마다 요소, option은 값 하나). '기타(  )'처럼 선택지 뒤에 손으로 쓰는 빈칸이 실제로 그려져 있을 때만 그 빈칸을 별도 text 요소로 추가하고, 빈칸이 없으면 만들지 않는다.\n"
 "- radio: □ 없이 나열된 배타적 텍스트 보기 중 하나를 골라 동그라미 치는 방식(보기마다 요소·option). 빈 양식에 ○가 안 그려져 있어도 이런 보기면 radio. 예: 성별 '남/여', 시각 'am/pm'(→am·pm 둘)·'오전/오후'. 항목·체크박스 라벨 뒤 괄호 안 택1 표기 '(L, R)'·'(좌, 우)'·'(am/pm)' 등(헤더에 '좌, 우 표시' 같은 지시가 있는 경우 포함)은, 그 항목(라벨은 괄호 뺀 이름)과 별도로 각 보기의 radio를 만든다. 예: 부위 '눈(L,R)' → 체크박스 '눈' + radio L·R, '□신고함 (am/pm)' → 체크박스 '신고함' + radio am·pm.\n"
 "- consent: 동의 조항 또는 동의 □.  - signature: '(서명 또는 인)' 문구영역.  - image: 사진 부착란(예 '칼라 사진').\n\n"
 "# 복합칸 (값 입력 + 선택)\n"
 "선택표시(am/pm·오전/오후 등) 바로 왼쪽에 값을 쓸 넓은 빈칸이 비어 있으면, 그 빈칸을 값 입력칸(time 등)으로도 뽑아 입력+선택 둘 다 요소로 만든다. 선택표시 왼쪽이 다른 글자나 □ 선택지로 차 있으면(예 '□안함 □신고함 (am/pm)') 입력칸은 만들지 말고 선택만.\n\n"
 "# 출력 형식\n"
 "JSON만: {\"elements\":[{\"region\":정수,\"key\":\"영문\",\"label\":\"한글\",\"type\":\"\",\"option\":\"\"?,\"unit\":\"급/세 등\"?,\"box\":[ymin,xmin,ymax,xmax]}]}\n"
 "box는 이미지 기준 0~1000 정규화, 단위·괄호 제외한 빈칸/□/문구의 사각형. type∈[text,textarea,number,date,time,email,phone,radio,checkbox_group,consent,signature,image].\n\n"
 "# 반드시 지킬 것\n"
 "- 이미지에 실제로 보이는 빈칸·□·○의 개수만큼만 요소를 만든다. 의미상 있을 법한, 이미지에 없는 필드는 만들지 않는다.\n"
#  "- 라벨·제목·안내문·표머리글·수신처('○○기관 귀중')·단위글자·괄호기호는 입력이 아니다.\n"
 "- 표 안 모든 빈 셀, 모든 □/○ 선택지, 반복행을 하나도 빠뜨리지 말 것.\n"
 "- 이미지를 신중히 보고 판단하라.")

# 판단 절차 + 일반화 타입 12종(설명·형태 예시만). stamp/select/datetime/file/checkbox 없음.
# ground()는 SYS_V2 사용.
SYS_V2 = """# 역할
빈 정부 서식 이미지에서, 작성자가 채워 제출해야 하는 입력만 추출한다.
접수기관이 채우거나 이미 인쇄된 고정 문구는 제외한다.
이미지에는 영역 번호 박스가 표시되어 있다.

# 판단 절차
1. 문서 맥락: 어떤 정보를 모으는 서식인지 파악한다.
2. 영역 맥락: 번호 영역이 고정 인쇄인지, 사용자 입력(빈칸·선택)인지 구분한다.
3. 요소화: 입력마다 요소 1개를 만들고 type·대략 box를 정한다.
4. 실제성: 이미지에 보이는 빈칸·□·선택 표시 개수만큼만 만든다. 없는 필드는 만들지 않는다.

# 타입
허용 type은 아래 12종만. 각 타입 = 설명 + 일반화 형태. 형태에 맞으면 그 type.
(checkbox·select·datetime·file·stamp 는 쓰지 않는다. 직인·도장은 signature.)

## text
한 줄 자유 입력.
형태: 밑줄·점선·빈 셀·괄호 안 빈칸 `(___)` · `○○○` 자리표시자.

## textarea
여러 줄 자유 입력.
형태: 세로로 큰 빈 칸·여러 줄 밑줄·큰 빈 셀.

## email
이메일 주소 입력.
형태: text와 같으나 라벨/맥락이 이메일.

## phone
전화번호 입력.
형태: text와 같으나 라벨/맥락이 전화·연락처.

## number
수량·금액·나이 등 **시각이 아닌** 숫자 값.
옆에 붙은 단위 글자는 unit에만 넣고, box는 단위 앞 빈칸. 단위 글자 자체는 요소로 만들지 않는다.
형태: `___` + 단위(급·원·회·점·세·시간(총량)·m …).
경계: `시`/`분`이 시각(몇 시 몇 분)을 쓰면 time. 총 시간·연령·급수·금액만 number.

## date
날짜에서 `년`·`월`·`일` 단위 글자가 실제로 인쇄되어 있으면, 각 단위 앞 입력칸을 별도 date 요소로 만든다.
단위 글자가 없이 한 칸에 합쳐 쓰는 날짜는 date 요소 1개로 만든다. 단위 글자 자체는 요소로 만들지 않는다.
기간도 실제로 인쇄된 단위와 입력칸을 기준으로 시작·끝을 각각 분리한다.
형태: `__년 __월 __일` → 3개 · `생년월일` 단일칸 → 1개 · `__년 __월 ~ __년 __월` → 4개.

## time
시각(하루 안의 시각). 시·분이 나뉘면 칸마다 요소.
형태: `__시 __분` · `__ : __` · 시각 빈칸 + `am/pm`(시각 빈칸은 time, am/pm은 radio).
경계: 단위가 `시`여도 시각이면 time이지 number가 아니다.

## checkbox_group
네모 □가 있는 선택. □마다 요소 1개, option=그 보기 값.
□ 뒤에 손으로 쓰는 빈칸이 실제로 있으면 그 빈칸은 별도 text.
복수 선택 가능·□가 보이면 checkbox_group.
형태: `□보기` · `□보기 (___ )`.

## radio
□ 없이, **하나만** 고르는 배타적 텍스트 보기(○·동그라미·슬래시·괄호 택1 포함).
보기마다 요소 1개, option=그 보기. 빈 양식에 ○가 없어도 배타 보기면 radio.
항목 라벨 뒤 괄호 택1은 본 항목과 별도로 각 보기 radio.
형태: `A/B` · `A · B` · `(A, B)` · `am/pm` · `남/여`. 슬래시·중점으로 나열된 각 낱개가 개별 보기(`am/pm`→am·pm 2개, `오전/오후`→오전·오후 2개, `남/여`→남·여 2개). 단 `쇼크/질식`·`식사/간식시간`처럼 한 항목명 안의 슬래시는 나누지 않는다.
경계: □가 있으면 checkbox_group. □ 없이 택1이면 radio.

## consent
동의 내용 또는 동의 □.
형태: 동의 문장 + `□동의` · 동의 체크란.

## signature
서명·날인·직인 위치. 인쇄된 서명/인/직인 안내 문구 영역.
형태: `(서명 또는 인)` · `(인)` · `인)` · 서명란·직인란 문구.

## image
사진·첨부 이미지를 붙이는 칸.
형태: 사진 규격 안내가 있는 빈 부착란.

# 출력
JSON만 (Structured Output schema). key는 만들지 않는다 — 서버가 region·type·option·unit으로 부여한다.
{"elements":[{"region":정수,"label":"한글","type":"","option":""?,"unit":""?,"box":[ymin,xmin,ymax,xmax]}]}
box: 이미지 0~1000 정규화. 입력 부분(빈칸·□·문구)의 사각형. 단위·장식 괄호는 box에서 제외.
type∈[text,textarea,email,phone,number,date,time,checkbox_group,radio,consent,signature,image].
"""

TYPES = ("text", "textarea", "email", "phone", "number", "date", "time",
         "checkbox_group", "radio", "consent", "signature", "image")

# OpenRouter json_schema — key 없음. option/unit은 null 허용.
ELEMENT_SCHEMA = {
    "type": "object",
    "properties": {
        "region": {"type": "integer", "description": "SoM 영역 번호"},
        "label": {"type": "string", "description": "한글 라벨"},
        "type": {"type": "string", "enum": list(TYPES)},
        "option": {"type": ["string", "null"], "description": "선택지 값(checkbox/radio)"},
        "unit": {"type": ["string", "null"], "description": "단위 글자(급/세/년/월 등)"},
        "box": {
            "type": "array",
            "items": {"type": "integer"},
            "minItems": 4,
            "maxItems": 4,
            "description": "[ymin,xmin,ymax,xmax] 0~1000",
        },
    },
    "required": ["region", "label", "type", "box"],
    "additionalProperties": False,
}

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "elements": {"type": "array", "items": ELEMENT_SCHEMA},
    },
    "required": ["elements"],
    "additionalProperties": False,
}


def load_key():
    """OPENROUTER_API_KEY — 환경변수 또는 상위 디렉토리 .env 탐색."""
    if os.environ.get("OPENROUTER_API_KEY"):
        return os.environ["OPENROUTER_API_KEY"]
    for base in [Path.cwd(), *Path(__file__).resolve().parents]:
        env = base / ".env"
        if env.exists():
            for ln in env.read_text().splitlines():
                if ln.startswith("OPENROUTER_API_KEY="):
                    return ln.split("=", 1)[1].strip()
    return ""


def _slug(s):
    """key용 짧은 토큰 — 공백·특수문자 제거."""
    s = re.sub(r"\s+", "", str(s))
    s = re.sub(r"[^\w가-힣]+", "", s, flags=re.UNICODE)
    return s[:24] or "x"


def assign_keys(elements):
    """LLM key 폐기 → region/type/option/unit + 동일그룹 순번으로 결정론 부여.
    box 순(region, ymin, xmin)으로 정렬해 호출 간 순서 흔들림을 줄인다."""
    from collections import defaultdict
    els = sorted(
        (dict(e) for e in elements),
        key=lambda e: (
            e.get("region") if e.get("region") is not None else 10**9,
            (e.get("box") or [0, 0, 0, 0])[0],
            (e.get("box") or [0, 0, 0, 0])[1],
            e.get("type") or "",
            str(e.get("option") or ""),
            str(e.get("unit") or ""),
        ),
    )
    counts = defaultdict(int)
    for e in els:
        e.pop("key", None)
        region = e.get("region")
        t = e.get("type") if e.get("type") in TYPES else "text"
        e["type"] = t
        base = f"r{region}_{t}"
        if e.get("option") not in (None, ""):
            base += f"_{_slug(e['option'])}"
        if e.get("unit") not in (None, ""):
            base += f"_{_slug(e['unit'])}"
        counts[base] += 1
        e["key"] = f"{base}_{counts[base]}"
    return els


def _parse_elements(raw_text):
    """모델 응답 텍스트 → elements 리스트. 실패 시 예외."""
    if isinstance(raw_text, list):
        # OpenRouter content parts
        raw_text = "".join(
            p.get("text", "") if isinstance(p, dict) else str(p) for p in raw_text
        )
    text = (raw_text or "").replace("```json", "").replace("```", "")
    data = json.loads(re.search(r"\{.*\}", text, re.S).group(0))
    return data.get("elements") or []


def _box_iou(a, b):
    """box=[ymin,xmin,ymax,xmax] IoU. 없거나 형식 틀리면 0."""
    if not a or not b or len(a) != 4 or len(b) != 4:
        return 0.0
    ay0, ax0, ay1, ax1 = a
    by0, bx0, by1, bx1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    if inter <= 0:
        return 0.0
    area_a = max(0, ax1 - ax0) * max(0, ay1 - ay0)
    area_b = max(0, bx1 - bx0) * max(0, by1 - by0)
    union = area_a + area_b - inter
    return inter / union if union else 0.0


def _elem_sig(e):
    """merge용 서명 — key/label 제외(표기 노이즈)."""
    opt = e.get("option")
    unit = e.get("unit")
    return (e.get("region"), e.get("type"),
            None if opt in (None, "") else str(opt),
            None if unit in (None, "") else str(unit))


def _majority_clusters(runs, iou_thresh=0.25):
    """runs → 클러스터 리스트. 각 항목: {runs:set, members:[(ri,e),...]}."""
    clusters = []
    for ri, els in enumerate(runs):
        for e in els:
            e = dict(e)
            sig = _elem_sig(e)
            best_i, best_iou = None, iou_thresh
            for ci, c in enumerate(clusters):
                if ri in c["runs"]:
                    continue
                if _elem_sig(c["members"][0][1]) != sig:
                    continue
                ious = [_box_iou(e.get("box"), m.get("box")) for _, m in c["members"]]
                iou = max(ious) if ious else 0.0
                if iou > best_iou:
                    best_iou, best_i = iou, ci
            if best_i is not None:
                clusters[best_i]["members"].append((ri, e))
                clusters[best_i]["runs"].add(ri)
            else:
                clusters.append({"members": [(ri, e)], "runs": {ri}})
    return clusters


def majority_merge(runs, min_votes=None, iou_thresh=0.25):
    """N회 elements 리스트 → 과반 이상 나온 요소만 남긴 1개 리스트.

    같은 (region, type, option, unit) + box IoU≥iou_thresh 이면 동일 슬롯.
    한 run에서 같은 클러스터에 두 번 들어가지 않음.
    box=중앙값, label=최빈. key는 assign_keys로 재부여.
    min_votes 기본 = N//2+1 (5회→3)."""
    from collections import Counter
    n = len(runs)
    if n == 0:
        return []
    if min_votes is None:
        min_votes = n // 2 + 1

    out = []
    for c in _majority_clusters(runs, iou_thresh=iou_thresh):
        votes = len(c["runs"])
        if votes < min_votes:
            continue
        members = [e for _, e in c["members"]]
        boxes = [m["box"] for m in members if m.get("box") and len(m["box"]) == 4]
        if boxes:
            box = [int(sorted(b[i] for b in boxes)[len(boxes) // 2]) for i in range(4)]
        else:
            box = members[0].get("box")
        label = Counter(m.get("label") or "" for m in members).most_common(1)[0][0]
        rep = members[0]
        out.append({
            "region": rep.get("region"),
            "type": rep.get("type"),
            "option": rep.get("option"),
            "unit": rep.get("unit"),
            "label": label,
            "box": box,
            "_votes": votes,
        })
    return assign_keys(out)


def majority_merge_trace(runs, min_votes=None, iou_thresh=0.25):
    """majority_merge와 동일 로직 + 클러스터 추적(시각화용).
    → {kept, dropped, clusters, min_votes, iou_thresh, n_runs}
    clusters[].kept True/False, median_box, label, sig, votes, members."""
    from collections import Counter
    n = len(runs)
    if min_votes is None:
        min_votes = n // 2 + 1 if n else 0
    raw = _majority_clusters(runs, iou_thresh=iou_thresh)
    clusters = []
    kept, dropped = [], []
    for i, c in enumerate(raw):
        votes = len(c["runs"])
        members = [e for _, e in c["members"]]
        boxes = [m["box"] for m in members if m.get("box") and len(m["box"]) == 4]
        if boxes:
            box = [int(sorted(b[j] for b in boxes)[len(boxes) // 2]) for j in range(4)]
        else:
            box = members[0].get("box")
        label = Counter(m.get("label") or "" for m in members).most_common(1)[0][0]
        rep = members[0]
        sig = _elem_sig(rep)
        is_kept = votes >= min_votes
        item = {
            "id": i,
            "votes": votes,
            "kept": is_kept,
            "sig": {"region": sig[0], "type": sig[1], "option": sig[2], "unit": sig[3]},
            "label": label,
            "median_box": box,
            "run_ids": sorted(c["runs"]),
            "members": [
                {"run": ri, "label": e.get("label"), "type": e.get("type"),
                 "option": e.get("option"), "unit": e.get("unit"),
                 "region": e.get("region"), "box": e.get("box"), "key": e.get("key")}
                for ri, e in c["members"]
            ],
        }
        clusters.append(item)
        el = {
            "region": rep.get("region"), "type": rep.get("type"),
            "option": rep.get("option"), "unit": rep.get("unit"),
            "label": label, "box": box, "_votes": votes, "_cluster": i,
        }
        (kept if is_kept else dropped).append(el)
    clusters.sort(key=lambda x: (-x["votes"], x["sig"]["region"] or 0, x["id"]))
    return {
        "kept": assign_keys(kept),
        "dropped": dropped,
        "clusters": clusters,
        "min_votes": min_votes,
        "iou_thresh": iou_thresh,
        "n_runs": n,
    }


def ground(marked_jpg_b64, n_atoms, model=MODEL, temperature=1.0, retries=3, system=None,
           reasoning_effort="low", focus_region=None, assign=True,
           focus_style="highlight", image_first=False, focus_ids=None, region_enum=None):
    """SoM 마킹된 JPEG(base64) → {"elements":[...]}.
    Structured Output(json_schema)로 type/box 형태 고정. key는 assign_keys로 코드 부여.
    reasoning_effort: OpenRouter reasoning.effort (= Gemini thinking_level) low|medium|high.
    focus_region: 단일 영역만 추출(실험). assign=False면 key 미부여.
    focus_style: "highlight"(픽셀 강조 가정) | "text"(같은 SoM 이미지, 번호로만 지정).
    image_first: True면 user content를 [이미지, 텍스트] 순 — 고정 이미지+가변 텍스트 캐시에 유리.
    파싱 실패 시 재시도, 최종 실패면 빈 리스트. system: 프롬프트(기본 SYS_V2).
    반환에 _meta(usage)가 붙을 수 있음(파이프라인은 elements만 사용)."""
    model = model or MODEL                          # 호출자가 model=None을 명시해도 기본 모델로(빈 model=API 400 방지)
    sys_prompt = SYS_V2 if system is None else system
    head = f"번호 0~{n_atoms-1}. key 필드는 넣지 말 것.\n"
    if focus_ids is not None:
        head = (
            f"번호 목록 {list(focus_ids)}에 해당하는 영역만 다룬다. 이 목록에 없는 번호의 영역은 "
            f"절대 요소로 만들지 않는다. 각 요소의 region은 반드시 이 목록 {list(focus_ids)} 중 하나여야 한다. "
            f"box는 그 번호 영역 사각형 안에만.\n"
            f"이 목록의 각 영역이 **키(라벨·제목·항목명)** 인지 "
            f"**값(작성자가 채우는 입력칸·선택지)** 인지 판단해, 값 성격일 때만 입력 요소를 만든다.\n"
            + head
        )
    elif focus_region is not None:
        if focus_style == "text":
            head = (
                f"이미지의 번호 박스 중 **영역 {focus_region}만** 다룬다. "
                f"다른 번호 영역의 입력은 만들지 않는다. 모든 요소의 region은 {focus_region}. "
                f"box는 영역 {focus_region} 사각형 안에만.\n"
                f"먼저 전체 서식 맥락에서 영역 {focus_region}이 **키(라벨·제목·항목명)** 인지 "
                f"**값(작성자가 채우는 입력칸·선택지)** 인지 판단한다. "
                f"키 성격이면 요소를 만들지 않는다. 값 성격일 때만 입력 요소를 만든다.\n"
                + head
            )
        else:
            head = (
                f"하이라이트된 영역 {focus_region}만 다룬다. 다른 영역은 요소로 만들지 않는다. "
                f"모든 요소의 region은 {focus_region}. box는 하이라이트 영역 안에만.\n"
                f"먼저 전체 서식 맥락에서 이 영역이 **키(라벨·제목·항목명)** 인지 "
                f"**값(작성자가 채우는 입력칸·선택지)** 인지 판단한다. "
                f"키 성격이면 요소를 만들지 않는다. 값 성격일 때만 입력 요소를 만든다.\n"
                + head
            )
    resp_schema = RESPONSE_SCHEMA
    if region_enum is not None:                            # region을 이 블록 번호들로 강제(생성 시점 봉쇄)
        import copy
        resp_schema = copy.deepcopy(RESPONSE_SCHEMA)
        resp_schema["properties"]["elements"]["items"]["properties"]["region"] = {
            "type": "integer", "enum": sorted(set(int(i) for i in region_enum))}
    img_part = {"type": "image_url",
                "image_url": {"url": "data:image/jpeg;base64," + marked_jpg_b64}}
    txt_part = {"type": "text", "text": head if image_first else (head + "[이미지]↓")}
    us = [img_part, txt_part] if image_first else [txt_part, img_part]
    body_obj = {
        "model": model,
        "temperature": temperature,
        "max_tokens": 20000,
        "reasoning": {"effort": reasoning_effort},
        "messages": [{"role": "system", "content": sys_prompt},
                     {"role": "user", "content": us}],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "form_elements",
                "strict": True,
                "schema": resp_schema,
            },
        },
    }
    body = json.dumps(body_obj).encode()
    last_err = None
    for attempt in range(retries):
        req = urllib.request.Request(API, data=body, headers={
            "Authorization": f"Bearer {load_key()}", "Content-Type": "application/json"})
        try:
            resp = json.load(urllib.request.urlopen(req, timeout=400))
            out = resp["choices"][0]["message"]["content"]
            els = _parse_elements(out)
            if assign:
                els = assign_keys(els)
            return {"elements": els, "_meta": {"usage": resp.get("usage"), "effort": reasoning_effort}}
        except Exception as ex:
            last_err = ex
            if attempt < retries - 1:
                time.sleep(1.5 * (attempt + 1))    # 레이트리밋·일시장애 백오프(병렬 콜에서 429 회복)
    try:
        body_obj["response_format"] = {"type": "json_object"}
        body = json.dumps(body_obj).encode()
        req = urllib.request.Request(API, data=body, headers={
            "Authorization": f"Bearer {load_key()}", "Content-Type": "application/json"})
        resp = json.load(urllib.request.urlopen(req, timeout=400))
        out = resp["choices"][0]["message"]["content"]
        els = _parse_elements(out)
        if assign:
            els = assign_keys(els)
        return {"elements": els,
                "_meta": {"usage": resp.get("usage"), "effort": reasoning_effort, "fallback": "json_object"}}
    except Exception:
        pass
    if last_err:
        pass
    return {"elements": []}


MERGE_SYS = """# 역할
같은 서식에 대해 N회 추출된 입력 요소 후보를 합친다. 이미지 좌표(box)와 type·label·option·unit을 보고
"같은 입력 슬롯"끼리 묶은 뒤, 충분히 반복해서 나온 것만 채택한다.

# 규칙
1. 입력에 없는 type·option·칸을 새로 만들지 않는다.
2. 같은 슬롯: 의미가 같고(예: 같은 체크 보기, 같은 기간 행의 시작년), box가 가까운 것.
   label 표기가 달라도(급/자격증 급수) 같은 칸이면 하나로 묶는다.
3. date/기간: 행마다 시작년·시작월·종료년·종료월처럼 실제로 반복된 칸 수만큼 유지. 합쳐 버리지 말 것.
4. checkbox: option(보기 값)이 같으면 같은 슬롯. box 미세 차이는 무시.
5. consent는 불안정하면 빼도 된다. signature·email·phone·image는 보통 1개.
6. 각 채택 항목의 sources에는 그 슬롯에 기여한 후보 id만 넣는다. 최소 min_votes개 이상.
7. box는 직접 새로 그리지 말고, sources로 가리킨 후보들의 좌표를 쓰게 둔다(서버가 median).

# 출력
JSON만. accepted 배열.
"""

# 병합 심판: sources = ["r0_e3", ...] 형태 (run_index + element_index)
MERGE_ACCEPTED_SCHEMA = {
    "type": "object",
    "properties": {
        "region": {"type": "integer"},
        "label": {"type": "string"},
        "type": {"type": "string", "enum": list(TYPES)},
        "option": {"type": ["string", "null"]},
        "unit": {"type": ["string", "null"]},
        "sources": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
            "description": "기여 후보 id. 예 r0_e12 = run0의 12번 요소",
        },
    },
    "required": ["region", "label", "type", "sources"],
    "additionalProperties": False,
}

MERGE_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "accepted": {"type": "array", "items": MERGE_ACCEPTED_SCHEMA},
    },
    "required": ["accepted"],
    "additionalProperties": False,
}


def _pack_merge_candidates(runs):
    """runs → id맵 + LLM에 줄 요약 리스트."""
    id_map = {}  # "r{ri}_e{ei}" -> element
    packed = []
    for ri, els in enumerate(runs):
        for ei, e in enumerate(els):
            cid = f"r{ri}_e{ei}"
            id_map[cid] = dict(e)
            packed.append({
                "id": cid,
                "run": ri,
                "region": e.get("region"),
                "type": e.get("type"),
                "label": e.get("label"),
                "option": e.get("option"),
                "unit": e.get("unit"),
                "box": e.get("box"),
            })
    return id_map, packed


def _median_box(boxes):
    boxes = [b for b in boxes if b and len(b) == 4]
    if not boxes:
        return None
    return [int(sorted(b[i] for b in boxes)[len(boxes) // 2]) for i in range(4)]


def ground_majority(marked_jpg_b64, n_atoms, n_runs=5, min_votes=None, reasoning_effort="low",
                    **kwargs):
    """ground()를 n_runs회 호출 후 majority_merge. kwargs는 ground()에 전달."""
    if min_votes is None:
        min_votes = n_runs // 2 + 1
    run_lists, metas = [], []
    for i in range(n_runs):
        res = ground(marked_jpg_b64, n_atoms, reasoning_effort=reasoning_effort, **kwargs)
        run_lists.append(res.get("elements") or [])
        metas.append(res.get("_meta"))
    merged = majority_merge(run_lists, min_votes=min_votes)
    return {
        "elements": merged,
        "_meta": {
            "mode": "majority",
            "n_runs": n_runs,
            "min_votes": min_votes,
            "per_run_counts": [len(r) for r in run_lists],
            "merged_count": len(merged),
            "runs": metas,
            "effort": reasoning_effort,
        },
        "_runs": run_lists,
    }


def merge_by_llm(runs, min_votes=None, model=MODEL, temperature=1.0, reasoning_effort="low",
                 retries=3):
    """N회 ground 결과 → LLM 심판 1콜로 채택 목록.
    sources id로 median box를 코드가 확정. min_votes 미만은 코드가 drop.
    → {"elements":[...], "_meta":{...}}"""
    n = len(runs)
    if n == 0:
        return {"elements": [], "_meta": {"mode": "llm_merge", "error": "empty"}}
    if min_votes is None:
        min_votes = n // 2 + 1

    id_map, packed = _pack_merge_candidates(runs)
    user_text = (
        f"runs={n}, min_votes={min_votes}. "
        f"후보는 아래 JSON. sources에는 후보 id만 쓰고, min_votes개 이상만 채택.\n"
        + json.dumps(packed, ensure_ascii=False)
    )
    body_obj = {
        "model": model,
        "temperature": temperature,
        "max_tokens": 20000,
        "reasoning": {"effort": reasoning_effort},
        "messages": [
            {"role": "system", "content": MERGE_SYS},
            {"role": "user", "content": user_text},
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "merge_accepted",
                "strict": True,
                "schema": MERGE_RESPONSE_SCHEMA,
            },
        },
    }
    body = json.dumps(body_obj).encode()
    raw_accepted = None
    usage = None
    last_err = None
    for _ in range(retries):
        req = urllib.request.Request(API, data=body, headers={
            "Authorization": f"Bearer {load_key()}", "Content-Type": "application/json"})
        try:
            resp = json.load(urllib.request.urlopen(req, timeout=400))
            usage = resp.get("usage")
            out = resp["choices"][0]["message"]["content"]
            if isinstance(out, list):
                out = "".join(
                    p.get("text", "") if isinstance(p, dict) else str(p) for p in out
                )
            text = (out or "").replace("```json", "").replace("```", "")
            data = json.loads(re.search(r"\{.*\}", text, re.S).group(0))
            raw_accepted = data.get("accepted") or []
            break
        except Exception as ex:
            last_err = ex
    if raw_accepted is None:
        return {"elements": [], "_meta": {"mode": "llm_merge", "error": str(last_err), "usage": usage}}

    out = []
    dropped_short = 0
    dropped_bad = 0
    for a in raw_accepted:
        sources = [s for s in (a.get("sources") or []) if s in id_map]
        runs_hit = set()
        boxes = []
        for s in sources:
            ri = int(s.split("_")[0][1:])
            runs_hit.add(ri)
            boxes.append(id_map[s].get("box"))
        if len(runs_hit) < min_votes:
            dropped_short += 1
            continue
        box = _median_box(boxes)
        if not box:
            dropped_bad += 1
            continue
        t = a.get("type") if a.get("type") in TYPES else "text"
        out.append({
            "region": a.get("region"),
            "label": a.get("label") or "",
            "type": t,
            "option": a.get("option"),
            "unit": a.get("unit"),
            "box": box,
            "_votes": len(runs_hit),
            "_sources": sources,
        })
    els = assign_keys(out)
    return {
        "elements": els,
        "_meta": {
            "mode": "llm_merge",
            "n_runs": n,
            "min_votes": min_votes,
            "per_run_counts": [len(r) for r in runs],
            "llm_accepted_raw": len(raw_accepted),
            "merged_count": len(els),
            "dropped_short_votes": dropped_short,
            "dropped_bad_box": dropped_bad,
            "usage": usage,
            "effort": reasoning_effort,
        },
    }
