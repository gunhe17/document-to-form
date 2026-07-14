"""③ LLM 그라운딩 — SoM 번호 이미지 → 입력 요소 상세 + 위치(box).

Gemini 3.1 Pro 비전. box=[ymin,xmin,ymax,xmax] 이미지 정규화 0~1000 (Gemini 그라운딩 규약).
temperature=1.0 (Gemini 3 권장; 낮추면 검출 저하·looping). reasoning effort=low.

LLM = 의미(무엇이 입력·타입·option·unit) + 대략 위치. 정확한 위치는 carve 가 잡는다.
"""
import os, json, re, urllib.request
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
 "- 라벨·제목·안내문·표머리글·수신처('○○기관 귀중')·단위글자·괄호기호는 입력이 아니다.\n"
 "- 표 안 모든 빈 셀, 모든 □/○ 선택지, 반복행을 하나도 빠뜨리지 말 것.\n"
 "- 이미지를 신중히 보고 판단하라.")


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


def ground(marked_jpg_b64, n_atoms, model=MODEL, temperature=1.0, retries=3):
    """SoM 마킹된 JPEG(base64) → {"elements":[...]}. 파싱 실패 시 재시도, 최종 실패면 빈 리스트."""
    us = [{"type": "text", "text": f"번호 0~{n_atoms-1}.\n[이미지]↓"},
          {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + marked_jpg_b64}}]
    body = json.dumps({"model": model, "temperature": temperature, "max_tokens": 20000,
                       "reasoning": {"effort": "low"},
                       "messages": [{"role": "system", "content": SYS}, {"role": "user", "content": us}]}).encode()
    for _ in range(retries):
        req = urllib.request.Request(API, data=body, headers={
            "Authorization": f"Bearer {load_key()}", "Content-Type": "application/json"})
        try:
            out = json.load(urllib.request.urlopen(req, timeout=400))["choices"][0]["message"]["content"]
            return json.loads(re.search(r"\{.*\}", out.replace("```", ""), re.S).group(0))
        except Exception:
            pass
    return {"elements": []}
