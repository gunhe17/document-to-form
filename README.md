# image-to-form

서식 이미지 → **입력 필드 위치 추출** → `FormSchema` JSON.
(image-to-md 가 "표 내용"이라면, image-to-form 은 "작성해야 할 입력 위치".)

## 파이프라인 (`pipeline.build`)

```
서식 이미지
 ① 영역분리   region_segment.segment      CV(LLM 없음) → frames/tables/cells/bands
 ② SoM 마킹   원자(cells+bands)에 번호      LLM 입력 이미지
 ③ 그라운딩   extract.ground              Gemini 3.1 Pro → 무엇·타입·option·unit·대략box
 ④ 배치       carve.place + 후처리         CV/OCR → 정확한 픽셀 위치
 ⑤ 병합/조립  merge_unit_fields → FormSchema
```

**원칙: LLM = 의미(무엇이 입력), CV/OCR = 위치(어디에).** LLM box는 대략치, 정밀 좌표는 carve 담당.

- **③ 그라운딩**: `temperature=1.0`(Gemini 3 권장·낮추면 검출저하), reasoning effort=low. box=`[ymin,xmin,ymax,xmax]` 0~1000 정규화. `OPENROUTER_API_KEY` 필요(.env 자동탐색). SoM region마다 `region` 번호도 반환.

- **④ 배치 규칙**(`carve.place`, 타입별) — LLM이 배정한 **SoM region 안으로 OCR 탐색 제한**(이웃 오앵커 방지):

| 타입 | 규칙 | 방법 |
|---|---|---|
| checkbox_group | `_cb_assign` | region 내 □ 검출 → LLM box 위치로 1:1 배정(중복 방지) |
| radio (보기) | `_fit_ink` | LLM box(중심 3px 정확)를 잉크에 조임 + 여백. OCR 안 씀 |
| radio (L/R·좌/우) | `_lr_pair` | region 부위별 글자(CC) 좌우 배정 |
| number/date/time + 단위 | `_ocr_anchor` | 단위글자(년/시…) EasyOCR 앵커 → 왼쪽 빈칸(글자 열-잉크로 높이 측정) |
| text/textarea | `_placeholder`(○○○/□□□/△△△ 도형)·`_after_label`(콜론뒤 빈칸)·`_fit_bounded`(셀 빈칸)·`_fill_cell`(여러 줄) | ink_frac로 분기 |
| signature | `_fit_ink` | "(서명 또는 인)" 문구 잉크 |
| image | `_snap_cell` | 표 셀 격자 스냅 |

- **⑤ 전역 후처리**: `carve_inline`(촘촘한 날짜행) · `merge_unit_fields`(급 등 단위 분할 병합) · **행 높이통일**(같은 행·타입·수평근접 요소 높이 median 통일).
- **OCR**: 모두 EasyOCR, 작은 글자는 **자동 업스케일**(년→녀 오독·단일글자 미검출 감소).

## 출력 형태 — FormSchema (3 평면)

imomtae `FormTemplate.schema` 계약. 검증기 통과가 합격선.

```
{
  "pages":    [ PageDef, … ],        // 배경 이미지 + 크기
  "fields":   { "<key>": FieldDef },  // DATA 평면 — 의미 (LLM 담당)
  "elements": [ ElementDef, … ]       // PRESENTATION 평면 — 좌표 (CV/도구 담당)
}
```

**원칙: LLM = 의미(fields), 도구 = 좌표(elements).** 둘은 `field_refs`로 연결.

### PageDef
`{ no, image, w, h }` — 페이지 번호 · 배경 이미지(S3) · 폭 · 높이(px).

### FieldDef (의미, 좌표 없음)
```
{ type, label, required=false, options?, validation? }
```
- **type (16종 고정)**: `text textarea email phone number date time datetime
  select radio checkbox checkbox_group consent signature file image`
  (폼 *생성* 시엔 consent/checkbox 제외 14종만 쓰고 나머진 normalize)
- **options** (select/radio/checkbox_group): `[{ value, label, allow_text=false }]`
  — `allow_text`=자유입력(예: "기타( )").

### ElementDef (좌표, 정규화 0..1)
```
{ id, page, rect:[x,y,w,h], z=0, widget, field_refs:[…], option?, slot? }
```
- **rect**: 각 성분 0..1 (페이지 w/h 무관). 정확히 4개.
- **field_refs**: 연결 필드 키. 0=장식(heading/divider), n=n:n.
- **option**: 이 위젯이 바인딩하는 단일 선택지 value (체크박스/라디오 하나).
- **slot**: 위치 슬롯 (예: 주민번호 자릿수 인덱스).

### 교차 검증 (validator)
- element `id` 페이지 내 유일.
- 모든 `field_refs` 키는 `fields`에 존재.
- `option` 지정 시: 참조 필드가 `options`를 정의하고 그 value에 포함.

## 예시
[examples/minimal.formschema.json](examples/minimal.formschema.json) — text/date/radio/checkbox_group
+ 각 요소 element. 체크박스는 **옵션마다 element 1개**(`option` 명시).

## 구조
```
image-to-form/
├── README.md
├── price.md                           그라운딩 실측 비용
├── examples/minimal.formschema.json   유효 예시
├── _verify/render.py                  절차별 검증 HTML 리포트 (→ step5_grounding.html)
└── core/
    ├── schema.py          FormSchema pydantic 계약 + validate_form_schema
    ├── region_segment.py  ① 영역분리 (CV)
    ├── extract.py         ③ LLM 그라운딩 (Gemini) — SYS 프롬프트 + ground()
    ├── carve.py           ④ 위치 도구함 (CV/OCR) — place() · fit_ink · ocr_anchor · fit_placeholder · letter_runs …
    ├── pipeline.py        오케스트레이션 — build() · place_elements() · to_form_schema()
    └── __init__.py        공개 API
```

## 사용
```python
from core import pipeline, validate_form_schema

built = pipeline.build("form.png", cache_path="cache.json")   # ①~⑤ 실행 (LLM 응답 캐시)
fs = pipeline.to_form_schema(built, "form.png")               # → FormSchema dict
validate_form_schema(fs)                                       # 계약 검증 (실패 시 ValidationError)
```
검증 리포트: `python _verify/render.py [문서번호…]` → `_verify/step5_grounding.html` (①분리 ②SoM ③LLM원본 ④배치).
