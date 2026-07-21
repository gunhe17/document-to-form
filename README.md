# image-to-form

서식 이미지 → **입력 필드 위치 추출** → `FormSchema` JSON.
(image-to-md 가 "표 내용"이라면, image-to-form 은 "작성해야 할 입력 위치".)

## 파이프라인

> 📊 **시각 자료**: [절차 다이어그램](_verify/pipeline_diagram.html) · [서식2호 단계별 입력→출력](_verify/stagedemo.html)

**원칙: LLM = 의미(무엇이 입력), CV/OCR = 위치(어디에).** LLM box는 대략치, 정밀 좌표는 carve 담당.

```
빈 서식 이미지
 ① 영역분리        region_segment.segment       CV·결정론 → atoms(셀/밴드)
 ② 대구획 검출      ground_focus.detect_focus_regions   LLM → 논리 대구획 번호목록
 ③ 구획별 그라운딩   mark_focus_multi + extract.ground   LLM → 구획 안 입력칸(type·region·대략box)
 ④ region-clamp    박스를 region 셀 안으로            결정론 후처리
 ⑤ carve          pipeline.place_elements           CV/OCR → 정밀 픽셀 위치
 → FormSchema
```

### 확정 조건 (실측 최적, 2026-07-21)
| 항목 | 값 | 근거 |
|---|---|---|
| temperature | **0.5** | raw 박스 위치정확 78%→100% (sweep 실측) |
| reasoning | **low** | 3.1 Pro 최저·최안정 (medium은 불일치↑) |
| 프롬프트 | **간결·직접** | gemini3devguide: Gemini 3은 장황한 프롬프트 과잉분석 |
| **대구획 스코핑** | 표=[머리행+본문]통째·제목별개·최대5~6 | 검출을 발산→수렴으로 = 일관성의 핵심 |
| 속도 | 대구획 병렬 17.9s (page 28.7s) | 지연 ∝ 출력토큰 → 쪼개 병렬 디코딩 |

- **① 영역분리**: `region_segment.segment(gray)` → atoms. 순수 CV, 결정론, 무료. box=`[ymin,xmin,ymax,xmax]` 0~1000 정규화 규약.
- **② 대구획 검출**: `detect_focus_regions(img)` — SoM 마킹 → **LLM 단독**이 논리 대구획 판단(`PARTITION_SYS`, temp 0.5·low). 표는 머리행+본문 통째, 제목 밴드는 별개, 반복행표는 통째, 최대 5~6.
- **③ 구획별 그라운딩**: 대구획마다 `mark_focus_multi`로 하이라이트 → `extract.ground(focus_ids=…)` 병렬 → 그 구획 안 입력칸 검출. region-filter로 구획 밖 드롭. `OPENROUTER_API_KEY` 필요(.env 자동탐색).
- **④ region-clamp**: LLM 박스가 영역 밖으로 샌 것을 region 셀 안으로 잘라 스팬·좌표복사 아티팩트 제거. 결정론·무료.
- **⑤ carve 배치 규칙**(`carve.place`, 타입별) — LLM 배정 region 안으로 OCR 탐색 제한:

| 타입 | 규칙 | 방법 |
|---|---|---|
| checkbox_group | `_cb_assign` | region 내 □ 검출 → LLM box 위치로 1:1 배정(중복 방지) |
| radio (보기) | `_fit_ink` | LLM box를 잉크에 조임 + 여백 |
| number/date/time + 단위 | `_ocr_anchor` | 단위글자(년/시…) EasyOCR 앵커 → 왼쪽 빈칸 |
| text/textarea | `_placeholder`·`_after_label`·`_fit_bounded`·`_fill_cell` | ink_frac로 분기 |
| signature | `_fit_ink` | "(서명 또는 인)" 문구 잉크 |
| image | `_snap_cell` | 표 셀 격자 스냅 |

### 연결 상태 (정직)
- ✅ **end-to-end**: `ground_focus.extract_form(img)` = ①영역분리 → ②대구획 → ③구획별 그라운딩(+④region-clamp). carve 전 raw 반환.
- ✅ **부품·동작**: `detect_focus_regions` · `ground_blocks`(구획별 하이라이트 그라운딩+clamp) · `place_elements`(carve)
- ✅ **전체 실행됨**: 18개 서식 raw 그라운딩 → `_verify/raw/*.json`(재사용, LLM 재호출 없이 carve/렌더 재생성). 러너 `_verify/run_all.py`(resumable·예산가드·불완전결과 미저장).
- 🔴 **미구현**: MD-context(전사모듈 제거) · `extract_form`+carve를 잇는 `to_form_schema` 자동화(수동으로는 `place_elements`로 재실행 가능)
- **레거시**: `pipeline.build()`는 page 모드(단일 그라운딩) — 여전히 동작, `_verify/render.py`가 사용

**OCR**: EasyOCR, 작은 글자 자동 업스케일. **일관성 원천은 파라미터보다 MD-context·대구획 스코핑**(검출→localization).

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
├── gemini3devguide.md                 Gemini 3 프롬프트 지침 (간결·직접)
├── examples/minimal.formschema.json   유효 예시
├── _verify/
│   ├── run_all.py                     확정 파이프라인 전체 서식 실행 → raw/*.json (resumable·예산가드)
│   ├── render_raw.py                  raw/*.json → 절차별 고화질 HTML (LLM 재호출 없음)
│   ├── raw/*.json                     ★재사용 raw 그라운딩 (carve 전, 서식당 1개)
│   ├── pipeline_raw.html              18서식 절차별 결과 리포트
│   ├── render.py                      page 모드 검증 HTML (→ step5_grounding.html)
│   ├── pipeline_diagram.html          절차 다이어그램 (연결 상태색)
│   └── stagedemo.html                 서식2호 단계별 입력→출력
└── core/
    ├── schema.py          FormSchema pydantic 계약 + validate_form_schema
    ├── region_segment.py  ① 영역분리 (CV)
    ├── ground_focus.py    ② 대구획 검출 + ③ 구획별 그라운딩 — detect_focus_regions() · mark_focus_multi() · PARTITION_SYS
    ├── extract.py         LLM 그라운딩 (Gemini) — SYS 프롬프트 · ground(focus_ids)
    ├── carve.py           ⑤ 위치 도구함 (CV/OCR) — place() · fit_ink · ocr_anchor · fit_placeholder …
    ├── budget.py          OpenRouter $20/일 가드
    ├── pipeline.py        오케스트레이션 — build()(page 레거시) · place_elements()(carve) · to_form_schema()
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
