"""image-to-form — 서식 이미지 → 입력필드 위치 → FormSchema.

안정 공개 표면 (서버/CLI/도구는 이 패키지만 import):

    from core import FormExtractor
    r = FormExtractor().extract("form.png")   # 이미지 → FormSchema
    r.validate()                              # 계약 검증

확정 파이프라인 (대구획, 무-MD): region_segment → detect_focus_regions(대구획·cap-split)
→ ground_blocks(하이라이트 그라운딩) → carve(place_elements) → FormSchema.
원칙: LLM = 의미(무엇·타입·option), CV/OCR = 위치(어디에).

내부 CV/LLM 상세(region_segment·ground_focus·extract·carve·pipeline)는 계약이 아니다.
"""
# 공개 API — 변환 진입점
from .convert import (
    FormExtractor, FormResult, extract_form_schema, carve_from_raw,
)
# 공개 API — FormSchema 계약
from .schema import (
    FormSchema, PageDef, FieldDef, OptionDef, ElementDef,
    validate_form_schema, FIELD_TYPES,
)
# 하위 모듈 (고급 사용·검증 도구용)
from . import ground_focus, pipeline, extract, carve
from .region_segment import segment

__all__ = [
    # 진입점
    "FormExtractor", "FormResult", "extract_form_schema", "carve_from_raw",
    # FormSchema 계약
    "FormSchema", "PageDef", "FieldDef", "OptionDef", "ElementDef",
    "validate_form_schema", "FIELD_TYPES",
    # 하위 모듈
    "ground_focus", "pipeline", "extract", "carve", "segment",
]
