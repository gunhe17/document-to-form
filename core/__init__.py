"""image-to-form core — 서식 이미지 → 입력 요소 위치 → FormSchema.

파이프라인 (pipeline.build):
  ① region_segment.segment  영역분리(CV)
  ② SoM 마킹                원자에 번호
  ③ extract.ground          LLM 그라운딩(의미·타입·대략위치)
  ④ carve.place + 후처리     정확한 위치(CV/OCR)
  ⑤ merge → to_form_schema  FormSchema

원칙: LLM=의미, CV/OCR=위치.
"""
from .schema import (
    FormSchema, PageDef, FieldDef, OptionDef, ElementDef,
    validate_form_schema, FIELD_TYPES,
)
from . import carve, extract, pipeline
from .region_segment import segment

__all__ = ["FormSchema", "PageDef", "FieldDef", "OptionDef", "ElementDef",
           "validate_form_schema", "FIELD_TYPES",
           "carve", "extract", "pipeline", "segment"]
