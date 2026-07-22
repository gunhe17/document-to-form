"""image-to-form 공개 API — 서식 이미지 → FormSchema (end-to-end).

    from core import FormExtractor
    r = FormExtractor().extract("form.png")
    r.schema      # FormSchema dict (validate_form_schema 통과)
    r.elements    # carve된 배치 요소 (좌표)
    r.raw         # carve 전 LLM 그라운딩 (재사용: carve만 재실행 가능)
    r.validate()  # 계약 검증

확정 파이프라인 (대구획, 무-MD):
    ① region_segment.segment          영역분리 (CV·결정론)
    ② ground_focus.detect_focus_regions   대구획 검출 (LLM) + cap-split·반복테이블유지
    ③ ground_focus.ground_blocks          구획별 하이라이트 그라운딩 (LLM) + region-clamp
    ⑤ pipeline.place_elements             carve — 정밀 배치 (CV/OCR)
    → pipeline.to_form_schema             FormSchema

원칙: LLM = 의미(무엇·타입·option), CV/OCR = 위치(어디에).
내부 CV/LLM 상세는 region_segment·ground_focus·extract·carve·pipeline 에 있고
이 모듈(+ __init__)이 안정 공개 표면이다.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cv2

from . import ground_focus, pipeline
from .region_segment import segment
from .schema import validate_form_schema


@dataclass(frozen=True)
class FormResult:
    """서식 1장 추출 결과 (image-to-md PageResult 대응)."""
    schema: dict[str, Any]              # FormSchema (pages·fields·elements)
    elements: list[dict]               # carve된 배치 요소 (rect·type·option…)
    raw: list[dict]                    # carve 전 LLM 그라운딩 (재사용·재실행 무료)
    groups: list[list[int]]            # 대구획 (SoM atom 번호 묶음)
    page: dict[str, int]               # {"w","h"} px
    meta: dict[str, Any] = field(default_factory=dict)
    source_path: str | None = None

    def validate(self) -> "FormResult":
        """FormSchema 계약 검증 (실패 시 ValidationError). 자기 자신 반환."""
        validate_form_schema(self.schema)
        return self

    @property
    def field_count(self) -> int:
        return len(self.schema.get("fields", {}))


def extract_form_schema(image_path: str, *, temperature: float = 0.5,
                        model: str | None = None) -> FormResult:
    """서식 이미지 경로 → FormResult. 확정 대구획 파이프라인 end-to-end.

    temperature·model 은 그라운딩(LLM) 파라미터 (기본 = 실측 확정값 temp 0.5)."""
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"이미지를 읽을 수 없음: {image_path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    IH, IW = gray.shape
    S = segment(gray)                                              # ①
    g = ground_focus.extract_form(img, temperature=temperature, model=model)  # ②③ (대구획→소블록→그라운딩)
    elements = pipeline.place_elements(gray, S, g["elements"])     # ⑤ carve
    built = {"page": {"w": IW, "h": IH}, "elements": elements}
    schema = pipeline.to_form_schema(built, image_name=str(image_path))
    return FormResult(schema=schema, elements=elements, raw=g["elements"],
                      groups=g["groups"], page={"w": IW, "h": IH},
                      meta=g.get("_meta", {}), source_path=str(image_path))


def carve_from_raw(image_path: str, raw_elements: list[dict]) -> FormResult:
    """저장해 둔 raw 그라운딩(재사용)으로 carve~FormSchema만 재실행 (LLM 재호출 없음).
    run_all.py 가 저장한 _verify/raw/*.json 의 'elements' 를 그대로 넣으면 된다."""
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"이미지를 읽을 수 없음: {image_path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    IH, IW = gray.shape
    S = segment(gray)
    elements = pipeline.place_elements(gray, S, raw_elements)
    built = {"page": {"w": IW, "h": IH}, "elements": elements}
    schema = pipeline.to_form_schema(built, image_name=str(image_path))
    return FormResult(schema=schema, elements=elements, raw=raw_elements,
                      groups=[], page={"w": IW, "h": IH}, source_path=str(image_path))


class FormExtractor:
    """서식 이미지 → FormSchema 변환기 (image-to-md Converter 대응).

        FormExtractor().extract("form.png").validate()
    """

    def __init__(self, *, temperature: float = 0.5, model: str | None = None):
        self.temperature = temperature
        self.model = model

    def extract(self, image_path: str) -> FormResult:
        """LLM 그라운딩 포함 end-to-end."""
        return extract_form_schema(image_path, temperature=self.temperature, model=self.model)

    def carve_from_raw(self, image_path: str, raw_elements: list[dict]) -> FormResult:
        """재사용 raw 로 carve~FormSchema 만 (LLM 없음)."""
        return carve_from_raw(image_path, raw_elements)
