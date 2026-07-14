"""FormSchema — image-to-form 출력 계약 (Pydantic v2).

★ imomtae `FormTemplate.schema` JSONB 계약의 vendored 사본.
  원본: imomtae-v3/TF/saas-center-platform/apps/api/app/modules/form/template/form_schema.py
  원본이 바뀌면 이 사본도 갱신할 것 (검증 통과가 합격선).

3개 평면(plane) + 관계(relations):
  pages       : 배경 이미지(스캔 서식)와 크기
  fields      : DATA 평면       — 의미 키(semantic key) → 필드 정의       [LLM 담당]
  elements    : PRESENTATION 평면 — 정규화 좌표(0..1)에 배치된 위젯        [도구/CV 담당]
  constraints : RELATION 평면    — 크로스필드 관계(범위유효성/계산)         [LLM 담당]

elements ↔ fields 는 ``field_refs`` 로 0~n:n 연결
  (0 = heading/divider 같은 장식, n = 한 위젯이 여러 필드 바인딩).
한 필드가 여러 박스에 걸치면(년/월/일, 주민번호 자릿수) element.slot 으로 분할.

값 관계는 flat fields 를 유지한 채 3가지로 표현(진짜 중첩 대신 태그/조건/규칙):
  FieldDef.group        : 논리 묶음 (예 "기간"=시작/종료, "활동시간"=유형/요일/시각)
  FieldDef.visible_when : 조건부 종속 (예 요일은 avail_type=="regular" 일 때만)
  FormSchema.constraints: 크로스필드 (lte/gte 범위유효성, sum/diff 계산)
원칙: LLM은 fields·constraints(의미/관계)만, CV/도구는 elements(좌표)만.
"""
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

# 필드 타입 16종 (계약 Literal 고정, 확장 금지).
# step6(폼 생성)에선 consent/checkbox 를 제외한 14종만 '사용'하고 결정론 normalize 로 변환.
FIELD_TYPES = (
    "text", "textarea", "email", "phone", "number", "date", "time", "datetime",
    "select", "radio", "checkbox", "checkbox_group", "consent", "signature",
    "file", "image",
)


class PageDef(BaseModel):
    """양식 페이지 (스캔 서식 배경)."""
    model_config = ConfigDict(extra="forbid")
    no: int = Field(..., description="페이지 번호")
    image: str = Field(..., description="배경 이미지 (S3 path/url)")
    w: int = Field(..., description="페이지 폭 (px)")
    h: int = Field(..., description="페이지 높이 (px)")


class OptionDef(BaseModel):
    """선택지 정의 (select/radio/checkbox_group 등)."""
    model_config = ConfigDict(extra="forbid")
    value: str = Field(..., description="저장 값")
    label: str = Field(..., description="표시 라벨")
    allow_text: bool = Field(default=False, description='자유 입력 허용 (예: "기타( )")')


class Condition(BaseModel):
    """단일 조건식 — visible_when 등. (다른 필드 값에 대한 판정)"""
    model_config = ConfigDict(extra="forbid")
    field: str = Field(..., description="참조 필드 키")
    op: Literal["eq", "ne", "in", "filled", "empty", "gt", "lt", "gte", "lte"] = Field(
        ..., description="판정 연산자 (filled/empty 는 value 불필요)")
    value: Any = Field(default=None, description="비교값 (eq/in/gt… 에서 사용)")


class FieldDef(BaseModel):
    """필드 정의 (DATA 평면 — 의미 키). 좌표 없음."""
    model_config = ConfigDict(extra="forbid")
    type: Literal[
        "text", "textarea", "email", "phone", "number", "date", "time", "datetime",
        "select", "radio", "checkbox", "checkbox_group", "consent", "signature",
        "file", "image",
    ] = Field(..., description="필드 타입 (14종)")
    label: str = Field(..., description="필드 라벨")
    required: bool = Field(default=False, description="필수 여부")
    options: list[OptionDef] | None = Field(default=None, description="선택지")
    validation: dict[str, Any] | None = Field(default=None, description="검증 규칙(min/max/pattern 등)")
    group: str | None = Field(default=None, description="논리 묶음/중첩 태그 (같은 값=한 그룹)")
    visible_when: Condition | None = Field(default=None, description="조건부 표시(종속). None=항상")


class ElementDef(BaseModel):
    """엘리먼트 정의 (PRESENTATION 평면 — 배치된 위젯). rect = 정규화 0..1."""
    model_config = ConfigDict(extra="forbid")
    id: str = Field(..., description="엘리먼트 ID (페이지 내 고유)")
    page: int = Field(..., description="배치 페이지 번호")
    rect: list[float] = Field(..., description="정규화 좌표 [x, y, w, h] (각 0..1)")
    z: int = Field(default=0, description="z-index")
    widget: str = Field(..., description="렌더 위젯 종류")
    field_refs: list[str] = Field(default_factory=list, description="연결 필드 키(0=장식, n=n:n)")
    option: str | None = Field(default=None, description="바인딩 단일 선택지 value(체크박스 하나)")
    slot: int | None = Field(default=None, description="위치 슬롯(예: 주민번호 자릿수)")

    @model_validator(mode="after")
    def _validate_rect(self) -> "ElementDef":
        if len(self.rect) != 4:
            raise ValueError(f"element '{self.id}'.rect must be [x,y,w,h], got {len(self.rect)}")
        for i, c in enumerate(self.rect):
            if not (0.0 <= c <= 1.0):
                raise ValueError(f"element '{self.id}'.rect[{i}]={c} out of [0,1]")
        return self


class Constraint(BaseModel):
    """크로스필드 관계 (RELATION 평면). 비교(범위유효성) 또는 계산(파생)."""
    model_config = ConfigDict(extra="forbid")
    type: Literal[
        "lte", "gte", "lt", "gt", "eq", "ne",   # 비교: fields[0] op fields[1] (범위유효성)
        "sum", "diff", "product",               # 계산: target = op(fields...)
    ] = Field(..., description="관계 종류")
    fields: list[str] = Field(..., description="관여 필드 키 (순서 의미: 비교는 [a,b])")
    target: str | None = Field(default=None, description="계산 결과 저장 필드 (sum/diff/product 필수)")
    message: str | None = Field(default=None, description="위반 시 메시지")

    @model_validator(mode="after")
    def _shape(self) -> "Constraint":
        compute = self.type in ("sum", "diff", "product")
        if len(self.fields) < 2:
            raise ValueError(f"constraint '{self.type}' needs >=2 fields")
        if compute and self.target is None:
            raise ValueError(f"compute constraint '{self.type}' requires target")
        if not compute and self.target is not None:
            raise ValueError(f"compare constraint '{self.type}' must not set target")
        return self


class FormSchema(BaseModel):
    """양식 스키마 루트 (FormTemplate.schema JSONB) — image-to-form 최종 출력."""
    model_config = ConfigDict(extra="forbid")
    pages: list[PageDef] = Field(default_factory=list)
    fields: dict[str, FieldDef] = Field(default_factory=dict)
    elements: list[ElementDef] = Field(default_factory=list)
    constraints: list[Constraint] = Field(default_factory=list, description="크로스필드 관계")

    @model_validator(mode="after")
    def _validate_cross_refs(self) -> "FormSchema":
        seen: set[str] = set()
        for el in self.elements:
            if el.id in seen:
                raise ValueError(f"duplicate element id: '{el.id}'")
            seen.add(el.id)
        for el in self.elements:
            for ref in el.field_refs:
                if ref not in self.fields:
                    raise ValueError(f"element '{el.id}' refs unknown field '{ref}'")
            if el.option is not None:
                if not el.field_refs:
                    raise ValueError(f"element '{el.id}' sets option but has no field_refs")
                for ref in el.field_refs:
                    field = self.fields[ref]
                    if not field.options:
                        raise ValueError(f"element '{el.id}' option but field '{ref}' has no options")
                    if el.option not in {o.value for o in field.options}:
                        raise ValueError(f"element '{el.id}'.option='{el.option}' not in field '{ref}' options")
        # 관계 참조 무결성: visible_when / constraints 가 실재 필드만 가리키게
        for key, fdef in self.fields.items():
            if fdef.visible_when and fdef.visible_when.field not in self.fields:
                raise ValueError(f"field '{key}'.visible_when refs unknown field '{fdef.visible_when.field}'")
        for i, con in enumerate(self.constraints):
            for ref in con.fields + ([con.target] if con.target else []):
                if ref not in self.fields:
                    raise ValueError(f"constraint[{i}] ('{con.type}') refs unknown field '{ref}'")
        return self


def validate_form_schema(data: dict) -> FormSchema:
    """dict → 파싱+검증. 실패 시 pydantic ValidationError."""
    return FormSchema.model_validate(data)
