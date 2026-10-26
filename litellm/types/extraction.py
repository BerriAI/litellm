from collections.abc import Mapping, Sequence
from typing import Annotated, Final, Literal, TypeAlias

from pydantic import (
    ConfigDict,
    Field,
    JsonValue,
    PrivateAttr,
    StrictBool,
    StrictFloat,
    StrictInt,
    StrictStr,
    model_validator,
    with_config,
)
from pydantic.fields import FieldInfo
from typing_extensions import ReadOnly, Required, Self, TypedDict

from litellm.types.llms.base import LiteLLMPydanticObjectBase

ExtractionFieldType: TypeAlias = Literal[
    "text", "number", "date", "time", "datetime", "duration", "list", "estimate", "forecast"
]
ExtractionFieldName: TypeAlias = Annotated[str, Field(pattern=r"^[A-Za-z_][A-Za-z0-9_.\-]{0,63}$")]
JsonNumber: TypeAlias = StrictInt | StrictFloat
ExtractionBound: TypeAlias = JsonNumber | StrictStr
EXTRACTION_STATE_REWRITTEN_METADATA_KEY: Final = "extraction_state_rewritten_by_guardrail"


def _is_none(value: object) -> bool:
    return value is None


_OMITTED_WHEN_NONE: Final = FieldInfo.from_field(exclude_if=_is_none)
OptionalJson: TypeAlias = Annotated[JsonValue, _OMITTED_WHEN_NONE]
OptionalFloat: TypeAlias = Annotated[float | None, _OMITTED_WHEN_NONE]
OptionalInt: TypeAlias = Annotated[int | None, _OMITTED_WHEN_NONE]


class _ClairicObject(LiteLLMPydanticObjectBase):
    model_config = ConfigDict(extra="allow", frozen=True)


class ExtractionField(_ClairicObject):
    type: Annotated[ExtractionFieldType, Field(description="The answer type for this field")]
    instructions: Annotated[str | None, Field(description="What the field should contain")] = None
    unit: Annotated[str | None, Field(description="Unit label. For durations, seconds through years")] = None
    integer: Annotated[StrictBool | None, Field(description="Numbers and durations only")] = None
    decimals: Annotated[JsonNumber | None, Field(description="Numbers, durations and number lists only, 0 to 6")] = None
    items: Annotated[
        Literal["text", "number", "date"] | None, Field(description="Item type, required for list fields")
    ] = None
    min: Annotated[ExtractionBound | None, Field(description="Estimate and forecast lower bound")] = None
    max: Annotated[ExtractionBound | None, Field(description="Estimate and forecast upper bound")] = None
    step: Annotated[ExtractionBound | None, Field(description="Estimate grid step, or month or year")] = None
    scale: Literal["linear", "log"] | None = None
    null_threshold: Annotated[
        JsonNumber | None, Field(description="Probability of null above which the value is null")
    ] = None
    horizon: Annotated[JsonNumber | None, Field(description="Forecast only, number of steps")] = None
    interval: Literal["step", "minute", "hour", "day", "week", "month", "quarter", "year"] | None = None

    @model_validator(mode="after")
    def _list_fields_name_their_item_type(self) -> Self:
        if self.type == "list" and self.items is None:
            raise ValueError("list fields need items set to text, number or date")
        return self


class ExtractionBase64Image(_ClairicObject):
    content_type: Annotated[str, Field(description="The image media type, such as image/png")]
    base64: Annotated[str, Field(description="Base64-encoded image bytes")]


ExtractionImage: TypeAlias = Annotated[str, Field(pattern=r"^[Dd][Aa][Tt][Aa]:")] | ExtractionBase64Image


class ExtractionOptions(_ClairicObject):
    include_quantiles: Annotated[StrictBool | None, Field(description="Add 99 percentiles to estimates")] = None
    return_state_text: Annotated[StrictBool | None, Field(description="Return the text that spans index into")] = None


class ExtractionRequestBody(_ClairicObject):
    state: Annotated[
        str,
        Field(description="The text to read. Must be non-empty unless images are sent. Text answers are spans of it"),
    ] = ""
    fields: Annotated[
        Mapping[ExtractionFieldName, ExtractionField],
        Field(min_length=1, description="Field name to typed field spec. Each name becomes a key in the answers"),
    ]
    images: Annotated[
        tuple[ExtractionImage, ...] | None,
        Field(max_length=4, description="Up to 4 images as data URLs or content_type plus base64, no remote URLs"),
    ] = None
    reference_date: Annotated[str | None, Field(description="ISO date that anchors dates written without a year")] = (
        None
    )
    options: ExtractionOptions | None = None


class ExtractionRequest(ExtractionRequestBody):
    model: Annotated[str, Field(description="An extraction model from the proxy model_list")]


class ExtractionAnswer(_ClairicObject):
    type: str
    value: Annotated[JsonValue, Field(description="The answer, null when the input does not determine it")]
    guess: OptionalJson = None
    p_null: OptionalFloat = None
    confidence: OptionalFloat = None
    span: Annotated[tuple[int, ...] | None, _OMITTED_WHEN_NONE] = None
    spans: Annotated[tuple[tuple[int, ...], ...] | None, _OMITTED_WHEN_NONE] = None
    mean: OptionalJson = None
    median: OptionalJson = None
    p10: OptionalJson = None
    p90: OptionalJson = None
    p_below: OptionalFloat = None
    p_above: OptionalFloat = None
    quantiles: Annotated[tuple[JsonValue, ...] | None, _OMITTED_WHEN_NONE] = None
    steps: Annotated[tuple[JsonValue, ...] | None, _OMITTED_WHEN_NONE] = None


class ExtractionUsage(_ClairicObject):
    input_tokens: int
    output_tokens: int
    state_tokens: OptionalInt = None
    image_tokens: OptionalInt = None
    schema_tokens: OptionalInt = None
    passes: OptionalInt = None


class ExtractionResponse(_ClairicObject):
    model: str
    answers: Mapping[str, ExtractionAnswer]
    truncated: bool
    state_text: Annotated[str | None, _OMITTED_WHEN_NONE] = None
    usage: ExtractionUsage

    _hidden_params: dict[str, object] = PrivateAttr(  # mutable-ok: @client and the proxy write cost into it in place
        default_factory=dict
    )

    @property
    def hidden_params(self) -> dict[str, object]:  # mutable-ok: API requires mutation
        return self._hidden_params

    def set_hidden_params(self, params: Mapping[str, object]) -> None:
        self._hidden_params.update(params)


@with_config(ConfigDict(extra="allow"))
class ExtractionCallParams(TypedDict, total=False):
    model: Required[ReadOnly[str]]
    state: Required[ReadOnly[str]]
    fields: Required[ReadOnly[Mapping[str, Mapping[str, object]]]]
    images: ReadOnly[Sequence[str | Mapping[str, object]] | None]
    reference_date: ReadOnly[str | None]
    api_key: ReadOnly[str | None]
    api_base: ReadOnly[str | None]
    timeout: ReadOnly[float | None]
    custom_llm_provider: ReadOnly[str | None]
    extra_headers: ReadOnly[Mapping[str, str] | None]
