from collections.abc import Mapping

from typing_extensions import ReadOnly, TypedDict

from litellm.types.guardrails import PiiEntityType


class PresidioAnalyzeRequest(TypedDict, total=False):
    text: str
    language: str | None
    ad_hoc_recognizers: list[str] | None
    entities: list[PiiEntityType | str] | None


class PresidioAnalyzeResponseItem(TypedDict, total=False):
    entity_type: PiiEntityType | str | None
    start: int | None
    end: int | None
    score: float | None
    analysis_explanation: ReadOnly[Mapping[str, object] | None]
    recognition_metadata: ReadOnly[Mapping[str, object] | None]
