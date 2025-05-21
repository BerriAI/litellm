from typing import Any, Dict, List, Optional, TypedDict, Union

from litellm.types.guardrails import PiiEntityType


class PresidioAnalyzeRequest(TypedDict, total=False):
    text: str
    language: Optional[str]
    ad_hoc_recognizers: Optional[List[str]]
    entities: Optional[List[PiiEntityType]]


class PresidioAnalyzeResponseItem(TypedDict, total=False):
    entity_type: Optional[Union[PiiEntityType, str]]
    start: Optional[int]
    end: Optional[int]
    score: Optional[float]
    analysis_explanation: Optional[Dict[str, Any]]
    recognition_metadata: Optional[Dict[str, Any]]
