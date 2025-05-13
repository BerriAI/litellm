from typing import List, Optional, TypedDict

from litellm.types.guardrails import PiiEntityType


class PresidioAnalyzeRequest(TypedDict, total=False):
    text: str
    language: Optional[str]
    ad_hoc_recognizers: Optional[List[str]]
    entities: Optional[List[PiiEntityType]]
