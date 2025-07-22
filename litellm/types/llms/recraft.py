from typing import Dict, List, Optional

from typing_extensions import TypedDict


class RecraftImageGenerationRequestParams(TypedDict, total=False):
    prompt: str
    text_layout: Optional[List[Dict]]
    n: Optional[int]
    style_id: Optional[str]
    style: Optional[str]
    substyle: Optional[str]
    model: Optional[str]
    response_format: Optional[str]
    size: Optional[str]
    negative_prompt: Optional[str]
    controls: Optional[Dict]