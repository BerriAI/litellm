from typing import Dict, List, Optional

from typing_extensions import TypedDict

from litellm.types.llms.openai import AllMessageValues


class LakeraAIRequest(TypedDict, total=False):
    messages: List[AllMessageValues]
    project_id: Optional[str]
    payload: Optional[bool]
    breakdown: Optional[bool]
    metadata: Optional[Dict]
    dev_info: Optional[bool]


class LakeraAIPayloadItem(TypedDict, total=False):
    start: Optional[int]
    end: Optional[int]
    text: Optional[str]
    detector_type: Optional[str]
    labels: Optional[List[str]]


class LakeraAIBreakdownItem(TypedDict, total=False):
    project_id: Optional[str]
    policy_id: Optional[str]
    detector_id: Optional[str]
    detector_type: Optional[str]
    detected: Optional[bool]


class LakeraAIDevInfo(TypedDict, total=False):
    git_revision: Optional[str]
    git_timestamp: Optional[str]
    model_version: Optional[str]
    version: Optional[str]


class LakeraAIResponse(TypedDict, total=False):
    flagged: Optional[bool]
    payload: Optional[List[LakeraAIPayloadItem]]
    breakdown: Optional[List[LakeraAIBreakdownItem]]
    dev_info: Optional[LakeraAIDevInfo]
