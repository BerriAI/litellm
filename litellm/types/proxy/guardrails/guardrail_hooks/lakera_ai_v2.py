from typing import Dict, List, Optional, TypedDict


class LakeraAIRequest(TypedDict, total=False):
    messages: List[Dict]
    project_id: Optional[str]
    payload: Optional[bool]
    breakdown: Optional[bool]
    metadata: Optional[Dict]
    dev_info: Optional[bool]


class LakeraAIPayloadItem(TypedDict):
    start: Optional[int]
    end: Optional[int]
    text: Optional[str]
    detector_type: Optional[str]
    labels: Optional[List[str]]


class LakeraAIBreakdownItem(TypedDict):
    project_id: Optional[str]
    policy_id: Optional[str]
    detector_id: Optional[str]
    detector_type: Optional[str]
    detected: Optional[bool]


class LakeraAIDevInfo(TypedDict):
    git_revision: Optional[str]
    git_timestamp: Optional[str]
    model_version: Optional[str]
    version: Optional[str]


class LakeraAIResponse(TypedDict):
    flagged: Optional[bool]
    payload: Optional[List[LakeraAIPayloadItem]]
    breakdown: Optional[List[LakeraAIBreakdownItem]]
    dev_info: Optional[LakeraAIDevInfo]
