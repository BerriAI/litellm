from typing_extensions import TypedDict

from litellm.types.llms.openai import AllMessageValues


class LakeraAIRequest(TypedDict, total=False):
    messages: list[AllMessageValues]
    project_id: str | None
    payload: bool | None
    breakdown: bool | None
    metadata: dict | None
    dev_info: bool | None


class LakeraAIPayloadItem(TypedDict, total=False):
    start: int | None
    end: int | None
    text: str | None
    detector_type: str | None
    labels: list[str] | None


class LakeraAIBreakdownItem(TypedDict, total=False):
    project_id: str | None
    policy_id: str | None
    detector_id: str | None
    detector_type: str | None
    detected: bool | None


class LakeraAIDevInfo(TypedDict, total=False):
    git_revision: str | None
    git_timestamp: str | None
    model_version: str | None
    version: str | None


class LakeraAIResponse(TypedDict, total=False):
    flagged: bool | None
    payload: list[LakeraAIPayloadItem] | None
    breakdown: list[LakeraAIBreakdownItem] | None
    dev_info: LakeraAIDevInfo | None
