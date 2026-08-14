"""
Type definitions for Aliyun AI Security Guardrail
阿里云AI安全护栏类型定义
Aliyun AI Guardrail supports the following detection types:
- contentModeration: Content safety moderation
- sensitiveData: Sensitive data detection (PII, etc.)
- promptAttack: Prompt injection attack detection
- maliciousUrl: Malicious URL detection
"""

from collections.abc import Sequence
from typing import Literal, TypeAlias

from pydantic import BaseModel, Field
from typing_extensions import NotRequired, ReadOnly, TypedDict

from ..base import GuardrailConfigModel


# Response types
class AliyunAIGuardrailResponseDetailResultExt(TypedDict, total=False):
    """Extended information in result"""

    Desensitization: ReadOnly[str | None]  # Desensitized text when action is mask


class AliyunAIGuardrailResponseDetailResult(TypedDict, total=False):
    """Result item in detail"""

    Confidence: ReadOnly[float | None]
    Label: ReadOnly[str | None]
    Ext: ReadOnly[AliyunAIGuardrailResponseDetailResultExt | None]
    # Per-result risk level. This is the shape documented for MultiModalGuard; the
    # ``_pro`` service codes report the severity on the parent Detail as ``Level``
    # instead, so both have to be honoured when deciding whether to block.
    RiskLevel: ReadOnly[str | None]


class AliyunAIGuardrailResponseDetail(TypedDict):
    """Detail item in response data"""

    Type: ReadOnly[str]  # contentModeration, sensitiveData, promptAttack, maliciousUrl
    Suggestion: ReadOnly[str]  # pass, block, mask
    Result: ReadOnly[Sequence[AliyunAIGuardrailResponseDetailResult]]
    # Risk level as returned by the ``_pro`` service codes (none/low/medium/high, or
    # S0-S4 for sensitiveData). Absent in the documented response shape, which carries
    # the severity as Result[].RiskLevel.
    Level: ReadOnly[NotRequired[str]]


class AliyunAIGuardrailResponseData(TypedDict, total=False):
    """Response data from Aliyun AI Guardrail API"""

    Suggestion: ReadOnly[str]  # Overall suggestion: pass, block, mask
    Detail: ReadOnly[Sequence[AliyunAIGuardrailResponseDetail] | None]


class AliyunAIGuardrailResponse(TypedDict):
    """Response from Aliyun AI Guardrail API"""

    RequestId: ReadOnly[str]
    Code: ReadOnly[int]
    Message: ReadOnly[str | None]
    Data: ReadOnly[AliyunAIGuardrailResponseData | None]


# Suggestion type
AliyunAIGuardrailSuggestion: TypeAlias = Literal["pass", "block", "watch"]

# Detection type
AliyunAIGuardrailDetectionType: TypeAlias = Literal[
    "contentModeration", "sensitiveData", "promptAttack", "maliciousUrl"
]


class AliyunAIGuardrailRequestParams(TypedDict, total=False):
    """Request parameters for Aliyun AI Guardrail API"""

    Action: ReadOnly[str]
    Version: ReadOnly[str]
    AccessKeyId: ReadOnly[str]
    Timestamp: ReadOnly[str]
    SignatureMethod: ReadOnly[str]
    SignatureVersion: ReadOnly[str]
    SignatureNonce: ReadOnly[str]
    Format: ReadOnly[str]
    Service: ReadOnly[str]
    ServiceParameters: ReadOnly[str]
    Signature: ReadOnly[str]


# Risk level literals
AliyunRiskLevel: TypeAlias = Literal["none", "low", "medium", "high"]

# Protection level literals
AliyunProtectionLevel: TypeAlias = Literal["low", "medium", "high", "max"]


# Configuration models
class AliyunAIGuardrailOptionalParams(BaseModel):
    """
    Optional parameters for Aliyun AI Guardrail.
    Credentials (access_key_id / access_key_secret) are configured
    in config.yaml on the AliyunAIGuardrailConfigModel and support os.environ/ references.
    """

    level: AliyunProtectionLevel | None = Field(
        default="medium",
        description="Protection level for risk filtering. 'low': block all risks (high protection), 'medium': block medium and high risks, 'high': block only high risks (low protection), 'max': observation mode (no blocking). Default: medium",
    )
    max_text_length: int | None = Field(
        default=2000,
        description="Maximum text length for a single API call. Text longer than this will be split.",
    )
    stream_window_size: int | None = Field(
        default=500,
        description="Sliding window size (in chars) for streaming output guardrail checks. Each check sends the most recent N chars to the API.",
    )
    stream_slide_step: int | None = Field(
        default=300,
        description="Sliding step (in chars) for streaming output guardrail checks. A check is triggered every time N new chars accumulate since the last check.",
    )
    stream_first_check_step: int | None = Field(
        default=50,
        description="First check threshold (in chars) for streaming output. The first guardrail check triggers earlier (at N chars) to reduce first-token latency, subsequent checks use stream_slide_step.",
    )
    region_id: str | None = Field(
        default="cn-shanghai",
        description="Aliyun region ID. Default: cn-shanghai",
    )
    service_input: str | None = Field(
        default="query_security_check_pro",
        description="Service code for input (pre-call) detection. Default: query_security_check_pro",
    )
    service_output: str | None = Field(
        default="response_security_check_pro",
        description="Service code for output (post-call) detection. Default: response_security_check_pro",
    )
    service_mcp: str | None = Field(
        default="query_security_check_pro",
        description="Service code for MCP tool call detection (pre_mcp_call and post_mcp_call). Default: query_security_check_pro",
    )


class AliyunAIGuardrailConfigModel(GuardrailConfigModel[AliyunAIGuardrailOptionalParams]):
    """
    Configuration model for Aliyun AI Guardrail.
    Credentials are configured in config.yaml and support os.environ/ references:
    - access_key_id: Aliyun Access Key ID
    - access_key_secret: Aliyun Access Key Secret
    """

    access_key_id: str | None = Field(
        default=None,
        description="Aliyun Access Key ID. Configure in config.yaml, supports os.environ/ reference",
    )
    access_key_secret: str | None = Field(
        default=None,
        description="Aliyun Access Key Secret. Configure in config.yaml, supports os.environ/ reference",
    )
    optional_params: AliyunAIGuardrailOptionalParams | None = Field(
        default_factory=AliyunAIGuardrailOptionalParams,
        description="Optional parameters for the Aliyun AI Guardrail",
    )

    @staticmethod
    def ui_friendly_name() -> str:
        return "Aliyun AI Security Guardrail"
