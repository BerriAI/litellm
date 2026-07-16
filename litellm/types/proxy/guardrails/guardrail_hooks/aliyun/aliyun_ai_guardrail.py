"""
Type definitions for Aliyun AI Security Guardrail
阿里云AI安全护栏类型定义
Aliyun AI Guardrail supports the following detection types:
- contentModeration: Content safety moderation
- sensitiveData: Sensitive data detection (PII, etc.)
- promptAttack: Prompt injection attack detection
- maliciousUrl: Malicious URL detection
"""

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from ..base import GuardrailConfigModel


# Response types
class AliyunAIGuardrailResponseDetailResultExt(TypedDict, total=False):
    """Extended information in result"""

    Desensitization: Optional[str]  # Desensitized text when action is mask


class AliyunAIGuardrailResponseDetailResult(TypedDict, total=False):
    """Result item in detail"""

    Confidence: Optional[float]
    Label: Optional[str]
    Ext: Optional[AliyunAIGuardrailResponseDetailResultExt]


class AliyunAIGuardrailResponseDetail(TypedDict):
    """Detail item in response data"""

    Type: str  # contentModeration, sensitiveData, promptAttack, maliciousUrl
    Suggestion: str  # pass, block, mask
    Result: List[AliyunAIGuardrailResponseDetailResult]


class AliyunAIGuardrailResponseData(TypedDict, total=False):
    """Response data from Aliyun AI Guardrail API"""

    Suggestion: str  # Overall suggestion: pass, block, mask
    Detail: Optional[List[AliyunAIGuardrailResponseDetail]]


class AliyunAIGuardrailResponse(TypedDict):
    """Response from Aliyun AI Guardrail API"""

    RequestId: str
    Code: int
    Message: Optional[str]
    Data: Optional[AliyunAIGuardrailResponseData]


# Suggestion type
AliyunAIGuardrailSuggestion = Literal["pass", "block", "watch"]

# Detection type
AliyunAIGuardrailDetectionType = Literal["contentModeration", "sensitiveData", "promptAttack", "maliciousUrl"]


class AliyunAIGuardrailRequestParams(TypedDict, total=False):
    """Request parameters for Aliyun AI Guardrail API"""

    Action: str
    Version: str
    AccessKeyId: str
    Timestamp: str
    SignatureMethod: str
    SignatureVersion: str
    SignatureNonce: str
    Format: str
    Service: str
    ServiceParameters: str
    Signature: str


# Risk level literals
AliyunRiskLevel = Literal["none", "low", "medium", "high"]

# Protection level literals
AliyunProtectionLevel = Literal["low", "medium", "high", "max"]


# Configuration models
class AliyunAIGuardrailOptionalParams(BaseModel):
    """
    Optional parameters for Aliyun AI Guardrail.
    Credentials (access_key_id / access_key_secret) are configured
    in config.yaml on the AliyunAIGuardrailConfigModel and support os.environ/ references.
    """

    level: Optional[AliyunProtectionLevel] = Field(
        default="medium",
        description="Protection level for risk filtering. 'low': block all risks (high protection), 'medium': block medium and high risks, 'high': block only high risks (low protection), 'max': observation mode (no blocking). Default: medium",
    )
    max_text_length: Optional[int] = Field(
        default=2000,
        description="Maximum text length for a single API call. Text longer than this will be split.",
    )
    stream_window_size: Optional[int] = Field(
        default=500,
        description="Sliding window size (in chars) for streaming output guardrail checks. Each check sends the most recent N chars to the API.",
    )
    stream_slide_step: Optional[int] = Field(
        default=300,
        description="Sliding step (in chars) for streaming output guardrail checks. A check is triggered every time N new chars accumulate since the last check.",
    )
    stream_first_check_step: Optional[int] = Field(
        default=50,
        description="First check threshold (in chars) for streaming output. The first guardrail check triggers earlier (at N chars) to reduce first-token latency, subsequent checks use stream_slide_step.",
    )
    region_id: Optional[str] = Field(
        default="cn-shanghai",
        description="Aliyun region ID. Default: cn-shanghai",
    )
    service_input: Optional[str] = Field(
        default="query_security_check_pro",
        description="Service code for input (pre-call) detection. Default: query_security_check_pro",
    )
    service_output: Optional[str] = Field(
        default="response_security_check_pro",
        description="Service code for output (post-call) detection. Default: response_security_check_pro",
    )
    service_mcp: Optional[str] = Field(
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

    access_key_id: Optional[str] = Field(
        default=None,
        description="Aliyun Access Key ID. Configure in config.yaml, supports os.environ/ reference",
    )
    access_key_secret: Optional[str] = Field(
        default=None,
        description="Aliyun Access Key Secret. Configure in config.yaml, supports os.environ/ reference",
    )
    optional_params: AliyunAIGuardrailOptionalParams = Field(
        default_factory=AliyunAIGuardrailOptionalParams,
        description="Optional parameters for the Aliyun AI Guardrail",
    )

    @staticmethod
    def ui_friendly_name() -> str:
        return "Aliyun AI Security Guardrail"
