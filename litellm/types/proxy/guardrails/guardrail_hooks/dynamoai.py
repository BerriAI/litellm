# Type definitions for DynamoAI Guardrails API

import enum
from typing import Any, Dict, List, Literal, Optional, TypedDict

from pydantic import Field

from .base import GuardrailConfigModel


class DynamoAIMessage(TypedDict):
    """Message structure for DynamoAI API"""
    role: str
    content: str

class DynamoRequestMetadata(TypedDict):
    endUserId: Optional[str]

class DynamoTextType(str, enum.Enum):
    MODEL_INPUT = "MODEL_INPUT"
    MODEL_RESPONSE = "MODEL_RESPONSE"


class PolicyMethod(str, enum.Enum):
    PII = "PII"
    TOXICITY = "TOXICITY"
    ALIGNMENT = "ALIGNMENT"
    HALLUCINATION = "HALLUCINATION"
    RAG_HALLUCINATION = "RAG_HALLUCINATION"


class PolicyApplicableTo(str, enum.Enum):
    INPUT = "INPUT"
    OUTPUT = "OUTPUT"
    ALL = "ALL"


class DynamoAIRequest(TypedDict, total=False):
    """Request structure for DynamoAI /moderation/analyze endpoint"""
    messages: List[Dict[str, Any]]
    textType: Optional[DynamoTextType]
    policyIds: List[str]
    modelId: Optional[str]
    clientId: Optional[str]
    metadata: Optional[DynamoRequestMetadata]


class PolicyInfo(TypedDict, total=False):
    """Policy information from DynamoAI response"""
    id: str
    name: str
    description: str
    method: PolicyMethod
    action: Literal["BLOCK", "WARN", "REDACT", "SANITIZE", "NONE"]
    methodParams: Dict[str, Any]
    decisionParams: Dict[str, Any]
    applicableTo: PolicyApplicableTo
    created_at: str
    creatorId: str


class PolicyOutputs(TypedDict, total=False):
    """Outputs from the policy"""
    action: Literal["BLOCK", "WARN", "REDACT", "SANITIZE", "NONE"]
    message: Optional[str]


class AppliedPolicyDto(TypedDict, total=False):
    """Applied policy details from DynamoAI response"""
    policy: PolicyInfo
    outputs: Optional[Dict[str, Any]]
    action: Optional[str]


class DynamoAIResponse(TypedDict, total=False):
    """Response structure from DynamoAI /moderation/analyze endpoint"""
    text: str
    textType: DynamoTextType
    finalAction: Literal["BLOCK", "WARN", "REDACT", "SANITIZE", "NONE"]
    appliedPolicies: List[AppliedPolicyDto]
    error: Optional[str]


class DynamoAIProcessedResult(TypedDict):
    """Processed result from DynamoAI guardrail check"""
    violations_detected: List[str]
    violation_details: Dict[str, Any]



class DynamoAIGuardrailConfigModel(GuardrailConfigModel):
    """Configuration model for DynamoAI Guardrails"""
    
    api_key: Optional[str] = Field(
        default=None,
        description="API key for DynamoAI Guardrails. If not provided, the `DYNAMOAI_API_KEY` environment variable is checked.",
    )
    api_base: Optional[str] = Field(
        default=None,
        description="Base URL for DynamoAI API. If not provided, the `DYNAMOAI_API_BASE` environment variable is checked, defaults to https://api.dynamo.ai",
    )
    policy_ids: Optional[List[str]] = Field(
        default=None,
        description="List of DynamoAI policy IDs to apply. If not provided, the `DYNAMOAI_POLICY_IDS` environment variable is checked (comma-separated).",
    )
    model_id: Optional[str] = Field(
        default=None,
        description="Model ID for tracking/logging purposes. If not provided, the `DYNAMOAI_MODEL_ID` environment variable is checked.",
    )
    guardrail_name: Optional[str] = Field(
        default=None,
        description="Name of the guardrail for identification in logs and traces.",
    )
    
    @staticmethod
    def ui_friendly_name() -> str:
        return "DynamoAI Guardrails"

