# Type definitions for DynamoAI Guardrails API

import enum
from typing import Any, Dict, List, Literal, Optional, TypedDict


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



class DynamoAIGuardrailConfigModel(TypedDict, total=False):
    """Configuration model for DynamoAI Guardrails"""
    api_key: Optional[str]
    api_base: Optional[str]
    model_id: Optional[str]
    guardrail_name: Optional[str]

