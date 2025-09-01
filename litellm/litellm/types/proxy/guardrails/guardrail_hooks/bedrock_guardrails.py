from typing import Any, Dict, List, Literal, Optional, Union

from typing_extensions import TypedDict


class BedrockTextContent(TypedDict, total=False):
    text: str


class BedrockContentItem(TypedDict, total=False):
    text: BedrockTextContent


class BedrockRequest(TypedDict, total=False):
    source: Literal["INPUT", "OUTPUT"]
    content: List[BedrockContentItem]


class BedrockGuardrailUsage(TypedDict, total=False):
    topicPolicyUnits: Optional[int]
    contentPolicyUnits: Optional[int]
    wordPolicyUnits: Optional[int]
    sensitiveInformationPolicyUnits: Optional[int]
    sensitiveInformationPolicyFreeUnits: Optional[int]
    contextualGroundingPolicyUnits: Optional[int]


class BedrockGuardrailOutput(TypedDict, total=False):
    text: Optional[str]


class BedrockGuardrailTopicPolicyItem(TypedDict, total=False):
    name: Optional[str]
    type: Optional[str]
    action: Optional[str]


class BedrockGuardrailTopicPolicy(TypedDict, total=False):
    topics: List[BedrockGuardrailTopicPolicyItem]


class BedrockGuardrailContentPolicyFilter(TypedDict, total=False):
    type: Optional[str]
    confidence: Optional[str]
    filterStrength: Optional[str]
    action: Optional[str]


class BedrockGuardrailContentPolicy(TypedDict, total=False):
    filters: List[BedrockGuardrailContentPolicyFilter]


class BedrockGuardrailWordPolicyCustomWord(TypedDict, total=False):
    match: str
    action: str


class BedrockGuardrailWordPolicyManagedWord(TypedDict, total=False):
    match: Optional[str]
    type: Optional[str]  # Note: There might be more types
    action: Optional[str]


class BedrockGuardrailWordPolicy(TypedDict, total=False):
    customWords: List[BedrockGuardrailWordPolicyCustomWord]
    managedWordLists: List[BedrockGuardrailWordPolicyManagedWord]


class BedrockGuardrailPiiEntity(TypedDict, total=False):
    type: Optional[str]  # Many PII types available per AWS docs
    match: Optional[str]
    action: Optional[str]


class BedrockGuardrailRegex(TypedDict, total=False):
    name: Optional[str]
    regex: Optional[str]
    match: Optional[str]
    action: Optional[str]


class BedrockGuardrailSensitiveInformationPolicy(TypedDict, total=False):
    piiEntities: Optional[List[BedrockGuardrailPiiEntity]]
    regexes: Optional[List[BedrockGuardrailRegex]]


class BedrockGuardrailContextualGroundingFilter(TypedDict, total=False):
    type: Optional[str]
    threshold: Optional[float]
    score: Optional[float]
    action: Optional[str]


class BedrockGuardrailContextualGroundingPolicy(TypedDict, total=False):
    filters: List[BedrockGuardrailContextualGroundingFilter]


class BedrockGuardrailCoverage(TypedDict, total=False):
    textCharacters: Dict[str, int]


class BedrockGuardrailInvocationMetrics(TypedDict, total=False):
    guardrailProcessingLatency: int
    usage: BedrockGuardrailUsage
    guardrailCoverage: BedrockGuardrailCoverage


class BedrockGuardrailAssessment(TypedDict, total=False):
    topicPolicy: Optional[BedrockGuardrailTopicPolicy]
    contentPolicy: Optional[BedrockGuardrailContentPolicy]
    wordPolicy: Optional[BedrockGuardrailWordPolicy]
    sensitiveInformationPolicy: Optional[BedrockGuardrailSensitiveInformationPolicy]
    contextualGroundingPolicy: Optional[BedrockGuardrailContextualGroundingPolicy]
    invocationMetrics: BedrockGuardrailInvocationMetrics
    guardrailCoverage: BedrockGuardrailCoverage


class BedrockGuardrailResponse(TypedDict, total=False):
    usage: Optional[BedrockGuardrailUsage]
    action: Optional[str]
    output: Optional[List[BedrockGuardrailOutput]]
    outputs: Optional[List[BedrockGuardrailOutput]]
    assessments: Optional[List[BedrockGuardrailAssessment]]
