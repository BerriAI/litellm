from typing import Dict, List, Literal, Optional

from typing_extensions import TypedDict

# Bedrock contextual grounding tags each content block so the guardrail knows
# which text is the reference source, the user question, and the content to grade.
BedrockGuardrailQualifier = Literal["grounding_source", "query", "guard_content"]


class BedrockTextContent(TypedDict, total=False):
    text: str
    qualifiers: List[BedrockGuardrailQualifier]


class BedrockContentItem(TypedDict, total=False):
    text: BedrockTextContent


class BedrockRequest(TypedDict, total=False):
    source: Literal["INPUT", "OUTPUT"]
    content: List[BedrockContentItem]
    outputScope: Literal["INTERVENTIONS", "FULL"]


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


# ---------------------------------------------------------------------------
# InvokeGuardrailChecks API (resource-less, detect-only)
# POST /guardrail-checks/invoke
# Unlike ApplyGuardrail, this API takes inline `checks` (no guardrail resource)
# and returns numeric scores per check; it never blocks/masks/rewrites content.
# ---------------------------------------------------------------------------


class BedrockChecksTextContent(TypedDict, total=False):
    text: str


class BedrockChecksMessage(TypedDict, total=False):
    role: Literal["user", "assistant", "system"]
    content: list[BedrockChecksTextContent]


class BedrockChecksScoreEntry(TypedDict, total=False):
    """A contentFilter/promptAttack result entry; severityScore is a float in [0,1]
    (Bedrock returns it in discrete steps: 0, 0.2, 0.4, 0.6, 0.8, 1.0)."""

    category: str | None
    severityScore: float | None


class BedrockChecksPiiEntry(TypedDict, total=False):
    """A sensitiveInformation result entry; confidence is in [0,1]."""

    type: str | None
    confidenceScore: float | None
    messageIndex: int | None
    contentIndex: int | None
    beginOffset: int | None
    endOffset: int | None


class BedrockChecksScoreResult(TypedDict, total=False):
    results: list[BedrockChecksScoreEntry]


class BedrockChecksSensitiveInformationResult(TypedDict, total=False):
    results: list[BedrockChecksPiiEntry]
    truncated: bool | None


class BedrockChecksResults(TypedDict, total=False):
    contentFilter: BedrockChecksScoreResult | None
    promptAttack: BedrockChecksScoreResult | None
    sensitiveInformation: BedrockChecksSensitiveInformationResult | None


class BedrockChecksViolation(TypedDict, total=False):
    """One over-threshold InvokeGuardrailChecks result; carries only the
    non-sensitive label and score, never offsets or matched text."""

    check: str
    category: str | None
    type: str | None
    severityScore: float
    confidenceScore: float
    truncated: bool


class BedrockChecksTextUnits(TypedDict, total=False):
    textUnits: int | None


class BedrockChecksUsage(TypedDict, total=False):
    contentFilter: BedrockChecksTextUnits | None
    promptAttack: BedrockChecksTextUnits | None
    sensitiveInformation: BedrockChecksTextUnits | None


class BedrockGuardrailChecksResponse(TypedDict, total=False):
    results: BedrockChecksResults | None
    usage: BedrockChecksUsage | None
