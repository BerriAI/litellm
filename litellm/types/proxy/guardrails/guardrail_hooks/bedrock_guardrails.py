from typing import Literal

from typing_extensions import TypedDict

# Bedrock contextual grounding tags each content block so the guardrail knows
# which text is the reference source, the user question, and the content to grade.
BedrockGuardrailQualifier = Literal["grounding_source", "query", "guard_content"]


class BedrockTextContent(TypedDict, total=False):
    text: str
    qualifiers: list[BedrockGuardrailQualifier]


class BedrockContentItem(TypedDict, total=False):
    text: BedrockTextContent


class BedrockRequest(TypedDict, total=False):
    source: Literal["INPUT", "OUTPUT"]
    content: list[BedrockContentItem]


class BedrockGuardrailUsage(TypedDict, total=False):
    topicPolicyUnits: int | None
    contentPolicyUnits: int | None
    wordPolicyUnits: int | None
    sensitiveInformationPolicyUnits: int | None
    sensitiveInformationPolicyFreeUnits: int | None
    contextualGroundingPolicyUnits: int | None
    contentPolicyImageUnits: int | None
    automatedReasoningPolicyUnits: int | None
    automatedReasoningPolicies: int | None


class BedrockGuardrailOutput(TypedDict, total=False):
    text: str | None


class BedrockGuardrailTopicPolicyItem(TypedDict, total=False):
    name: str | None
    type: str | None
    action: str | None


class BedrockGuardrailTopicPolicy(TypedDict, total=False):
    topics: list[BedrockGuardrailTopicPolicyItem]


class BedrockGuardrailContentPolicyFilter(TypedDict, total=False):
    type: str | None
    confidence: str | None
    filterStrength: str | None
    action: str | None


class BedrockGuardrailContentPolicy(TypedDict, total=False):
    filters: list[BedrockGuardrailContentPolicyFilter]


class BedrockGuardrailWordPolicyCustomWord(TypedDict, total=False):
    match: str
    action: str


class BedrockGuardrailWordPolicyManagedWord(TypedDict, total=False):
    match: str | None
    type: str | None  # Note: There might be more types
    action: str | None


class BedrockGuardrailWordPolicy(TypedDict, total=False):
    customWords: list[BedrockGuardrailWordPolicyCustomWord]
    managedWordLists: list[BedrockGuardrailWordPolicyManagedWord]


class BedrockGuardrailPiiEntity(TypedDict, total=False):
    type: str | None  # Many PII types available per AWS docs
    match: str | None
    action: str | None


class BedrockGuardrailRegex(TypedDict, total=False):
    name: str | None
    regex: str | None
    match: str | None
    action: str | None


class BedrockGuardrailSensitiveInformationPolicy(TypedDict, total=False):
    piiEntities: list[BedrockGuardrailPiiEntity] | None
    regexes: list[BedrockGuardrailRegex] | None


class BedrockGuardrailContextualGroundingFilter(TypedDict, total=False):
    type: str | None
    threshold: float | None
    score: float | None
    action: str | None


class BedrockGuardrailContextualGroundingPolicy(TypedDict, total=False):
    filters: list[BedrockGuardrailContextualGroundingFilter]


class BedrockGuardrailCoverage(TypedDict, total=False):
    textCharacters: dict[str, int]


class BedrockGuardrailInvocationMetrics(TypedDict, total=False):
    guardrailProcessingLatency: int
    usage: BedrockGuardrailUsage
    guardrailCoverage: BedrockGuardrailCoverage


class BedrockGuardrailAssessment(TypedDict, total=False):
    topicPolicy: BedrockGuardrailTopicPolicy | None
    contentPolicy: BedrockGuardrailContentPolicy | None
    wordPolicy: BedrockGuardrailWordPolicy | None
    sensitiveInformationPolicy: BedrockGuardrailSensitiveInformationPolicy | None
    contextualGroundingPolicy: BedrockGuardrailContextualGroundingPolicy | None
    invocationMetrics: BedrockGuardrailInvocationMetrics
    guardrailCoverage: BedrockGuardrailCoverage


class BedrockGuardrailResponse(TypedDict, total=False):
    usage: BedrockGuardrailUsage | None
    action: str | None
    output: list[BedrockGuardrailOutput] | None
    outputs: list[BedrockGuardrailOutput] | None
    assessments: list[BedrockGuardrailAssessment] | None


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
