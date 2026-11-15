from dataclasses import dataclass
from enum import StrEnum
from typing import TypeAlias


class ProviderFamily(StrEnum):
    ANTHROPIC = "anthropic"
    BEDROCK = "bedrock"
    VERTEX_AI = "vertex_ai"
    AZURE_AI = "azure_ai"


class WireShape(StrEnum):
    ANTHROPIC_MESSAGES = "anthropic_messages"
    BEDROCK_CONVERSE = "bedrock_converse"


class ClientSurface(StrEnum):
    MESSAGES = "messages"
    CHAT = "chat"
    RESPONSES = "responses"


class Runtime(StrEnum):
    PYTHON = "python"
    RUST = "rust"


class CacheIntent(StrEnum):
    CLIENT = "client"
    GATEWAY = "gateway"
    CLIENT_AND_GATEWAY = "client_and_gateway"
    UNATTRIBUTED = "unattributed"


class UnassessedReason(StrEnum):
    PROVIDER_OUT_OF_SCOPE = "provider_out_of_scope"
    MODEL_OUT_OF_SCOPE = "model_out_of_scope"
    SURFACE_OUT_OF_SCOPE = "surface_out_of_scope"
    WIRE_SHAPE_UNRECOGNIZED = "wire_shape_unrecognized"
    RUNTIME_UNKNOWN = "runtime_unknown"
    PATH_NOT_VERIFIED = "path_not_verified"
    REQUEST_BODY_UNAVAILABLE = "request_body_unavailable"
    REQUEST_REDACTED = "request_redacted"
    BREAKPOINT_UNRECOGNIZED = "breakpoint_unrecognized"
    TTL_UNRECOGNIZED = "ttl_unrecognized"
    CACHE_INTENT_NOT_SENT = "cache_intent_not_sent"
    PREFIX_KEY_UNAVAILABLE = "prefix_key_unavailable"
    DEPLOYMENT_SETTINGS_UNAVAILABLE = "deployment_settings_unavailable"
    UPSTREAM_FAILED = "upstream_failed"


class UsageUnknownReason(StrEnum):
    STREAM_USAGE_NOT_CAPTURED = "stream_usage_not_captured"
    BODY_UNAVAILABLE = "body_unavailable"
    BODY_MALFORMED = "body_malformed"
    INCOMPLETE = "incomplete"
    MISSING = "missing"
    PARTIAL = "partial"
    MALFORMED = "malformed"
    CONTRADICTORY = "contradictory"
    ITERATED = "iterated"


@dataclass(frozen=True, slots=True)
class ObservationPath:
    provider: ProviderFamily
    wire: WireShape
    surface: ClientSurface
    stream: bool
    runtime: Runtime


@dataclass(frozen=True, slots=True)
class CacheCounters:
    read_tokens: int
    write_tokens: int

    @property
    def active(self) -> bool:
        return self.read_tokens > 0 or self.write_tokens > 0


@dataclass(frozen=True, slots=True)
class UsageUnknown:
    reason: UsageUnknownReason


UpstreamCacheUsage: TypeAlias = CacheCounters | UsageUnknown


@dataclass(frozen=True, slots=True)
class CacheablePrefix:
    fingerprint: str
    ttl_seconds: int
    breakpoints: int


@dataclass(frozen=True, slots=True)
class NoCacheBreakpoint:
    pass


@dataclass(frozen=True, slots=True)
class PrefixUnassessed:
    reason: UnassessedReason


PrefixEvidence: TypeAlias = CacheablePrefix | NoCacheBreakpoint | PrefixUnassessed


@dataclass(frozen=True, slots=True)
class DeploymentCacheIdentity:
    deployment_id: str
    config_fingerprint: str
    provider: str
    provider_model: str
    min_prefix_tokens: int | None


@dataclass(frozen=True, slots=True)
class Assessed:
    path: ObservationPath
    prefix: CacheablePrefix
    usage: UpstreamCacheUsage


@dataclass(frozen=True, slots=True)
class Unassessed:
    reason: UnassessedReason
    path: ObservationPath | None
    prefix: CacheablePrefix | None


@dataclass(frozen=True, slots=True)
class PromptCacheObservation:
    """Retained evidence: opaque identities, counters and reasons only, never prompt content."""

    call_id: str
    deployment: DeploymentCacheIdentity
    intent: CacheIntent
    observed_at: float
    outcome: Assessed | Unassessed
