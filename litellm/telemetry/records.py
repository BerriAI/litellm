from dataclasses import dataclass, field
from enum import Enum
from typing import Final


class TelemetryGroup(str, Enum):
    """
    A telemetry group describes a group of metrics that can be toggled on/off separately. One
    group may have dependencies on another.
    """

    HEARTBEAT = "heartbeat"
    REQUEST_SUCCESS = "request_success"
    TOKEN_INFO = "token_info"
    REQUEST_TAXONOMY = "request_taxonomy"
    EVENT_DETAILS = "event_details"
    INSTANCE_CONFIGURATION = "instance_configuration"
    PAGE_NAVIGATION = "page_navigation"


class StatusClass(str, Enum):
    NONE = "none"
    SUCCESS = "2xx"
    REDIRECT = "3xx"
    CLIENT_ERROR = "4xx"
    SERVER_ERROR = "5xx"

    @classmethod
    def from_status_code(cls, status_code: int | None) -> "StatusClass":
        match status_code:
            case int() if 200 <= status_code < 300:
                return cls.SUCCESS
            case int() if 300 <= status_code < 400:
                return cls.REDIRECT
            case int() if 400 <= status_code < 500:
                return cls.CLIENT_ERROR
            case int() if 500 <= status_code < 600:
                return cls.SERVER_ERROR
            case _:
                return cls.NONE


class BlockType(str, Enum):
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    FILE = "file"
    TOOL_USE = "tool_use"
    TOOL_RESULT = "tool_result"
    THINKING = "thinking"
    OTHER = "other"


class UIAction(str, Enum):
    VIEW = "view"
    CLICK = "click"
    SUBMIT = "submit"


OTHER_HEADER_KEY: Final = "other"

ALLOWED_HEADER_KEYS: Final = frozenset(
    {
        "anthropic-beta",
        "anthropic-version",
        "openai-beta",
        "openai-organization",
        "x-litellm-api-key",
        "x-litellm-disable-callbacks",
        "x-litellm-enable-message-redaction",
        "x-litellm-num-retries",
        "x-litellm-tags",
        "x-litellm-timeout",
        "x-stainless-lang",
        "x-stainless-package-version",
    }
)


@dataclass(frozen=True, slots=True)
class TokenCounts:
    input: int = 0
    output: int = 0
    cache_read: int = 0


@dataclass(frozen=True, slots=True)
class BlockCounts:
    total: int
    by_type: tuple[tuple[BlockType, int], ...] = ()


@dataclass(frozen=True, slots=True)
class InstanceInfo:
    instance_id: str
    litellm_version: str
    groups: frozenset[TelemetryGroup] = frozenset()
    config_keys: frozenset[str] = frozenset()


@dataclass(frozen=True, slots=True)
class RequestRecord:
    endpoint: str
    stream: bool
    litellm_status: StatusClass
    provider: str | None = None
    deployment_hash: str | None = None
    provider_status: StatusClass = StatusClass.NONE
    litellm_cache_hit: bool = False
    handled_by_rust: bool = False
    provider_cache_hit: bool = False
    provider_attempts: int = 0
    tokens: TokenCounts = field(default_factory=TokenCounts)
    latency_to_headers_ms: float | None = None
    latency_to_first_byte_ms: float | None = None
    blocks: BlockCounts | None = None
    header_keys: frozenset[str] = frozenset()


@dataclass(frozen=True, slots=True)
class AttemptRecord:
    provider: str
    provider_status: StatusClass
    stream: bool
    deployment_hash: str | None = None
    latency_to_first_token_ms: float | None = None


@dataclass(frozen=True, slots=True)
class UIEvent:
    page: str
    action: UIAction
    target: str | None = None
