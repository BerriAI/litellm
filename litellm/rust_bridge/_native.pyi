from asyncio import Future
from collections.abc import Mapping, Sequence
from typing import Literal, Never, NotRequired, TypedDict, final

import httpx
from pydantic import JsonValue

from litellm.rust_bridge.execution import Execution
from litellm.rust_bridge.public_call import NativeCall

class RustBridgeDeclined(Exception): ...
class RustUpstreamError(Exception): ...
class ForkedAfterNativeRuntimeStarted(RuntimeError): ...
class ProcessReservedForForking(RuntimeError): ...

def trace_encode_error(message: str) -> bytes: ...
def bundled_model_cost_map() -> bytes: ...

@final
class NativeClickHouseSpendConfig:
    def __new__(cls, database: str, url: str, retention_days: int) -> NativeClickHouseSpendConfig: ...

@final
class NativeClickHouseSpendStorage:
    def __new__(cls, config: NativeClickHouseSpendConfig) -> NativeClickHouseSpendStorage: ...
    def ensure_schema(self) -> Future[None]: ...
    def insert_rows(self, rows: Sequence[Mapping[str, object]]) -> Future[None]: ...

@final
class NativeDatabase:
    @staticmethod
    def connect() -> Future[NativeDatabase]: ...
    def close(self) -> Future[None]: ...

@final
class NativeDiagnosticProcessor:
    def __new__(cls, minimum_custom_key_length: int) -> NativeDiagnosticProcessor: ...
    def redact_text(self, text: str) -> str: ...
    def redact_structured_text(self, key: str | None, text: str) -> str: ...
    def redact_client_message(self, text: str) -> str: ...
    def process_diagnostic(
        self,
        message: str,
        exception: str | None,
        stack: str | None,
        leaves: Sequence[tuple[str | None, str]],
        policy: tuple[bool, int, int],
    ) -> tuple[str, str | None, str | None, list[str], bool]: ...
    def scrub_access_arguments(self, arguments: Sequence[str]) -> list[str]: ...

def ocr(
    call: NativeCall,
) -> Execution: ...
def aocr(
    call: NativeCall,
) -> Execution: ...
def ocr_health_check_document(model: str, custom_llm_provider: str | None) -> dict[str, object]: ...
def ocr_passthrough_response(model: str, endpoint: str, body: bytes) -> dict[str, object] | None: ...
def embedding(
    call: NativeCall,
) -> Execution: ...
def aembedding(
    call: NativeCall,
) -> Execution: ...
def transcription(
    call: NativeCall,
) -> dict[str, object]: ...
def atranscription(
    call: NativeCall,
) -> Future[dict[str, object]]: ...
def completion(
    call: NativeCall,
) -> Execution: ...
def acompletion(
    call: NativeCall,
) -> Execution: ...
def responses(
    call: NativeCall,
) -> Execution: ...
def aresponses(
    call: NativeCall,
) -> Execution: ...
def messages(
    call: NativeCall,
) -> Execution: ...
def amessages(
    call: NativeCall,
) -> Execution: ...

@final
class ResponsesWebSocketConnection:
    def __new__(cls, _uninstantiable: Never, /) -> Never: ...
    @classmethod
    def connect(
        cls,
        url: str,
        headers: Mapping[str, str] | None = None,
        timeout_seconds: float | None = None,
    ) -> Future[ResponsesWebSocketConnection]: ...
    def send_text(self, text: str) -> Future[None]: ...
    def recv_text(self) -> Future[str | None]: ...
    def close(self) -> Future[None]: ...

@final
class TokenCounter:
    @staticmethod
    def from_tokenizer(tokenizer: Tokenizer, fast: bool = False) -> TokenCounter: ...
    def acount_request(self, body: bytes) -> Future[dict[str, object]]: ...

@final
class Tokenizer:
    @staticmethod
    def from_tiktoken(encoding: str) -> Tokenizer: ...
    @staticmethod
    def from_json(tokenizer_json: str) -> Tokenizer: ...
    @staticmethod
    def from_pretrained(
        identifier: str,
        revision: str = "main",
        token: str | None = None,
    ) -> Tokenizer: ...
    def encode(self, text: str) -> list[int]: ...
    def decode(self, ids: Sequence[int], skip_special_tokens: bool = True) -> str: ...
    def count(self, text: str, fast: bool = False) -> int: ...
    # tiktoken encodings
    def encode_special(self, text: str, allowed: Sequence[str]) -> list[int]: ...
    def encode_with_unstable(self, text: str, allowed: Sequence[str]) -> tuple[list[int], list[list[int]]]: ...
    def encode_single_token(self, piece: bytes) -> int: ...
    def special_tokens(self) -> dict[str, int]: ...
    def max_token_value(self) -> int: ...
    def is_special_token(self, token: int) -> bool: ...
    def token_byte_values(self) -> list[bytes]: ...
    def decode_bytes(self, ids: Sequence[int]) -> bytes: ...
    # Hugging Face tokenizers
    def to_json(self, pretty: bool = False) -> str: ...
    def token_to_id(self, token: str) -> int | None: ...
    def id_to_token(self, id: int) -> str | None: ...
    def get_vocab(self, with_added_tokens: bool = True) -> dict[str, int]: ...
    def get_vocab_size(self, with_added_tokens: bool = True) -> int: ...
    def added_tokens_decoder(self) -> list[tuple[int, tuple[str, bool, bool, bool, bool, bool]]]: ...
    def padding(self) -> dict[str, object] | None: ...
    def truncation(self) -> dict[str, object] | None: ...
    def num_special_tokens_to_add(self, is_pair: bool) -> int: ...
    def encode_special_tokens(self) -> bool: ...
    def encode_huggingface(
        self,
        sequence: str | Sequence[str],
        pair: str | Sequence[str] | None = None,
        is_pretokenized: bool = False,
        add_special_tokens: bool = True,
        fast: bool = False,
    ) -> HuggingFaceEncoding: ...
    def encode_batch_huggingface(
        self,
        inputs: Sequence[tuple[str | Sequence[str], str | Sequence[str] | None]],
        is_pretokenized: bool = False,
        add_special_tokens: bool = True,
        fast: bool = False,
    ) -> list[HuggingFaceEncoding]: ...
    @property
    def name(self) -> str: ...

@final
class HuggingFaceEncoding:
    def __new__(cls, json: str | None = None) -> HuggingFaceEncoding: ...
    @staticmethod
    def merge(encodings: Sequence[HuggingFaceEncoding], growing_offsets: bool = True) -> HuggingFaceEncoding: ...
    def __len__(self) -> int: ...
    def __reduce__(self) -> tuple[type[HuggingFaceEncoding], tuple[str]]: ...
    def word_to_tokens(self, word_index: int, sequence_index: int = 0) -> tuple[int, int] | None: ...
    def word_to_chars(self, word_index: int, sequence_index: int = 0) -> tuple[int, int] | None: ...
    def token_to_sequence(self, token_index: int) -> int | None: ...
    def token_to_chars(self, token_index: int) -> tuple[int, int] | None: ...
    def token_to_word(self, token_index: int) -> int | None: ...
    def char_to_token(self, char_pos: int, sequence_index: int = 0) -> int | None: ...
    def char_to_word(self, char_pos: int, sequence_index: int = 0) -> int | None: ...
    def set_sequence_id(self, sequence_id: int) -> None: ...
    def pad(
        self,
        length: int,
        direction: str = "right",
        pad_id: int = 0,
        pad_type_id: int = 0,
        pad_token: str = "[PAD]",
    ) -> None: ...
    def truncate(self, max_length: int, stride: int = 0, direction: str = "right") -> None: ...
    @property
    def ids(self) -> list[int]: ...
    @property
    def tokens(self) -> list[str]: ...
    @property
    def offsets(self) -> list[tuple[int, int]]: ...
    @property
    def type_ids(self) -> list[int]: ...
    @property
    def attention_mask(self) -> list[int]: ...
    @property
    def special_tokens_mask(self) -> list[int]: ...
    @property
    def word_ids(self) -> list[int | None]: ...
    @property
    def sequence_ids(self) -> list[int | None]: ...
    @property
    def overflowing(self) -> list[HuggingFaceEncoding]: ...
    @property
    def n_sequences(self) -> int: ...

def gil_stats() -> dict[str, int]: ...
def process_state_started() -> bool: ...
def reserve_process_for_forking() -> None: ...

__all__ = [
    "ForkedAfterNativeRuntimeStarted",
    "HuggingFaceEncoding",
    "NativeClickHouseSpendConfig",
    "NativeClickHouseSpendStorage",
    "NativeDatabase",
    "NativeDiagnosticProcessor",
    "ProcessReservedForForking",
    "ResponsesWebSocketConnection",
    "RouterCall",
    "RouterCoreHandle",
    "RustBridgeDeclined",
    "RustUpstreamError",
    "TokenCounter",
    "Tokenizer",
    "acompletion",
    "aembedding",
    "amessages",
    "aocr",
    "aresponses",
    "atranscription",
    "bundled_model_cost_map",
    "completion",
    "embedding",
    "gil_stats",
    "messages",
    "ocr",
    "ocr_health_check_document",
    "ocr_passthrough_response",
    "process_state_started",
    "reserve_process_for_forking",
    "responses",
    "trace_encode_error",
    "transcription",
]

@final
class _SecretManagerRuntime:
    @staticmethod
    def from_config(
        system: str,
        environment: Mapping[str, str],
        settings: Mapping[str, object] | None = None,
        enterprise_enabled: bool = False,
    ) -> _SecretManagerRuntime: ...
    @staticmethod
    def from_client(client: object) -> _SecretManagerRuntime | None: ...
    @property
    def system(self) -> str: ...
    def read_secret(self, name: str, settings: Mapping[str, object] | None = None) -> JsonValue: ...
    def read_secret_async(self, name: str, settings: Mapping[str, object] | None = None) -> Future[JsonValue]: ...
    def async_write_secret(
        self,
        secret_name: str,
        secret_value: str,
        description: str | None = None,
        optional_params: Mapping[str, object] | None = None,
        timeout: float | httpx.Timeout | None = None,
        tags: object = None,
    ) -> Future[dict[str, JsonValue]]: ...
    def async_delete_secret(
        self,
        secret_name: str,
        recovery_window_in_days: int | None = None,
        optional_params: Mapping[str, object] | None = None,
        timeout: float | httpx.Timeout | None = None,
    ) -> Future[dict[str, JsonValue]]: ...
    def async_rotate_secret(
        self,
        current_secret_name: str,
        new_secret_name: str,
        new_secret_value: str,
        optional_params: Mapping[str, object] | None = None,
        timeout: float | httpx.Timeout | None = None,
    ) -> Future[dict[str, JsonValue]]: ...
    def sync_read_secret(
        self,
        secret_name: str,
        optional_params: Mapping[str, object] | None = None,
        timeout: float | httpx.Timeout | None = None,
        primary_secret_name: str | None = None,
    ) -> JsonValue: ...
    def async_read_secret(
        self,
        secret_name: str,
        optional_params: Mapping[str, object] | None = None,
        timeout: float | httpx.Timeout | None = None,
        primary_secret_name: str | None = None,
    ) -> Future[JsonValue]: ...

@final
class NativeCacheHandle:
    def __new__(cls, _uninstantiable: Never, /) -> Never: ...
    @staticmethod
    def memory(
        *,
        ttl: float = 600.0,
        capacity: int = 200,
        max_entry_bytes: int = 4194304,
    ) -> NativeCacheHandle: ...
    @staticmethod
    def redis(
        url: str,
        *,
        namespace: str,
        ttl: float = 600.0,
        max_entry_bytes: int = 4194304,
    ) -> NativeCacheHandle: ...
    def get(self, key: str) -> object: ...
    def set(self, key: str, value: object, *, ttl: float | None = None) -> None: ...
    def async_get(self, key: str) -> Future[object]: ...
    def async_set(self, key: str, value: object, *, ttl: float | None = None) -> Future[None]: ...
    def async_set_many(self, entries: Sequence[tuple[str, object]], *, ttl: float | None = None) -> Future[None]: ...
    def flush(self) -> None: ...
    def async_flush(self) -> Future[None]: ...
    def ping(self) -> Future[bool]: ...
    def disconnect(self) -> Future[None]: ...
    def delete(self, keys: Sequence[str]) -> Future[None]: ...

# The FailureKind names an attempt outcome's `kind` takes (router-core `FailureKind`).
_FailureKind = Literal[
    "RateLimitError",
    "ContextWindowExceededError",
    "ContentPolicyViolationError",
    "AuthenticationError",
    "PermissionDeniedError",
    "NotFoundError",
    "BadRequestError",
    "Timeout",
    "APIConnectionError",
    "InternalServerError",
    "ServiceUnavailableError",
    "GuardrailIntervention",
    "Declined",
    "CallerCredentialError",
    "APIError",
]

class _AttemptStep(TypedDict):
    kind: Literal["attempt"]
    deployment_id: str
    # The deployment as a `model_list` entry, with its id in `model_info`.
    deployment: dict[str, JsonValue]
    # The name this attempt serves: the requested model or the fallback target.
    model: str
    model_group: str
    target_order: int | None
    candidate_ids: list[str]
    retry: int
    fallback_depth: int
    fallback_level: int
    timeout: float | None

class _SleepStep(TypedDict):
    kind: Literal["sleep"]
    seconds: float

class _AllowFallbackStep(TypedDict):
    kind: Literal["allow_fallback"]
    target: str

class _SelectStep(TypedDict):
    kind: Literal["select"]
    stage: Literal["plugins", "strategy"]
    strategy: str | None
    model_group: str
    candidates: list[str]

# One RouterCache op against the router's DualCache. `key` is set for every op but `batch_get`,
# which sets `keys`; `value` is the JSON to store (`set`) or the amount (`increment`); `ttl` is
# in seconds. Outcomes: get -> the JSON value or None; batch_get -> one JSON value or None per
# key, in order; increment -> the new int or float; set and delete -> None.
class _CacheStep(TypedDict):
    kind: Literal["cache"]
    op: Literal["get", "set", "increment", "batch_get", "delete"]
    key: str | None
    keys: list[str] | None
    value: JsonValue
    ttl: float | None

class _DoneStep(TypedDict):
    kind: Literal["done"]
    deployment_id: str
    model_group: str
    attempted_retries: int
    attempted_fallbacks: int
    total_tokens: int

class _FailedStep(TypedDict):
    kind: Literal["failed"]
    error: str
    error_kind: Literal["config", "unknown_model", "no_healthy_deployments", "attempt", "cache", "retired"]
    attempt_failed: bool
    status: int | None
    failure_kind: _FailureKind | None

# The end of a `RouterCoreHandle.select`.
class _SelectedStep(TypedDict):
    kind: Literal["done"]
    deployment_id: str
    model_group: str

# The end of a `RouterCoreHandle.healthy`.
class _HealthyStep(TypedDict):
    kind: Literal["done"]
    deployment_ids: list[str]

# The end of a `RouterCoreHandle.report_failure`.
class _ReportedStep(TypedDict):
    kind: Literal["done"]
    cooled: bool

# One `os.environ/` reference a `secrets` step asks for: a top-level param (`key`), or an
# env-only `dataSources[source].parameters[key]` (`env_only`).
class _SecretRef(TypedDict):
    name: str
    key: str
    env_only: bool
    source: NotRequired[int]

# The deployment build's `os.environ/` references: answered with one value per ref, in order.
class _SecretsStep(TypedDict):
    kind: Literal["secrets"]
    index: int
    refs: list[_SecretRef]

# The deployment build's steps for one deployment. `validate` and `added` are answered with
# `{"entry": <model_list entry>}`, `{"skip": True}` or `{"invalid": True}`; `provider` with
# `{"provider": str | None}` or `{"invalid": True}`.
class _BuildStep(TypedDict):
    kind: Literal["validate", "provider", "added"]
    path: Literal["create", "add"]
    index: int
    entry: dict[str, JsonValue]
    declared_id: str | None
    duplicate_id: bool
    provider: str | None

class _BuiltDeployment(TypedDict):
    index: int
    id: str | None
    provider: str | None

# The end of a `RouterCoreHandle.build`.
class _BuiltStep(TypedDict):
    kind: Literal["done"]
    built: list[_BuiltDeployment]

# `RouterCoreHandle.resolve`: what a name resolves to.
class _Resolution(TypedDict):
    model_group: str
    kind: str
    ids: list[str]

_RouterStep = _AttemptStep | _SleepStep | _AllowFallbackStep | _SelectStep | _CacheStep | _DoneStep | _FailedStep

class _AttemptSucceeded(TypedDict):
    ok: Literal[True]
    total_tokens: NotRequired[int]

class _AttemptFailed(TypedDict):
    ok: Literal[False]
    kind: _FailureKind
    status: NotRequired[int | None]
    retry_after: NotRequired[float | None]
    num_retries: NotRequired[int | None]
    detail: NotRequired[str | None]

# Each document is a mapping or its JSON text.
_Document = Mapping[str, object] | str

@final
class RouterCoreHandle:
    def __new__(cls, config: _Document, dynamic: _Document) -> RouterCoreHandle: ...
    def start(self, op: str, request: _Document) -> RouterCall: ...
    # Driven like `start`; its `done` step is a `_SelectedStep`.
    def select(self, model: str, request: _Document) -> RouterCall: ...
    # Driven like `select`; its `done` step is a `_HealthyStep`.
    def healthy(self, model: str, request: _Document) -> RouterCall: ...
    # `{"replace" | "append" | "upsert": [deployments], "delete": [ids], "settings": {dynamic document}, "index": {...}}`.
    def apply(self, changes: _Document) -> int: ...
    # Build model_list entries through the host, driven like `start`: `_SecretsStep` and
    # `_BuildStep` steps, then a `_BuiltStep`. `path`: "create" (`set_model_list`) or "add";
    # `install`: "replace", "append" or "upsert".
    def build(self, entries: Sequence[object], path: str, resolve_secrets: bool, install: str) -> RouterCall: ...
    # Keep the providers `other`'s builds resolved for the same deployments.
    def adopt_providers(self, other: RouterCoreHandle) -> None: ...
    # `get_router_model_info`, driven like `start`: `model_info` lookup steps (a `lookup` field
    # and its arguments), then a done step with `info` (the merged map or None) and `user_keys`.
    def router_model_info(self, query: _Document) -> RouterCall: ...
    # `get_deployment_model_info`, driven like `router_model_info`; its done step has `info`.
    def deployment_model_info(self, model_id: str, model_name: object) -> RouterCall: ...
    # `_set_model_group_info` for a `GroupInfoQuery` document, driven like `router_model_info`; its
    # done step has `value`: None, or `{"assignments": {field: value}}`.
    def model_group_info(self, query: _Document) -> RouterCall: ...
    # `get_configured_token_limits`, driven like `router_model_info`; `value` is `[input, output]`.
    def configured_token_limits(self, id: object, configured: _Document) -> RouterCall: ...
    # `get_model_listing_info`, driven like `router_model_info`; `value` has `cost_map_keys`,
    # `max_input_tokens` and `max_output_tokens`.
    def model_listing_info(self, deployments: list[object]) -> RouterCall: ...
    # A configured string that is not blank (`get_configured_mode`, `get_configured_display_name`).
    @staticmethod
    def configured_text(value: object) -> str | None: ...
    # Each model_info's `service_tiers`, unvalidated.
    @staticmethod
    def configured_service_tiers(model_infos: list[object]) -> list[object]: ...
    # The routing-cache key a deployment counts `metric` (tpm, rpm, itpm, otpm) under in `minute`.
    def usage_key(self, id: str, model: str, metric: str, minute: str) -> str: ...
    def registry_version(self) -> int: ...
    # The version the deployments before the newest appends last changed at.
    def registry_rewritten(self) -> int: ...
    # A failure the router did not see itself, driven like `start`: cache steps, then a
    # `_ReportedStep`.
    def report_failure(
        self,
        id: str,
        failure: Mapping[str, object],
        deployment_cooldown: bool = False,
        count_only: bool = False,
        requested_group: str | None = None,
        cooldown_time: float | None = None,
        uncounted: bool = False,
    ) -> RouterCall: ...
    # A success a logging callback saw, driven like `report_failure`; `done` carries `key`.
    def report_success(self, id: str, total_tokens: float, requests: float) -> RouterCall: ...
    # Count a success against any id, driven like `report_failure`; `done` carries `key`.
    def count_success(self, id: str) -> RouterCall: ...
    # [(id, tpm_key, rpm_key)] for this minute, or None for a group with no deployment.
    def usage_keys(self, model_group: str) -> list[tuple[str, str, str]] | None: ...
    # Every deployment as a `model_list` entry, in config order.
    def deployments(self) -> list[dict[str, JsonValue]]: ...
    def deployments_from(self, start: int) -> list[dict[str, JsonValue]]: ...
    def deployment(self, id: str) -> dict[str, JsonValue] | None: ...
    # Lookups over the registry indexes, answering what each index holds as written.
    # Registry lookups (router-core `lookup.rs`): rows the host reads its stored entries at
    # (`{"at": position, "name"?, "group"?, "member"?}`, `{"pattern": entry, "name"}`, or a
    # `{"raw": value}` / `{"raw_positions": value}` its own read raises at).
    def lookup_all(
        self, model_name: str, team_id: str | None = None, rename: str | None = None
    ) -> list[dict[str, JsonValue]]: ...
    def lookup_across_teams(self, model: str) -> tuple[list[dict[str, JsonValue]], int]: ...
    def lookup_config(self) -> list[dict[str, JsonValue]]: ...
    def lookup_model_ids(
        self, model_name: str | None = None, exclude_team_models: bool = False
    ) -> tuple[list[str], dict[str, JsonValue] | None]: ...
    def lookup_model_list(
        self, model_name: str | None, team_id: str | None, scope: Mapping[str, object]
    ) -> list[dict[str, JsonValue]]: ...
    def lookup_model_names(
        self, team_id: str | None, scope: Mapping[str, object]
    ) -> tuple[list[str], dict[str, JsonValue] | None]: ...
    def lookup_access_groups(
        self, model_name: str | None, access_group: str | None, team_id: str | None, scope: Mapping[str, object]
    ) -> tuple[list[tuple[str, list[str]]], dict[str, JsonValue] | None]: ...
    def lookup_name_serving(
        self, model_id: str, scope: Mapping[str, object]
    ) -> tuple[bool, str | None, dict[str, JsonValue] | None]: ...
    def has_model_name(self, name: str) -> bool: ...
    def lookup_team_model_has_alternatives(self, id: str) -> tuple[bool, dict[str, JsonValue] | None]: ...
    def lookup_group_usable_by_team(
        self, model_group: str, team_id: str | None
    ) -> tuple[int | None, dict[str, JsonValue] | None]: ...
    def lookup_team_public_name(
        self, model_id: str, team_id: str
    ) -> tuple[int | None, dict[str, JsonValue] | None]: ...
    def lookup_routable(self, model_group: str, team_id: str | None) -> list[dict[str, JsonValue]]: ...
    def lookup_wildcard_route(
        self, model: str, team_id: str | None, requests: list[str]
    ) -> list[dict[str, JsonValue]]: ...
    def lookup_candidate_ids(
        self, resolved: str, team_id: str | None, members: list[str] | None, requests: list[str]
    ) -> tuple[list[str], dict[str, JsonValue] | None]: ...
    def has_model_id(self, id: str) -> bool: ...
    def id_position(self, id: str) -> JsonValue | None: ...
    def id_positions(self) -> list[tuple[str, JsonValue]]: ...
    def named_positions(self, name: str) -> JsonValue | None: ...
    def team_positions(self, team: str, public: str) -> JsonValue | None: ...
    def team_public_names_of(self, team: str) -> list[str]: ...
    def litellm_model_positions(self, model: str) -> list[int]: ...
    # The registry indexes the host hands out (`HostIndexes`), as JSON.
    def host_indexes(self) -> dict[str, JsonValue]: ...
    def provider_default_ids(self) -> list[str]: ...
    # `PatternMatchRouter.route` over the wildcard router, or `team`'s.
    def pattern_route(
        self, team: str | None, request: str, filtered: list[str] | None = None
    ) -> list[dict[str, JsonValue]] | None: ...
    # `get_models_by_pattern` over the wildcard router, or `team`'s.
    def models_by_pattern(self, team: str | None, requests: list[str]) -> list[str] | None: ...
    # The provider the host resolved when it built deployment `id` (its `provider` build step).
    def provider(self, id: str) -> str | None: ...
    def resolve(self, name: str, specific_deployment: bool = False) -> _Resolution | None: ...
    def routable(self, model: str, request: _Document) -> list[dict[str, JsonValue]]: ...
    # `_get_routing_context` for a pick of `model`: (strategy, scope, group).
    def routing_context(self, model: str, routing_strategy: str | None = None) -> tuple[str, str, str | None]: ...
    def retire(self) -> None: ...
    # A stream that failed before content: what may still take over it, and the retry budget for
    # a failure of `kind`: {"can_retry", "can_fall_back", "refusal_fallback": bool,
    # "retry_budget": int | None, "policy_applies": bool}.
    def stream_policy(
        self, request: _Document, origin: _Document, kind: str | None = None
    ) -> dict[str, bool | int | None]: ...
    # The same-group retry and the fallback walk for such a stream, driven like `start`.
    def retry_stream(
        self, op: str, request: _Document, origin: _Document, failure: Mapping[str, object]
    ) -> RouterCall: ...
    def fallback_stream(
        self, op: str, request: _Document, origin: _Document, failure: Mapping[str, object]
    ) -> RouterCall: ...
    def retry_policy_retries(self, failure: Mapping[str, object], model_group: str | None = None) -> int | None: ...
    def allowed_fails_for(self, failure: Mapping[str, object]) -> int | None: ...
    def should_retry(
        self, failure: Mapping[str, object], counts: tuple[int, int], fallbacks: tuple[bool, bool, bool]
    ) -> bool: ...

@final
class RouterCall:
    # attempt: _AttemptSucceeded | _AttemptFailed; sleep: None; allow_fallback: bool;
    # select: the remaining candidate ids, pick first; cache: see _CacheStep; first call: None.
    # A call from `select` steps through select and cache only, and ends in _SelectedStep or failed.
    def next(
        self, outcome: _AttemptSucceeded | _AttemptFailed | bool | Sequence[str] | JsonValue | None
    ) -> _RouterStep | _SelectedStep | _HealthyStep | _ReportedStep | _SecretsStep | _BuildStep | _BuiltStep: ...
