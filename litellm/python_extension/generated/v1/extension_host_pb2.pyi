from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class ExtensionKind(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    EXTENSION_KIND_UNSPECIFIED: _ClassVar[ExtensionKind]
    EXTENSION_KIND_CALLBACK: _ClassVar[ExtensionKind]
    EXTENSION_KIND_GUARDRAIL: _ClassVar[ExtensionKind]

class HookPhase(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    HOOK_PHASE_UNSPECIFIED: _ClassVar[HookPhase]
    HOOK_PHASE_PRE_CALL: _ClassVar[HookPhase]
    HOOK_PHASE_DURING_CALL: _ClassVar[HookPhase]
    HOOK_PHASE_POST_CALL: _ClassVar[HookPhase]

class GuardrailDecision(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    GUARDRAIL_DECISION_UNSPECIFIED: _ClassVar[GuardrailDecision]
    GUARDRAIL_DECISION_ALLOW: _ClassVar[GuardrailDecision]
    GUARDRAIL_DECISION_REPLACE_REQUEST: _ClassVar[GuardrailDecision]
    GUARDRAIL_DECISION_REPLACE_RESPONSE: _ClassVar[GuardrailDecision]
    GUARDRAIL_DECISION_BLOCK: _ClassVar[GuardrailDecision]
    GUARDRAIL_DECISION_ERROR: _ClassVar[GuardrailDecision]

class CallbackEventKind(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    CALLBACK_EVENT_KIND_UNSPECIFIED: _ClassVar[CallbackEventKind]
    CALLBACK_EVENT_KIND_SUCCESS: _ClassVar[CallbackEventKind]
    CALLBACK_EVENT_KIND_FAILURE: _ClassVar[CallbackEventKind]

class StreamFrameKind(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    STREAM_FRAME_KIND_UNSPECIFIED: _ClassVar[StreamFrameKind]
    STREAM_FRAME_KIND_OPEN: _ClassVar[StreamFrameKind]
    STREAM_FRAME_KIND_INPUT_CHUNK: _ClassVar[StreamFrameKind]
    STREAM_FRAME_KIND_OUTPUT_CHUNK: _ClassVar[StreamFrameKind]
    STREAM_FRAME_KIND_END: _ClassVar[StreamFrameKind]
    STREAM_FRAME_KIND_ERROR: _ClassVar[StreamFrameKind]

class ErrorCode(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    ERROR_CODE_UNSPECIFIED: _ClassVar[ErrorCode]
    ERROR_CODE_INVALID_ARGUMENT: _ClassVar[ErrorCode]
    ERROR_CODE_ALREADY_EXISTS: _ClassVar[ErrorCode]
    ERROR_CODE_NOT_FOUND: _ClassVar[ErrorCode]
    ERROR_CODE_NOT_ACTIVE: _ClassVar[ErrorCode]
    ERROR_CODE_LOAD_FAILED: _ClassVar[ErrorCode]
    ERROR_CODE_EXTENSION_FAILED: _ClassVar[ErrorCode]
    ERROR_CODE_UNSUPPORTED_CAPABILITY: _ClassVar[ErrorCode]
    ERROR_CODE_UNSUPPORTED_HOOK: _ClassVar[ErrorCode]
    ERROR_CODE_SERIALIZATION_FAILED: _ClassVar[ErrorCode]
    ERROR_CODE_TRANSPORT_FAILED: _ClassVar[ErrorCode]
    ERROR_CODE_UNAUTHENTICATED: _ClassVar[ErrorCode]
    ERROR_CODE_VERSION_MISMATCH: _ClassVar[ErrorCode]
EXTENSION_KIND_UNSPECIFIED: ExtensionKind
EXTENSION_KIND_CALLBACK: ExtensionKind
EXTENSION_KIND_GUARDRAIL: ExtensionKind
HOOK_PHASE_UNSPECIFIED: HookPhase
HOOK_PHASE_PRE_CALL: HookPhase
HOOK_PHASE_DURING_CALL: HookPhase
HOOK_PHASE_POST_CALL: HookPhase
GUARDRAIL_DECISION_UNSPECIFIED: GuardrailDecision
GUARDRAIL_DECISION_ALLOW: GuardrailDecision
GUARDRAIL_DECISION_REPLACE_REQUEST: GuardrailDecision
GUARDRAIL_DECISION_REPLACE_RESPONSE: GuardrailDecision
GUARDRAIL_DECISION_BLOCK: GuardrailDecision
GUARDRAIL_DECISION_ERROR: GuardrailDecision
CALLBACK_EVENT_KIND_UNSPECIFIED: CallbackEventKind
CALLBACK_EVENT_KIND_SUCCESS: CallbackEventKind
CALLBACK_EVENT_KIND_FAILURE: CallbackEventKind
STREAM_FRAME_KIND_UNSPECIFIED: StreamFrameKind
STREAM_FRAME_KIND_OPEN: StreamFrameKind
STREAM_FRAME_KIND_INPUT_CHUNK: StreamFrameKind
STREAM_FRAME_KIND_OUTPUT_CHUNK: StreamFrameKind
STREAM_FRAME_KIND_END: StreamFrameKind
STREAM_FRAME_KIND_ERROR: StreamFrameKind
ERROR_CODE_UNSPECIFIED: ErrorCode
ERROR_CODE_INVALID_ARGUMENT: ErrorCode
ERROR_CODE_ALREADY_EXISTS: ErrorCode
ERROR_CODE_NOT_FOUND: ErrorCode
ERROR_CODE_NOT_ACTIVE: ErrorCode
ERROR_CODE_LOAD_FAILED: ErrorCode
ERROR_CODE_EXTENSION_FAILED: ErrorCode
ERROR_CODE_UNSUPPORTED_CAPABILITY: ErrorCode
ERROR_CODE_UNSUPPORTED_HOOK: ErrorCode
ERROR_CODE_SERIALIZATION_FAILED: ErrorCode
ERROR_CODE_TRANSPORT_FAILED: ErrorCode
ERROR_CODE_UNAUTHENTICATED: ErrorCode
ERROR_CODE_VERSION_MISMATCH: ErrorCode

class GetCapabilitiesRequest(_message.Message):
    __slots__ = ("protocol_major", "protocol_minor")
    PROTOCOL_MAJOR_FIELD_NUMBER: _ClassVar[int]
    PROTOCOL_MINOR_FIELD_NUMBER: _ClassVar[int]
    protocol_major: int
    protocol_minor: int
    def __init__(self, protocol_major: _Optional[int] = ..., protocol_minor: _Optional[int] = ...) -> None: ...

class HostCapabilities(_message.Message):
    __slots__ = ("protocol_major", "protocol_minor", "supported_hooks", "supports_duplex_streaming", "supports_callback_batching", "supports_cache", "max_callback_batch_size")
    PROTOCOL_MAJOR_FIELD_NUMBER: _ClassVar[int]
    PROTOCOL_MINOR_FIELD_NUMBER: _ClassVar[int]
    SUPPORTED_HOOKS_FIELD_NUMBER: _ClassVar[int]
    SUPPORTS_DUPLEX_STREAMING_FIELD_NUMBER: _ClassVar[int]
    SUPPORTS_CALLBACK_BATCHING_FIELD_NUMBER: _ClassVar[int]
    SUPPORTS_CACHE_FIELD_NUMBER: _ClassVar[int]
    MAX_CALLBACK_BATCH_SIZE_FIELD_NUMBER: _ClassVar[int]
    protocol_major: int
    protocol_minor: int
    supported_hooks: _containers.RepeatedScalarFieldContainer[str]
    supports_duplex_streaming: bool
    supports_callback_batching: bool
    supports_cache: bool
    max_callback_batch_size: int
    def __init__(self, protocol_major: _Optional[int] = ..., protocol_minor: _Optional[int] = ..., supported_hooks: _Optional[_Iterable[str]] = ..., supports_duplex_streaming: bool = ..., supports_callback_batching: bool = ..., supports_cache: bool = ..., max_callback_batch_size: _Optional[int] = ...) -> None: ...

class ExtensionSpec(_message.Message):
    __slots__ = ("id", "kind", "entrypoint", "constructor_json")
    ID_FIELD_NUMBER: _ClassVar[int]
    KIND_FIELD_NUMBER: _ClassVar[int]
    ENTRYPOINT_FIELD_NUMBER: _ClassVar[int]
    CONSTRUCTOR_JSON_FIELD_NUMBER: _ClassVar[int]
    id: str
    kind: ExtensionKind
    entrypoint: str
    constructor_json: bytes
    def __init__(self, id: _Optional[str] = ..., kind: _Optional[_Union[ExtensionKind, str]] = ..., entrypoint: _Optional[str] = ..., constructor_json: _Optional[bytes] = ...) -> None: ...

class ExtensionDescriptor(_message.Message):
    __slots__ = ("id", "kind", "hooks", "callable", "async_callable")
    ID_FIELD_NUMBER: _ClassVar[int]
    KIND_FIELD_NUMBER: _ClassVar[int]
    HOOKS_FIELD_NUMBER: _ClassVar[int]
    CALLABLE_FIELD_NUMBER: _ClassVar[int]
    ASYNC_CALLABLE_FIELD_NUMBER: _ClassVar[int]
    id: str
    kind: ExtensionKind
    hooks: _containers.RepeatedScalarFieldContainer[str]
    callable: bool
    async_callable: bool
    def __init__(self, id: _Optional[str] = ..., kind: _Optional[_Union[ExtensionKind, str]] = ..., hooks: _Optional[_Iterable[str]] = ..., callable: bool = ..., async_callable: bool = ...) -> None: ...

class PrepareRevisionRequest(_message.Message):
    __slots__ = ("revision_id", "extensions")
    REVISION_ID_FIELD_NUMBER: _ClassVar[int]
    EXTENSIONS_FIELD_NUMBER: _ClassVar[int]
    revision_id: str
    extensions: _containers.RepeatedCompositeFieldContainer[ExtensionSpec]
    def __init__(self, revision_id: _Optional[str] = ..., extensions: _Optional[_Iterable[_Union[ExtensionSpec, _Mapping]]] = ...) -> None: ...

class PrepareRevisionResponse(_message.Message):
    __slots__ = ("operation", "extensions")
    OPERATION_FIELD_NUMBER: _ClassVar[int]
    EXTENSIONS_FIELD_NUMBER: _ClassVar[int]
    operation: OperationResult
    extensions: _containers.RepeatedCompositeFieldContainer[ExtensionDescriptor]
    def __init__(self, operation: _Optional[_Union[OperationResult, _Mapping]] = ..., extensions: _Optional[_Iterable[_Union[ExtensionDescriptor, _Mapping]]] = ...) -> None: ...

class CommitRevisionRequest(_message.Message):
    __slots__ = ("revision_id",)
    REVISION_ID_FIELD_NUMBER: _ClassVar[int]
    revision_id: str
    def __init__(self, revision_id: _Optional[str] = ...) -> None: ...

class RetireRevisionRequest(_message.Message):
    __slots__ = ("revision_id",)
    REVISION_ID_FIELD_NUMBER: _ClassVar[int]
    revision_id: str
    def __init__(self, revision_id: _Optional[str] = ...) -> None: ...

class InvocationContext(_message.Message):
    __slots__ = ("request_id", "invocation_id", "active_revision", "api_surface", "call_type", "trace_context")
    class TraceContextEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    INVOCATION_ID_FIELD_NUMBER: _ClassVar[int]
    ACTIVE_REVISION_FIELD_NUMBER: _ClassVar[int]
    API_SURFACE_FIELD_NUMBER: _ClassVar[int]
    CALL_TYPE_FIELD_NUMBER: _ClassVar[int]
    TRACE_CONTEXT_FIELD_NUMBER: _ClassVar[int]
    request_id: str
    invocation_id: str
    active_revision: str
    api_surface: str
    call_type: str
    trace_context: _containers.ScalarMap[str, str]
    def __init__(self, request_id: _Optional[str] = ..., invocation_id: _Optional[str] = ..., active_revision: _Optional[str] = ..., api_surface: _Optional[str] = ..., call_type: _Optional[str] = ..., trace_context: _Optional[_Mapping[str, str]] = ...) -> None: ...

class AuthContext(_message.Message):
    __slots__ = ("key_hash", "user_id", "team_id", "request_metadata")
    class RequestMetadataEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    KEY_HASH_FIELD_NUMBER: _ClassVar[int]
    USER_ID_FIELD_NUMBER: _ClassVar[int]
    TEAM_ID_FIELD_NUMBER: _ClassVar[int]
    REQUEST_METADATA_FIELD_NUMBER: _ClassVar[int]
    key_hash: str
    user_id: str
    team_id: str
    request_metadata: _containers.ScalarMap[str, str]
    def __init__(self, key_hash: _Optional[str] = ..., user_id: _Optional[str] = ..., team_id: _Optional[str] = ..., request_metadata: _Optional[_Mapping[str, str]] = ...) -> None: ...

class CacheRef(_message.Message):
    __slots__ = ("invocation_id", "opaque_handle")
    INVOCATION_ID_FIELD_NUMBER: _ClassVar[int]
    OPAQUE_HANDLE_FIELD_NUMBER: _ClassVar[int]
    invocation_id: str
    opaque_handle: str
    def __init__(self, invocation_id: _Optional[str] = ..., opaque_handle: _Optional[str] = ...) -> None: ...

class GuardrailInvocation(_message.Message):
    __slots__ = ("context", "plugin_id", "hook_phase", "request_json", "response_json", "auth", "cache")
    CONTEXT_FIELD_NUMBER: _ClassVar[int]
    PLUGIN_ID_FIELD_NUMBER: _ClassVar[int]
    HOOK_PHASE_FIELD_NUMBER: _ClassVar[int]
    REQUEST_JSON_FIELD_NUMBER: _ClassVar[int]
    RESPONSE_JSON_FIELD_NUMBER: _ClassVar[int]
    AUTH_FIELD_NUMBER: _ClassVar[int]
    CACHE_FIELD_NUMBER: _ClassVar[int]
    context: InvocationContext
    plugin_id: str
    hook_phase: HookPhase
    request_json: bytes
    response_json: bytes
    auth: AuthContext
    cache: CacheRef
    def __init__(self, context: _Optional[_Union[InvocationContext, _Mapping]] = ..., plugin_id: _Optional[str] = ..., hook_phase: _Optional[_Union[HookPhase, str]] = ..., request_json: _Optional[bytes] = ..., response_json: _Optional[bytes] = ..., auth: _Optional[_Union[AuthContext, _Mapping]] = ..., cache: _Optional[_Union[CacheRef, _Mapping]] = ...) -> None: ...

class PublicError(_message.Message):
    __slots__ = ("type", "message", "status_code", "param")
    TYPE_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    STATUS_CODE_FIELD_NUMBER: _ClassVar[int]
    PARAM_FIELD_NUMBER: _ClassVar[int]
    type: str
    message: str
    status_code: int
    param: str
    def __init__(self, type: _Optional[str] = ..., message: _Optional[str] = ..., status_code: _Optional[int] = ..., param: _Optional[str] = ...) -> None: ...

class GuardrailResult(_message.Message):
    __slots__ = ("operation", "decision", "request_json", "response_json", "public_error")
    OPERATION_FIELD_NUMBER: _ClassVar[int]
    DECISION_FIELD_NUMBER: _ClassVar[int]
    REQUEST_JSON_FIELD_NUMBER: _ClassVar[int]
    RESPONSE_JSON_FIELD_NUMBER: _ClassVar[int]
    PUBLIC_ERROR_FIELD_NUMBER: _ClassVar[int]
    operation: OperationResult
    decision: GuardrailDecision
    request_json: bytes
    response_json: bytes
    public_error: PublicError
    def __init__(self, operation: _Optional[_Union[OperationResult, _Mapping]] = ..., decision: _Optional[_Union[GuardrailDecision, str]] = ..., request_json: _Optional[bytes] = ..., response_json: _Optional[bytes] = ..., public_error: _Optional[_Union[PublicError, _Mapping]] = ...) -> None: ...

class CallbackEvent(_message.Message):
    __slots__ = ("context", "plugin_id", "kind", "standard_logging_payload_json", "response_json", "error_json", "start_time_seconds", "end_time_seconds", "auth", "cache", "streaming")
    CONTEXT_FIELD_NUMBER: _ClassVar[int]
    PLUGIN_ID_FIELD_NUMBER: _ClassVar[int]
    KIND_FIELD_NUMBER: _ClassVar[int]
    STANDARD_LOGGING_PAYLOAD_JSON_FIELD_NUMBER: _ClassVar[int]
    RESPONSE_JSON_FIELD_NUMBER: _ClassVar[int]
    ERROR_JSON_FIELD_NUMBER: _ClassVar[int]
    START_TIME_SECONDS_FIELD_NUMBER: _ClassVar[int]
    END_TIME_SECONDS_FIELD_NUMBER: _ClassVar[int]
    AUTH_FIELD_NUMBER: _ClassVar[int]
    CACHE_FIELD_NUMBER: _ClassVar[int]
    STREAMING_FIELD_NUMBER: _ClassVar[int]
    context: InvocationContext
    plugin_id: str
    kind: CallbackEventKind
    standard_logging_payload_json: bytes
    response_json: bytes
    error_json: bytes
    start_time_seconds: float
    end_time_seconds: float
    auth: AuthContext
    cache: CacheRef
    streaming: bool
    def __init__(self, context: _Optional[_Union[InvocationContext, _Mapping]] = ..., plugin_id: _Optional[str] = ..., kind: _Optional[_Union[CallbackEventKind, str]] = ..., standard_logging_payload_json: _Optional[bytes] = ..., response_json: _Optional[bytes] = ..., error_json: _Optional[bytes] = ..., start_time_seconds: _Optional[float] = ..., end_time_seconds: _Optional[float] = ..., auth: _Optional[_Union[AuthContext, _Mapping]] = ..., cache: _Optional[_Union[CacheRef, _Mapping]] = ..., streaming: bool = ...) -> None: ...

class PublishCallbackEventsRequest(_message.Message):
    __slots__ = ("events",)
    EVENTS_FIELD_NUMBER: _ClassVar[int]
    events: _containers.RepeatedCompositeFieldContainer[CallbackEvent]
    def __init__(self, events: _Optional[_Iterable[_Union[CallbackEvent, _Mapping]]] = ...) -> None: ...

class PublishCallbackEventsResponse(_message.Message):
    __slots__ = ("operations",)
    OPERATIONS_FIELD_NUMBER: _ClassVar[int]
    operations: _containers.RepeatedCompositeFieldContainer[OperationResult]
    def __init__(self, operations: _Optional[_Iterable[_Union[OperationResult, _Mapping]]] = ...) -> None: ...

class StreamOpen(_message.Message):
    __slots__ = ("context", "plugin_id", "request_json", "auth", "cache", "iterator_hook")
    CONTEXT_FIELD_NUMBER: _ClassVar[int]
    PLUGIN_ID_FIELD_NUMBER: _ClassVar[int]
    REQUEST_JSON_FIELD_NUMBER: _ClassVar[int]
    AUTH_FIELD_NUMBER: _ClassVar[int]
    CACHE_FIELD_NUMBER: _ClassVar[int]
    ITERATOR_HOOK_FIELD_NUMBER: _ClassVar[int]
    context: InvocationContext
    plugin_id: str
    request_json: bytes
    auth: AuthContext
    cache: CacheRef
    iterator_hook: bool
    def __init__(self, context: _Optional[_Union[InvocationContext, _Mapping]] = ..., plugin_id: _Optional[str] = ..., request_json: _Optional[bytes] = ..., auth: _Optional[_Union[AuthContext, _Mapping]] = ..., cache: _Optional[_Union[CacheRef, _Mapping]] = ..., iterator_hook: bool = ...) -> None: ...

class StreamFrame(_message.Message):
    __slots__ = ("kind", "stream_id", "open", "chunk_json", "error")
    KIND_FIELD_NUMBER: _ClassVar[int]
    STREAM_ID_FIELD_NUMBER: _ClassVar[int]
    OPEN_FIELD_NUMBER: _ClassVar[int]
    CHUNK_JSON_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    kind: StreamFrameKind
    stream_id: str
    open: StreamOpen
    chunk_json: bytes
    error: PublicError
    def __init__(self, kind: _Optional[_Union[StreamFrameKind, str]] = ..., stream_id: _Optional[str] = ..., open: _Optional[_Union[StreamOpen, _Mapping]] = ..., chunk_json: _Optional[bytes] = ..., error: _Optional[_Union[PublicError, _Mapping]] = ...) -> None: ...

class CacheGetRequest(_message.Message):
    __slots__ = ("cache", "key", "local_only")
    CACHE_FIELD_NUMBER: _ClassVar[int]
    KEY_FIELD_NUMBER: _ClassVar[int]
    LOCAL_ONLY_FIELD_NUMBER: _ClassVar[int]
    cache: CacheRef
    key: str
    local_only: bool
    def __init__(self, cache: _Optional[_Union[CacheRef, _Mapping]] = ..., key: _Optional[str] = ..., local_only: bool = ...) -> None: ...

class CacheGetResponse(_message.Message):
    __slots__ = ("operation", "value_json")
    OPERATION_FIELD_NUMBER: _ClassVar[int]
    VALUE_JSON_FIELD_NUMBER: _ClassVar[int]
    operation: OperationResult
    value_json: bytes
    def __init__(self, operation: _Optional[_Union[OperationResult, _Mapping]] = ..., value_json: _Optional[bytes] = ...) -> None: ...

class CacheSetRequest(_message.Message):
    __slots__ = ("cache", "key", "value_json", "ttl_seconds", "local_only")
    CACHE_FIELD_NUMBER: _ClassVar[int]
    KEY_FIELD_NUMBER: _ClassVar[int]
    VALUE_JSON_FIELD_NUMBER: _ClassVar[int]
    TTL_SECONDS_FIELD_NUMBER: _ClassVar[int]
    LOCAL_ONLY_FIELD_NUMBER: _ClassVar[int]
    cache: CacheRef
    key: str
    value_json: bytes
    ttl_seconds: float
    local_only: bool
    def __init__(self, cache: _Optional[_Union[CacheRef, _Mapping]] = ..., key: _Optional[str] = ..., value_json: _Optional[bytes] = ..., ttl_seconds: _Optional[float] = ..., local_only: bool = ...) -> None: ...

class OperationResult(_message.Message):
    __slots__ = ("ok", "error_code", "error_message")
    OK_FIELD_NUMBER: _ClassVar[int]
    ERROR_CODE_FIELD_NUMBER: _ClassVar[int]
    ERROR_MESSAGE_FIELD_NUMBER: _ClassVar[int]
    ok: bool
    error_code: ErrorCode
    error_message: str
    def __init__(self, ok: bool = ..., error_code: _Optional[_Union[ErrorCode, str]] = ..., error_message: _Optional[str] = ...) -> None: ...
