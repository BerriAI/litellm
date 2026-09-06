from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar


@dataclass(frozen=True, slots=True)
class NativeBedrockOptions:
    aws_access_key_id: str | None = None
    aws_secret_access_key: str | None = None
    aws_session_token: str | None = None
    aws_region_name: str | None = None
    aws_session_name: str | None = None
    aws_profile_name: str | None = None
    aws_role_name: str | None = None
    aws_web_identity_token: str | None = None
    aws_sts_endpoint: str | None = None
    aws_external_id: str | None = None
    aws_bedrock_runtime_endpoint: str | None = None
    request_metadata_fields: tuple[str, ...] = ()
    request_metadata: Mapping[str, str] | None = None


@dataclass(frozen=True, slots=True)
class NativeAnthropicOptions:
    user_id: str | None = None


@dataclass(frozen=True, slots=True)
class NativeVertexOptions:
    project: str | None = None
    location: str | None = None


def bedrock_options(params: Mapping[str, object]) -> NativeBedrockOptions:
    def string(name: str) -> str | None:
        value = params.get(name)
        return value if isinstance(value, str) else None

    return NativeBedrockOptions(
        aws_access_key_id=string("aws_access_key_id"),
        aws_secret_access_key=string("aws_secret_access_key"),
        aws_session_token=string("aws_session_token"),
        aws_region_name=string("aws_region_name"),
        aws_session_name=string("aws_session_name"),
        aws_profile_name=string("aws_profile_name"),
        aws_role_name=string("aws_role_name"),
        aws_web_identity_token=string("aws_web_identity_token"),
        aws_sts_endpoint=string("aws_sts_endpoint"),
        aws_external_id=string("aws_external_id"),
        aws_bedrock_runtime_endpoint=string("aws_bedrock_runtime_endpoint"),
    )


def anthropic_options(litellm_params: Mapping[str, object] | None) -> NativeAnthropicOptions:
    metadata = None if litellm_params is None else litellm_params.get("metadata")
    user_id = metadata.get("user_id") if isinstance(metadata, Mapping) else None
    return NativeAnthropicOptions(user_id=user_id if isinstance(user_id, str) else None)


def vertex_options(params: Mapping[str, object]) -> NativeVertexOptions:
    project = params.get("vertex_project") or params.get("vertex_ai_project")
    location = params.get("vertex_location") or params.get("vertex_ai_location")
    return NativeVertexOptions(
        project=project if isinstance(project, str) else None,
        location=location if isinstance(location, str) else None,
    )


@dataclass(frozen=True, slots=True)
class NativeRequestOptions:
    api_key: str | None = None
    api_base: str | None = None
    custom_llm_provider: str | None = None
    extra_headers: Mapping[str, object] | None = None
    extra_query: Mapping[str, object] | None = None
    timeout_seconds: float | None = None
    bedrock: NativeBedrockOptions | None = None
    anthropic: NativeAnthropicOptions | None = None
    vertex: NativeVertexOptions | None = None


@dataclass(frozen=True, slots=True)
class RequestAttribution:
    user_api_key_hash: str | None = None
    user_api_key_user_id: str | None = None
    user_api_key_team_id: str | None = None


@dataclass(frozen=True, slots=True)
class NativeRequestCapabilities:
    execution_mode: str | None = None
    stream: bool = False
    has_agentic_hook: bool = False
    has_custom_client: bool = False
    request_format: str | None = None
    input_source_kind: str | None = None
    native_response_format: bool = False
    websocket_mode: str | None = None
    requires_connection: bool = False


@dataclass(frozen=True, slots=True)
class NativeRequestContext:
    litellm_call_id: str | None = None
    trace_id: str | None = None
    request_model: str | None = None
    attribution: RequestAttribution = RequestAttribution()
    capabilities: NativeRequestCapabilities = NativeRequestCapabilities()


RequestT = TypeVar("RequestT")
RequestContraT = TypeVar("RequestContraT", contravariant=True)
ResultT = TypeVar("ResultT", covariant=True)


@dataclass(frozen=True, slots=True)
class PreparedNativeCall(Generic[RequestT]):
    request: RequestT
    options: NativeRequestOptions = NativeRequestOptions()
    context: NativeRequestContext = NativeRequestContext()


class NativeFunction(Protocol[RequestContraT, ResultT]):
    def __call__(
        self,
        request: RequestContraT,
        *,
        options: NativeRequestOptions,
        context: NativeRequestContext,
    ) -> ResultT: ...


def call_native(native: NativeFunction[RequestT, ResultT], prepared: PreparedNativeCall[RequestT]) -> ResultT:
    return native(prepared.request, options=prepared.options, context=prepared.context)


@dataclass(frozen=True, slots=True)
class NativeChatCompletionsRequest:
    model: str
    messages: Sequence[object]
    optional_params: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class NativeMessagesRequest:
    model: str
    body: dict[str, object]


@dataclass(frozen=True, slots=True)
class NativeOCRRequest:
    model: str
    document: dict[str, object]
    optional_params: dict[str, object]


@dataclass(frozen=True, slots=True)
class NativeTranscriptionRequest:
    model: str
    audio: dict[str, object]
    optional_params: dict[str, object]


@dataclass(frozen=True, slots=True)
class NativeResponsesWebSocketRequest:
    url: str
