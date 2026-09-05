from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final, Generic, Protocol

from typing_extensions import ReadOnly, TypedDict, TypeVar

from .callbacks import OneShotCallbackHandle


class NativePreCallDetails(TypedDict):
    complete_input_dict: ReadOnly[Mapping[str, object]]
    api_base: ReadOnly[str]
    headers: ReadOnly[Mapping[str, object] | None]


@dataclass(frozen=True, slots=True)
class NativeRequestOptions:
    api_key: str | None = None
    api_base: str | None = None
    custom_llm_provider: str | None = None
    extra_headers: Mapping[str, object] | None = None
    extra_query: Mapping[str, object] | None = None
    timeout_seconds: float | None = None
    provider_connection: Mapping[str, object] | None = None


@dataclass(frozen=True, slots=True)
class RequestAttribution:
    user_api_key_hash: str | None = None
    user_api_key_user_id: str | None = None
    user_api_key_team_id: str | None = None


@dataclass(frozen=True, slots=True)
class NativeRequestContext:
    metadata: Mapping[str, object] | None = None
    litellm_metadata: Mapping[str, object] | None = None
    request_metadata_fields: tuple[str, ...] = ()
    litellm_call_id: str | None = None
    request_model: str | None = None
    attribution: RequestAttribution = RequestAttribution()


RequestT = TypeVar("RequestT")
RequestContraT = TypeVar("RequestContraT", contravariant=True)
ResultT = TypeVar("ResultT", covariant=True)
CallbackT = TypeVar("CallbackT", default=OneShotCallbackHandle)
CallbackContraT = TypeVar("CallbackContraT", contravariant=True, default=OneShotCallbackHandle)


@dataclass(frozen=True, slots=True)
class PreparedNativeCall(Generic[RequestT, CallbackT]):
    request: RequestT
    context: NativeRequestContext = NativeRequestContext()
    callback_adapter: CallbackT | None = None


class NativeFunction(Protocol[RequestContraT, ResultT, CallbackContraT]):
    def __call__(
        self, request: RequestContraT, *, context: NativeRequestContext, callback_adapter: CallbackContraT | None = None
    ) -> ResultT: ...


def call_native(
    native: NativeFunction[RequestT, ResultT, CallbackT], prepared: PreparedNativeCall[RequestT, CallbackT]
) -> ResultT:
    return native(prepared.request, context=prepared.context, callback_adapter=prepared.callback_adapter)


_PROVIDER_CONNECTION_FIELDS: Final = frozenset(
    (
        "aws_access_key_id",
        "aws_secret_access_key",
        "aws_session_token",
        "aws_region_name",
        "aws_session_name",
        "aws_profile_name",
        "aws_role_name",
        "aws_web_identity_token",
        "aws_sts_endpoint",
        "aws_external_id",
        "aws_bedrock_runtime_endpoint",
        "vertex_project",
        "vertex_ai_project",
        "vertex_location",
        "vertex_ai_location",
    )
)


def provider_connection_params(params: Mapping[str, object]) -> dict[str, object]:
    return {  # mutable-ok: PyO3 boundary payload
        key: value for key, value in params.items() if key in _PROVIDER_CONNECTION_FIELDS
    }


def provider_request_params(params: Mapping[str, object]) -> dict[str, object]:
    return {  # mutable-ok: PyO3 boundary payload
        key: value for key, value in params.items() if key not in _PROVIDER_CONNECTION_FIELDS
    }


@dataclass(frozen=True, slots=True)
class NativeChatCompletionsRequest:
    model: str
    messages: Sequence[object]
    optional_params: Mapping[str, object]
    options: NativeRequestOptions


@dataclass(frozen=True, slots=True)
class NativeMessagesRequest:
    model: str
    body: dict[str, object]
    options: NativeRequestOptions


@dataclass(frozen=True, slots=True)
class NativeOCRRequest:
    model: str
    document: dict[str, object]
    optional_params: dict[str, object]
    options: NativeRequestOptions


@dataclass(frozen=True, slots=True)
class NativeTranscriptionRequest:
    model: str
    audio: dict[str, object]
    optional_params: dict[str, object]
    options: NativeRequestOptions


@dataclass(frozen=True, slots=True)
class NativeResponsesWebSocketRequest:
    url: str
    options: NativeRequestOptions
