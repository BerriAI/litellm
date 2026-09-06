from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, Protocol

import httpx
from pydantic import TypeAdapter

from litellm._logging import verbose_logger
from litellm.exceptions import APIError
from litellm.rust_bridge.configuration import rust_enabled
from litellm.rust_bridge.loader import get_native_bridge
from litellm.rust_bridge.timeouts import timeout_to_seconds
from litellm.types.llms.openai import ResponsesAPIResponse
from litellm.types.responses.main import GenericResponseOutputItem

RUST_RESPONSES_PROVIDERS: Final = frozenset({"openai", "anthropic", "bedrock"})
_GENERIC_OUTPUT_ADAPTER: Final = TypeAdapter(list[GenericResponseOutputItem])


class RustResponses(Protocol):
    def __call__(
        self,
        *,
        model: str,
        input: str | Sequence[object],
        optional_params: Mapping[str, object] | None,
        api_key: str | None,
        api_base: str | None,
        custom_llm_provider: str | None,
        extra_headers: Mapping[str, object] | None,
        timeout_seconds: float | None,
        use_chat_completions_api: bool | None,
    ) -> Mapping[str, object]: ...


class RustResponsesDecline(Protocol):
    def __call__(
        self,
        *,
        model: str,
        input: str | Sequence[object],
        optional_params: Mapping[str, object] | None,
        custom_llm_provider: str | None,
        use_chat_completions_api: bool,
    ) -> str | None: ...


class _Unset:
    pass


_UNSET: Final = _Unset()


@dataclass(slots=True)
class _RustResponsesState:
    responses: RustResponses | None = None
    decline: RustResponsesDecline | None = None


_STATE: Final = _RustResponsesState()


def set_rust_responses(
    *,
    responses: RustResponses | None | _Unset = _UNSET,
    decline: RustResponsesDecline | None | _Unset = _UNSET,
) -> None:
    if not isinstance(responses, _Unset):
        _STATE.responses = responses
    if not isinstance(decline, _Unset):
        _STATE.decline = decline


def _load_responses() -> RustResponses | None:
    if _STATE.responses is not None:
        return _STATE.responses
    native_bridge: Final = get_native_bridge()
    if native_bridge is None:
        return None
    loaded: Final[RustResponses | None] = getattr(native_bridge, "responses", None)
    return loaded


def _load_decline() -> RustResponsesDecline | None:
    if _STATE.decline is not None:
        return _STATE.decline
    native_bridge: Final = get_native_bridge()
    if native_bridge is None:
        return None
    loaded: Final[RustResponsesDecline | None] = getattr(native_bridge, "responses_decline", None)
    return loaded


def rust_responses_accepts(
    *,
    model: str,
    input: str | Sequence[object],
    optional_params: Mapping[str, object],
    custom_llm_provider: str | None,
    use_chat_completions_api: bool,
    stream: object,
    request_override: bool | None,
    extra_body: Mapping[str, object] | None,
    extra_query: Mapping[str, object] | None,
) -> bool:
    if custom_llm_provider not in RUST_RESPONSES_PROVIDERS or stream:
        return False
    if extra_body or extra_query or not rust_enabled(request_override=request_override):
        return False
    decline: Final = _load_decline()
    if decline is None:
        return False
    try:
        reason: Final = decline(
            model=model,
            input=input,
            optional_params=optional_params,
            custom_llm_provider=custom_llm_provider,
            use_chat_completions_api=use_chat_completions_api,
        )
    except Exception as rust_error:  # noqa: BLE001  # rollout gate failures safely stay on Python
        verbose_logger.debug("Rust Responses gate raised %s", type(rust_error).__name__)
        return False
    if reason is not None:
        verbose_logger.debug("Rust Responses declined (%s)", reason)
        return False
    return True


def _raise_or_decline(error: BaseException, *, model: str, provider: str | None) -> None:
    native_bridge: Final = get_native_bridge()
    upstream: Final = getattr(native_bridge, "RustUpstreamError", None) if native_bridge is not None else None
    declined: Final = getattr(native_bridge, "RustBridgeDeclined", None) if native_bridge is not None else None
    if isinstance(upstream, type) and isinstance(error, upstream):
        args: Final = error.args
        status: Final = args[0] if args else 0
        message: Final = args[1] if len(args) > 1 else ""
        raise APIError(
            status_code=int(status) or 500,
            message=f"litellm rust responses: {message}",
            llm_provider=provider or "",
            model=model,
        )
    if isinstance(declined, type) and isinstance(error, declined):
        return
    raise error


def _build_response(
    response_data: Mapping[str, object],
    *,
    native_openai: bool,
) -> ResponsesAPIResponse:
    if native_openai:
        return ResponsesAPIResponse.model_validate(response_data)
    output: Final = _GENERIC_OUTPUT_ADAPTER.validate_python(response_data.get("output"))
    return ResponsesAPIResponse.model_validate(MappingProxyType({**response_data, "output": output}))


def responses(
    *,
    model: str,
    input: str | Sequence[object],
    optional_params: Mapping[str, object],
    api_key: str | None,
    api_base: str | None,
    custom_llm_provider: str | None,
    extra_headers: Mapping[str, object] | None,
    timeout: float | httpx.Timeout | None,
    use_chat_completions_api: bool,
) -> ResponsesAPIResponse | None:
    rust_responses: Final = _load_responses()
    if rust_responses is None:
        return None
    try:
        rust_response: Final = rust_responses(
            model=model,
            input=input,
            optional_params=optional_params,
            api_key=api_key,
            api_base=api_base,
            custom_llm_provider=custom_llm_provider,
            extra_headers=extra_headers,
            timeout_seconds=timeout_to_seconds(timeout),
            use_chat_completions_api=use_chat_completions_api,
        )
    except Exception as rust_error:  # noqa: BLE001  # upstream failures must not retry and double bill
        _raise_or_decline(rust_error, model=model, provider=custom_llm_provider)
        return None
    built: Final = _build_response(
        rust_response,
        native_openai=custom_llm_provider == "openai" and not use_chat_completions_api,
    )
    return built
