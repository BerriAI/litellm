from __future__ import annotations

from typing import Final

import pytest

from litellm.rust_bridge import _native
from litellm.rust_bridge.catalog import NATIVE_EXPORTS
from litellm.rust_bridge.configuration import ComponentName, ExecutionDecision
from litellm.rust_bridge.route import ComponentExecution
from litellm.rust_bridge.runtime import BridgeErrorContext, invoke

pytestmark = pytest.mark.requires_rust_extension

UNIMPLEMENTED: Final = {
    ComponentName.EMBEDDINGS: ("embedding", "aembedding"),
    ComponentName.RERANK: ("rerank", "arerank"),
    ComponentName.IMAGE_GENERATION: ("image_generation", "aimage_generation"),
    ComponentName.IMAGE_EDIT: ("image_edit", "aimage_edit"),
    ComponentName.SPEECH: ("speech", "aspeech"),
    ComponentName.MODERATION: ("moderation", "amoderation"),
    ComponentName.RESPONSES: ("responses", "aresponses"),
}


class UntouchedInput:
    def __getattribute__(self, name: str) -> object:
        raise AssertionError(f"unimplemented route inspected {name}")


class HostileValue:
    def __getattribute__(self, name: str) -> object:
        raise AssertionError(f"admission inspected {name}")


class RaisingHost:
    def __init__(self, error: BaseException) -> None:
        self.error: Final = error

    def invoke(self, *args: object) -> object:
        raise self.error


def test_catalog_exports_are_registered() -> None:
    assert all(hasattr(_native, export) for export in NATIVE_EXPORTS)


@pytest.mark.parametrize(("route_name", "exports"), tuple(UNIMPLEMENTED.items()))
def test_package_lifecycle_binding_declines_without_input_reads(
    route_name: ComponentName,
    exports: tuple[str, str],
) -> None:
    for export in exports:
        native: Final = getattr(_native, export)
        request: Final = UntouchedInput()
        with pytest.raises(
            _native.RustBridgeDeclined,
            match=f"^{route_name.value} native lifecycle is not implemented$",
        ):
            native(request, (request,), {"callback": request, "file": request}, request)


@pytest.mark.parametrize(("route_name", "exports"), tuple(UNIMPLEMENTED.items()))
def test_package_stub_decline_selects_python(
    route_name: ComponentName,
    exports: tuple[str, str],
) -> None:
    native: Final = getattr(_native, exports[0])
    execution: Final = ComponentExecution(
        route_name=route_name,
        decision=ExecutionDecision.RUST_WITH_FALLBACK,
    )
    result: Final = invoke(
        execution=execution,
        native_call=lambda: native(UntouchedInput(), (), {}, UntouchedInput()),
        python_fallback=lambda: "python",
        adapt=str,
        context=BridgeErrorContext(route=route_name.value, model="unused", provider="unused"),
    )
    assert result == "python"


def test_transcription_admission_declines_unsupported_provider_before_host_work() -> None:
    request: Final = {
        "model": "model",
        "audio": {"format": "wav", "data": "YQ=="},
        "custom_llm_provider": "unsupported",
    }
    for binding in (_native.transcription, _native.atranscription):
        with pytest.raises(_native.RustBridgeDeclined):
            binding(request, (), {}, UntouchedInput())


@pytest.mark.parametrize("asynchronous", (False, True))
def test_chat_admission_declines_unsupported_provider_before_host_work(asynchronous: bool) -> None:
    binding: Final = _native.achat_completions if asynchronous else _native.chat_completions
    request: Final = {
        "model": "model",
        "messages": [{"role": "user", "content": "hi"}],
        "custom_llm_provider": "unsupported",
    }
    with pytest.raises(_native.RustBridgeDeclined):
        binding(request, (), {}, UntouchedInput())


@pytest.mark.parametrize("asynchronous", (False, True))
def test_chat_admission_declines_opaque_messages_without_touching_them(asynchronous: bool) -> None:
    binding: Final = _native.achat_completions if asynchronous else _native.chat_completions
    request: Final = {
        "model": "anthropic/model",
        "messages": [HostileValue()],
    }
    with pytest.raises(_native.RustBridgeDeclined, match="cannot be inspected"):
        binding(request, (), {}, UntouchedInput())


def test_chat_host_reserved_error_is_terminal_and_preserves_identity() -> None:
    error: Final = _native.RustBridgeDeclined("raised by host")
    request: Final = {
        "model": "anthropic/model",
        "messages": [{"role": "user", "content": "hi"}],
    }
    with pytest.raises(_native.RustHostCallbackError) as caught:
        _native.chat_completions(request, (), request, RaisingHost(error))
    assert caught.value.__cause__ is error


@pytest.mark.parametrize("asynchronous", (False, True))
def test_messages_declines_required_host_hook_before_preparation(asynchronous: bool) -> None:
    binding: Final = _native.amessages if asynchronous else _native.messages
    request: Final = {
        "model": "anthropic/model",
        "body": {},
        "custom_llm_provider": "anthropic",
        "has_agentic_hook": True,
    }
    with pytest.raises(_native.RustBridgeDeclined, match="host operations"):
        binding(request, (), {}, UntouchedInput())


@pytest.mark.parametrize(
    ("provider", "facts", "headers"),
    (
        ("anthropic", {"stream": True}, {}),
        ("anthropic", {"anthropic_user_id": True}, {}),
        ("bedrock", {"bedrock_metadata_owned": True}, {}),
        ("bedrock", {}, {"x-amz-date": "forwarded"}),
    ),
)
def test_chat_entrypoints_decline_before_the_host_callback(provider: str, facts: dict, headers: dict) -> None:
    for binding in (_native.chat_completions, _native.achat_completions):
        request: Final = {
            "model": "model",
            "messages": [{"role": "user", "content": "hi"}],
            "custom_llm_provider": provider,
            "host_facts": facts,
            "extra_headers": headers,
        }
        with pytest.raises(_native.RustBridgeDeclined):
            binding(request, (), {}, UntouchedInput())


def test_transcription_lifecycle_declines_audio_format_before_host_work() -> None:
    request: Final = {
        "model": "model",
        "audio": {"format": "unsupported", "data": "YQ=="},
        "custom_llm_provider": "bedrock",
    }
    for binding in (_native.transcription, _native.atranscription):
        with pytest.raises(_native.RustBridgeDeclined, match="audio format"):
            binding(request, (), {}, UntouchedInput())


def test_websocket_declines_before_parsing_or_dialing_url() -> None:
    with pytest.raises(_native.RustBridgeDeclined):
        _native.ResponsesWebSocketConnection.connect("not a URL", custom_llm_provider="azure")


def test_tokenizer_initialization_unavailability_is_not_a_request_decline() -> None:
    with pytest.raises(_native.RustBridgeUnavailable):
        _native.count_input_tokens(
            b'{"input":"hello"}',
            "anthropic",
            "",
            False,
            False,
            lambda _tokenizer: "{}",
        )
