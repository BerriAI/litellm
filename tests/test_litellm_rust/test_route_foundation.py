from __future__ import annotations

from typing import Final

import pytest

from litellm.rust_bridge import _native
from litellm.rust_bridge.bindings import NativeBinding
from litellm.rust_bridge.catalog import NATIVE_EXPORTS
from litellm.rust_bridge.chat_completions.lifecycle import LIFECYCLE as CHAT_COMPLETIONS
from litellm.rust_bridge.configuration import ExecutionDecision, ComponentName
from litellm.rust_bridge.embeddings.lifecycle import LIFECYCLE as EMBEDDINGS
from litellm.rust_bridge.image_edit.lifecycle import LIFECYCLE as IMAGE_EDIT
from litellm.rust_bridge.image_generation.lifecycle import LIFECYCLE as IMAGE_GENERATION
from litellm.rust_bridge.messages.lifecycle import LIFECYCLE as MESSAGES
from litellm.rust_bridge.moderation.lifecycle import LIFECYCLE as MODERATION
from litellm.rust_bridge.rerank.lifecycle import LIFECYCLE as RERANK
from litellm.rust_bridge.responses.lifecycle import LIFECYCLE as RESPONSES
from litellm.rust_bridge.route import ComponentExecution, NativeLifecycle
from litellm.rust_bridge.runtime import BridgeErrorContext, invoke
from litellm.rust_bridge.speech.lifecycle import LIFECYCLE as SPEECH
from litellm.rust_bridge.transcription.lifecycle import LIFECYCLE as TRANSCRIPTION

pytestmark = pytest.mark.requires_rust_extension

UNIMPLEMENTED: Final[dict[ComponentName, NativeBinding[NativeLifecycle[object, object]]]] = {
    ComponentName.MESSAGES: MESSAGES,
    ComponentName.CHAT_COMPLETIONS: CHAT_COMPLETIONS,
    ComponentName.TRANSCRIPTION: TRANSCRIPTION,
    ComponentName.EMBEDDINGS: EMBEDDINGS,
    ComponentName.RERANK: RERANK,
    ComponentName.IMAGE_GENERATION: IMAGE_GENERATION,
    ComponentName.IMAGE_EDIT: IMAGE_EDIT,
    ComponentName.SPEECH: SPEECH,
    ComponentName.MODERATION: MODERATION,
    ComponentName.RESPONSES: RESPONSES,
}


class UntouchedInput:
    def __getattribute__(self, name: str) -> object:
        raise AssertionError(f"unimplemented route inspected {name}")


def test_catalog_exports_are_registered() -> None:
    assert all(hasattr(_native, export) for export in NATIVE_EXPORTS)


@pytest.mark.parametrize(("route_name", "binding"), tuple(UNIMPLEMENTED.items()))
@pytest.mark.parametrize("asynchronous", (False, True))
def test_package_lifecycle_binding_declines_without_input_reads(
    route_name: ComponentName,
    binding: NativeBinding[NativeLifecycle[object, object]],
    asynchronous: bool,
) -> None:
    native: Final = binding.load()
    assert native is not None
    request: Final = UntouchedInput()
    with pytest.raises(_native.RustBridgeDeclined, match=f"^{route_name.value} native lifecycle is not implemented$"):
        native(request, (request,), {"callback": request, "file": request}, asynchronous, request)


@pytest.mark.parametrize(("route_name", "binding"), tuple(UNIMPLEMENTED.items()))
def test_package_stub_decline_selects_python(
    route_name: ComponentName,
    binding: NativeBinding[NativeLifecycle[object, object]],
) -> None:
    native: Final[NativeLifecycle[object, object] | None] = binding.load()
    assert native is not None
    execution: Final = ComponentExecution(
        route_name=route_name,
        decision=ExecutionDecision.RUST_WITH_FALLBACK,
    )
    result: Final = invoke(
        execution=execution,
        native_call=lambda: native(UntouchedInput(), (), {}, False, UntouchedInput()),
        python_fallback=lambda: "python",
        adapt=str,
        context=BridgeErrorContext(route=route_name.value, model="unused", provider="unused"),
    )
    assert result == "python"


@pytest.mark.parametrize("asynchronous", (False, True))
@pytest.mark.parametrize("route", ("messages", "chat_completions", "transcription", "ocr"))
def test_value_admission_declines_unsupported_provider_before_credentials(route: str, asynchronous: bool) -> None:
    binding: Final = getattr(_native, ("a" if asynchronous else "") + route)
    payload: Final = [{"role": "user", "content": "hi"}] if route == "chat_completions" else {}
    with pytest.raises(_native.RustBridgeDeclined):
        binding("model", payload, custom_llm_provider="unsupported", api_base="http://127.0.0.1:1")


@pytest.mark.parametrize("asynchronous", (False, True))
def test_messages_declines_required_host_hook_before_preparation(asynchronous: bool) -> None:
    binding: Final = _native.amessages if asynchronous else _native.messages
    with pytest.raises(_native.RustBridgeDeclined, match="host operations"):
        binding("model", {}, custom_llm_provider="anthropic", has_agentic_hook=True)


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
    calls: Final[list[bool]] = []
    for binding in (_native.chat_completions, _native.achat_completions):
        with pytest.raises(_native.RustBridgeDeclined):
            binding(
                "model",
                [{"role": "user", "content": "hi"}],
                custom_llm_provider=provider,
                host_facts=facts,
                extra_headers=headers,
                on_request=lambda: calls.append(True),
            )
    assert calls == []


def test_transcription_declines_audio_format_before_credentials() -> None:
    with pytest.raises(_native.RustBridgeDeclined, match="audio format"):
        _native.transcription("model", {"format": "unsupported", "data": "YQ=="}, custom_llm_provider="bedrock")


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
