import pytest

import litellm
from tests.test_litellm_rust.support.recording_server import RecordingServer
from tests.test_litellm_rust.support.routing import call_public

pytestmark = pytest.mark.requires_rust_extension


@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
async def test_noncallable_native_export_is_unavailable_before_provider_io(
    server: RecordingServer, monkeypatch: pytest.MonkeyPatch, asynchronous: bool
) -> None:
    from litellm.rust_bridge import _native

    server.expected_requests = 0
    monkeypatch.setattr(_native, "atokenize" if asynchronous else "tokenize", object())
    with pytest.raises(RuntimeError, match=r"Rust tokenize bridge.*unavailable"):
        await call_public(
            litellm.tokenize,
            litellm.atokenize,
            {"model": "hosted_vllm/acme/model", "prompt": "hello"},
            server,
            asynchronous=asynchronous,
        )
    assert server.requests == []


@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
async def test_missing_prompt_reports_the_public_function_name_before_provider_io(
    server: RecordingServer, asynchronous: bool
) -> None:
    server.expected_requests = 0
    with pytest.raises(TypeError) as caught:
        await call_public(
            litellm.tokenize,
            litellm.atokenize,
            {"model": "hosted_vllm/acme/model"},
            server,
            asynchronous=asynchronous,
        )
    assert str(caught.value) == (
        f"{'atokenize' if asynchronous else 'tokenize'}() missing 1 required positional argument: 'prompt'"
    )
    assert server.requests == []
