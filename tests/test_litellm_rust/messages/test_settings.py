from typing import Final

import pytest

import litellm
from litellm.types.integrations.anthropic_cache_control_hook import GATEWAY_INJECTED_CACHE_METADATA_KEY
from tests.test_litellm_rust.messages.data import call
from tests.test_litellm_rust.support.recording_server import RecordingServer
from tests.test_litellm_rust.support.routing import assert_served_natively

pytestmark = pytest.mark.requires_rust_extension


@pytest.mark.asyncio
async def test_global_prompt_caching_marks_system_and_latest_message(
    server: RecordingServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    model: Final = "claude-test-cache-control"
    monkeypatch.setitem(
        litellm.model_cost,
        model,
        {"litellm_provider": "anthropic", "mode": "chat", "supports_prompt_caching": True},
    )
    monkeypatch.setattr(litellm, "enable_anthropic_prompt_caching", True)
    monkeypatch.setattr(litellm, "anthropic_prompt_caching_ttl", "invalid")

    await call(
        server,
        model=f"anthropic/{model}",
        messages=[{"role": "user", "content": "latest turn"}],
        system="stable prefix",
    )

    assert_served_natively(server)
    body: Final = server.requests[0].body
    assert isinstance(body, dict)
    assert body["system"] == [{"type": "text", "text": "stable prefix", "cache_control": {"type": "ephemeral"}}]
    assert body["messages"] == [
        {
            "role": "user",
            "content": [{"type": "text", "text": "latest turn", "cache_control": {"type": "ephemeral"}}],
        }
    ]


@pytest.mark.asyncio
async def test_request_prompt_caching_flag_uses_global_ttl_setting(
    server: RecordingServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    model: Final = "claude-test-request-cache-control"
    monkeypatch.setitem(
        litellm.model_cost,
        model,
        {"litellm_provider": "anthropic", "mode": "chat", "supports_prompt_caching": True},
    )
    monkeypatch.setattr(litellm, "enable_anthropic_prompt_caching", False)
    monkeypatch.setattr(litellm, "anthropic_prompt_caching_ttl", "1h")

    await call(
        server,
        model=f"anthropic/{model}",
        messages=[{"role": "user", "content": "latest turn"}],
        system="stable prefix",
        enable_prompt_caching=True,
    )

    assert_served_natively(server)
    body: Final = server.requests[0].body
    assert isinstance(body, dict)
    assert body["system"][0]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}
    assert body["messages"][0]["content"][0]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}


@pytest.mark.asyncio
async def test_configured_cache_control_points_are_applied_on_the_native_route(
    server: RecordingServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    model: Final = "claude-test-configured-cache-control"
    monkeypatch.setitem(
        litellm.model_cost,
        model,
        {"litellm_provider": "anthropic", "mode": "chat", "supports_prompt_caching": True},
    )
    monkeypatch.setattr(litellm, "enable_anthropic_prompt_caching", False)

    await call(
        server,
        model=f"anthropic/{model}",
        messages=[{"role": "user", "content": "latest turn"}],
        cache_control_injection_points=[
            {"location": "message", "index": "-1", "control": {"type": "ephemeral", "ttl": "5m"}}
        ],
    )

    assert_served_natively(server)
    body: Final = server.requests[0].body
    assert isinstance(body, dict)
    assert body["messages"][0]["content"][0]["cache_control"] == {"type": "ephemeral", "ttl": "5m"}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("system", "expected"),
    [
        ("stable prefix", "deployment-1"),
        ([{"type": "text", "text": "marked", "cache_control": {"type": "ephemeral"}}], None),
    ],
)
async def test_native_injection_names_the_deployment_in_request_metadata(
    server: RecordingServer, monkeypatch: pytest.MonkeyPatch, system: object, expected: str | None
) -> None:
    model: Final = "claude-test-gateway-injection-metadata"
    monkeypatch.setitem(
        litellm.model_cost,
        model,
        {"litellm_provider": "anthropic", "mode": "chat", "supports_prompt_caching": True},
    )
    monkeypatch.setattr(litellm, "enable_anthropic_prompt_caching", True)
    metadata: Final[dict[str, object]] = {"existing": "kept"}

    await call(
        server,
        model=f"anthropic/{model}",
        messages=[{"role": "user", "content": "latest turn"}],
        system=system,
        litellm_metadata=metadata,
        model_info={"id": "deployment-1"},
    )

    assert_served_natively(server)
    assert metadata.get(GATEWAY_INJECTED_CACHE_METADATA_KEY) == expected
    assert metadata["existing"] == "kept"
