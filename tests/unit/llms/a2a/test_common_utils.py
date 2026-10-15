"""Tests for litellm/llms/a2a/common_utils.py."""

from collections.abc import Mapping
from types import MappingProxyType

import pytest

from litellm.llms.a2a.common_utils import extract_text_from_a2a_response, resolve_a2a_hop_auth_header


class _RecordingEntraResolver:
    def __init__(self) -> None:
        self.calls: list[Mapping[str, object]] = []

    async def __call__(self, litellm_params: Mapping[str, object]) -> Mapping[str, str]:
        self.calls.append(litellm_params)
        return MappingProxyType({"Authorization": "Bearer minted-entra-token"})


_SERVICE_PRINCIPAL = MappingProxyType({"tenant_id": "tenant", "client_id": "client", "client_secret": "sp-secret"})


@pytest.mark.asyncio
async def test_entra_agent_gets_a_minted_bearer_for_the_a2a_hop():
    resolver = _RecordingEntraResolver()

    header = await resolve_a2a_hop_auth_header(_SERVICE_PRINCIPAL, None, resolver)

    assert header == {"Authorization": "Bearer minted-entra-token"}
    assert resolver.calls == [_SERVICE_PRINCIPAL]


@pytest.mark.asyncio
async def test_completion_bridge_agent_keeps_its_entra_credentials_for_the_model_provider():
    """A bridged agent's tenant_id/client_id/client_secret authenticate the model it bridges to, so the A2A hop
    must not spend them on a bearer of its own."""
    resolver = _RecordingEntraResolver()

    header = await resolve_a2a_hop_auth_header(_SERVICE_PRINCIPAL, "azure_ai", resolver)

    assert header is None
    assert resolver.calls == []


@pytest.mark.asyncio
async def test_agent_without_entra_credentials_gets_no_bearer():
    resolver = _RecordingEntraResolver()

    header = await resolve_a2a_hop_auth_header({"api_base": "https://agent.example.com"}, None, resolver)

    assert header is None
    assert resolver.calls == []


def _task(*, status_role: str | None = None, status_text: str = "", artifact_text: str | None = None) -> dict:
    result: dict = {"kind": "task"}
    if status_role is not None:
        result["status"] = {
            "state": "completed",
            "message": {"role": status_role, "parts": [{"kind": "text", "text": status_text}]},
        }
    if artifact_text is not None:
        result["artifacts"] = [{"parts": [{"kind": "text", "text": artifact_text}]}]
    return {"jsonrpc": "2.0", "id": "1", "result": result}


@pytest.mark.parametrize(
    "response, expected",
    [
        # Regression: a task echoing the caller in status.message while carrying the real
        # answer in artifacts must not come back empty. Skipping a caller-authored message
        # has to fall through to the agent's own output, not short-circuit extraction.
        pytest.param(
            _task(status_role="user", status_text="check active alarms", artifact_text="THE ANSWER"),
            "THE ANSWER",
            id="user_echo_falls_through_to_artifacts",
        ),
        pytest.param(
            _task(status_role="agent", status_text="agent status", artifact_text="THE ANSWER"),
            "agent status",
            id="agent_status_message_preferred",
        ),
        pytest.param(_task(artifact_text="THE ANSWER"), "THE ANSWER", id="artifacts_only"),
        pytest.param(_task(status_role="user", status_text="echo"), "", id="user_echo_alone_is_empty"),
        pytest.param(
            {"result": {"kind": "message", "role": "user", "parts": [{"kind": "text", "text": "echo"}]}},
            "",
            id="direct_user_message_is_empty",
        ),
        pytest.param(
            {"result": {"kind": "message", "role": "agent", "parts": [{"kind": "text", "text": "hi"}]}},
            "hi",
            id="direct_agent_message",
        ),
    ],
)
def test_extract_text_skips_caller_authored_messages(response, expected):
    """Caller-authored parts are skipped, never returned, and never hide the agent's reply."""
    assert extract_text_from_a2a_response(response) == expected
