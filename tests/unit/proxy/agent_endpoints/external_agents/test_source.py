import base64
import hashlib
import hmac
import json
from collections.abc import Mapping
from typing import Final

import pytest

from litellm.proxy.agent_endpoints.external_agents.models import HookFailure, ObservedAgentEvent
from litellm.proxy.agent_endpoints.external_agents.source import GrokBotHookSource, verify_ingest

SECRET: Final = "whsec_" + base64.b64encode(b"governance-test-signing-key").decode()
NOW: Final = 1800000000


@pytest.mark.parametrize("authorization", ("", "Bearer wrong", "Basic ingest", "Bearer ingest"))
def test_ingest_authentication_requires_exact_bearer_key(authorization: str) -> None:
    result: Final = verify_ingest(authorization, "ingest")
    assert result == (
        None if authorization == "Bearer ingest" else HookFailure(status=401, reason="Invalid ingest key")
    )


def test_missing_stored_ingest_key_fails_closed() -> None:
    assert verify_ingest("Bearer ", None) == HookFailure(status=401, reason="Invalid ingest key")


def signed_headers(body: bytes, secret: str = SECRET, timestamp: int = NOW) -> Mapping[str, str]:
    signed: Final = f"event-1.{timestamp}.".encode() + body
    signature: Final = base64.b64encode(
        hmac.new(base64.b64decode(secret[6:]), signed, hashlib.sha256).digest()
    ).decode()
    return {"webhook-id": "event-1", "webhook-timestamp": str(timestamp), "webhook-signature": f"v1,{signature}"}


@pytest.mark.parametrize("previous", [False, True])
def test_signed_body_accepts_current_or_previous_secret(previous: bool) -> None:
    body: Final = b'{"request_id":"event-1"}'
    other: Final = "whsec_" + base64.b64encode(b"other-key").decode()
    secrets: Final = (other, SECRET) if previous else (SECRET,)
    assert GrokBotHookSource().verify(body, signed_headers(body), secrets, NOW) is None


@pytest.mark.parametrize("body,timestamp", [(b"tampered", NOW), (b"original", NOW - 301), (b"original", NOW + 301)])
def test_invalid_or_stale_signature_is_rejected(body: bytes, timestamp: int) -> None:
    result: Final = GrokBotHookSource().verify(body, signed_headers(b"original", timestamp=timestamp), (SECRET,), NOW)
    assert isinstance(result, HookFailure)
    assert result.status == 401


def test_missing_signature_is_rejected_even_with_proxy_identity_headers() -> None:
    result: Final = GrokBotHookSource().verify(
        b"{}",
        {
            "webhook-id": "event-1",
            "webhook-timestamp": str(NOW),
            "x-litellm-user-role": "proxy_admin",
            "authorization": "Bearer spoofed",
        },
        (SECRET,),
        NOW,
    )
    assert result == HookFailure(status=401, reason="Invalid webhook signature")


def test_non_ascii_signature_is_rejected() -> None:
    headers: Final = {**signed_headers(b"{}"), "webhook-signature": "v1,\u00e9"}
    assert GrokBotHookSource().verify(b"{}", headers, (SECRET,), NOW) == HookFailure(
        status=401, reason="Invalid webhook signature"
    )


def test_normalization_extracts_resolved_tools_and_mcp_toolset_without_content() -> None:
    body: Final = json.dumps(
        {
            "type": "tool_call",
            "request_id": "event-1",
            "tenant_id": "tenant",
            "session_id": "shared-bot",
            "actor": {"type": "user", "id": "member", "email_address": None},
            "source": {"application": "new-vendor-app"},
            "future_field": True,
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "private prompt"},
                        {
                            "type": "tool_use",
                            "tool_name": "CallMcpTool",
                            "input": {"secret": "private input"},
                            "tool_info": {"type": "mcp", "toolset_name": "restricted-server"},
                        },
                        {"type": "tool_use", "tool_name": "Shell"},
                        {"type": "attachment", "url": "private attachment"},
                    ],
                }
            ],
        }
    ).encode()
    event: Final = GrokBotHookSource().parse(body, "event-1")
    assert event == ObservedAgentEvent(
        request_id="event-1",
        tenant_id="tenant",
        external_agent_id="shared-bot",
        actor_id="member",
        frame_type="tool_call",
        application="new-vendor-app",
        tool_names=("CallMcpTool", "restricted-server", "Shell"),
    )
    assert "private" not in event.model_dump_json()


@pytest.mark.parametrize("body", [b"not-json", b"{}"])
def test_invalid_frame_is_an_error_value(body: bytes) -> None:
    assert GrokBotHookSource().parse(body, "event-1") == HookFailure(status=400, reason="Invalid hook frame")
