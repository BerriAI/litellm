"""Live e2e: custom pass-through endpoints inject configured headers and honor
x-pass-* client headers (prefix stripped) on the way to the upstream.

The upstream is the real Anthropic Messages API rather than an echo service:
Anthropic doesn't echo request headers back, but it does gate real behavior on
two of them, which is enough to prove forwarding without a mock. A static
x-api-key configured on the pass-through endpoint (the caller never supplies
one) must reach upstream, or every call 401s; an invalid x-pass-anthropic-version
sent by the caller must reach upstream with the prefix stripped, and Anthropic
echoes the exact value back in its 400 body, so a unique-per-run marker proves
this specific request's header - not a stale or cached one - got there.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel, Field

from e2e_config import unique_marker
from e2e_http import AuthHeaders, NoBody, require_successful_call, unwrap
from lifecycle import ResourceManager
from models import AnthropicMessagesResponse, ChatMessage, KeyGenerateBody
from passthrough_client import PassthroughClient

pytestmark = pytest.mark.e2e

ANTHROPIC_MESSAGES_TARGET = "https://api.anthropic.com/v1/messages"
MODEL = "claude-haiku-4-5-20251001"


class PassThroughCreateBody(BaseModel):
    path: str
    target: str
    headers: dict[str, str] = {}
    auth: bool = True
    include_subpath: bool = False


class PassThroughEndpoint(BaseModel):
    id: str | None = None
    path: str
    target: str


class PassThroughCreateResponse(BaseModel):
    endpoints: list[PassThroughEndpoint]


class PassThroughDeleteParams(BaseModel):
    endpoint_id: str


class AnthropicPassThroughHeaders(AuthHeaders):
    content_type: str = Field(default="application/json", serialization_alias="Content-Type")
    x_pass_anthropic_version: str = Field(serialization_alias="x-pass-anthropic-version")


class AnthropicMessagesBody(BaseModel):
    model: str
    max_tokens: int = 8
    messages: list[ChatMessage]


def _create_passthrough(client: PassthroughClient, *, path: str) -> PassThroughEndpoint:
    created = unwrap(
        client.proxy.transport.post(
            "/config/pass_through_endpoint",
            headers=client.proxy.transport.master,
            json=PassThroughCreateBody(
                path=path,
                target=ANTHROPIC_MESSAGES_TARGET,
                headers={"x-api-key": "os.environ/ANTHROPIC_API_KEY"},
            ),
            response_type=PassThroughCreateResponse,
        )
    )
    assert created.endpoints, "create returned no endpoints"
    endpoint = created.endpoints[0]
    assert endpoint.id, "created pass-through endpoint has no id"
    return endpoint


def _delete_passthrough(client: PassthroughClient, endpoint_id: str) -> None:
    _ = client.proxy.transport.delete(
        "/config/pass_through_endpoint",
        headers=client.proxy.transport.master,
        json=NoBody(),
        params=PassThroughDeleteParams(endpoint_id=endpoint_id),
        response_type=PassThroughCreateResponse,
    )


def _messages_body() -> AnthropicMessagesBody:
    return AnthropicMessagesBody(model=MODEL, messages=[ChatMessage(role="user", content="Say hi.")])


class TestPassthroughHeaders:
    @pytest.mark.covers(
        "other.config.passthrough.headers_forwarded",
        exercised_on=[],
    )
    def test_static_and_x_pass_headers_reach_upstream(
        self, client: PassthroughClient, resources: ResourceManager
    ) -> None:
        marker = unique_marker()
        path = f"/e2e-passthrough-headers-{marker}"

        endpoint = _create_passthrough(client, path=path)
        assert endpoint.id is not None
        resources.defer(lambda: _delete_passthrough(client, endpoint.id or ""))

        key = client.proxy.generate_key(
            KeyGenerateBody(
                models=[],
                allowed_passthrough_routes=[path],
                user_id=f"e2e-pass-headers-{marker}",
            )
        )
        resources.defer(lambda: client.proxy.delete_key(key))

        result = client.proxy.transport.send(
            path,
            headers=AnthropicPassThroughHeaders(
                authorization=f"Bearer {key}",
                x_pass_anthropic_version="2023-06-01",
            ),
            json=_messages_body(),
        )
        require_successful_call(result)
        completion = AnthropicMessagesResponse.model_validate_json(result.body)
        text = "".join(block.text or "" for block in (completion.content or []))
        assert text.strip(), (
            f"static x-api-key must reach Anthropic for the call to succeed at all; got {result.body[:300]}"
        )

        invalid_version = f"e2e-passhdr-{unique_marker()}"
        blocked = client.proxy.transport.send(
            path,
            headers=AnthropicPassThroughHeaders(
                authorization=f"Bearer {key}",
                x_pass_anthropic_version=invalid_version,
            ),
            json=_messages_body(),
        )
        assert blocked.status_code == 400, (
            f"expected Anthropic to reject the invalid anthropic-version, got "
            f"{blocked.status_code}: {blocked.body[:300]}"
        )
        assert invalid_version in blocked.body, (
            f"x-pass-anthropic-version must reach upstream with the prefix stripped; "
            f"marker missing from Anthropic's error body: {blocked.body[:300]}"
        )
