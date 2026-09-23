"""Live e2e: an upstream that hangs up mid-stream on /v1/messages surfaces to
Anthropic clients as an `event: error` SSE frame, not an OpenAI-shaped
`data: {"error": ...}` frame they silently drop.

A per-test live provider edge forwards the deployment's provider traffic and
truncates the upstream's chunked response mid-stream. Covers Anthropic-direct
and Bedrock deployments. Customer report.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Final

import anthropic
import pytest
from anthropic.types import MessageParam
from e2e_config import PROVIDER_EDGE_ADVERTISE_HOST, PROVIDER_EDGE_BIND_HOST, unique_marker
from lifecycle import ResourceManager
from models import AnthropicErrorEvent, AnthropicMessagesBody, ChatMessage, LiteLLMParamsBody
from provider_edge import LiveEdge, RunningEdge, start_provider_edge
from provider_edge_bedrock import bedrock_signer
from proxy_client import ProxyClient
from pydantic import JsonValue, TypeAdapter, ValidationError
from sdk_clients import NO_PROXY_CACHE, SdkClients

pytestmark = pytest.mark.e2e

ANTHROPIC_BACKEND: Final = "anthropic/claude-haiku-4-5"
BEDROCK_BACKEND: Final = "bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0"
BEDROCK_EDGE_REGION: Final = "us-east-1"

_PROMPT: Final = "Count from 1 to 100, one number per line."


def _user_turn(text: str) -> MessageParam:
    return {"role": "user", "content": text}


def _start_edge(provider: str, truncate_after: int) -> RunningEdge:
    if provider == "anthropic":
        return start_provider_edge(
            LiveEdge(truncate_after=truncate_after),
            mounts=MappingProxyType({"anthropic": "https://api.anthropic.com"}),
            bind_host=PROVIDER_EDGE_BIND_HOST,
            advertise_host=PROVIDER_EDGE_ADVERTISE_HOST,
        )
    return start_provider_edge(
        LiveEdge(truncate_after=truncate_after, sign=bedrock_signer(BEDROCK_EDGE_REGION)),
        mounts=MappingProxyType(
            {f"bedrock/{BEDROCK_EDGE_REGION}": f"https://bedrock-runtime.{BEDROCK_EDGE_REGION}.amazonaws.com"}
        ),
        bind_host=PROVIDER_EDGE_BIND_HOST,
        advertise_host=PROVIDER_EDGE_ADVERTISE_HOST,
    )


def _deployment_params(provider: str, edge: RunningEdge) -> LiteLLMParamsBody:
    if provider == "anthropic":
        return LiteLLMParamsBody(
            model=ANTHROPIC_BACKEND,
            api_key="os.environ/ANTHROPIC_API_KEY",
            api_base=edge.edge.api_base("anthropic"),
        )
    return LiteLLMParamsBody(
        model=BEDROCK_BACKEND,
        api_base=edge.edge.api_base(f"bedrock/{BEDROCK_EDGE_REGION}"),
        aws_access_key_id="os.environ/AWS_ACCESS_KEY_ID",
        aws_secret_access_key="os.environ/AWS_SECRET_ACCESS_KEY",
        aws_region_name=BEDROCK_EDGE_REGION,
    )


def _register(
    proxy: ProxyClient, resources: ResourceManager, provider: str, truncate_after: int
) -> tuple[str, str]:
    edge: Final = _start_edge(provider, truncate_after)
    resources.defer(edge.shutdown)
    model: Final = f"e2e-messages-truncated-{unique_marker()}"
    model_id: Final = proxy.create_model(model, _deployment_params(provider, edge))
    resources.defer(lambda: proxy.delete_model(model_id))
    return model, resources.key()


_FRAME_PAYLOAD: Final[TypeAdapter[JsonValue]] = TypeAdapter(JsonValue)


def _bare_error_frame(frame: str) -> bool:
    """Whether a `data:` payload is an error object missing the Anthropic
    envelope: the OpenAI-shaped {"error": ...} Anthropic clients drop."""
    payload: Final = _FRAME_PAYLOAD.validate_json(frame)
    return isinstance(payload, dict) and "error" in payload and payload.get("type") != "error"


@pytest.mark.provider_edge_host
@pytest.mark.provider_live
class TestAnthropicMessagesUpstreamStreamFailure:
    @pytest.mark.covers("llm.messages.anthropic.upstream_stream_failure.stream.error_event")
    @pytest.mark.parametrize("provider", ["anthropic", "bedrock"])
    def test_interrupted_upstream_stream_raises_in_the_anthropic_sdk(
        self,
        proxy: ProxyClient,
        resources: ResourceManager,
        sdk: SdkClients,
        provider: str,
    ) -> None:
        model, key = _register(proxy, resources, provider, truncate_after=2)
        client: Final = sdk.anthropic(key)

        try:
            for _ in client.messages.create(
                model=model,
                max_tokens=300,
                stream=True,
                messages=[_user_turn(_PROMPT)],
                extra_body=NO_PROXY_CACHE,
            ):
                pass
        except anthropic.APIError:
            return
        pytest.fail(
            "the edge hung up the upstream stream but the Anthropic SDK saw a clean end of stream: "
            "the proxy emitted an OpenAI-shaped data: {\"error\": ...} frame with no event: error line, "
            "which Anthropic clients silently drop"
        )

    @pytest.mark.covers("llm.messages.anthropic.upstream_stream_failure.stream.error_event")
    @pytest.mark.parametrize("provider", ["anthropic", "bedrock"])
    def test_interrupted_upstream_stream_is_an_anthropic_error_event(
        self,
        proxy: ProxyClient,
        resources: ResourceManager,
        provider: str,
    ) -> None:
        model, key = _register(proxy, resources, provider, truncate_after=2)

        outcome: Final = proxy.messages_stream(
            key,
            AnthropicMessagesBody(
                model=model,
                max_tokens=300,
                stream=True,
                messages=[ChatMessage(role="user", content=_PROMPT)],
            ),
        )
        frames: Final = outcome.stream_events
        assert outcome.is_streaming, (
            f"/v1/messages did not answer with an SSE stream: status={outcome.status_code} body={outcome.body}"
        )
        assert frames, (
            f"the proxy sent no SSE data frames although the upstream hung up; "
            f"stream_error={outcome.stream_error!r}"
        )
        assert outcome.stream_error == "event: error", (
            f"the interrupted stream was not announced by an 'event: error' line Anthropic clients read; "
            f"stream_error={outcome.stream_error!r} frames={frames}"
        )
        try:
            AnthropicErrorEvent.model_validate_json(frames[-1])
        except ValidationError:
            pytest.fail(
                f"the last SSE frame was not an Anthropic {{\"type\": \"error\", \"error\": ...}} envelope; "
                f"frames={frames}"
            )
        bare: Final = tuple(frame for frame in frames if _bare_error_frame(frame))
        assert not bare, (
            f"the proxy emitted error frames without the Anthropic envelope, which Anthropic clients drop: "
            f"{bare}; all frames={frames}"
        )
