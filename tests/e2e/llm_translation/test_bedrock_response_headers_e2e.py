"""Live e2e: Bedrock chat completions surface the upstream AWS request ids (LIT-5145).

Every provider response header is re-emitted to the caller as `llm_provider-<header>`,
which is how a customer correlates a LiteLLM request with the provider's own record of
it. Bedrock was the exception: the chat integration never captured the upstream response
headers, so there was nothing to forward and no `llm_provider-*` header reached the
client on either the streaming or the non-streaming path. That left `x-litellm-call-id`
as the only correlation handle, which AWS support cannot look up, and it blocked a
customer's support case for two months across two separate reports.

`x-amzn-requestid` is the header AWS asks for, so it is the one asserted here, on both
paths because both were reported broken and they capture headers in different places
(`converse_handler` for the buffered response, the streaming wrapper for the streamed
one).
"""

from __future__ import annotations

import pytest
from e2e_config import unique_marker
from e2e_http import require_successful_call
from lifecycle import ResourceManager
from models import ChatBody, ChatMessage, LiteLLMParamsBody
from proxy_client import ProxyClient

pytestmark = pytest.mark.e2e

BEDROCK_BACKEND = "bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0"
AWS_REQUEST_ID_HEADER = "llm_provider-x-amzn-requestid"


def _register_bedrock(proxy: ProxyClient, resources: ResourceManager) -> tuple[str, str]:
    model = f"e2e-bedrock-headers-{unique_marker()}"
    model_id = proxy.create_model(
        model,
        LiteLLMParamsBody(
            model=BEDROCK_BACKEND,
            aws_access_key_id="os.environ/AWS_ACCESS_KEY_ID",
            aws_secret_access_key="os.environ/AWS_SECRET_ACCESS_KEY",
            aws_region_name="os.environ/AWS_REGION",
        ),
    )
    resources.defer(lambda: proxy.delete_model(model_id))
    return model, resources.key()


def _chat_body(model: str, *, stream: bool) -> ChatBody:
    return ChatBody(
        model=model,
        messages=[ChatMessage(role="user", content=f"Reply with the single word pong. {unique_marker()}")],
        max_tokens=16,
        stream=stream or None,
    )


def _assert_carries_aws_request_id(headers: dict[str, str], *, path: str) -> None:
    request_id = headers.get(AWS_REQUEST_ID_HEADER)
    assert request_id, (
        f"a {path} bedrock call returned no {AWS_REQUEST_ID_HEADER} header, so the caller has no "
        f"AWS-side identifier to give support (LIT-5145). Provider headers reach the client as "
        f"llm_provider-*; got the llm_provider headers "
        f"{sorted(name for name in headers if name.startswith('llm_provider-'))}"
    )
    assert request_id.strip(), (
        f"{AWS_REQUEST_ID_HEADER} was present but empty on the {path} path, which is no more "
        f"usable to AWS support than its absence"
    )


class TestBedrockResponseHeaders:
    @pytest.mark.covers(
        "llm.chat_completions.bedrock_converse.basic.nonstream.forwards_provider_headers",
        exercised_on=["chat_completions"],
    )
    def test_nonstreaming_chat_forwards_aws_request_id(
        self, proxy: ProxyClient, resources: ResourceManager
    ) -> None:
        model, key = _register_bedrock(proxy, resources)

        result = proxy.transport.send(
            "/chat/completions",
            headers=proxy.transport.bearer(key),
            json=_chat_body(model, stream=False),
        )

        require_successful_call(result)
        _assert_carries_aws_request_id(result.headers, path="non-streaming")

    @pytest.mark.covers(
        "llm.chat_completions.bedrock_converse.basic.stream.forwards_provider_headers",
        exercised_on=["chat_completions"],
    )
    def test_streaming_chat_forwards_aws_request_id(
        self, proxy: ProxyClient, resources: ResourceManager
    ) -> None:
        model, key = _register_bedrock(proxy, resources)

        result = proxy.chat_stream(key, _chat_body(model, stream=True))

        require_successful_call(result)
        assert result.is_streaming, f"expected an SSE stream, got content-type {result.content_type!r}"
        _assert_carries_aws_request_id(result.headers, path="streaming")
