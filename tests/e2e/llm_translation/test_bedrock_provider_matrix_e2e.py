"""Live e2e for the Bedrock cells of the provider-feature matrix: provider
response headers on /chat/completions and regional inference-profile model ids
(us.anthropic.*) over the invoke route.

Header forwarding is the #37003 contract: the proxy surfaces Bedrock's response
headers prefixed llm_provider- (llm_provider-x-amzn-requestid above all) so a
caller can hand AWS support the request id behind a completion. Regional
inference-profile ids are the deployment shape most Bedrock customers run; a
v1.90.0 regression timed them out, and the Converse route keeps them covered in
test_chat_completions_regression_e2e.py, so the invoke route carries its own
rows here.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from e2e_config import unique_marker
from e2e_http import StreamingResponse, unwrap
from lifecycle import ResourceManager
from models import ChatBody, ChatMessage, ChatResponse, LiteLLMParamsBody
from passthrough_client import PassthroughClient

pytestmark = pytest.mark.e2e

CONVERSE_REGIONAL_BACKEND = "bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0"
INVOKE_REGIONAL_BACKEND = "bedrock/invoke/us.anthropic.claude-haiku-4-5-20251001-v1:0"
PROVIDER_HEADER_PREFIX = "llm_provider-"
BEDROCK_REQUEST_ID_HEADER = "llm_provider-x-amzn-requestid"


class _StreamDelta(BaseModel):
    content: str | None = None


class _StreamChoice(BaseModel):
    delta: _StreamDelta = _StreamDelta()


class _StreamChunk(BaseModel):
    choices: list[_StreamChoice] = []


def _streamed_text(events: list[str]) -> str:
    chunks = [_StreamChunk.model_validate_json(event) for event in events]
    return "".join(choice.delta.content or "" for chunk in chunks for choice in chunk.choices)


def _assert_streamed_completion(result: StreamingResponse) -> None:
    assert result.ok and result.is_streaming, f"stream was not established: {result}"
    assert result.stream_error is None, f"stream carried an error event: {result.stream_error}"
    assert len(result.stream_events) > 1, f"stream did not deliver multiple data events: {result}"
    assert _streamed_text(result.stream_events).strip(), (
        f"stream completed with no content deltas: {result.stream_events[:3]}"
    )


def _assert_request_id_header(result: StreamingResponse) -> None:
    forwarded = [name for name in result.headers if name.startswith(PROVIDER_HEADER_PREFIX)]
    assert result.headers.get(BEDROCK_REQUEST_ID_HEADER), (
        f"missing {BEDROCK_REQUEST_ID_HEADER}; forwarded provider headers: {forwarded}"
    )


def _assert_completion(response: ChatResponse) -> None:
    assert response.choices, f"completion returned no choices: {response}"
    message = response.choices[0].message
    content = (message.content if message else None) or ""
    assert content.strip(), f"completion carried no content: {response}"


def _register_bedrock_model(
    client: PassthroughClient, resources: ResourceManager, prefix: str, backend: str
) -> str:
    model = f"{prefix}-{unique_marker()}"
    model_id = client.proxy.create_model(
        model,
        LiteLLMParamsBody(
            model=backend,
            aws_access_key_id="os.environ/AWS_ACCESS_KEY_ID",
            aws_secret_access_key="os.environ/AWS_SECRET_ACCESS_KEY",
            aws_region_name="os.environ/AWS_REGION",
        ),
    )
    resources.defer(lambda: client.proxy.delete_model(model_id))
    return model


def _prompt() -> list[ChatMessage]:
    return [ChatMessage(role="user", content="reply with one word")]


class TestBedrockResponseHeaders:
    @pytest.mark.covers(
        "llm.chat_completions.bedrock_converse.response_headers.nonstream.works",
        exercised_on=[],
    )
    def test_bedrock_request_id_header_surfaces(
        self, client: PassthroughClient, resources: ResourceManager
    ) -> None:
        model = _register_bedrock_model(client, resources, "e2e-bedrock-headers", CONVERSE_REGIONAL_BACKEND)
        key = resources.key()

        result = client.proxy.transport.send(
            "/chat/completions",
            headers=client.proxy.transport.bearer(key),
            json=ChatBody(model=model, messages=_prompt(), max_tokens=64),
        )

        assert result.ok, f"chat call failed: {result.status_code} {result.body[:300]}"
        _assert_request_id_header(result)

    @pytest.mark.covers(
        "llm.chat_completions.bedrock_converse.response_headers.stream.works",
        exercised_on=[],
    )
    def test_bedrock_request_id_header_surfaces_on_stream(
        self, client: PassthroughClient, resources: ResourceManager
    ) -> None:
        model = _register_bedrock_model(
            client, resources, "e2e-bedrock-headers-stream", CONVERSE_REGIONAL_BACKEND
        )
        key = resources.key()

        result = client.proxy.chat_stream(
            key, ChatBody(model=model, messages=_prompt(), stream=True, max_tokens=64)
        )

        _assert_streamed_completion(result)
        _assert_request_id_header(result)


class TestBedrockInvokeRegionalModelIds:
    @pytest.mark.covers("llm.chat_completions.bedrock_invoke.basic.nonstream.works", exercised_on=[])
    def test_invoke_regional_id_completes(
        self, client: PassthroughClient, resources: ResourceManager
    ) -> None:
        model = _register_bedrock_model(client, resources, "e2e-bedrock-invoke", INVOKE_REGIONAL_BACKEND)
        key = resources.key()

        response = unwrap(client.proxy.chat(key, ChatBody(model=model, messages=_prompt(), max_tokens=64)))

        _assert_completion(response)

    @pytest.mark.covers("llm.chat_completions.bedrock_invoke.basic.stream.works", exercised_on=[])
    def test_invoke_regional_id_streams(
        self, client: PassthroughClient, resources: ResourceManager
    ) -> None:
        model = _register_bedrock_model(client, resources, "e2e-bedrock-invoke-stream", INVOKE_REGIONAL_BACKEND)
        key = resources.key()

        result = client.proxy.chat_stream(
            key, ChatBody(model=model, messages=_prompt(), stream=True, max_tokens=64)
        )

        _assert_streamed_completion(result)
