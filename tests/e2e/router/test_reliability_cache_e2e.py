"""An exact cache hit preserves the full choices and usage without another provider call.

Response IDs, creation timestamps and proxy headers are transport metadata;
compare every field within choices and usage, including provider extensions.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Final

import pytest
from complexity_router_client import ComplexityRouterClient
from e2e_config import (
    FIXTURE_DIR,
    FIXTURE_MODE_RAW,
    PROVIDER_EDGE_ADVERTISE_HOST,
    PROVIDER_EDGE_BIND_HOST,
    REQUEST_TIMEOUT,
    unique_marker,
)
from e2e_http import StreamingResponse, require_successful_call, unwrap
from lifecycle import ResourceManager
from models import (
    AnthropicMessagesBody,
    ChatBody,
    ChatMessage,
    ChatResponse,
    ChatStreamOptions,
    ChatTool,
    ChatToolFunction,
    LiteLLMParamsBody,
)
from provider_edge import ProviderEdge, ProviderRequestObservation, observed_provider_edge
from pydantic import BaseModel, JsonValue

pytestmark = pytest.mark.e2e

_OPENAI_CHAT_MODEL: Final = "openai/gpt-5.6"
_OPENAI_EMBEDDING_MODEL: Final = "openai/text-embedding-3-small"
_OPENAI_TRANSCRIPTION_MODEL: Final = "openai/gpt-4o-mini-transcribe"
_OPENAI_TEXT_COMPLETION_MODEL: Final = "text-completion-openai/gpt-3.5-turbo-instruct"
_ANTHROPIC_MESSAGES_MODEL: Final = "anthropic/claude-haiku-4-5"
_COHERE_RERANK_MODEL: Final = "cohere/rerank-v3.5"
_WEATHER_WAV: Final = Path(__file__).resolve().parents[1] / "llm_translation/realtime/fixtures/weather_question_24k.wav"


class _CacheChatBody(ChatBody):
    ttl: int = 600


class _CachedAnswer(BaseModel):
    model: str
    choices: tuple[dict[str, JsonValue], ...]
    usage: dict[str, JsonValue]


class _CachedMessageContent(BaseModel):
    type: str | None = None
    text: str | None = None


class _CachedMessagesUsage(BaseModel):
    input_tokens: int | None = None
    output_tokens: int | None = None


class _CachedMessagesResponse(BaseModel):
    model: str | None = None
    content: tuple[_CachedMessageContent, ...] = ()
    usage: _CachedMessagesUsage | None = None


class _CachedEmbedding(BaseModel):
    index: int
    embedding: tuple[float, ...]


class _CachedEmbeddingsResponse(BaseModel):
    data: tuple[_CachedEmbedding, ...]


class _CachedRerankResult(BaseModel):
    index: int | None = None
    relevance_score: float | None = None


class _CachedRerankResponse(BaseModel):
    results: tuple[_CachedRerankResult, ...]


class _CachedTranscriptionResponse(BaseModel):
    text: str


class _CachedResponsesContent(BaseModel):
    type: str | None = None
    text: str | None = None


class _CachedResponsesOutput(BaseModel):
    type: str | None = None
    content: tuple[_CachedResponsesContent, ...] = ()


class _CachedResponsesResponse(BaseModel):
    model: str | None = None
    status: str | None = None
    output: tuple[_CachedResponsesOutput, ...] = ()


class _CachedTextCompletionChoice(BaseModel):
    text: str | None = None


class _CachedTextCompletionResponse(BaseModel):
    choices: tuple[_CachedTextCompletionChoice, ...]


class _CacheEmbeddingsBody(BaseModel):
    model: str
    input: str | tuple[str, ...]
    cache: Mapping[str, bool] | None = None


class _CacheRerankBody(BaseModel):
    model: str
    query: str
    documents: tuple[str, ...]
    top_n: int
    cache: Mapping[str, bool] | None = None


class _CacheTranscriptionForm(BaseModel):
    model: str
    response_format: str = "json"


class _CacheResponsesBody(BaseModel):
    model: str
    input: str
    stream: bool = False
    cache: Mapping[str, bool] | None = None


class _CacheTextCompletionBody(BaseModel):
    model: str
    prompt: str
    max_tokens: int = 64
    stream: bool = False
    cache: Mapping[str, bool] | None = None


class _ChatStreamDelta(BaseModel):
    content: str | None = None


class _ChatStreamChoice(BaseModel):
    delta: _ChatStreamDelta | None = None
    finish_reason: str | None = None


class _ChatStreamChunk(BaseModel):
    choices: tuple[_ChatStreamChoice, ...] = ()


class _MessagesStreamDelta(BaseModel):
    text: str | None = None
    stop_reason: str | None = None


class _MessagesStreamEvent(BaseModel):
    type: str
    delta: _MessagesStreamDelta | None = None


class _ResponsesStreamEvent(BaseModel):
    type: str
    delta: str | None = None


class _TextCompletionStreamChoice(BaseModel):
    text: str | None = None
    delta: _ChatStreamDelta | None = None
    finish_reason: str | None = None


class _TextCompletionStreamChunk(BaseModel):
    choices: tuple[_TextCompletionStreamChoice, ...] = ()


class _StructuredCacheResponse(BaseModel):
    answer: str
    count: int


class _LanternToolArguments(BaseModel):
    lantern: str


def _openai_params(edge: ProviderEdge, model: str) -> LiteLLMParamsBody:
    return LiteLLMParamsBody(
        model=model,
        api_key="os.environ/OPENAI_API_KEY",
        api_base=f"{edge.api_base('openai')}/v1",
    )


def _anthropic_params(edge: ProviderEdge) -> LiteLLMParamsBody:
    return LiteLLMParamsBody(
        model=_ANTHROPIC_MESSAGES_MODEL,
        api_key="os.environ/ANTHROPIC_API_KEY",
        api_base=edge.api_base("anthropic"),
    )


def _assert_cache_miss(response: StreamingResponse, call_type: str) -> None:
    require_successful_call(response)
    assert "x-litellm-cache-key" not in response.headers, f"{call_type}: first call must be a cache miss"


def _assert_cache_hit(response: StreamingResponse, call_type: str) -> None:
    require_successful_call(response)


def _assert_one_provider_call(observation: ProviderRequestObservation, call_type: str) -> None:
    assert observation.count == 1, f"{call_type}: two successful requests must invoke the provider exactly once"


def _chat_stream_text(response: StreamingResponse) -> str:
    assert response.is_streaming, f"chat: expected SSE response, got {response.content_type!r}"
    assert response.stream_error is None, f"chat: stream carried an error: {response.stream_error}"
    chunks: Final = tuple(_ChatStreamChunk.model_validate_json(event) for event in response.stream_events)
    content: Final = "".join(
        choice.delta.content or "" for chunk in chunks for choice in chunk.choices if choice.delta is not None
    )
    finish_reasons: Final = tuple(
        choice.finish_reason for chunk in chunks for choice in chunk.choices if choice.finish_reason is not None
    )
    assert content.strip(), "chat: stream assembled to an empty answer"
    assert finish_reasons == ("stop",), f"chat: unexpected stream finish reasons: {finish_reasons}"
    return content


def _messages_stream_text(response: StreamingResponse) -> str:
    assert response.is_streaming, f"messages: expected SSE response, got {response.content_type!r}"
    assert response.stream_error is None, f"messages: stream carried an error: {response.stream_error}"
    events: Final = tuple(_MessagesStreamEvent.model_validate_json(event) for event in response.stream_events)
    content: Final = "".join(
        event.delta.text or "" for event in events if event.type == "content_block_delta" and event.delta is not None
    )
    finish_reasons: Final = tuple(
        event.delta.stop_reason
        for event in events
        if event.type == "message_delta" and event.delta is not None and event.delta.stop_reason is not None
    )
    assert content.strip(), "messages: stream assembled to an empty answer"
    assert finish_reasons == ("end_turn",), f"messages: unexpected stream finish reasons: {finish_reasons}"
    assert events[-1].type == "message_stop", f"messages: stream did not terminate with message_stop: {events[-1].type}"
    return content


def _responses_stream_text(response: StreamingResponse) -> str:
    assert response.is_streaming, f"responses: expected SSE response, got {response.content_type!r}"
    assert response.stream_error is None, f"responses: stream carried an error: {response.stream_error}"
    events: Final = tuple(_ResponsesStreamEvent.model_validate_json(event) for event in response.stream_events)
    content: Final = "".join(event.delta or "" for event in events if event.type == "response.output_text.delta")
    assert content.strip(), "responses: stream assembled to an empty answer"
    assert events[-1].type == "response.completed", "responses: stream did not terminate with response.completed"
    return content


def _text_completion_stream_text(response: StreamingResponse) -> str:
    assert response.is_streaming, f"text completions: expected SSE response, got {response.content_type!r}"
    assert response.stream_error is None, f"text completions: stream carried an error: {response.stream_error}"
    chunks: Final = tuple(_TextCompletionStreamChunk.model_validate_json(event) for event in response.stream_events)
    content: Final = "".join(
        choice.text or (choice.delta.content if choice.delta is not None else "") or ""
        for chunk in chunks
        for choice in chunk.choices
    )
    finish_reasons: Final = tuple(
        choice.finish_reason for chunk in chunks for choice in chunk.choices if choice.finish_reason is not None
    )
    assert content.strip(), "text completions: stream assembled to an empty answer"
    assert finish_reasons == ("stop",), f"text completions: unexpected stream finish reasons: {finish_reasons}"
    return content


def _assert_cached_stream(
    first: StreamingResponse,
    second: StreamingResponse,
    observation: ProviderRequestObservation,
    call_type: str,
    stream_text: Callable[[StreamingResponse], str],
) -> None:
    _assert_cache_miss(first, call_type)
    first_text: Final = stream_text(first)
    _assert_cache_hit(second, call_type)
    assert stream_text(second) == first_text, f"{call_type}: cached stream changed assembled content"
    _assert_one_provider_call(observation, call_type)


class TestReliabilityCache:
    @pytest.mark.covers("reliability.cache.exact.returns_cached")
    @pytest.mark.replayable
    def test_exact_cache_returns_cached(
        self, client: ComplexityRouterClient, resources: ResourceManager, scoped_key: str
    ) -> None:
        marker: Final = unique_marker()
        model: Final = f"e2e-cache-{marker}"
        prompt: Final = f"Reply with a short sentence about a blue lantern. Request marker: {marker}"
        observation: Final = ProviderRequestObservation(marker)

        with observed_provider_edge(
            observation,
            mode_raw=FIXTURE_MODE_RAW,
            bundle_dir=FIXTURE_DIR,
            bind_host=PROVIDER_EDGE_BIND_HOST,
            advertise_host=PROVIDER_EDGE_ADVERTISE_HOST,
            forward_timeout=REQUEST_TIMEOUT,
        ) as edge:
            model_id: Final = client.proxy.create_model(model, _openai_params(edge, _OPENAI_CHAT_MODEL))
            resources.defer(lambda: client.proxy.delete_model(model_id))
            body: Final = _CacheChatBody(
                model=model,
                messages=[ChatMessage(role="user", content=prompt)],
                max_completion_tokens=512,
                reasoning_effort="none",
                cache=None,
            )
            first: Final = client.proxy.transport.send(
                "/chat/completions", headers=client.proxy.transport.bearer(scoped_key), json=body
            )
            _assert_cache_miss(first, "chat")
            answer: Final = ChatResponse.model_validate_json(first.body)
            assert len(answer.choices) == 1
            choice: Final = answer.choices[0]
            assert choice.message is not None and choice.message.role == "assistant"
            assert choice.message.content is not None and choice.message.content.strip(), "first answer is empty"
            assert choice.finish_reason == "stop"
            assert answer.usage is not None
            assert answer.usage.prompt_tokens is not None and answer.usage.prompt_tokens > 0
            assert answer.usage.completion_tokens is not None and answer.usage.completion_tokens > 0
            assert answer.usage.total_tokens == answer.usage.prompt_tokens + answer.usage.completion_tokens
            assert observation.count == 1, "first miss must invoke the provider exactly once"

            second: Final = client.proxy.transport.send(
                "/chat/completions", headers=client.proxy.transport.bearer(scoped_key), json=body
            )
            _assert_cache_hit(second, "chat")
            assert _CachedAnswer.model_validate_json(second.body) == _CachedAnswer.model_validate_json(first.body), (
                "cache hit changed the answer, finish reason or usage"
            )
            _assert_one_provider_call(observation, "chat")

    @pytest.mark.covers("reliability.cache.exact.returns_cached")
    @pytest.mark.replayable
    def test_messages_exact_cache_returns_cached(
        self, client: ComplexityRouterClient, resources: ResourceManager, scoped_key: str
    ) -> None:
        marker: Final = unique_marker()
        model: Final = f"e2e-cache-messages-{marker}"
        observation: Final = ProviderRequestObservation(marker)
        with observed_provider_edge(
            observation,
            mode_raw=FIXTURE_MODE_RAW,
            bundle_dir=FIXTURE_DIR,
            bind_host=PROVIDER_EDGE_BIND_HOST,
            advertise_host=PROVIDER_EDGE_ADVERTISE_HOST,
            forward_timeout=REQUEST_TIMEOUT,
        ) as edge:
            model_id: Final = client.proxy.create_model(model, _anthropic_params(edge))
            resources.defer(lambda: client.proxy.delete_model(model_id))
            body: Final = AnthropicMessagesBody(
                model=model,
                max_tokens=128,
                messages=[ChatMessage(role="user", content=f"Reply with one word. Request marker: {marker}")],
                cache=None,
            )
            first: Final = client.proxy.transport.send(
                "/v1/messages", headers=client.proxy.transport.bearer(scoped_key), json=body
            )
            _assert_cache_miss(first, "messages")
            first_answer: Final = _CachedMessagesResponse.model_validate_json(first.body)
            assert "".join(block.text or "" for block in first_answer.content).strip(), (
                "messages: first answer is empty"
            )
            assert first_answer.usage is not None, "messages: first response omitted usage"

            second: Final = client.proxy.transport.send(
                "/v1/messages", headers=client.proxy.transport.bearer(scoped_key), json=body
            )
            _assert_cache_hit(second, "messages")
            assert _CachedMessagesResponse.model_validate_json(second.body) == first_answer, (
                "messages: cache hit changed content or usage"
            )
            _assert_one_provider_call(observation, "messages")

    @pytest.mark.covers("reliability.cache.exact.returns_cached")
    @pytest.mark.replayable
    def test_embeddings_exact_cache_returns_cached(
        self, client: ComplexityRouterClient, resources: ResourceManager, scoped_key: str
    ) -> None:
        marker: Final = unique_marker()
        model: Final = f"e2e-cache-embeddings-{marker}"
        observation: Final = ProviderRequestObservation(marker)
        with observed_provider_edge(
            observation,
            mode_raw=FIXTURE_MODE_RAW,
            bundle_dir=FIXTURE_DIR,
            bind_host=PROVIDER_EDGE_BIND_HOST,
            advertise_host=PROVIDER_EDGE_ADVERTISE_HOST,
            forward_timeout=REQUEST_TIMEOUT,
        ) as edge:
            model_id: Final = client.proxy.create_model(model, _openai_params(edge, _OPENAI_EMBEDDING_MODEL))
            resources.defer(lambda: client.proxy.delete_model(model_id))
            body: Final = _CacheEmbeddingsBody(model=model, input=f"embedding request marker: {marker}")
            first: Final = client.proxy.transport.send(
                "/embeddings", headers=client.proxy.transport.bearer(scoped_key), json=body
            )
            _assert_cache_miss(first, "embeddings")
            first_answer: Final = _CachedEmbeddingsResponse.model_validate_json(first.body)
            assert first_answer.data and first_answer.data[0].embedding, "embeddings: first response has no vector"

            second: Final = client.proxy.transport.send(
                "/embeddings", headers=client.proxy.transport.bearer(scoped_key), json=body
            )
            _assert_cache_hit(second, "embeddings")
            assert _CachedEmbeddingsResponse.model_validate_json(second.body) == first_answer, (
                "embeddings: cache hit changed the returned vectors"
            )
            _assert_one_provider_call(observation, "embeddings")

    @pytest.mark.covers("reliability.cache.exact.returns_cached")
    def test_rerank_exact_cache_returns_cached(
        self, client: ComplexityRouterClient, resources: ResourceManager, scoped_key: str
    ) -> None:
        marker: Final = unique_marker()
        model: Final = f"e2e-cache-rerank-{marker}"
        observation: Final = ProviderRequestObservation(marker)
        with observed_provider_edge(
            observation,
            mode_raw=FIXTURE_MODE_RAW,
            bundle_dir=FIXTURE_DIR,
            bind_host=PROVIDER_EDGE_BIND_HOST,
            advertise_host=PROVIDER_EDGE_ADVERTISE_HOST,
            forward_timeout=REQUEST_TIMEOUT,
            mounts={"cohere": "https://api.cohere.com"},
        ) as edge:
            model_id: Final = client.proxy.create_model(
                model,
                LiteLLMParamsBody(
                    model=_COHERE_RERANK_MODEL,
                    api_key="os.environ/COHERE_API_KEY",
                    api_base=edge.api_base("cohere"),
                ),
            )
            resources.defer(lambda: client.proxy.delete_model(model_id))
            body: Final = _CacheRerankBody(
                model=model,
                query=f"Which document mentions a blue lantern? {marker}",
                documents=(
                    f"A blue lantern is lit near the harbor. {marker}",
                    f"A red sail crosses the river. {marker}",
                ),
                top_n=2,
            )
            first: Final = client.proxy.transport.send(
                "/v1/rerank", headers=client.proxy.transport.bearer(scoped_key), json=body
            )
            _assert_cache_miss(first, "rerank")
            first_answer: Final = _CachedRerankResponse.model_validate_json(first.body)
            assert first_answer.results and first_answer.results[0].relevance_score is not None, (
                "rerank: first response has no scored result"
            )

            second: Final = client.proxy.transport.send(
                "/v1/rerank", headers=client.proxy.transport.bearer(scoped_key), json=body
            )
            _assert_cache_hit(second, "rerank")
            assert _CachedRerankResponse.model_validate_json(second.body) == first_answer, (
                "rerank: cache hit changed ranked results"
            )
            _assert_one_provider_call(observation, "rerank")

    @pytest.mark.covers("reliability.cache.exact.returns_cached")
    @pytest.mark.replayable
    def test_transcriptions_exact_cache_returns_cached(
        self, client: ComplexityRouterClient, resources: ResourceManager, scoped_key: str
    ) -> None:
        marker: Final = unique_marker()
        model: Final = f"e2e-cache-transcription-{marker}"
        observation: Final = ProviderRequestObservation(marker)
        with observed_provider_edge(
            observation,
            mode_raw=FIXTURE_MODE_RAW,
            bundle_dir=FIXTURE_DIR,
            bind_host=PROVIDER_EDGE_BIND_HOST,
            advertise_host=PROVIDER_EDGE_ADVERTISE_HOST,
            forward_timeout=REQUEST_TIMEOUT,
        ) as edge:
            model_id: Final = client.proxy.create_model(model, _openai_params(edge, _OPENAI_TRANSCRIPTION_MODEL))
            resources.defer(lambda: client.proxy.delete_model(model_id))
            first: Final = unwrap(
                client.proxy.transport.upload(
                    "/v1/audio/transcriptions",
                    headers=client.proxy.transport.bearer(scoped_key),
                    form=_CacheTranscriptionForm(model=model),
                    filename=f"cache-{marker}.wav",
                    content=_WEATHER_WAV.read_bytes(),
                    file_content_type="audio/wav",
                    response_type=_CachedTranscriptionResponse,
                )
            )
            assert first.text.strip(), "transcriptions: first response is empty"
            second: Final = unwrap(
                client.proxy.transport.upload(
                    "/v1/audio/transcriptions",
                    headers=client.proxy.transport.bearer(scoped_key),
                    form=_CacheTranscriptionForm(model=model),
                    filename=f"cache-{marker}.wav",
                    content=_WEATHER_WAV.read_bytes(),
                    file_content_type="audio/wav",
                    response_type=_CachedTranscriptionResponse,
                )
            )
            assert second == first, "transcriptions: cache hit changed transcript text"
            _assert_one_provider_call(observation, "transcriptions")

    @pytest.mark.covers("reliability.cache.exact.returns_cached")
    @pytest.mark.replayable
    def test_responses_exact_cache_returns_cached(
        self, client: ComplexityRouterClient, resources: ResourceManager, scoped_key: str
    ) -> None:
        marker: Final = unique_marker()
        model: Final = f"e2e-cache-responses-{marker}"
        observation: Final = ProviderRequestObservation(marker)
        with observed_provider_edge(
            observation,
            mode_raw=FIXTURE_MODE_RAW,
            bundle_dir=FIXTURE_DIR,
            bind_host=PROVIDER_EDGE_BIND_HOST,
            advertise_host=PROVIDER_EDGE_ADVERTISE_HOST,
            forward_timeout=REQUEST_TIMEOUT,
        ) as edge:
            model_id: Final = client.proxy.create_model(model, _openai_params(edge, _OPENAI_CHAT_MODEL))
            resources.defer(lambda: client.proxy.delete_model(model_id))
            body: Final = _CacheResponsesBody(model=model, input=f"Reply with one word. Request marker: {marker}")
            first: Final = client.proxy.transport.send(
                "/v1/responses", headers=client.proxy.transport.bearer(scoped_key), json=body
            )
            _assert_cache_miss(first, "responses")
            first_answer: Final = _CachedResponsesResponse.model_validate_json(first.body)
            assert "".join(
                content.text or "" for output in first_answer.output for content in output.content
            ).strip(), "responses: first response is empty"

            second: Final = client.proxy.transport.send(
                "/v1/responses", headers=client.proxy.transport.bearer(scoped_key), json=body
            )
            _assert_cache_hit(second, "responses")
            assert _CachedResponsesResponse.model_validate_json(second.body) == first_answer, (
                "responses: cache hit changed output or status"
            )
            _assert_one_provider_call(observation, "responses")

    @pytest.mark.covers("reliability.cache.exact.returns_cached")
    @pytest.mark.replayable
    def test_text_completions_exact_cache_returns_cached(
        self, client: ComplexityRouterClient, resources: ResourceManager, scoped_key: str
    ) -> None:
        marker: Final = unique_marker()
        model: Final = f"e2e-cache-completions-{marker}"
        observation: Final = ProviderRequestObservation(marker)
        with observed_provider_edge(
            observation,
            mode_raw=FIXTURE_MODE_RAW,
            bundle_dir=FIXTURE_DIR,
            bind_host=PROVIDER_EDGE_BIND_HOST,
            advertise_host=PROVIDER_EDGE_ADVERTISE_HOST,
            forward_timeout=REQUEST_TIMEOUT,
        ) as edge:
            model_id: Final = client.proxy.create_model(model, _openai_params(edge, _OPENAI_TEXT_COMPLETION_MODEL))
            resources.defer(lambda: client.proxy.delete_model(model_id))
            body: Final = _CacheTextCompletionBody(
                model=model,
                prompt=f"Finish this sentence in a few words: the lantern shines. Request marker: {marker}",
            )
            first: Final = client.proxy.transport.send(
                "/v1/completions", headers=client.proxy.transport.bearer(scoped_key), json=body
            )
            _assert_cache_miss(first, "text completions")
            first_answer: Final = _CachedTextCompletionResponse.model_validate_json(first.body)
            assert first_answer.choices and (first_answer.choices[0].text or "").strip(), (
                "text completions: first response is empty"
            )

            second: Final = client.proxy.transport.send(
                "/v1/completions", headers=client.proxy.transport.bearer(scoped_key), json=body
            )
            _assert_cache_hit(second, "text completions")
            assert _CachedTextCompletionResponse.model_validate_json(second.body) == first_answer, (
                "text completions: cache hit changed completion text"
            )
            _assert_one_provider_call(observation, "text completions")

    @pytest.mark.covers("reliability.cache.exact.returns_cached")
    @pytest.mark.replayable
    def test_chat_stream_exact_cache_replays_content(
        self, client: ComplexityRouterClient, resources: ResourceManager, scoped_key: str
    ) -> None:
        marker: Final = unique_marker()
        model: Final = f"e2e-cache-chat-stream-{marker}"
        observation: Final = ProviderRequestObservation(marker)
        with observed_provider_edge(
            observation,
            mode_raw=FIXTURE_MODE_RAW,
            bundle_dir=FIXTURE_DIR,
            bind_host=PROVIDER_EDGE_BIND_HOST,
            advertise_host=PROVIDER_EDGE_ADVERTISE_HOST,
            forward_timeout=REQUEST_TIMEOUT,
        ) as edge:
            model_id: Final = client.proxy.create_model(model, _openai_params(edge, _OPENAI_CHAT_MODEL))
            resources.defer(lambda: client.proxy.delete_model(model_id))
            body: Final = _CacheChatBody(
                model=model,
                messages=[ChatMessage(role="user", content=f"Reply with one sentence. Request marker: {marker}")],
                max_completion_tokens=256,
                stream=True,
                stream_options=ChatStreamOptions(include_usage=True),
                cache=None,
            )
            first: Final = client.proxy.transport.send(
                "/chat/completions", headers=client.proxy.transport.bearer(scoped_key), json=body, stream=True
            )
            second: Final = client.proxy.transport.send(
                "/chat/completions", headers=client.proxy.transport.bearer(scoped_key), json=body, stream=True
            )
            _assert_cached_stream(first, second, observation, "chat stream", _chat_stream_text)

    @pytest.mark.covers("reliability.cache.exact.returns_cached")
    @pytest.mark.replayable
    def test_messages_stream_exact_cache_replays_content(
        self, client: ComplexityRouterClient, resources: ResourceManager, scoped_key: str
    ) -> None:
        marker: Final = unique_marker()
        model: Final = f"e2e-cache-messages-stream-{marker}"
        observation: Final = ProviderRequestObservation(marker)
        with observed_provider_edge(
            observation,
            mode_raw=FIXTURE_MODE_RAW,
            bundle_dir=FIXTURE_DIR,
            bind_host=PROVIDER_EDGE_BIND_HOST,
            advertise_host=PROVIDER_EDGE_ADVERTISE_HOST,
            forward_timeout=REQUEST_TIMEOUT,
        ) as edge:
            model_id: Final = client.proxy.create_model(model, _anthropic_params(edge))
            resources.defer(lambda: client.proxy.delete_model(model_id))
            body: Final = AnthropicMessagesBody(
                model=model,
                max_tokens=128,
                stream=True,
                messages=[ChatMessage(role="user", content=f"Reply with one sentence. Request marker: {marker}")],
                cache=None,
            )
            first: Final = client.proxy.transport.send(
                "/v1/messages", headers=client.proxy.transport.bearer(scoped_key), json=body, stream=True
            )
            second: Final = client.proxy.transport.send(
                "/v1/messages", headers=client.proxy.transport.bearer(scoped_key), json=body, stream=True
            )
            _assert_cached_stream(first, second, observation, "messages stream", _messages_stream_text)

    @pytest.mark.covers("reliability.cache.exact.returns_cached")
    @pytest.mark.replayable
    def test_responses_stream_exact_cache_replays_content(
        self, client: ComplexityRouterClient, resources: ResourceManager, scoped_key: str
    ) -> None:
        marker: Final = unique_marker()
        model: Final = f"e2e-cache-responses-stream-{marker}"
        observation: Final = ProviderRequestObservation(marker)
        with observed_provider_edge(
            observation,
            mode_raw=FIXTURE_MODE_RAW,
            bundle_dir=FIXTURE_DIR,
            bind_host=PROVIDER_EDGE_BIND_HOST,
            advertise_host=PROVIDER_EDGE_ADVERTISE_HOST,
            forward_timeout=REQUEST_TIMEOUT,
        ) as edge:
            model_id: Final = client.proxy.create_model(model, _openai_params(edge, _OPENAI_CHAT_MODEL))
            resources.defer(lambda: client.proxy.delete_model(model_id))
            body: Final = _CacheResponsesBody(
                model=model,
                input=f"Reply with one sentence. Request marker: {marker}",
                stream=True,
            )
            first: Final = client.proxy.transport.send(
                "/v1/responses", headers=client.proxy.transport.bearer(scoped_key), json=body, stream=True
            )
            second: Final = client.proxy.transport.send(
                "/v1/responses", headers=client.proxy.transport.bearer(scoped_key), json=body, stream=True
            )
            _assert_cached_stream(first, second, observation, "responses stream", _responses_stream_text)

    @pytest.mark.covers("reliability.cache.exact.returns_cached")
    @pytest.mark.replayable
    def test_text_completions_stream_exact_cache_replays_content(
        self, client: ComplexityRouterClient, resources: ResourceManager, scoped_key: str
    ) -> None:
        marker: Final = unique_marker()
        model: Final = f"e2e-cache-completions-stream-{marker}"
        observation: Final = ProviderRequestObservation(marker)
        with observed_provider_edge(
            observation,
            mode_raw=FIXTURE_MODE_RAW,
            bundle_dir=FIXTURE_DIR,
            bind_host=PROVIDER_EDGE_BIND_HOST,
            advertise_host=PROVIDER_EDGE_ADVERTISE_HOST,
            forward_timeout=REQUEST_TIMEOUT,
        ) as edge:
            model_id: Final = client.proxy.create_model(model, _openai_params(edge, _OPENAI_TEXT_COMPLETION_MODEL))
            resources.defer(lambda: client.proxy.delete_model(model_id))
            body: Final = _CacheTextCompletionBody(
                model=model,
                prompt=f"Complete one sentence about a lantern. Request marker: {marker}",
                stream=True,
            )
            first: Final = client.proxy.transport.send(
                "/v1/completions", headers=client.proxy.transport.bearer(scoped_key), json=body, stream=True
            )
            second: Final = client.proxy.transport.send(
                "/v1/completions", headers=client.proxy.transport.bearer(scoped_key), json=body, stream=True
            )
            _assert_cached_stream(first, second, observation, "text completions stream", _text_completion_stream_text)

    @pytest.mark.covers("reliability.cache.exact.returns_cached")
    @pytest.mark.replayable
    def test_tool_arguments_survive_an_exact_cache_hit(
        self, client: ComplexityRouterClient, resources: ResourceManager, scoped_key: str
    ) -> None:
        marker: Final = unique_marker()
        model: Final = f"e2e-cache-tool-{marker}"
        observation: Final = ProviderRequestObservation(marker)
        tool: Final = ChatTool(
            function=ChatToolFunction(
                name="get_lantern_status",
                description="Return the status of a lantern",
                parameters={
                    "type": "object",
                    "properties": {"lantern": {"type": "string"}},
                    "required": ["lantern"],
                },
            )
        )
        with observed_provider_edge(
            observation,
            mode_raw=FIXTURE_MODE_RAW,
            bundle_dir=FIXTURE_DIR,
            bind_host=PROVIDER_EDGE_BIND_HOST,
            advertise_host=PROVIDER_EDGE_ADVERTISE_HOST,
            forward_timeout=REQUEST_TIMEOUT,
        ) as edge:
            model_id: Final = client.proxy.create_model(model, _openai_params(edge, _OPENAI_CHAT_MODEL))
            resources.defer(lambda: client.proxy.delete_model(model_id))
            body: Final = _CacheChatBody(
                model=model,
                messages=[
                    ChatMessage(
                        role="user",
                        content=f"Call get_lantern_status with lantern set to harbor. Request marker: {marker}",
                    )
                ],
                max_completion_tokens=256,
                reasoning_effort="none",
                tools=[tool],
                tool_choice="required",
                cache=None,
            )
            first: Final = client.proxy.transport.send(
                "/chat/completions", headers=client.proxy.transport.bearer(scoped_key), json=body
            )
            _assert_cache_miss(first, "tool calls")
            first_answer: Final = ChatResponse.model_validate_json(first.body)
            assert first_answer.choices
            first_choice: Final = first_answer.choices[0]
            first_message: Final = first_choice.message
            assert first_message is not None
            first_calls: Final = first_message.tool_calls
            assert first_choice.finish_reason == "tool_calls", (
                f"tool calls: unexpected finish reason {first_choice.finish_reason!r}"
            )
            assert first_calls and first_calls[0].function.name == "get_lantern_status", (
                "tool calls: missing function call"
            )
            first_arguments: Final = first_calls[0].function.arguments
            assert first_arguments is not None
            assert _LanternToolArguments.model_validate_json(first_arguments) == _LanternToolArguments(lantern="harbor")

            second: Final = client.proxy.transport.send(
                "/chat/completions", headers=client.proxy.transport.bearer(scoped_key), json=body
            )
            _assert_cache_hit(second, "tool calls")
            second_answer: Final = ChatResponse.model_validate_json(second.body)
            assert second_answer.choices and second_answer.choices[0].message is not None
            assert second_answer.choices[0].finish_reason == "tool_calls"
            assert second_answer.choices[0].message.tool_calls == first_calls, "tool calls: cache hit changed arguments"
            assert _CachedAnswer.model_validate_json(second.body) == _CachedAnswer.model_validate_json(first.body)
            _assert_one_provider_call(observation, "tool calls")

    @pytest.mark.covers("reliability.cache.exact.returns_cached")
    @pytest.mark.replayable
    def test_structured_output_survives_an_exact_cache_hit(
        self, client: ComplexityRouterClient, resources: ResourceManager, scoped_key: str
    ) -> None:
        marker: Final = unique_marker()
        model: Final = f"e2e-cache-structured-{marker}"
        observation: Final = ProviderRequestObservation(marker)
        response_format: Final[dict[str, object]] = {
            "type": "json_schema",
            "json_schema": {
                "name": "cache_result",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {"answer": {"type": "string"}, "count": {"type": "integer"}},
                    "required": ["answer", "count"],
                    "additionalProperties": False,
                },
            },
        }
        with observed_provider_edge(
            observation,
            mode_raw=FIXTURE_MODE_RAW,
            bundle_dir=FIXTURE_DIR,
            bind_host=PROVIDER_EDGE_BIND_HOST,
            advertise_host=PROVIDER_EDGE_ADVERTISE_HOST,
            forward_timeout=REQUEST_TIMEOUT,
        ) as edge:
            model_id: Final = client.proxy.create_model(model, _openai_params(edge, _OPENAI_CHAT_MODEL))
            resources.defer(lambda: client.proxy.delete_model(model_id))
            body: Final = _CacheChatBody(
                model=model,
                messages=[
                    ChatMessage(
                        role="user",
                        content=f'Return JSON with answer "ready" and count 3. Request marker: {marker}',
                    )
                ],
                max_completion_tokens=128,
                response_format=response_format,
                cache=None,
            )
            first: Final = client.proxy.transport.send(
                "/chat/completions", headers=client.proxy.transport.bearer(scoped_key), json=body
            )
            _assert_cache_miss(first, "structured output")
            first_answer: Final = ChatResponse.model_validate_json(first.body)
            assert first_answer.choices and first_answer.choices[0].message is not None
            first_content: Final = first_answer.choices[0].message.content
            assert first_content is not None
            assert _StructuredCacheResponse.model_validate_json(first_content) == _StructuredCacheResponse(
                answer="ready", count=3
            )

            second: Final = client.proxy.transport.send(
                "/chat/completions", headers=client.proxy.transport.bearer(scoped_key), json=body
            )
            _assert_cache_hit(second, "structured output")
            second_answer: Final = ChatResponse.model_validate_json(second.body)
            assert second_answer.choices and second_answer.choices[0].message is not None
            second_content: Final = second_answer.choices[0].message.content
            assert second_content is not None
            assert _StructuredCacheResponse.model_validate_json(second_content) == _StructuredCacheResponse(
                answer="ready", count=3
            )
            assert _CachedAnswer.model_validate_json(second.body) == _CachedAnswer.model_validate_json(first.body)
            _assert_one_provider_call(observation, "structured output")
