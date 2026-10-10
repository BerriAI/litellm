import asyncio
import json
import os
import uuid
from typing import Final
from urllib.parse import unquote

import httpx
import pytest
import redis
import respx
from integration._support.upstream import _aws_event_frame
from integration._support.wire import Reply, Request, wire_server

import litellm
from litellm import Router, acompletion, aembedding, completion
from litellm.caching.caching import Cache, LiteLLMCacheType
from litellm.caching.caching_handler import _PENDING_CACHE_WRITES
from litellm.caching.dual_cache import DualCache
from litellm.caching.in_memory_cache import InMemoryCache
from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER
from litellm.proxy.hooks.batch_redis_get import PROXY_BatchRedisRequests


async def _drain_cache_writes() -> None:
    await GLOBAL_LOGGING_WORKER.flush()
    await asyncio.gather(*_PENDING_CACHE_WRITES)


async def _wait_for_stream_cache_entry(cache: Cache, model: str, messages: list[dict[str, str]]) -> None:
    key: Final = cache.get_cache_key(model=model, messages=messages, stream=True)
    for _ in range(200):
        if await cache.cache.async_get_cache(key) is not None:
            return
        await asyncio.sleep(0.025)
    pytest.fail(f"no cache entry was written for {model} within 5s")


@pytest.fixture
def redis_response_cache(monkeypatch: pytest.MonkeyPatch) -> Cache:
    cache: Final = Cache(
        type=LiteLLMCacheType.REDIS,
        host=os.environ["REDIS_HOST"],
        port=os.environ["REDIS_PORT"],
    )
    monkeypatch.setattr(litellm, "cache", cache)
    return cache


@pytest.mark.asyncio
async def test_batch_get_cache_with_none_keys(redis_response_cache: Cache) -> None:
    redis_cache: Final = redis_response_cache.cache
    keys: Final = (None, f"missing-{uuid.uuid4().hex}", None, f"missing-{uuid.uuid4().hex}")
    expected: Final = {key: None for key in keys if key is not None}

    assert redis_cache.batch_get_cache(key_list=keys) == expected
    assert await redis_cache.async_batch_get_cache(key_list=keys) == expected


@pytest.mark.asyncio
async def test_cache_control_overrides(redis_response_cache: Cache) -> None:
    messages: Final = [{"role": "user", "content": f"cache control {uuid.uuid4().hex}"}]
    first: Final = await acompletion(
        model="gpt-4o-mini",
        messages=messages,
        caching=True,
        mock_response="cached",
    )
    bypassed: Final = await acompletion(
        model="gpt-4o-mini",
        messages=messages,
        caching=True,
        cache={"no-cache": True},
        mock_response="not cached",
    )

    assert first.id != bypassed.id


def test_caching_dynamic_args(redis_response_cache: Cache) -> None:
    messages: Final = [{"role": "user", "content": f"dynamic args {uuid.uuid4().hex}"}]
    first: Final = completion(
        model="gpt-4o-mini",
        messages=messages,
        caching=True,
        mock_response="cached",
    )
    second: Final = completion(
        model="gpt-4o-mini",
        messages=messages,
        caching=True,
        mock_response="not cached",
    )

    assert second.id == first.id
    assert second.choices[0].message.content == first.choices[0].message.content


def test_caching_redis_simple(redis_response_cache: Cache) -> None:
    messages: Final = [{"role": "user", "content": f"redis simple {uuid.uuid4().hex}"}]
    first: Final = completion(
        model="gpt-4o-mini",
        messages=messages,
        caching=True,
        mock_response="cached",
        stream=True,
    )
    first_chunks: Final = tuple(first)
    second: Final = completion(
        model="gpt-4o-mini",
        messages=messages,
        caching=True,
        mock_response="not cached",
        stream=True,
    )
    second_chunks: Final = tuple(second)

    assert "".join(chunk.choices[0].delta.content or "" for chunk in first_chunks) == "".join(
        chunk.choices[0].delta.content or "" for chunk in second_chunks
    )
    assert first_chunks[-1].id == second_chunks[-1].id


@pytest.mark.asyncio
async def test_dual_cache_batch_get_cache(redis_response_cache: Cache) -> None:
    dual_cache: Final = DualCache(
        in_memory_cache=InMemoryCache(),
        redis_cache=redis_response_cache.cache,
    )
    in_memory_key: Final = f"memory-{uuid.uuid4().hex}"
    redis_key: Final = f"redis-{uuid.uuid4().hex}"
    missing_key: Final = f"missing-{uuid.uuid4().hex}"
    dual_cache.in_memory_cache.set_cache(in_memory_key, {"source": "memory"})
    await redis_response_cache.cache.async_set_cache(redis_key, {"source": "redis"})

    result: Final = await dual_cache.async_batch_get_cache(
        keys=[in_memory_key, redis_key, missing_key]
    )

    assert result == [{"source": "memory"}, {"source": "redis"}, None]


@pytest.mark.asyncio
async def test_embedding_caching_base_64(redis_response_cache: Cache) -> None:
    inputs: Final = [f"base64 embedding {uuid.uuid4().hex}"]
    first: Final = await aembedding(
        model="text-embedding-3-small",
        input=inputs,
        caching=True,
        encoding_format="base64",
        mock_response="0.1,0.2,0.3",
    )
    second: Final = await aembedding(
        model="text-embedding-3-small",
        input=inputs,
        caching=True,
        encoding_format="base64",
        mock_response="0.4,0.5,0.6",
    )

    assert second._hidden_params["cache_hit"] is True
    assert second.data[0].embedding == first.data[0].embedding


@pytest.mark.asyncio
async def test_redis_batch_cache_write(monkeypatch: pytest.MonkeyPatch) -> None:
    redis_response_cache: Final = Cache(
        type=LiteLLMCacheType.REDIS,
        host=os.environ["REDIS_HOST"],
        port=os.environ["REDIS_PORT"],
        redis_flush_size=2,
    )
    monkeypatch.setattr(litellm, "cache", redis_response_cache)
    messages: Final = [{"role": "user", "content": f"batch write {uuid.uuid4().hex}"}]
    first: Final = await acompletion(
        model="gpt-4o-mini",
        messages=messages,
        mock_response="first",
    )
    await acompletion(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": f"flush {uuid.uuid4().hex}"}],
        mock_response="second",
    )
    await _drain_cache_writes()
    cached: Final = await acompletion(
        model="gpt-4o-mini",
        messages=messages,
        mock_response="third",
    )

    assert cached.id == first.id


@pytest.mark.asyncio
async def test_redis_cache_acompletion_stream(redis_response_cache: Cache) -> None:
    messages: Final = [{"role": "user", "content": f"async stream {uuid.uuid4().hex}"}]
    first_stream: Final = await acompletion(
        model="gpt-4o-mini",
        messages=messages,
        stream=True,
        mock_response="streamed cache response",
    )
    first_chunks: Final = tuple([chunk async for chunk in first_stream])
    await _wait_for_stream_cache_entry(redis_response_cache, "gpt-4o-mini", messages)
    second_stream: Final = await acompletion(
        model="gpt-4o-mini",
        messages=messages,
        stream=True,
        mock_response="different response",
    )
    second_chunks: Final = tuple([chunk async for chunk in second_stream])

    assert first_chunks[-1].id == second_chunks[-1].id
    assert "".join(chunk.choices[0].delta.content or "" for chunk in first_chunks) == "".join(
        chunk.choices[0].delta.content or "" for chunk in second_chunks
    )


@pytest.mark.asyncio
async def test_redis_cache_acompletion_stream_bedrock(
    redis_response_cache: Cache,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("AWS_BEARER_TOKEN_BEDROCK", raising=False)
    model_id: Final = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
    model: Final = f"bedrock/{model_id}"
    prompt: Final = f"bedrock stream {uuid.uuid4().hex}"
    messages: Final = [{"role": "user", "content": prompt}]
    response_text: Final = "scripted Bedrock cache response"
    expected_request_body: Final = {
        "messages": [{"role": "user", "content": [{"text": prompt}]}],
        "inferenceConfig": {"maxTokens": 40, "temperature": 1},
    }
    response_body: Final = b"".join(
        _aws_event_frame(event_type, payload, "cache-stream", "cache-stream")
        for event_type, payload in (
            ("messageStart", {"role": "assistant"}),
            ("contentBlockDelta", {"delta": {"text": response_text}, "contentBlockIndex": 0}),
            ("contentBlockStop", {"contentBlockIndex": 0}),
            ("messageStop", {"stopReason": "end_turn"}),
            ("metadata", {"usage": {"inputTokens": 11, "outputTokens": 4, "totalTokens": 15}}),
        )
    )

    def respond(request: Request) -> Reply:
        assert request.method == "POST"
        assert unquote(request.target) == f"/model/{model_id}/converse-stream", request.target
        assert json.loads(request.body) == expected_request_body
        assert "/us-east-1/bedrock/aws4_request" in request.headers.get("authorization", ""), "wrong SigV4 region"
        return Reply(body=response_body, content_type="application/vnd.amazon.eventstream")

    with wire_server(respond) as wire:
        first_stream: Final = await acompletion(
            model=model,
            messages=messages,
            max_tokens=40,
            temperature=1,
            stream=True,
            api_base=wire.url,
            aws_access_key_id="AKIASCRIPTEDPROVIDER",
            aws_secret_access_key="scripted-secret",
            aws_region_name="us-east-1",
        )
        first_chunks: Final = tuple([chunk async for chunk in first_stream])
        first_text: Final = "".join(chunk.choices[0].delta.content or "" for chunk in first_chunks)
        assert first_text == response_text
        first_requests: Final = wire.drain()
        assert len(first_requests) == 1
        first_request: Final = first_requests[0]

        first_cache_handler: Final = first_stream.logging_obj.llm_caching_handler
        assert first_cache_handler is not None
        first_cache_key: Final = first_cache_handler.preset_cache_key
        assert first_cache_key is not None
        for _ in range(10):
            await asyncio.sleep(0)
            await _drain_cache_writes()
            if await redis_response_cache.async_get_cache(cache_key=first_cache_key) is not None:
                break
        stored_response: Final = await redis_response_cache.async_get_cache(cache_key=first_cache_key)

        second_stream: Final = await acompletion(
            model=model,
            messages=messages,
            max_tokens=40,
            temperature=1,
            stream=True,
            api_base=wire.url,
            aws_access_key_id="AKIASCRIPTEDPROVIDER",
            aws_secret_access_key="scripted-secret",
            aws_region_name="us-east-1",
        )
        second_chunks: Final = tuple([chunk async for chunk in second_stream])
        second_text: Final = "".join(chunk.choices[0].delta.content or "" for chunk in second_chunks)
        second_cache_handler: Final = second_stream.logging_obj.llm_caching_handler
        assert second_cache_handler is not None
        second_cache_key: Final = second_cache_handler.preset_cache_key
        assert second_cache_key is not None

        assert unquote(first_request.target) == f"/model/{model_id}/converse-stream", first_request.target
        assert first_cache_key == second_cache_key
        assert isinstance(stored_response, dict)
        stored_text: Final = stored_response["choices"][0]["message"]["content"]
        assert first_text == stored_text == second_text
        assert len(wire.drain()) == 0, "second call should hit Redis"


@pytest.mark.asyncio
async def test_redis_cache_atext_completion(redis_response_cache: Cache) -> None:
    prompt: Final = f"cached text completion {uuid.uuid4().hex}"
    first: Final = await litellm.atext_completion(
        model="gpt-3.5-turbo-instruct",
        prompt=prompt,
        mock_response="cached",
    )
    await _drain_cache_writes()
    second: Final = await litellm.atext_completion(
        model="gpt-3.5-turbo-instruct",
        prompt=prompt,
        mock_response="different",
    )

    assert first.id == second.id


def test_redis_cache_basic(redis_response_cache: Cache) -> None:
    messages: Final = [{"role": "user", "content": f"redis basic {uuid.uuid4().hex}"}]
    first: Final = completion(
        model="gpt-4o-mini",
        messages=messages,
        caching=True,
        mock_response="cached",
    )
    second: Final = completion(
        model="gpt-4o-mini",
        messages=messages,
        caching=True,
        mock_response="different",
    )

    assert first.id == second.id


def test_redis_cache_completion(redis_response_cache: Cache) -> None:
    messages: Final = [{"role": "user", "content": f"redis completion {uuid.uuid4().hex}"}]
    first: Final = completion(
        model="gpt-4o-mini", messages=messages, caching=True, mock_response="first"
    )
    cached: Final = completion(
        model="gpt-4o-mini", messages=messages, caching=True, mock_response="cached"
    )
    changed_params: Final = completion(
        model="gpt-4o-mini",
        messages=messages,
        caching=True,
        temperature=0.5,
        mock_response="different params",
    )
    changed_model: Final = completion(
        model="gpt-4.1-mini", messages=messages, caching=True, mock_response="different model"
    )

    assert cached.id == first.id
    assert changed_params.id != first.id
    assert changed_model.id != first.id


def test_redis_cache_completion_stream(redis_response_cache: Cache) -> None:
    messages: Final = [{"role": "user", "content": f"redis stream {uuid.uuid4().hex}"}]
    first_chunks: Final = tuple(
        completion(
            model="gpt-4o-mini",
            messages=messages,
            caching=True,
            stream=True,
            mock_response="cached stream",
        )
    )
    second_chunks: Final = tuple(
        completion(
            model="gpt-4o-mini",
            messages=messages,
            caching=True,
            stream=True,
            mock_response="different stream",
        )
    )

    assert first_chunks[-1].id == second_chunks[-1].id
    assert "".join(chunk.choices[0].delta.content or "" for chunk in first_chunks) == "".join(
        chunk.choices[0].delta.content or "" for chunk in second_chunks
    )


@pytest.mark.asyncio
async def test_redis_get_ttl(redis_response_cache: Cache) -> None:
    redis_backend: Final = redis_response_cache.cache
    key: Final = f"ttl-{uuid.uuid4().hex}"
    redis_backend.set_cache(key, "stored", ttl=60)

    ttl: Final = await redis_backend.async_get_ttl(key)

    assert ttl is not None
    assert 0 < ttl <= 60


def test_redis_increment_pipeline(redis_response_cache: Cache) -> None:
    redis_backend: Final = redis_response_cache.cache
    key: Final = f"increment-{uuid.uuid4().hex}"

    assert redis_backend.increment_cache(key, value=1) == 1
    assert redis_backend.increment_cache(key, value=2) == 3


@pytest.mark.asyncio
async def test_redis_proxy_batch_redis_get_cache(redis_response_cache: Cache) -> None:
    hook: Final = PROXY_BatchRedisRequests()
    hook.in_memory_cache = InMemoryCache()
    messages: Final = [{"role": "user", "content": f"proxy cache {uuid.uuid4().hex}"}]
    first: Final = await acompletion(
        model="gpt-4o-mini",
        messages=messages,
        mock_response="first",
    )
    assert first is not None
    await _drain_cache_writes()
    second: Final = await acompletion(
        model="gpt-4o-mini",
        messages=messages,
        mock_response="second",
    )

    assert "cache_key" not in first._hidden_params
    assert "cache_key" in second._hidden_params
    assert second.id == first.id


def test_sync_cache_control_overrides(redis_response_cache: Cache) -> None:
    messages: Final = [{"role": "user", "content": f"sync cache control {uuid.uuid4().hex}"}]
    first: Final = completion(
        model="gpt-4o-mini", messages=messages, caching=True, mock_response="cached"
    )
    bypassed: Final = completion(
        model="gpt-4o-mini",
        messages=messages,
        caching=True,
        cache={"no-cache": True},
        mock_response="not cached",
    )

    assert first.id != bypassed.id


@pytest.fixture
def redis_search_enabled() -> None:
    with redis.Redis(
        host=os.environ["REDIS_HOST"],
        port=int(os.environ["REDIS_PORT"]),
        password=os.environ.get("REDIS_PASSWORD"),
    ) as client:
        try:
            client.execute_command("FT._LIST")
        except redis.exceptions.ResponseError:
            pytest.skip("Redis Search is required for semantic caching")


@pytest.mark.asyncio
async def test_redis_semantic_cache_acompletion(
    monkeypatch: pytest.MonkeyPatch, redis_search_enabled: None
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "scripted-embedding-key")
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    index_name: Final = f"semantic-{uuid.uuid4().hex}"
    monkeypatch.setattr(
        litellm,
        "cache",
        Cache(
            type=LiteLLMCacheType.REDIS_SEMANTIC,
            redis_url=f"redis://{os.environ['REDIS_HOST']}:{os.environ['REDIS_PORT']}/0",
            redis_semantic_cache_index_name=index_name,
            similarity_threshold=0.8,
        ),
    )
    embedding_response: Final = {
        "object": "list",
        "data": [{"object": "embedding", "embedding": [0.1] * 1536, "index": 0}],
        "model": "text-embedding-ada-002",
        "usage": {"prompt_tokens": 1, "total_tokens": 1},
    }
    with respx.mock(base_url="https://api.openai.com") as upstream:
        embeddings: Final = upstream.post("/v1/embeddings").mock(
            return_value=httpx.Response(200, json=embedding_response)
        )
        first: Final = await acompletion(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "write a poem about summer"}],
            max_tokens=20,
            mock_response="Summer sun shines bright and warm.",
        )
        await _drain_cache_writes()
        second: Final = await acompletion(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "write a poem about summertime"}],
            max_tokens=20,
            mock_response="A different summer poem.",
        )

    embedded_inputs: Final = {json.loads(call.request.content)["input"] for call in embeddings.calls}
    assert {"write a poem about summer", "write a poem about summertime"} <= embedded_inputs
    assert first.id == second.id
    assert second.choices[0].message.content == "Summer sun shines bright and warm."


def test_redis_semantic_cache_completion(
    monkeypatch: pytest.MonkeyPatch, redis_search_enabled: None
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "scripted-embedding-key")
    monkeypatch.setattr(
        litellm,
        "cache",
        Cache(
            type=LiteLLMCacheType.REDIS_SEMANTIC,
            redis_url=f"redis://{os.environ['REDIS_HOST']}:{os.environ['REDIS_PORT']}/0",
            redis_semantic_cache_index_name=f"semantic-{uuid.uuid4().hex}",
            similarity_threshold=0.8,
        ),
    )
    embedding_response: Final = {
        "object": "list",
        "data": [{"object": "embedding", "embedding": [0.1] * 1536, "index": 0}],
        "model": "text-embedding-ada-002",
        "usage": {"prompt_tokens": 1, "total_tokens": 1},
    }
    with respx.mock(base_url="https://api.openai.com") as upstream:
        embeddings: Final = upstream.post("/v1/embeddings").mock(
            return_value=httpx.Response(200, json=embedding_response)
        )
        first: Final = completion(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "write a poem about summer"}],
            max_tokens=20,
            mock_response="Summer sun shines bright and warm.",
        )
        second: Final = completion(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "write a poem about summertime"}],
            max_tokens=20,
            mock_response="A different summer poem.",
        )

    embedded_inputs: Final = {json.loads(call.request.content)["input"] for call in embeddings.calls}
    assert {"write a poem about summer", "write a poem about summertime"} <= embedded_inputs
    assert first.id == second.id
    assert second.choices[0].message.content == "Summer sun shines bright and warm."


@pytest.mark.asyncio
async def test_acompletion_caching_on_router(redis_response_cache: Cache) -> None:
    router: Final = Router(
        model_list=[
            {
                "model_name": "cached-model",
                "litellm_params": {"model": "gpt-4o-mini", "mock_response": "router response"},
            }
        ],
        redis_host=os.environ["REDIS_HOST"],
        redis_port=os.environ["REDIS_PORT"],
        cache_responses=True,
        routing_strategy="simple-shuffle",
    )
    messages: Final = [{"role": "user", "content": f"router cache {uuid.uuid4().hex}"}]

    first: Final = await router.acompletion(model="cached-model", messages=messages)
    await _drain_cache_writes()
    second: Final = await router.acompletion(model="cached-model", messages=messages)

    assert second.id == first.id
    assert second.choices[0].message.content == first.choices[0].message.content


@pytest.mark.asyncio
async def test_acompletion_caching_on_router_caching_groups(redis_response_cache: Cache) -> None:
    router: Final = Router(
        model_list=[
            {
                "model_name": "primary",
                "litellm_params": {"model": "gpt-4o-mini", "mock_response": "router response"},
            },
            {
                "model_name": "secondary",
                "litellm_params": {"model": "gpt-4o-mini", "mock_response": "different response"},
            },
        ],
        redis_host=os.environ["REDIS_HOST"],
        redis_port=os.environ["REDIS_PORT"],
        cache_responses=True,
        caching_groups=[("primary", "secondary")],
    )
    messages: Final = [{"role": "user", "content": f"router groups {uuid.uuid4().hex}"}]

    first: Final = await router.acompletion(model="primary", messages=messages)
    await _drain_cache_writes()
    second: Final = await router.acompletion(model="secondary", messages=messages)

    assert second.id == first.id
    assert second.choices[0].message.content == first.choices[0].message.content


@pytest.mark.asyncio
async def test_acompletion_caching_with_ttl_on_router(redis_response_cache: Cache) -> None:
    router: Final = Router(
        model_list=[
            {
                "model_name": "ttl-model",
                "litellm_params": {"model": "gpt-4o-mini", "mock_response": "router response"},
            }
        ],
        redis_host=os.environ["REDIS_HOST"],
        redis_port=os.environ["REDIS_PORT"],
        cache_responses=True,
    )
    messages: Final = [{"role": "user", "content": f"router ttl {uuid.uuid4().hex}"}]

    first: Final = await router.acompletion(model="ttl-model", messages=messages, ttl=0)
    await _drain_cache_writes()
    second: Final = await router.acompletion(model="ttl-model", messages=messages, ttl=0)
    await _drain_cache_writes()
    stored: Final = await router.acompletion(model="ttl-model", messages=messages, ttl=60)
    await _drain_cache_writes()
    replayed: Final = await router.acompletion(model="ttl-model", messages=messages, ttl=60)

    assert second.id != first.id
    assert replayed.id == stored.id


@pytest.mark.asyncio
async def test_completion_caching_on_router(redis_response_cache: Cache) -> None:
    router: Final = Router(
        model_list=[
            {
                "model_name": "completion-model",
                "litellm_params": {"model": "gpt-4o-mini", "mock_response": "router response"},
            }
        ],
        redis_host=os.environ["REDIS_HOST"],
        redis_port=os.environ["REDIS_PORT"],
        cache_responses=True,
    )
    messages: Final = [{"role": "user", "content": f"router completion {uuid.uuid4().hex}"}]

    first: Final = await router.acompletion(model="completion-model", messages=messages)
    await _drain_cache_writes()
    second: Final = await router.acompletion(model="completion-model", messages=messages)

    assert second.id == first.id
