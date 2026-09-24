"""Differential tests: every scenario runs through the Python `Cache` and the native one, and the
native facade must observe exactly what the Python facade observes."""

import asyncio
import threading
import time
from collections.abc import Awaitable, Callable, Generator
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from typing import Final, TypeAlias, cast
from uuid import uuid4

import fakeredis
import pytest

import litellm
from litellm.caching.caching import Cache
from litellm.rust_bridge import _native
from litellm.types.caching import LiteLLMCacheType
from litellm.types.llms.custom_llm import CustomLLMItem
from litellm.types.utils import EmbeddingResponse, Usage
from tests.test_litellm_rust.support.isolation import isolated_callback_registries, rebound

pytestmark: Final = pytest.mark.requires_rust_extension

NativeCache: Final = _native.Cache
_CacheResolver: Final = _native._CacheResolver  # pyright: ignore[reportPrivateUsage]  # the resolver native routes use has no public module name

BACKENDS: Final = (LiteLLMCacheType.LOCAL, LiteLLMCacheType.DISK, LiteLLMCacheType.REDIS)
EMBEDDING_PROVIDER: Final = "parity-embedding"

CacheBuilder: TypeAlias = Callable[..., Cache]
Scenario: TypeAlias = Callable[[Cache], Awaitable[object]]


def resolved_kind(cache: object) -> str:
    return _CacheResolver(SimpleNamespace(cache=cache)).resolve().kind


def fake_redis_url(stack: ExitStack) -> str:
    server: Final = fakeredis.TcpFakeServer(("127.0.0.1", 0), server_type="redis")
    worker: Final = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()

    def stop() -> None:
        server.shutdown()
        server.server_close()
        worker.join(timeout=5)

    stack.callback(stop)
    return f"127.0.0.1:{server.server_address[1]}"


@pytest.fixture
def build(request: pytest.FixtureRequest, tmp_path: Path) -> Generator[CacheBuilder]:
    """Builds a facade over its own empty store, so neither facade can read the other's entries."""
    backend: Final = cast(LiteLLMCacheType, request.param)
    with ExitStack() as stack:

        def construct(facade_class: type[Cache], **settings: object) -> Cache:
            match backend:
                case LiteLLMCacheType.LOCAL:
                    facade = facade_class(type=backend, **settings)
                case LiteLLMCacheType.DISK:
                    facade = facade_class(type=backend, disk_cache_dir=str(tmp_path / uuid4().hex), **settings)
                case LiteLLMCacheType.REDIS:
                    host, _, port = fake_redis_url(stack).partition(":")
                    facade = facade_class(type=backend, host=host, port=port, **settings)
                case _:
                    raise AssertionError(f"no parity factory for {backend}")
            expected_kind: Final = "native" if facade_class is NativeCache else "python_callback"
            assert resolved_kind(facade) == expected_kind
            return facade

        yield construct


async def observe_both(build: CacheBuilder, scenario: Scenario, **settings: object) -> object:
    """Runs `scenario` on both facades, asserts they agree, and returns what Python observed."""
    python: Final = await scenario(build(Cache, **settings))
    native: Final = await scenario(build(NativeCache, **settings))
    assert native == python
    return python


def completion(label: str, **extra: object) -> dict[str, object]:
    return {"model": "gpt-4o", "messages": [{"role": "user", "content": f"{label} {uuid4().hex}"}], **extra}


KEY_CASES: Final[tuple[tuple[str, dict[str, object]], ...]] = (
    ("plain", {"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]}),
    (
        "params",
        {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "hi"}],
            "temperature": 0.2,
            "max_tokens": 10,
            "tools": [{"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}],
        },
    ),
    (
        "anthropic",
        {
            "model": "claude",
            "messages": [{"role": "user", "content": "hi"}],
            "system": "be brief",
            "thinking": {"type": "enabled", "budget_tokens": 1024},
            "tool_choice": {"type": "auto"},
        },
    ),
    ("embedding", {"model": "text-embedding-3-small", "input": ["a", "b"]}),
    ("dynamic namespace", {"model": "gpt-4o", "input": "x", "cache": {"namespace": "tenant"}}),
    ("metadata namespace", {"model": "gpt-4o", "input": "x", "metadata": {"redis_namespace": "tenant"}}),
    (
        "caching groups",
        {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "hi"}],
            "metadata": {"model_group": "group", "caching_groups": [("group", "other")]},
        },
    ),
    ("ignored internals", {"model": "gpt-4o", "input": "x", "litellm_call_id": "id", "api_key": "sk-x"}),
)


@pytest.mark.parametrize("namespace", [None, "team"])
@pytest.mark.parametrize(("case", "kwargs"), KEY_CASES, ids=[case for case, _ in KEY_CASES])
def test_cache_keys_and_preset_key_side_effects_match(
    case: str, kwargs: dict[str, object], namespace: str | None
) -> None:
    del case

    def derive(facade_class: type[Cache]) -> tuple[str, object]:
        call_kwargs: Final = {**kwargs, "litellm_params": {}}
        key: Final = facade_class(type=LiteLLMCacheType.LOCAL, namespace=namespace).get_cache_key(**call_kwargs)
        return key, call_kwargs["litellm_params"]

    python: Final = derive(Cache)
    assert derive(NativeCache) == python
    assert python[1] == {"preset_cache_key": python[0]}


SEMANTIC_METADATA: Final[tuple[dict[str, object], ...]] = (
    {},
    {"user_api_key": "hash-a"},
    {"user_api_key": "hash-a", "user_api_key_team_id": "team-1", "user_api_key_org_id": "org-1"},
    {"user_api_key": "hash-a", "user_api_key_end_user_id": "end-user-1"},
    {"user_api_key_end_user_id": "end-user-1"},
)


@pytest.mark.parametrize("scope", ["key", "end_user"])
@pytest.mark.parametrize("metadata_field", ["metadata", "litellm_metadata"])
@pytest.mark.parametrize("metadata", SEMANTIC_METADATA)
def test_semantic_scope_keys_match(scope: str, metadata_field: str, metadata: dict[str, object]) -> None:
    kwargs: Final = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": f"prompt {uuid4().hex}"}],
        "temperature": 0.1,
        metadata_field: metadata,
    }

    def derive(facade_class: type[Cache]) -> str:
        return facade_class(
            type=LiteLLMCacheType.REDIS_SEMANTIC,
            redis_url="redis://127.0.0.1:6379",
            similarity_threshold=0.8,
            semantic_cache_scope=scope,
        ).get_cache_key(**kwargs)

    assert derive(NativeCache) == derive(Cache)


async def read_and_write_round(facade: Cache) -> object:
    stored: Final = completion("stored")
    facade.add_cache({"answer": "sync"}, **stored)
    async_stored: Final = completion("async-stored")
    await facade.async_add_cache({"answer": "async"}, **async_stored)
    return (
        facade.get_cache(**stored),
        await facade.async_get_cache(**stored),
        facade.get_cache(**async_stored),
        await facade.async_get_cache(**async_stored),
        facade.get_cache(**completion("absent")),
        await facade.async_get_cache(**completion("absent")),
        facade.get_cache(**{**stored, "model": "gpt-4o-mini"}),
    )


MAX_AGE_CONTROLS: Final[tuple[dict[str, object], ...]] = (
    {},
    {"s-maxage": 0},
    {"s-max-age": 0},
    {"s-maxage": 1e-9},
    {"s-max-age": 1e-9},
    {"s-maxage": 3600},
    {"s-max-age": 3600},
    {"s-maxage": 1e-9, "s-max-age": 3600},
    {"s-maxage": 3600, "s-max-age": 1e-9},
    {"s-maxage": 0, "s-max-age": 1e-9},
)


async def max_age_round(facade: Cache) -> object:
    kwargs: Final = completion("max-age")
    facade.add_cache({"answer": 1}, **kwargs)
    time.sleep(0.01)
    return [
        (facade.get_cache(**kwargs, cache=control), await facade.async_get_cache(**kwargs, cache=control))
        for control in MAX_AGE_CONTROLS
    ]


async def ttl_round(facade: Cache) -> object:
    configured: Final = completion("configured-ttl")
    facade.add_cache({"answer": "configured"}, **configured)
    per_request: Final = completion("request-ttl")
    await facade.async_add_cache({"answer": "request"}, **per_request, cache={"ttl": 1})
    long_lived: Final = completion("long-ttl")
    facade.add_cache({"answer": "long"}, **long_lived, cache={"ttl": 3600})
    immediately: Final = (facade.get_cache(**configured), facade.get_cache(**per_request))
    await asyncio.sleep(1.5)
    return (
        immediately,
        facade.get_cache(**configured),
        await facade.async_get_cache(**per_request),
        facade.get_cache(**long_lived),
    )


async def namespace_round(facade: Cache) -> object:
    kwargs: Final = completion("namespace")
    facade.add_cache({"answer": "tenant"}, **kwargs, cache={"namespace": "tenant"})
    return (
        facade.get_cache(**kwargs),
        facade.get_cache(**kwargs, cache={"namespace": "tenant"}),
        await facade.async_get_cache(**kwargs, cache={"namespace": "tenant"}),
        facade.get_cache(**kwargs, cache={"namespace": "other"}),
    )


async def opt_in_round(facade: Cache) -> object:
    silent: Final = completion("silent")
    facade.add_cache({"answer": "silent"}, **silent)
    opted: Final = completion("opted")
    await facade.async_add_cache({"answer": "opted"}, **opted, cache={"use-cache": True})
    return (
        facade.get_cache(**silent, cache={"use-cache": True}),
        facade.get_cache(**opted),
        facade.get_cache(**opted, cache={"use-cache": True}),
        await facade.async_get_cache(**opted, cache={"use-cache": True}),
    )


@pytest.mark.parametrize("build", BACKENDS, indirect=True)
async def test_reads_and_writes_match(build: CacheBuilder) -> None:
    observed: Final = cast(tuple[object, ...], await observe_both(build, read_and_write_round))
    assert observed[:4] == ({"answer": "sync"}, {"answer": "sync"}, {"answer": "async"}, {"answer": "async"})


@pytest.mark.parametrize("build", BACKENDS, indirect=True)
async def test_max_age_controls_match(build: CacheBuilder) -> None:
    observed: Final = cast(list[tuple[object, object]], await observe_both(build, max_age_round))
    values: Final = [value for pair in observed for value in pair]
    assert None in values
    assert {"answer": 1} in values


@pytest.mark.parametrize("build", BACKENDS, indirect=True)
@pytest.mark.parametrize("ttl", [None, 1])
async def test_ttl_matches(build: CacheBuilder, ttl: int | None) -> None:
    observed: Final = cast(tuple[object, ...], await observe_both(build, ttl_round, ttl=ttl))
    assert observed[0] == ({"answer": "configured"}, {"answer": "request"})
    assert observed[-1] == {"answer": "long"}


@pytest.mark.parametrize("build", BACKENDS, indirect=True)
async def test_namespaces_match(build: CacheBuilder) -> None:
    observed: Final = await observe_both(build, namespace_round)
    assert observed == (None, {"answer": "tenant"}, {"answer": "tenant"}, None)


@pytest.mark.parametrize("build", BACKENDS, indirect=True)
async def test_default_off_mode_matches(build: CacheBuilder) -> None:
    observed: Final = await observe_both(build, opt_in_round, mode="default_off")
    assert observed == (None, None, {"answer": "opted"}, {"answer": "opted"})


def embedding_response(
    texts: list[str], prompt_tokens: int | None, details: dict[str, int] | None
) -> EmbeddingResponse:
    response: Final = EmbeddingResponse(
        model="text-embedding-3-small",
        data=[{"object": "embedding", "index": index, "embedding": [float(index), 0.5]} for index in range(len(texts))],
    )
    if prompt_tokens is not None:
        response.usage = Usage(prompt_tokens=prompt_tokens, completion_tokens=0, prompt_tokens_details=details)
    return response


def embedding_round(input_count: int, prompt_tokens: int | None, details: dict[str, int] | None) -> Scenario:
    async def run(facade: Cache) -> object:
        texts: Final = [f"text {index} {uuid4().hex}" for index in range(input_count)]
        await facade.async_add_cache_pipeline(
            embedding_response(texts, prompt_tokens, details),
            model="text-embedding-3-small",
            input=texts if input_count != 1 else texts[0],
        )
        entries: Final = [await facade.async_get_cache(model="text-embedding-3-small", input=text) for text in texts]
        return [
            {key: value for key, value in cast(dict[str, object], entry).items() if key != "timestamp"}
            if isinstance(entry, dict)
            else entry
            for entry in entries
        ]

    return run


@pytest.mark.parametrize("build", BACKENDS, indirect=True)
@pytest.mark.parametrize(
    ("input_count", "prompt_tokens", "details"),
    [
        (1, 7, None),
        (3, 10, None),
        (3, 10, {"text_tokens": 7, "image_count": 2}),
        (2, None, None),
    ],
    ids=["single-string", "split-with-remainder", "split-details", "no-usage"],
)
async def test_embedding_pipeline_matches(
    build: CacheBuilder, input_count: int, prompt_tokens: int | None, details: dict[str, int] | None
) -> None:
    observed: Final = cast(
        list[object], await observe_both(build, embedding_round(input_count, prompt_tokens, details))
    )
    assert all(isinstance(entry, dict) for entry in observed)


@pytest.mark.parametrize("build", BACKENDS, indirect=True)
async def test_embedding_count_mismatch_skips_the_write_on_both(build: CacheBuilder) -> None:
    async def run(facade: Cache) -> object:
        texts: Final = [f"a {uuid4().hex}", f"b {uuid4().hex}"]
        await facade.async_add_cache_pipeline(
            embedding_response(texts[:1], 3, None), model="text-embedding-3-small", input=texts
        )
        return [await facade.async_get_cache(model="text-embedding-3-small", input=text) for text in texts]

    assert await observe_both(build, run) == [None, None]


class CountingEmbedding(litellm.CustomLLM):
    def __init__(self) -> None:
        super().__init__()
        self.inputs: list[list[str]] = []

    async def aembedding(
        self,
        model: str,
        input: list[str],
        model_response: EmbeddingResponse,
        print_verbose: Callable[..., object],
        logging_obj: object,
        optional_params: dict[str, object],
        api_key: object = None,
        api_base: object = None,
        timeout: object = None,
        litellm_params: object = None,
    ) -> EmbeddingResponse:
        self.inputs.append(list(input))
        model_response.model = model
        model_response.data = [
            {"object": "embedding", "index": index, "embedding": [float(len(text)), 1.0]}
            for index, text in enumerate(input)
        ]
        model_response.usage = Usage(prompt_tokens=len(input) * 3, completion_tokens=0)
        return model_response


def through_litellm(scenario: Callable[[CountingEmbedding], Awaitable[object]]) -> Scenario:
    """Installs `facade` as `litellm.cache` so the real caching handler drives it."""

    async def run(facade: Cache) -> object:
        provider: Final = CountingEmbedding()
        with ExitStack() as stack:
            stack.enter_context(isolated_callback_registries())
            stack.enter_context(rebound(litellm, "cache", facade))
            stack.enter_context(
                rebound(
                    litellm,
                    "custom_provider_map",
                    [cast(CustomLLMItem, {"provider": EMBEDDING_PROVIDER, "custom_handler": provider})],
                )
            )
            stack.enter_context(
                rebound(
                    litellm,
                    "_custom_providers",  # pyright: ignore[reportPrivateUsage]  # no public provider-registration hook
                    [*litellm._custom_providers, EMBEDDING_PROVIDER],  # pyright: ignore[reportPrivateUsage]  # no public provider-registration hook
                )
            )
            stack.enter_context(rebound(litellm, "provider_list", [*litellm.provider_list, EMBEDDING_PROVIDER]))
            observed: Final = await scenario(provider)
            await asyncio.sleep(0.2)
            return observed

    return run


async def stream_text(messages: list[dict[str, str]], reply: str) -> str:
    stream: Final = await litellm.acompletion(
        model="gpt-4o", messages=messages, mock_response=reply, caching=True, stream=True
    )
    return "".join([chunk.choices[0].delta.content or "" async for chunk in stream])


async def completion_flow(provider: CountingEmbedding) -> object:
    del provider
    prompt: Final = [{"role": "user", "content": f"hello {uuid4().hex}"}]
    first: Final = await litellm.acompletion(model="gpt-4o", messages=prompt, mock_response="hi there", caching=True)
    await asyncio.sleep(0.2)
    second: Final = await litellm.acompletion(model="gpt-4o", messages=prompt, mock_response="changed", caching=True)
    bypass: Final = await litellm.acompletion(
        model="gpt-4o", messages=prompt, mock_response="fresh", caching=True, cache={"no-cache": True}
    )
    unstored_prompt: Final = [{"role": "user", "content": f"unstored {uuid4().hex}"}]
    await litellm.acompletion(
        model="gpt-4o", messages=unstored_prompt, mock_response="once", caching=True, cache={"no-store": True}
    )
    await asyncio.sleep(0.2)
    unstored: Final = await litellm.acompletion(
        model="gpt-4o", messages=unstored_prompt, mock_response="twice", caching=True
    )
    first_stream: Final = await stream_text(prompt, "streamed once")
    await asyncio.sleep(0.2)
    second_stream: Final = await stream_text(prompt, "streamed twice")
    return (
        second.id == first.id,
        second.choices[0].message.content,
        second._hidden_params.get("cache_hit"),  # pyright: ignore[reportPrivateUsage]  # cache_hit is only exposed through hidden params
        bypass.choices[0].message.content,
        unstored.choices[0].message.content,
        [first_stream, second_stream],
    )


async def embedding_flow(provider: CountingEmbedding) -> object:
    model: Final = f"{EMBEDDING_PROVIDER}/model"
    tag: Final = uuid4().hex
    first: Final = await litellm.aembedding(model=model, input=[f"a {tag}", f"bb {tag}"], caching=True)
    await asyncio.sleep(0.2)
    second: Final = await litellm.aembedding(model=model, input=[f"bb {tag}", f"ccc {tag}", f"a {tag}"], caching=True)
    return (
        [[text.split(" ")[0] for text in call] for call in provider.inputs],
        [item["embedding"] for item in first.data],
        [item["embedding"] for item in second.data],
        None if second.usage is None else second.usage.prompt_tokens,
    )


@pytest.mark.parametrize("build", BACKENDS, indirect=True)
async def test_completion_caching_through_litellm_matches(build: CacheBuilder) -> None:
    observed: Final = await observe_both(build, through_litellm(completion_flow))
    assert observed == (True, "hi there", True, "fresh", "twice", ["streamed once", "streamed once"])


@pytest.mark.parametrize("build", BACKENDS, indirect=True)
async def test_embedding_caching_through_litellm_matches(build: CacheBuilder) -> None:
    observed: Final = cast(tuple[object, ...], await observe_both(build, through_litellm(embedding_flow)))
    assert observed[0] == [["a", "bb"], ["ccc"]]
