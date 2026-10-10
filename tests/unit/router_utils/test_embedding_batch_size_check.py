import json
from types import MappingProxyType
from typing import Final

import httpx
import pytest
import respx

import litellm
from litellm import Router
from litellm.router_utils.embedding_batch_size_check import (
    count_embedding_inputs,
    effective_embedding_input,
    raise_if_embedding_batch_too_large,
)

_MODEL: Final = "embed-group"
_PROVIDER: Final = "openai"
_OPENAI_EMBED_BASE: Final = "https://batch-limit-embed.local/v1"
_VLLM_EMBED_BASE: Final = "https://batch-limit-vllm.local/v1"


def _model_info(max_embedding_batch_size: int | None) -> dict[str, object]:
    if max_embedding_batch_size is None:
        return {}
    return {"max_embedding_batch_size": max_embedding_batch_size}


def _raise_batch(input_value: str | list, limit: int | None) -> None:
    raise_if_embedding_batch_too_large(
        input=input_value,
        model_info=_model_info(limit),
        model=_MODEL,
        llm_provider=_PROVIDER,
    )


def _embedding_router(
    *,
    provider_model: str,
    api_base: str,
    max_embedding_batch_size: int,
    deployment_id: str,
    litellm_params_extra: dict[str, object] | None = None,
) -> Router:
    litellm_params: dict[str, object] = {
        "model": provider_model,
        "api_key": "sk-fake",
        "api_base": api_base,
    }
    if litellm_params_extra:
        litellm_params.update(litellm_params_extra)
    return Router(
        model_list=[
            {
                "model_name": "embed",
                "litellm_params": litellm_params,
                "model_info": {"id": deployment_id, "max_embedding_batch_size": max_embedding_batch_size},
            }
        ],
        num_retries=0,
    )


async def _call_embedding(router: Router, *, sync: bool, **kwargs: object) -> None:
    if sync:
        router.embedding(**kwargs)
    else:
        await router.aembedding(**kwargs)


def _vllm_embedding_response(input_count: int) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "object": "list",
            "data": [{"object": "embedding", "index": i, "embedding": [0.1, 0.2]} for i in range(input_count)],
            "model": "bge-m3",
            "usage": {"prompt_tokens": input_count, "total_tokens": input_count},
        },
    )


class TestCountEmbeddingInputs:
    @pytest.mark.parametrize(
        ("input_value", "expected_count"),
        [
            ("hello", 1),
            (["a", "b"], 2),
            ([1, 2, 3, 4, 5], 1),
            ([[1, 2, 3], [4, 5, 6]], 2),
        ],
    )
    def test_openai_input_shapes(self, input_value: object, expected_count: int) -> None:
        assert count_embedding_inputs(input_value) == expected_count

    def test_batch_count_is_independent_of_string_length(self) -> None:
        long_text: Final = "word " * 500
        assert count_embedding_inputs([long_text, long_text]) == 2


class TestEffectiveEmbeddingInput:
    _MIXED_EXTRA_BODY: Final = {
        "truncate": "END",
        "encoding_format": "float",
        "input": ["a", "b", "c"],
        "user": "batch-job",
    }

    @pytest.mark.parametrize(
        ("top_level", "extra_body", "expected"),
        [
            ("ok", None, "ok"),
            ("ok", {"input": ["a", "b", "c"]}, ["a", "b", "c"]),
            (["a", "b"], {"truncate": "END"}, ["a", "b"]),
            ("ignored-top-level", _MIXED_EXTRA_BODY, ["a", "b", "c"]),
            (
                ["would", "be", "three"],
                {"dimensions": 256, "input": "single-string", "model": "ignored"},
                ["would", "be", "three"],
            ),
            ("ok", {"truncate": "NONE", "input": ("a", "b")}, ("a", "b")),
            (["x", "y", "z"], {"input": [[1, 2], [3, 4]], "foo": "bar"}, ["x", "y", "z"]),
            (
                ["a", "b", "c"],
                {"input": "ok"},
                ["a", "b", "c"],
            ),
            (["a", "b"], {}, ["a", "b"]),
            (["a", "b", "c"], {"input": None, "truncate": "END"}, ["a", "b", "c"]),
            (["a", "b"], {"input": {"nested": "dict"}, "truncate": "END"}, ["a", "b"]),
            (["a", "b"], {"input": 42, "dimensions": 128}, ["a", "b"]),
        ],
    )
    def test_resolves_provider_bound_input(
        self,
        top_level: str | list,
        extra_body: dict[str, object] | None,
        expected: object,
    ) -> None:
        assert effective_embedding_input(top_level, extra_body) == expected

    @pytest.mark.parametrize(
        "extra_body",
        ["not-a-mapping", ["input", "list"]],
    )
    def test_non_mapping_extra_body_keeps_top_level(self, extra_body: object) -> None:
        top_level: Final = ["a", "b", "c"]
        assert effective_embedding_input(top_level, extra_body) == top_level

    def test_readonly_mapping_extra_body_with_input(self) -> None:
        extra_body: Final = MappingProxyType({"input": ["a", "b"], "encoding_format": "float"})
        assert effective_embedding_input("ok", extra_body) == ["a", "b"]


class TestRaiseIfEmbeddingBatchTooLarge:
    @pytest.mark.parametrize(
        "input_value",
        [
            "hello",
            ["a", "b"],
            ["a", "b", "c"],
            [1, 2, 3, 4, 5],
            [[1, 2, 3], [4, 5, 6]],
        ],
    )
    def test_accepts_inputs_at_or_below_limit(self, input_value: str | list) -> None:
        limit: Final = max(1, count_embedding_inputs(input_value))
        _raise_batch(input_value, limit)

    @pytest.mark.parametrize(
        ("input_value", "limit", "message_match"),
        [
            (["a", "b", "c", "d"], 3, "4 inputs"),
            ([[1], [2], [3]], 2, None),
            (
                effective_embedding_input(
                    "ok",
                    {"truncate": "END", "input": ["a", "b", "c", "d"], "user": "x"},
                ),
                3,
                "4 inputs",
            ),
            (
                effective_embedding_input(
                    ["a", "b", "c"],
                    {"input": {"bad": "shape"}, "encoding_format": "float"},
                ),
                2,
                None,
            ),
        ],
    )
    def test_rejects_inputs_above_limit(
        self,
        input_value: str | list,
        limit: int,
        message_match: str | None,
    ) -> None:
        with pytest.raises(litellm.BadRequestError, match=message_match) as excinfo:
            _raise_batch(input_value, limit)
        assert excinfo.value.status_code == 400
        if message_match is not None:
            assert f"max_embedding_batch_size={limit}" in excinfo.value.message

    def test_no_limit_configured_is_no_op(self) -> None:
        _raise_batch(["a"] * 100, None)


class TestRouterEmbeddingBatchLimit:
    @pytest.fixture(autouse=True)
    def _disable_aiohttp(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)

    @pytest.mark.parametrize("sync", [False, True], ids=["async", "sync"])
    @pytest.mark.asyncio
    async def test_oversized_top_level_input_rejects_before_provider_call(self, sync: bool) -> None:
        router: Final = _embedding_router(
            provider_model="openai/text-embedding-3-small",
            api_base=_OPENAI_EMBED_BASE,
            max_embedding_batch_size=2,
            deployment_id="embed-batch-capped",
        )
        embeddings_url: Final = f"{_OPENAI_EMBED_BASE}/embeddings"
        empty_ok: Final = httpx.Response(200, json={"object": "list", "data": [], "model": "x", "usage": {}})

        with respx.mock(assert_all_called=False) as respx_mock:
            route: Final = respx_mock.post(embeddings_url).mock(return_value=empty_ok)
            with pytest.raises(litellm.BadRequestError, match="max_embedding_batch_size=2"):
                await _call_embedding(
                    router,
                    sync=sync,
                    model="embed",
                    input=["one", "two", "three"],
                )
            assert route.call_count == 0

    @pytest.mark.parametrize("sync", [False, True], ids=["async", "sync"])
    @pytest.mark.asyncio
    async def test_extra_body_input_override_rejects_before_provider_call(self, sync: bool) -> None:
        router: Final = _embedding_router(
            provider_model="hosted_vllm/bge-m3",
            api_base=_VLLM_EMBED_BASE,
            max_embedding_batch_size=2,
            deployment_id="vllm-embed-batch-capped",
        )
        embeddings_url: Final = f"{_VLLM_EMBED_BASE}/embeddings"
        extra_body: Final = {"truncate": "END", "input": ["a", "b", "c"]}

        with respx.mock(assert_all_called=False) as respx_mock:
            route: Final = respx_mock.post(embeddings_url).mock(return_value=_vllm_embedding_response(3))
            with pytest.raises(litellm.BadRequestError, match="3 inputs"):
                await _call_embedding(
                    router,
                    sync=sync,
                    model="embed",
                    input="ok",
                    extra_body=extra_body,
                )
            assert route.call_count == 0

    @pytest.mark.asyncio
    async def test_deployment_extra_body_input_counts_toward_batch_limit(self) -> None:
        router: Final = Router(
            model_list=[
                {
                    "model_name": "embed",
                    "litellm_params": {
                        "model": "hosted_vllm/bge-m3",
                        "api_key": "sk-fake",
                        "api_base": _VLLM_EMBED_BASE,
                        "extra_body": {"input": ["a", "b", "c"]},
                    },
                    "model_info": {
                        "id": "vllm-embed-deployment-extra-body",
                        "max_embedding_batch_size": 2,
                    },
                }
            ],
            num_retries=0,
        )
        embeddings_url: Final = f"{_VLLM_EMBED_BASE}/embeddings"

        with respx.mock(assert_all_called=False) as respx_mock:
            route: Final = respx_mock.post(embeddings_url).mock(return_value=_vllm_embedding_response(3))
            with pytest.raises(litellm.BadRequestError, match="3 inputs"):
                await router.aembedding(model="embed", input="ok")
            assert route.call_count == 0

    @pytest.mark.asyncio
    async def test_extra_body_input_at_limit_reaches_provider(self) -> None:
        router: Final = _embedding_router(
            provider_model="hosted_vllm/bge-m3",
            api_base=_VLLM_EMBED_BASE,
            max_embedding_batch_size=2,
            deployment_id="vllm-embed-batch-capped",
        )
        embeddings_url: Final = f"{_VLLM_EMBED_BASE}/embeddings"

        with respx.mock(assert_all_called=False) as respx_mock:
            route: Final = respx_mock.post(embeddings_url).mock(return_value=_vllm_embedding_response(2))
            await router.aembedding(model="embed", input="ok", extra_body={"input": ["a", "b"]})
            assert route.call_count == 1
            assert json.loads(route.calls.last.request.content)["input"] == ["a", "b"]

    @pytest.mark.asyncio
    async def test_unrelated_extra_body_does_not_bypass_top_level_batch_limit(self) -> None:
        router: Final = _embedding_router(
            provider_model="hosted_vllm/bge-m3",
            api_base=_VLLM_EMBED_BASE,
            max_embedding_batch_size=2,
            deployment_id="vllm-embed-batch-capped",
        )
        embeddings_url: Final = f"{_VLLM_EMBED_BASE}/embeddings"

        with respx.mock(assert_all_called=False) as respx_mock:
            route: Final = respx_mock.post(embeddings_url).mock(return_value=_vllm_embedding_response(3))
            with pytest.raises(litellm.BadRequestError, match="3 inputs"):
                await _call_embedding(
                    router,
                    sync=False,
                    model="embed",
                    input=["a", "b", "c"],
                    extra_body={"truncate": "END", "encoding_format": "float"},
                )
            assert route.call_count == 0

    @pytest.mark.parametrize("sync", [False, True], ids=["async", "sync"])
    @pytest.mark.asyncio
    async def test_bedrock_ignores_extra_body_input_override_for_provider_but_guard_uses_top_level(
        self,
        sync: bool,
    ) -> None:
        router: Final = _embedding_router(
            provider_model="bedrock/amazon.titan-embed-text-v1",
            api_base="",
            max_embedding_batch_size=2,
            deployment_id="bedrock-titan-embed-batch-capped",
            litellm_params_extra={
                "aws_access_key_id": "AKIAFAKE",
                "aws_secret_access_key": "fake-secret",
                "aws_region_name": "us-east-1",
                "custom_llm_provider": "bedrock",
            },
        )

        with pytest.raises(litellm.BadRequestError, match="3 inputs"):
            await _call_embedding(
                router,
                sync=sync,
                model="embed",
                input=["a", "b", "c"],
                extra_body={"input": "ok"},
            )

    @pytest.mark.parametrize("sync", [False, True], ids=["async", "sync"])
    @pytest.mark.asyncio
    async def test_explicit_custom_llm_provider_in_deployment_rejects_oversized_batch(
        self,
        sync: bool,
    ) -> None:
        router: Final = _embedding_router(
            provider_model="openai/text-embedding-3-small",
            api_base=_OPENAI_EMBED_BASE,
            max_embedding_batch_size=2,
            deployment_id="embed-batch-capped-explicit-provider",
            litellm_params_extra={"custom_llm_provider": "openai"},
        )
        embeddings_url: Final = f"{_OPENAI_EMBED_BASE}/embeddings"
        empty_ok: Final = httpx.Response(200, json={"object": "list", "data": [], "model": "x", "usage": {}})

        with respx.mock(assert_all_called=False) as respx_mock:
            route: Final = respx_mock.post(embeddings_url).mock(return_value=empty_ok)
            with pytest.raises(litellm.BadRequestError, match="max_embedding_batch_size=2"):
                await _call_embedding(
                    router,
                    sync=sync,
                    model="embed",
                    input=["one", "two", "three"],
                )
            assert route.call_count == 0
