import json
import time
from collections.abc import Iterator
from types import MappingProxyType
from typing import Final

import httpx
import pytest
import respx

import litellm
from litellm import Router
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.litellm_logging import Logging as LitellmLogging
from litellm.types.router import RouterRateLimitError
from litellm.router_utils.embedding_batch_size_check import (
    count_embedding_inputs,
    deployments_admitting_embedding_batch,
    effective_embedding_input,
    embedding_batch_too_large_error,
    embedding_deployment_selection,
)

_MODEL: Final = "embed-group"
_PROVIDER: Final = "openai"
_OPENAI_EMBED_BASE: Final = "https://batch-limit-embed.local/v1"
_VLLM_EMBED_BASE: Final = "https://batch-limit-vllm.local/v1"
_UNCAPPED_EMBED_BASE: Final = "https://batch-limit-uncapped.local/v1"


def _model_info(max_embedding_batch_size: int | None) -> dict[str, object]:
    if max_embedding_batch_size is None:
        return {}
    return {"max_embedding_batch_size": max_embedding_batch_size}


def _deployment(limit: int | None, extra_body: dict[str, object] | None = None) -> dict[str, object]:
    return {
        "model_info": _model_info(limit),
        "litellm_params": {} if extra_body is None else {"extra_body": extra_body},
    }


def _batch_error(
    input_value: object,
    limit: int | None,
    request_kwargs: dict[str, object] | None = None,
) -> litellm.BadRequestError | None:
    return embedding_batch_too_large_error(
        deployment={"model_info": _model_info(limit), "litellm_params": {"custom_llm_provider": _PROVIDER}},
        input=input_value,
        request_kwargs=request_kwargs or {},
        model=_MODEL,
    )


def _embedding_router(
    *,
    provider_model: str,
    api_base: str,
    max_embedding_batch_size: int | None,
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
                "model_info": {"id": deployment_id, **_model_info(max_embedding_batch_size)},
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
            (None, {"input": ["a", "b"]}, ["a", "b"]),
            (7, {"input": ["a", "b"]}, ["a", "b"]),
            (None, None, None),
        ],
    )
    def test_resolves_provider_bound_input(
        self,
        top_level: object,
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


class TestDeploymentsAdmittingEmbeddingBatch:
    @pytest.fixture(autouse=True)
    def _embedding_selection(self) -> Iterator[None]:
        with embedding_deployment_selection():
            yield

    @pytest.mark.parametrize(
        ("input_value", "limit"),
        [
            ("hello", 1),
            (["a", "b"], 2),
            (["a", "b", "c"], 3),
            ([1, 2, 3, 4, 5], 1),
            ([[1, 2, 3], [4, 5, 6]], 2),
        ],
    )
    def test_keeps_deployments_whose_limit_admits_the_batch(self, input_value: str | list, limit: int) -> None:
        capped: Final = _deployment(limit)
        uncapped: Final = _deployment(None)
        assert deployments_admitting_embedding_batch([capped, uncapped], input_value, None) == [capped, uncapped]

    def test_drops_deployments_whose_limit_is_below_the_batch(self) -> None:
        capped: Final = _deployment(2)
        uncapped: Final = _deployment(None)
        assert deployments_admitting_embedding_batch([capped, uncapped], ["a", "b", "c"], None) == [uncapped]

    def test_keeps_every_deployment_when_none_admits_the_batch(self) -> None:
        smaller: Final = _deployment(1)
        small: Final = _deployment(2)
        assert deployments_admitting_embedding_batch([smaller, small], ["a", "b", "c"], None) == [smaller, small]

    def test_request_without_embedding_input_keeps_every_deployment(self) -> None:
        capped: Final = _deployment(1)
        uncapped: Final = _deployment(None)
        assert deployments_admitting_embedding_batch([capped, uncapped], None, None) == [capped, uncapped]

    def test_request_extra_body_replaces_the_deployment_extra_body(self) -> None:
        capped: Final = _deployment(2, extra_body={"input": ["a", "b", "c"]})
        uncapped: Final = _deployment(None)
        assert deployments_admitting_embedding_batch([capped, uncapped], "ok", None) == [uncapped]
        assert deployments_admitting_embedding_batch([capped, uncapped], "ok", {"extra_body": {}}) == [capped, uncapped]

    @pytest.mark.parametrize("top_level", [None, 7, {"text": "a"}])
    def test_extra_body_input_routes_when_top_level_input_is_not_a_list(self, top_level: object) -> None:
        capped: Final = _deployment(2)
        uncapped: Final = _deployment(None)
        request_kwargs: Final = {"extra_body": {"input": ["a", "b", "c"]}}
        assert deployments_admitting_embedding_batch([capped, uncapped], top_level, request_kwargs) == [uncapped]


class TestEmbeddingBatchTooLargeError:

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
        error: Final = _batch_error(input_value, limit)
        assert isinstance(error, litellm.BadRequestError)
        assert error.status_code == 400
        if message_match is not None:
            assert message_match in error.message
            assert f"max_embedding_batch_size={limit}" in error.message

    @pytest.mark.parametrize("top_level", [None, 7, {"text": "a"}])
    def test_extra_body_override_counts_when_top_level_input_is_not_a_list(self, top_level: object) -> None:
        error: Final = _batch_error(top_level, 2, {"extra_body": {"input": ["a", "b", "c"]}})
        assert error is not None
        assert "3 inputs" in error.message


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
    async def test_oversized_batch_rejection_reaches_failure_callbacks(self, sync: bool) -> None:
        failures: Final[list[Exception]] = []

        class FailureRecorder(CustomLogger):
            def log_failure_event(self, kwargs, response_obj, start_time, end_time):
                failures.append(kwargs["exception"])

            async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
                failures.append(kwargs["exception"])

        recorder: Final = FailureRecorder()
        logging_obj: Final = LitellmLogging(
            model="embed",
            messages=[],
            stream=False,
            call_type="embedding" if sync else "aembedding",
            start_time=time.time(),
            litellm_call_id="embed-batch-rejected",
            function_id="embed-batch-rejected",
            dynamic_failure_callbacks=[recorder] if sync else None,
            dynamic_async_failure_callbacks=None if sync else [recorder],
        )
        logging_obj.model_call_details["litellm_params"] = {"metadata": {}, "proxy_server_request": {}}
        router: Final = _embedding_router(
            provider_model="openai/text-embedding-3-small",
            api_base=_OPENAI_EMBED_BASE,
            max_embedding_batch_size=2,
            deployment_id="embed-batch-capped",
        )

        with respx.mock(assert_all_called=False) as respx_mock:
            route: Final = respx_mock.post(f"{_OPENAI_EMBED_BASE}/embeddings").mock(
                return_value=httpx.Response(200, json={"object": "list", "data": [], "model": "x", "usage": {}})
            )
            with pytest.raises(litellm.BadRequestError, match="3 inputs") as excinfo:
                await _call_embedding(
                    router,
                    sync=sync,
                    model="embed",
                    input=["one", "two", "three"],
                    litellm_logging_obj=logging_obj,
                )
            assert route.call_count == 0
        assert failures == [excinfo.value]

    @pytest.mark.parametrize("sync", [False, True], ids=["async", "sync"])
    @pytest.mark.asyncio
    async def test_no_limit_configured_forwards_any_batch(self, sync: bool) -> None:
        router: Final = _embedding_router(
            provider_model="hosted_vllm/bge-m3",
            api_base=_VLLM_EMBED_BASE,
            max_embedding_batch_size=None,
            deployment_id="vllm-embed-uncapped",
        )

        with respx.mock(assert_all_called=False) as respx_mock:
            route: Final = respx_mock.post(f"{_VLLM_EMBED_BASE}/embeddings").mock(
                return_value=_vllm_embedding_response(100)
            )
            await _call_embedding(router, sync=sync, model="embed", input=["a"] * 100)
            assert len(json.loads(route.calls.last.request.content)["input"]) == 100

    @pytest.mark.parametrize("sync", [False, True], ids=["async", "sync"])
    @pytest.mark.asyncio
    async def test_oversized_batch_goes_to_the_group_deployment_that_admits_it(self, sync: bool) -> None:
        router: Final = Router(
            model_list=[
                {
                    "model_name": "embed",
                    "litellm_params": {
                        "model": "hosted_vllm/bge-m3",
                        "api_key": "sk-fake",
                        "api_base": _VLLM_EMBED_BASE,
                        "weight": 1000,
                    },
                    "model_info": {"id": "vllm-embed-capped", "max_embedding_batch_size": 2},
                },
                {
                    "model_name": "embed",
                    "litellm_params": {
                        "model": "hosted_vllm/bge-m3",
                        "api_key": "sk-fake",
                        "api_base": _UNCAPPED_EMBED_BASE,
                        "weight": 1,
                    },
                    "model_info": {"id": "vllm-embed-uncapped"},
                },
            ],
            num_retries=0,
        )

        with respx.mock(assert_all_called=False) as respx_mock:
            capped_route: Final = respx_mock.post(f"{_VLLM_EMBED_BASE}/embeddings").mock(
                return_value=_vllm_embedding_response(3)
            )
            uncapped_route: Final = respx_mock.post(f"{_UNCAPPED_EMBED_BASE}/embeddings").mock(
                return_value=_vllm_embedding_response(3)
            )
            for _ in range(5):
                await _call_embedding(router, sync=sync, model="embed", input=["a", "b", "c"])
            assert (capped_route.call_count, uncapped_route.call_count) == (0, 5)

    @pytest.mark.asyncio
    async def test_non_embedding_call_with_input_ignores_embedding_batch_limit(self) -> None:
        router: Final = Router(
            model_list=[
                {
                    "model_name": "embed",
                    "litellm_params": {
                        "model": "openai/text-embedding-3-small",
                        "api_key": "sk-fake",
                        "api_base": _OPENAI_EMBED_BASE,
                        "order": 1,
                    },
                    "model_info": {"id": "capped", "max_embedding_batch_size": 2},
                },
                {
                    "model_name": "embed",
                    "litellm_params": {
                        "model": "openai/text-embedding-3-small",
                        "api_key": "sk-fake",
                        "api_base": _UNCAPPED_EMBED_BASE,
                        "order": 2,
                    },
                    "model_info": {"id": "uncapped"},
                },
            ],
            num_retries=0,
        )
        responses_input: Final = [{"role": "user", "content": str(turn)} for turn in range(3)]

        async_pick: Final = await router.async_get_available_deployment(
            model="embed", request_kwargs={}, input=responses_input
        )
        sync_pick: Final = router.get_available_deployment(model="embed", request_kwargs={}, input=responses_input)

        assert (async_pick["model_info"]["id"], sync_pick["model_info"]["id"]) == ("capped", "capped")

    @pytest.mark.parametrize("sync", [False, True], ids=["async", "sync"])
    @pytest.mark.asyncio
    async def test_oversized_batch_waits_for_a_cooling_down_deployment_that_admits_it(self, sync: bool) -> None:
        router: Final = Router(
            model_list=[
                {
                    "model_name": "embed",
                    "litellm_params": {"model": "hosted_vllm/bge-m3", "api_key": "sk-fake", "api_base": _VLLM_EMBED_BASE},
                    "model_info": {"id": "vllm-embed-capped", "max_embedding_batch_size": 2},
                },
                {
                    "model_name": "embed",
                    "litellm_params": {
                        "model": "hosted_vllm/bge-m3",
                        "api_key": "sk-fake",
                        "api_base": _UNCAPPED_EMBED_BASE,
                    },
                    "model_info": {"id": "vllm-embed-uncapped"},
                },
            ],
            num_retries=0,
        )

        with respx.mock(assert_all_called=False) as respx_mock:
            capped_route: Final = respx_mock.post(f"{_VLLM_EMBED_BASE}/embeddings").mock(
                return_value=_vllm_embedding_response(3)
            )
            uncapped_route: Final = respx_mock.post(f"{_UNCAPPED_EMBED_BASE}/embeddings").mock(
                return_value=httpx.Response(429, json={"error": {"message": "rate limited"}})
            )
            with pytest.raises(litellm.RateLimitError):
                await _call_embedding(router, sync=sync, model="embed", input=["a", "b", "c"])
            with pytest.raises(RouterRateLimitError):
                await _call_embedding(router, sync=sync, model="embed", input=["a", "b", "c"])
            assert (capped_route.call_count, uncapped_route.call_count) == (0, 1)

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
