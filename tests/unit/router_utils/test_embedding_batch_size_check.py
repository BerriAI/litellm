from typing import Final

import pytest

import litellm
from litellm.router_utils.embedding_batch_size_check import (
    count_embedding_inputs,
    raise_if_embedding_batch_too_large,
)

_MODEL: Final = "embed-group"
_PROVIDER: Final = "openai"


def _model_info(max_embedding_batch_size: int | None) -> dict[str, object]:
    if max_embedding_batch_size is None:
        return {}
    return {"max_embedding_batch_size": max_embedding_batch_size}


class TestCountEmbeddingInputs:
    def test_string_input_counts_as_one(self) -> None:
        assert count_embedding_inputs("hello") == 1

    def test_list_of_strings_counts_items(self) -> None:
        assert count_embedding_inputs(["a", "b"]) == 2

    def test_pretokenized_single_input_counts_as_one(self) -> None:
        assert count_embedding_inputs([1, 2, 3, 4, 5]) == 1

    def test_pretokenized_batch_counts_rows(self) -> None:
        assert count_embedding_inputs([[1, 2, 3], [4, 5, 6]]) == 2

    def test_batch_count_is_independent_of_string_length(self) -> None:
        long_text: Final = "word " * 500
        assert count_embedding_inputs([long_text, long_text]) == 2


class TestRaiseIfEmbeddingBatchTooLarge:
    def test_accepts_string_at_limit_one(self) -> None:
        raise_if_embedding_batch_too_large(
            input="hello",
            model_info=_model_info(1),
            model=_MODEL,
            llm_provider=_PROVIDER,
        )

    def test_accepts_list_below_limit(self) -> None:
        raise_if_embedding_batch_too_large(
            input=["a", "b"],
            model_info=_model_info(3),
            model=_MODEL,
            llm_provider=_PROVIDER,
        )

    def test_accepts_list_at_limit(self) -> None:
        raise_if_embedding_batch_too_large(
            input=["a", "b", "c"],
            model_info=_model_info(3),
            model=_MODEL,
            llm_provider=_PROVIDER,
        )

    def test_rejects_list_above_limit(self) -> None:
        with pytest.raises(litellm.BadRequestError) as excinfo:
            raise_if_embedding_batch_too_large(
                input=["a", "b", "c", "d"],
                model_info=_model_info(3),
                model=_MODEL,
                llm_provider=_PROVIDER,
            )
        assert excinfo.value.status_code == 400
        assert "4 inputs" in excinfo.value.message
        assert "max_embedding_batch_size=3" in excinfo.value.message

    def test_accepts_pretokenized_single_at_limit_one(self) -> None:
        raise_if_embedding_batch_too_large(
            input=[1, 2, 3, 4, 5],
            model_info=_model_info(1),
            model=_MODEL,
            llm_provider=_PROVIDER,
        )

    def test_accepts_pretokenized_batch_at_limit(self) -> None:
        raise_if_embedding_batch_too_large(
            input=[[1, 2, 3], [4, 5, 6]],
            model_info=_model_info(2),
            model=_MODEL,
            llm_provider=_PROVIDER,
        )

    def test_rejects_pretokenized_batch_above_limit(self) -> None:
        with pytest.raises(litellm.BadRequestError):
            raise_if_embedding_batch_too_large(
                input=[[1], [2], [3]],
                model_info=_model_info(2),
                model=_MODEL,
                llm_provider=_PROVIDER,
            )

    def test_no_limit_configured_is_no_op(self) -> None:
        raise_if_embedding_batch_too_large(
            input=["a"] * 100,
            model_info=_model_info(None),
            model=_MODEL,
            llm_provider=_PROVIDER,
        )


@pytest.mark.asyncio
async def test_router_embedding_rejects_oversized_batch_before_provider_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import httpx
    import respx

    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    from litellm import Router

    router: Final = Router(
        model_list=[
            {
                "model_name": "embed",
                "litellm_params": {
                    "model": "openai/text-embedding-3-small",
                    "api_key": "sk-fake",
                    "api_base": "https://batch-limit-embed.local/v1",
                },
                "model_info": {"id": "embed-batch-capped", "max_embedding_batch_size": 2},
            }
        ],
        num_retries=0,
    )

    with respx.mock(assert_all_called=False) as respx_mock:
        route: Final = respx_mock.post("https://batch-limit-embed.local/v1/embeddings").mock(
            return_value=httpx.Response(200, json={"object": "list", "data": [], "model": "x", "usage": {}})
        )
        with pytest.raises(litellm.BadRequestError, match="max_embedding_batch_size=2"):
            await router.aembedding(model="embed", input=["one", "two", "three"])
        assert route.call_count == 0
