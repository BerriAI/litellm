from typing import Final

import pytest

import litellm
from litellm import Router
from litellm.types.utils import OpenAIBatchListResponse

TOKEN_BY_KEY: Final = {"key-with-more-pages": "1", "key-on-last-page": None}


def _deployment(api_key: str) -> dict:
    return {
        "model_name": "mistral-ocr",
        "litellm_params": {"model": "mistral/mistral-ocr-latest", "api_key": api_key},
        "model_info": {"id": api_key},
    }


async def _fake_alist_batches(**kwargs: object) -> OpenAIBatchListResponse:
    if kwargs.get("custom_llm_provider") != "mistral":
        raise ValueError("a mistral key sent down the default openai list path")
    token: Final = TOKEN_BY_KEY[str(kwargs["api_key"])]
    return OpenAIBatchListResponse(
        data=(), first_id=None, last_id=None, has_more=token is not None, next_page_token=token
    )


@pytest.mark.asyncio
async def test_alist_batches_keeps_the_page_token_a_deployment_returned(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "alist_batches", _fake_alist_batches)
    router: Final = Router(model_list=[_deployment("key-with-more-pages"), _deployment("key-on-last-page")])

    result: Final = await router.alist_batches(model="mistral-ocr", limit=3)

    assert result["has_more"] is True
    assert result["next_page_token"] == "1"


@pytest.mark.asyncio
async def test_alist_batches_lists_with_each_deployments_own_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "alist_batches", _fake_alist_batches)
    router: Final = Router(model_list=[_deployment("key-with-more-pages")])

    result: Final = await router.alist_batches(model="mistral-ocr", limit=3)

    assert result["has_more"] is True


async def _fake_openai_default_alist_batches(**kwargs: object) -> OpenAIBatchListResponse:
    provider: Final = kwargs.get("custom_llm_provider", "openai")
    if provider != "openai":
        raise ValueError(f"LiteLLM doesn't support {provider} for 'list_batch'")
    return OpenAIBatchListResponse(data=(), first_id="batch_a", last_id="batch_a", has_more=False)


@pytest.mark.asyncio
async def test_alist_batches_keeps_the_openai_default_for_providers_that_cannot_list(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(litellm, "alist_batches", _fake_openai_default_alist_batches)
    router: Final = Router(
        model_list=[
            {
                "model_name": "instruct",
                "litellm_params": {"model": "text-completion-openai/gpt-3.5-turbo-instruct", "api_key": "sk-test"},
            }
        ]
    )

    result: Final = await router.alist_batches(model="instruct", limit=3)

    assert result["first_id"] == "batch_a"
    assert "next_page_token" not in result
