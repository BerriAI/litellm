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
