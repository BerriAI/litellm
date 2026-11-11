import base64
import json
from typing import Final, cast

import httpx
import pytest
from respx import MockRouter
from typing_extensions import ReadOnly, TypedDict

from litellm import embedding
from litellm.utils import get_optional_params_embeddings


class _Kwargs(TypedDict, total=False):
    model: ReadOnly[str]
    api_key: ReadOnly[str]
    api_base: ReadOnly[str]
    api_version: ReadOnly[str]
    aws_access_key_id: ReadOnly[str]
    aws_secret_access_key: ReadOnly[str]
    aws_region_name: ReadOnly[str]


class _Case(TypedDict):
    id: ReadOnly[str]
    provider: ReadOnly[str]
    kwargs: ReadOnly[_Kwargs]
    url: ReadOnly[str]


_AZURE_BASE: Final = "https://offline-embed.openai.azure.com"
_AZURE_URL: Final = f"{_AZURE_BASE}/openai/deployments/text-embedding-ada-002/embeddings?api-version=2024-02-15-preview"
_TITAN_URL: Final = "https://bedrock-runtime.us-west-2.amazonaws.com/model/amazon.titan-embed-image-v1/invoke"

_CASES: Final[tuple[_Case, ...]] = (
    {
        "id": "azure_text_embedding",
        "provider": "azure",
        "kwargs": {
            "model": "azure/text-embedding-ada-002",
            "api_key": "azure-offline-key",
            "api_base": _AZURE_BASE,
            "api_version": "2024-02-15-preview",
        },
        "url": _AZURE_URL,
    },
    {
        "id": "bedrock_titan_image",
        "provider": "bedrock",
        "kwargs": cast(
            _Kwargs,
            {
                "model": "bedrock/amazon.titan-embed-image-v1",
                "aws_access_key_id": "AKIAFAKE",
                "aws_secret_access_key": "fakesecret",
                "aws_region_name": "us-west-2",
            },
        ),
        "url": _TITAN_URL,
    },
)


_MAX_RETRIES_KWARGS: Final[tuple[_Kwargs, ...]] = (
    *(case["kwargs"] for case in _CASES),
    {"model": "volcengine/doubao-embedding-text-240715"},
    {"model": "voyage/voyage-3-lite"},
)


def _canned_response(case: _Case) -> httpx.Response:
    vector: Final = [0.11, 0.22, 0.33]
    if case["provider"] == "azure":
        return httpx.Response(
            200,
            json={
                "object": "list",
                "data": [{"object": "embedding", "index": 0, "embedding": vector}],
                "model": "text-embedding-ada-002",
                "usage": {"prompt_tokens": 2, "total_tokens": 2},
            },
        )
    return httpx.Response(200, json={"embedding": vector, "inputTextTokenCount": 4})


@pytest.fixture(autouse=True)
def _httpx_only_transport(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DISABLE_AIOHTTP_TRANSPORT", "True")


@pytest.mark.parametrize("kwargs", _MAX_RETRIES_KWARGS, ids=lambda k: k["model"])
def test_embedding_optional_params_max_retries(kwargs: _Kwargs) -> None:
    optional_params: Final = get_optional_params_embeddings(**dict(kwargs), max_retries=20)
    assert optional_params["max_retries"] == 20


@pytest.mark.parametrize("case", _CASES, ids=lambda c: c["id"])
def test_image_embedding(case: _Case, respx_mock: MockRouter) -> None:
    png_b64: Final = base64.b64encode(b"\x89PNG\r\n\x1a\nFAKEPIXELS").decode()
    data_url: Final = f"data:image/png;base64,{png_b64}"
    route: Final = respx_mock.post(case["url"]).mock(return_value=_canned_response(case))
    response: Final = embedding(**dict(case["kwargs"]), input=[data_url])
    body: Final = json.loads(route.calls.last.request.content)
    if case["provider"] == "azure":
        assert body["input"] == [data_url]
    else:
        assert body["inputImage"] == png_b64
    assert response.data[0]["embedding"] == [0.11, 0.22, 0.33]
