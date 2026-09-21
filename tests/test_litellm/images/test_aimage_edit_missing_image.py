"""Regression tests for https://github.com/BerriAI/litellm/issues/42185

POST /v1/images/edits without a multipart `image` used to TypeError inside
aimage_edit (required positional arg) and surface as HTTP 500. A missing client
field must be BadRequestError 400.
"""

from collections.abc import Mapping

import pytest

import litellm
from litellm.images.main import aimage_edit


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kwargs",
    [
        {"model": "openai/gpt-image-1", "prompt": "add a hat"},
        {"model": "openai/gpt-image-1", "prompt": "add a hat", "image": None},
        {"model": "openai/gpt-image-1", "prompt": "add a hat", "image": []},
    ],
    ids=["omitted", "none", "empty_list"],
)
async def test_aimage_edit_missing_image_raises_bad_request(kwargs: Mapping[str, object]) -> None:
    with pytest.raises(litellm.BadRequestError) as exc_info:
        await litellm.aimage_edit(**kwargs)

    err = exc_info.value
    message = str(err)
    assert err.status_code == 400
    assert "Missing required parameter: 'image'" in message
    assert "positional argument" not in message
    assert "TypeError" not in message
    assert getattr(err, "param", None) == "image"
    assert err.model == "openai/gpt-image-1"
    assert err.llm_provider == "openai"


@pytest.mark.asyncio
async def test_aimage_edit_direct_import_missing_image_raises_bad_request() -> None:
    with pytest.raises(litellm.BadRequestError) as exc_info:
        await aimage_edit(model="gpt-image-1", prompt="add a hat")

    err = exc_info.value
    assert err.status_code == 400
    assert "Missing required parameter: 'image'" in str(err)
    assert "positional argument" not in str(err)


@pytest.mark.asyncio
async def test_aimage_edit_missing_image_keeps_custom_llm_provider() -> None:
    with pytest.raises(litellm.BadRequestError) as exc_info:
        await litellm.aimage_edit(
            model="gpt-image-1",
            prompt="add a hat",
            custom_llm_provider="azure",
        )

    err = exc_info.value
    assert err.status_code == 400
    assert err.llm_provider == "azure"
    assert err.model == "gpt-image-1"
