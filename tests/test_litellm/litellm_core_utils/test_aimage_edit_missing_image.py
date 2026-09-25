"""Regression tests for https://github.com/BerriAI/litellm/issues/42185

Lives under litellm_core_utils so the core-utils CI shard collects it (misc currently lists missing paths and collects zero tests).

POST /v1/images/edits without a multipart `image` used to TypeError inside
aimage_edit (required positional arg) and surface as HTTP 500. A missing client
field must be BadRequestError 400.
"""

from collections.abc import Mapping

import pytest

import litellm
from litellm.images.main import aimage_edit

_MISSING_IMAGE_MESSAGE = "Missing required parameter: 'image'."
_MISSING_IMAGE_BODY = {
    "message": _MISSING_IMAGE_MESSAGE,
    "type": "invalid_request_error",
    "param": "image",
    "code": "missing_required_parameter",
}


def _assert_missing_image_400(
    err: litellm.BadRequestError,
    *,
    model: str,
    llm_provider: str,
) -> None:
    message = str(err)
    assert err.status_code == 400
    assert _MISSING_IMAGE_MESSAGE in message
    assert "positional argument" not in message
    assert "TypeError" not in message
    assert err.model == model
    assert err.llm_provider == llm_provider
    assert err.param == "image"
    body = err.body
    assert isinstance(body, Mapping)
    for key, value in _MISSING_IMAGE_BODY.items():
        assert body[key] == value


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

    _assert_missing_image_400(
        exc_info.value,
        model="openai/gpt-image-1",
        llm_provider="openai",
    )


@pytest.mark.asyncio
async def test_aimage_edit_direct_import_missing_image_raises_bad_request() -> None:
    with pytest.raises(litellm.BadRequestError) as exc_info:
        await aimage_edit(model="gpt-image-1", prompt="add a hat")

    _assert_missing_image_400(
        exc_info.value,
        model="gpt-image-1",
        llm_provider="openai",
    )


@pytest.mark.asyncio
async def test_aimage_edit_missing_image_keeps_custom_llm_provider() -> None:
    with pytest.raises(litellm.BadRequestError) as exc_info:
        await litellm.aimage_edit(
            model="gpt-image-1",
            prompt="add a hat",
            custom_llm_provider="azure",
        )

    _assert_missing_image_400(
        exc_info.value,
        model="gpt-image-1",
        llm_provider="azure",
    )


@pytest.mark.asyncio
async def test_aimage_edit_missing_image_defaults_model_to_unknown() -> None:
    with pytest.raises(litellm.BadRequestError) as exc_info:
        await aimage_edit(prompt="add a hat")

    _assert_missing_image_400(
        exc_info.value,
        model="unknown",
        llm_provider="openai",
    )
