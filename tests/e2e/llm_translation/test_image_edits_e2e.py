"""Live e2e: POST /v1/images/edits returns an edited image.

Registers an OpenAI image model, then sends a small PNG plus an edit prompt
through the real OpenAI SDK (LIT-4577) to /v1/images/edits and asserts the
response carries an image (url or base64). /images/edits is a distinct native
route from /images/generations: it is multipart file upload with the image sent
as the `image` part, not a JSON body. The fixture image is a small generated
64x64 PNG, so no external asset is needed.
"""

from __future__ import annotations

import base64

import openai
import pytest
from e2e_config import SLOW_PROVIDER_TIMEOUT_SECONDS, unique_marker
from lifecycle import ResourceManager
from models import LiteLLMParamsBody
from proxy_client import ProxyClient
from sdk_clients import SdkClients

pytestmark = pytest.mark.e2e

_TEST_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAIAAAAlC+aJAAAAS0lEQVR42u3PMQ0AAAwDoPo3"
    "3UrYvQQckD4XAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEB"
    "AYHLAMpT0sIcNbcEAAAAAElFTkSuQmCC"
)


def _register_image_model(proxy: ProxyClient, resources: ResourceManager) -> tuple[str, str]:
    model = f"e2e-image-edit-{unique_marker()}"
    model_id = proxy.create_model(
        model,
        LiteLLMParamsBody(model="openai/gpt-image-1", api_key="os.environ/OPENAI_API_KEY"),
    )
    resources.defer(lambda: proxy.delete_model(model_id))
    return model, resources.key()


def _image_part(content: bytes) -> tuple[str, bytes, str]:
    return ("image.png", content, "image/png")


def _assert_client_error(error: openai.APIStatusError, context: str) -> None:
    assert 400 <= error.status_code < 500, f"{context}: expected 4xx, got {error.status_code}: {error.message}"


class TestImageEdit:
    @pytest.mark.covers("llm.images_edits.openai.basic.nonstream.works")
    def test_image_edit_returns_image(self, proxy: ProxyClient, resources: ResourceManager, sdk: SdkClients) -> None:
        model, key = _register_image_model(proxy, resources)
        client = sdk.openai(key)

        edited = client.images.edit(
            model=model,
            image=_image_part(_TEST_PNG),
            prompt="Add a small red circle in the center",
            timeout=SLOW_PROVIDER_TIMEOUT_SECONDS,
        )
        assert edited.data, f"/images/edits returned no data: {edited!r}"
        first = edited.data[0]
        assert first.b64_json or first.url, f"edited image has neither b64_json nor url: {first!r}"

    @pytest.mark.covers("llm.images_edits.openai.input_validation.nonstream.works")
    def test_empty_prompt_returns_error(self, proxy: ProxyClient, resources: ResourceManager, sdk: SdkClients) -> None:
        model, key = _register_image_model(proxy, resources)
        client = sdk.openai(key)

        with pytest.raises(openai.APIStatusError) as raised:
            client.images.edit(model=model, image=_image_part(_TEST_PNG), prompt="")
        _assert_client_error(raised.value, "empty image-edit prompt")

    @pytest.mark.covers("llm.images_edits.openai.input_validation.nonstream.works")
    def test_empty_image_returns_error(self, proxy: ProxyClient, resources: ResourceManager, sdk: SdkClients) -> None:
        model, key = _register_image_model(proxy, resources)
        client = sdk.openai(key)

        with pytest.raises(openai.APIStatusError) as raised:
            client.images.edit(model=model, image=_image_part(b""), prompt="add a red circle")
        _assert_client_error(raised.value, "empty image-edit file")
