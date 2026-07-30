"""Live e2e: POST /v1/images/edits returns an edited image.

Registers an OpenAI image model, then sends a small PNG plus an edit prompt as a
multipart request to /v1/images/edits and asserts the response carries an image
(url or base64). /images/edits is a distinct native route from
/images/generations: it is multipart file upload with the image sent as the
`image` part, not a JSON body. The fixture image is a small generated 64x64 PNG,
so no external asset is needed.
"""

from __future__ import annotations

import base64

import pytest

from e2e_config import unique_marker
from e2e_http import unwrap
from endpoints_client import EndpointsClient
from lifecycle import ResourceManager
from models import LiteLLMParamsBody

pytestmark = pytest.mark.e2e

_TEST_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAIAAAAlC+aJAAAAS0lEQVR42u3PMQ0AAAwDoPo3"
    "3UrYvQQckD4XAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEB"
    "AYHLAMpT0sIcNbcEAAAAAElFTkSuQmCC"
)


class TestImageEdit:
    @pytest.mark.covers("llm.images_edits.openai.basic.nonstream.works")
    def test_image_edit_returns_image(
        self, endpoints_client: EndpointsClient, resources: ResourceManager
    ) -> None:
        model = f"e2e-image-edit-{unique_marker()}"
        model_id = endpoints_client.create_model(
            model,
            LiteLLMParamsBody(model="openai/gpt-image-1", api_key="os.environ/OPENAI_API_KEY"),
        )
        resources.defer(lambda: endpoints_client.delete_model(model_id))
        key = resources.key()

        edited = unwrap(
            endpoints_client.image_edit(
                key, model, "Add a small red circle in the center", _TEST_PNG
            )
        )
        assert edited.data, f"/images/edits returned no data: {edited}"
        first = edited.data[0]
        assert first.b64_json or first.url, (
            f"edited image has neither b64_json nor url: {first}"
        )
