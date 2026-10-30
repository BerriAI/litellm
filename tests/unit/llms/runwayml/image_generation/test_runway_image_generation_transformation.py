from typing import Final
from unittest.mock import MagicMock

import httpx
import pytest

from litellm.llms.runwayml.image_generation.transformation import RunwayMLImageGenerationConfig
from litellm.types.utils import ImageResponse

TASK_URL = "https://api.dev.runwayml.com/v1/tasks/task_1"


@pytest.mark.respx()
def test_output_dict_with_non_string_values_still_yields_the_image(respx_mock):
    respx_mock.get(TASK_URL).respond(
        json={
            "id": "task_1",
            "status": "SUCCEEDED",
            "output": [{"url": "https://cdn.example/image.png", "b64_json": None, "width": 1024}],
        },
        status_code=200,
    )
    created: Final = httpx.Response(
        200,
        json={"id": "task_1", "status": "PENDING"},
        request=httpx.Request(
            "POST", "https://api.dev.runwayml.com/v1/text_to_image", headers={"Authorization": "Bearer k"}
        ),
    )

    image_response: Final = RunwayMLImageGenerationConfig().transform_image_generation_response(
        model="gen4_image",
        raw_response=created,
        model_response=ImageResponse(),
        logging_obj=MagicMock(),
        request_data={},
        optional_params={},
        litellm_params={},
        encoding=None,
    )

    assert [image.url for image in image_response.data] == ["https://cdn.example/image.png"]


def test_non_object_task_body_raises_the_base_error_class():
    created: Final = httpx.Response(
        200,
        json=["unexpected"],
        request=httpx.Request(
            "POST", "https://api.dev.runwayml.com/v1/text_to_image", headers={"Authorization": "Bearer k"}
        ),
    )

    with pytest.raises(AttributeError):
        RunwayMLImageGenerationConfig().transform_image_generation_response(
            model="gen4_image",
            raw_response=created,
            model_response=ImageResponse(),
            logging_obj=MagicMock(),
            request_data={},
            optional_params={},
            litellm_params={},
            encoding=None,
        )


@pytest.mark.asyncio
async def test_async_non_object_task_body_raises_the_base_error_class():
    created: Final = httpx.Response(
        200,
        json=["unexpected"],
        request=httpx.Request(
            "POST", "https://api.dev.runwayml.com/v1/text_to_image", headers={"Authorization": "Bearer k"}
        ),
    )

    with pytest.raises(AttributeError):
        await RunwayMLImageGenerationConfig().async_transform_image_generation_response(
            model="gen4_image",
            raw_response=created,
            model_response=ImageResponse(),
            logging_obj=MagicMock(),
            request_data={},
            optional_params={},
            litellm_params={},
            encoding=None,
        )
