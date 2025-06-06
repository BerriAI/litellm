import asyncio
import os
import sys
from unittest import mock

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath("../../.."))

from litellm.proxy.proxy_server import app, initialize

example_image_generation_result = {
    "created": 1589478378,
    "data": [{"url": "https://example.com/image.png"}],
}

example_image_edit_result = {
    "created": 1589478400,
    "data": [
        {
            "b64_json": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
        }
    ],
}


def mock_patch_aimage_generation():
    return mock.patch(
        "litellm.proxy.proxy_server.llm_router.aimage_generation",
        return_value=example_image_generation_result,
    )


def mock_patch_aimage_edit():
    return mock.patch(
        "litellm.proxy.proxy_server.llm_router.aimage_edit",
        return_value=example_image_edit_result,
    )


@pytest.fixture(scope="function")
def client_no_auth():
    from litellm.proxy.proxy_server import cleanup_router_config_variables

    cleanup_router_config_variables()
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_fp = os.path.join(base_dir, "test_configs", "test_config_no_auth.yaml")
    asyncio.run(initialize(config=config_fp, debug=True))
    return TestClient(app)


@mock_patch_aimage_generation()
def test_azure_image_generation_route(mock_aimage_generation, client_no_auth):
    test_data = {"prompt": "A cute baby sea otter", "n": 1, "size": "1024x1024"}
    response = client_no_auth.post(
        "/openai/deployments/dall-e-3/images/generations", json=test_data
    )

    mock_aimage_generation.assert_called_once_with(
        model="dall-e-3",
        prompt="A cute baby sea otter",
        n=1,
        size="1024x1024",
        metadata=mock.ANY,
        proxy_server_request=mock.ANY,
    )
    assert response.status_code == 200
    assert response.json()["data"]


@mock_patch_aimage_edit()
def test_azure_image_edit_route(mock_aimage_edit, client_no_auth):
    image_path = os.path.join(
        os.path.dirname(__file__),
        "../../../image_gen_tests/test_image.png",
    )
    with open(image_path, "rb") as f:
        files = {"image": ("test_image.png", f, "image/png")}
        data = {"prompt": "A cute baby sea otter"}
        response = client_no_auth.post(
            "/openai/deployments/dall-e-3/images/edits", files=files, data=data
        )

    mock_aimage_edit.assert_called_once()
    called_kwargs = mock_aimage_edit.call_args.kwargs
    assert called_kwargs["model"] == "dall-e-3"
    assert called_kwargs["prompt"] == "A cute baby sea otter"
    assert response.status_code == 200
    assert response.json()["data"]
