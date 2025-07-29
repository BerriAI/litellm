import asyncio
import os
import sys
from pathlib import Path
from unittest import mock

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath("../../.."))
import litellm
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
    """Patch the underlying image generation call used by the Router."""
    mock_obj = mock.AsyncMock(return_value=example_image_generation_result)
    mock_obj.__name__ = "aimage_generation"
    return mock.patch(
        "litellm.aimage_generation",
        new_callable=lambda: mock_obj,
    )


def mock_patch_aimage_edit():
    """Patch the underlying image edit call used by the Router."""
    mock_obj = mock.AsyncMock(return_value=example_image_edit_result)
    mock_obj.__name__ = "aimage_edit"
    return mock.patch(
        "litellm.aimage_edit",
        new_callable=lambda: mock_obj,
    )


@pytest.fixture(scope="function")
def client_no_auth():
    from litellm.proxy.proxy_server import cleanup_router_config_variables

    cleanup_router_config_variables()
    repo_root = Path(__file__).resolve().parents[4]
    config_fp = (
        repo_root
        / "tests"
        / "proxy_unit_tests"
        / "test_configs"
        / "test_config_no_auth.yaml"
    )
    config_fp = str(config_fp)

    # Create mock objects with __name__ attribute
    mock_generation = mock.AsyncMock(return_value=example_image_generation_result)
    mock_generation.__name__ = "aimage_generation"

    mock_edit = mock.AsyncMock(return_value=example_image_edit_result)
    mock_edit.__name__ = "aimage_edit"

    with mock.patch(
        "litellm.aimage_generation",
        new_callable=lambda: mock_generation,
    ) as patched_generation, mock.patch(
        "litellm.aimage_edit",
        new_callable=lambda: mock_edit,
    ) as patched_edit:
        asyncio.run(initialize(config=config_fp, debug=True))
        client = TestClient(app)
        yield client, patched_generation, patched_edit


def test_azure_image_generation_route(client_no_auth):
    client, mock_aimage_generation, _ = client_no_auth
    test_data = {"prompt": "A cute baby sea otter", "n": 1, "size": "1024x1024"}
    response = client.post(
        "/openai/deployments/dall-e-3/images/generations", json=test_data
    )

    mock_aimage_generation.assert_called_once()
    call_kwargs = mock_aimage_generation.call_args.kwargs
    assert "dall-e-3" in call_kwargs["model"]
    assert call_kwargs["prompt"] == "A cute baby sea otter"
    assert call_kwargs["n"] == 1
    assert call_kwargs["size"] == "1024x1024"
    assert response.status_code == 200
    assert response.json()["data"]


def test_azure_image_edit_route(client_no_auth):
    litellm._turn_on_debug()
    client, _, mock_aimage_edit = client_no_auth
    image_path = os.path.join(
        os.path.dirname(__file__),
        "../../../image_gen_tests/test_image.png",
    )
    with open(image_path, "rb") as f:
        files = {"image": ("test_image.png", f, "image/png")}
        data = {"prompt": "A cute baby sea otter"}
        response = client.post(
            "/openai/deployments/dall-e-3/images/edits", files=files, data=data
        )

    mock_aimage_edit.assert_called_once()
    called_kwargs = mock_aimage_edit.call_args.kwargs
    assert "dall-e-3" in called_kwargs["model"]
    assert called_kwargs["prompt"] == "A cute baby sea otter"
    assert response.status_code == 200
    assert response.json()["data"]
