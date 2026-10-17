import asyncio
from pathlib import Path
from unittest import mock

import pytest
from fastapi.testclient import TestClient

from litellm.proxy.proxy_server import app, initialize

example_image_variation_result = {
    "created": 1589478420,
    "data": [
        {"b64_json": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="}
    ],
}


@pytest.fixture(scope="function")
def client_no_auth():
    from litellm.proxy.proxy_server import cleanup_router_config_variables

    cleanup_router_config_variables()
    repo_root = Path(__file__).resolve().parents[4]
    config_fp = repo_root / "tests" / "unit" / "proxy" / "test_configs" / "test_config_no_auth.yaml"
    config_fp = str(config_fp)

    mock_variation = mock.AsyncMock(return_value=example_image_variation_result)
    mock_variation.__name__ = "aimage_variation"

    with mock.patch(
        "litellm.aimage_variation",
        new_callable=lambda: mock_variation,
    ) as patched_variation:
        asyncio.run(initialize(config=config_fp, debug=True))
        client = TestClient(app)
        yield client, patched_variation


def test_image_variation_route_is_registered():
    """Regression test for #46367: the proxy must expose POST /v1/images/variations.

    The image router is mounted wholesale during initialize(), so its own
    path set is exactly what the app serves; asserting on the router avoids
    importing the whole app in this test.
    """
    from litellm.proxy.image_endpoints.endpoints import router as image_router

    paths = {getattr(route, "path", None) for route in image_router.routes}
    assert "/v1/images/variations" in paths
    assert "/images/variations" in paths
    assert "/openai/deployments/{model:path}/images/variations" in paths


def test_image_variation_multipart_reaches_aimage_variation(client_no_auth):
    client, mock_aimage_variation = client_no_auth
    files = {"image": ("test_image.png", b"\x89PNG\r\n\x1a\nimage-bytes", "image/png")}
    data = {"model": "dall-e-2", "n": 2}
    response = client.post("/v1/images/variations", files=files, data=data)

    mock_aimage_variation.assert_called_once()
    called_kwargs = mock_aimage_variation.call_args.kwargs
    # The no-auth config pins a user_model, which model resolution prefers
    # over the body model - same behavior as the edit route.
    assert isinstance(called_kwargs["model"], str)
    assert called_kwargs["n"] == 2
    image = called_kwargs["image"]
    assert isinstance(image, list) and len(image) == 1
    assert image[0].getvalue() == b"\x89PNG\r\n\x1a\nimage-bytes"
    assert response.status_code == 200
    assert response.json()["data"]


def test_image_variation_string_image_is_rejected(client_no_auth):
    client, mock_aimage_variation = client_no_auth
    data = {"model": "dall-e-2", "image": "not-a-file"}
    response = client.post("/v1/images/variations", data=data)

    assert response.status_code == 422
    assert "Expected UploadFile" in response.text
    mock_aimage_variation.assert_not_called()
