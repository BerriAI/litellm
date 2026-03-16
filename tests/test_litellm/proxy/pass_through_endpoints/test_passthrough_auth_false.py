"""
Test that pass-through endpoints with auth: false work without Authorization header.

Regression test for the bug where auth: false pass-through routes still required
an Authorization header and failed with 'NoneType' object has no attribute 'split'
when no header was provided.
"""
import asyncio
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath("../../.."))


def _initialize_proxy_with_config(config: dict, tmp_path) -> TestClient:
    """Initialize proxy with config and return TestClient."""
    from litellm.proxy.proxy_server import (
        app,
        cleanup_router_config_variables,
        initialize,
    )

    cleanup_router_config_variables()
    config_fp = tmp_path / "proxy_config.yaml"
    config_fp.write_text(yaml.safe_dump(config))
    asyncio.run(initialize(config=str(config_fp), debug=True))
    return TestClient(app, raise_server_exceptions=False)


def test_passthrough_auth_false_without_authorization_header(tmp_path):
    """
    With auth: false, a pass-through request without Authorization header should succeed.

    Previously this returned 401 with 'NoneType' object has no attribute 'split'
    when JWT or OAuth2 auth was enabled.
    """
    config = {
        "model_list": [
            {
                "model_name": "gpt-4o-mini",
                "litellm_params": {
                    "model": "openai/gpt-4o-mini",
                    "api_key": "sk-fake",
                    "api_base": "https://httpbin.org",
                },
            }
        ],
        "general_settings": {
            "master_key": "sk-1234",
            "enable_jwt_auth": True,  # Triggers the bug path when auth runs
            "pass_through_endpoints": [
                {
                    "path": "/v1/cuopt/request",
                    "target": "https://httpbin.org/post",
                    "auth": False,
                    "headers": {"content-type": "application/json"},
                    "forward_headers": True,
                }
            ],
        },
    }

    with patch(
        "litellm.proxy.pass_through_endpoints.pass_through_endpoints.get_async_httpx_client"
    ) as mock_get_client:
        # Use MagicMock to avoid httpx.Response raise_for_status requiring request
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "application/json"}
        mock_response.content = b'{"json": {"test": "data"}}'
        mock_response.text = '{"json": {"test": "data"}}'
        mock_response.aread = AsyncMock(return_value=mock_response.content)
        mock_response.raise_for_status = MagicMock()  # no-op for 200
        mock_response.json = MagicMock(return_value={"json": {"test": "data"}})
        mock_async_client = MagicMock()
        mock_async_client.request = AsyncMock(return_value=mock_response)
        mock_client_obj = MagicMock()
        mock_client_obj.client = mock_async_client
        mock_get_client.return_value = mock_client_obj

        client = _initialize_proxy_with_config(config=config, tmp_path=tmp_path)

        response = client.post(
            "/v1/cuopt/request",
            json={"test": "data"},
            headers={"Content-Type": "application/json"},
        )

        assert response.status_code != 401, (
            f"Expected success but got 401: {response.text}"
        )
        assert "split" not in response.text, (
            f"Should not contain NoneType.split error: {response.text}"
        )
        assert response.status_code == 200
