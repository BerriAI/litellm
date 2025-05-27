import asyncio
import importlib
import json
import os
import socket
import subprocess
import sys
from unittest import mock
from unittest.mock import AsyncMock, MagicMock, mock_open, patch

import click
import httpx
import pytest
import yaml
from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system-path

import litellm
from litellm.proxy.proxy_server import app, initialize

example_embedding_result = {
    "object": "list",
    "data": [
        {
            "object": "embedding",
            "index": 0,
            "embedding": [
                -0.006929283495992422,
                -0.005336422007530928,
                -4.547132266452536e-05,
                -0.024047505110502243,
                -0.006929283495992422,
                -0.005336422007530928,
                -4.547132266452536e-05,
                -0.024047505110502243,
                -0.006929283495992422,
                -0.005336422007530928,
                -4.547132266452536e-05,
                -0.024047505110502243,
            ],
        }
    ],
    "model": "text-embedding-3-small",
    "usage": {"prompt_tokens": 5, "total_tokens": 5},
}


def mock_patch_aembedding():
    return mock.patch(
        "litellm.proxy.proxy_server.llm_router.aembedding",
        return_value=example_embedding_result,
    )


@pytest.fixture(scope="function")
def client_no_auth():
    # Assuming litellm.proxy.proxy_server is an object
    from litellm.proxy.proxy_server import cleanup_router_config_variables

    cleanup_router_config_variables()
    filepath = os.path.dirname(os.path.abspath(__file__))
    config_fp = f"{filepath}/test_configs/test_config_no_auth.yaml"
    # initialize can get run in parallel, it sets specific variables for the fast api app, sinc eit gets run in parallel different tests use the wrong variables
    asyncio.run(initialize(config=config_fp, debug=True))
    return TestClient(app)


@pytest.mark.asyncio
async def test_initialize_scheduled_jobs_credentials(monkeypatch):
    """
    Test that get_credentials is only called when store_model_in_db is True
    """
    monkeypatch.delenv("DISABLE_PRISMA_SCHEMA_UPDATE", raising=False)
    monkeypatch.delenv("STORE_MODEL_IN_DB", raising=False)
    from litellm.proxy.proxy_server import ProxyStartupEvent
    from litellm.proxy.utils import ProxyLogging

    # Mock dependencies
    mock_prisma_client = MagicMock()
    mock_proxy_logging = MagicMock(spec=ProxyLogging)
    mock_proxy_logging.slack_alerting_instance = MagicMock()
    mock_proxy_config = AsyncMock()

    with patch("litellm.proxy.proxy_server.proxy_config", mock_proxy_config), patch(
        "litellm.proxy.proxy_server.store_model_in_db", False
    ):  # set store_model_in_db to False
        # Test when store_model_in_db is False
        await ProxyStartupEvent.initialize_scheduled_background_jobs(
            general_settings={},
            prisma_client=mock_prisma_client,
            proxy_budget_rescheduler_min_time=1,
            proxy_budget_rescheduler_max_time=2,
            proxy_batch_write_at=5,
            proxy_logging_obj=mock_proxy_logging,
        )

        # Verify get_credentials was not called
        mock_proxy_config.get_credentials.assert_not_called()

    # Now test with store_model_in_db = True
    with patch("litellm.proxy.proxy_server.proxy_config", mock_proxy_config), patch(
        "litellm.proxy.proxy_server.store_model_in_db", True
    ), patch("litellm.proxy.proxy_server.get_secret_bool", return_value=True):
        await ProxyStartupEvent.initialize_scheduled_background_jobs(
            general_settings={},
            prisma_client=mock_prisma_client,
            proxy_budget_rescheduler_min_time=1,
            proxy_budget_rescheduler_max_time=2,
            proxy_batch_write_at=5,
            proxy_logging_obj=mock_proxy_logging,
        )

        # Verify get_credentials was called both directly and scheduled
        assert mock_proxy_config.get_credentials.call_count == 1  # Direct call

        # Verify a scheduled job was added for get_credentials
        mock_scheduler_calls = [
            call[0] for call in mock_proxy_config.get_credentials.mock_calls
        ]
        assert len(mock_scheduler_calls) > 0


# Mock Prisma
class MockPrisma:
    def __init__(self, database_url=None, proxy_logging_obj=None, http_client=None):
        self.database_url = database_url
        self.proxy_logging_obj = proxy_logging_obj
        self.http_client = http_client

    async def connect(self):
        pass

    async def disconnect(self):
        pass


mock_prisma = MockPrisma()


@patch(
    "litellm.proxy.proxy_server.ProxyStartupEvent._setup_prisma_client",
    return_value=mock_prisma,
)
@pytest.mark.asyncio
async def test_aaaproxy_startup_master_key(mock_prisma, monkeypatch, tmp_path):
    """
    Test that master_key is correctly loaded from either config.yaml or environment variables
    """
    import yaml
    from fastapi import FastAPI

    # Import happens here - this is when the module probably reads the config path
    from litellm.proxy.proxy_server import proxy_startup_event

    # Mock the Prisma import
    monkeypatch.setattr("litellm.proxy.proxy_server.PrismaClient", MockPrisma)

    # Create test app
    app = FastAPI()

    # Test Case 1: Master key from config.yaml
    test_master_key = "sk-12345"
    test_config = {"general_settings": {"master_key": test_master_key}}

    # Create a temporary config file
    config_path = tmp_path / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(test_config, f)

    print(f"SET ENV VARIABLE - CONFIG_FILE_PATH, str(config_path): {str(config_path)}")
    # Second setting of CONFIG_FILE_PATH to a different value
    monkeypatch.setenv("CONFIG_FILE_PATH", str(config_path))
    print(f"config_path: {config_path}")
    print(f"os.getenv('CONFIG_FILE_PATH'): {os.getenv('CONFIG_FILE_PATH')}")
    async with proxy_startup_event(app):
        from litellm.proxy.proxy_server import master_key

        assert master_key == test_master_key

    # Test Case 2: Master key from environment variable
    test_env_master_key = "sk-67890"

    # Create empty config
    empty_config = {"general_settings": {}}
    with open(config_path, "w") as f:
        yaml.dump(empty_config, f)

    monkeypatch.setenv("LITELLM_MASTER_KEY", test_env_master_key)
    print("test_env_master_key: {}".format(test_env_master_key))
    async with proxy_startup_event(app):
        from litellm.proxy.proxy_server import master_key

        assert master_key == test_env_master_key

    # Test Case 3: Master key with os.environ prefix
    test_resolved_key = "sk-resolved-key"
    test_config_with_prefix = {
        "general_settings": {"master_key": "os.environ/CUSTOM_MASTER_KEY"}
    }

    # Create config with os.environ prefix
    with open(config_path, "w") as f:
        yaml.dump(test_config_with_prefix, f)

    monkeypatch.setenv("CUSTOM_MASTER_KEY", test_resolved_key)
    async with proxy_startup_event(app):
        from litellm.proxy.proxy_server import master_key

        assert master_key == test_resolved_key


def test_team_info_masking():
    """
    Test that sensitive team information is properly masked

    Ref: https://huntr.com/bounties/661b388a-44d8-4ad5-862b-4dc5b80be30a
    """
    from litellm.proxy.proxy_server import ProxyConfig

    proxy_config = ProxyConfig()
    # Test team object with sensitive data
    team1_info = {
        "success_callback": "['langfuse', 's3']",
        "langfuse_secret": "secret-test-key",
        "langfuse_public_key": "public-test-key",
    }

    with pytest.raises(Exception) as exc_info:
        proxy_config._get_team_config(
            team_id="test_dev",
            all_teams_config=[team1_info],
        )

    print("Got exception: {}".format(exc_info.value))
    assert "secret-test-key" not in str(exc_info.value)
    assert "public-test-key" not in str(exc_info.value)


@mock_patch_aembedding()
def test_embedding_input_array_of_tokens(mock_aembedding, client_no_auth):
    """
    Test to bypass decoding input as array of tokens for selected providers

    Ref: https://github.com/BerriAI/litellm/issues/10113
    """
    try:
        test_data = {
            "model": "vllm_embed_model",
            "input": [[2046, 13269, 158208]],
        }

        response = client_no_auth.post("/v1/embeddings", json=test_data)

        mock_aembedding.assert_called_once_with(
            model="vllm_embed_model",
            input=[[2046, 13269, 158208]],
            metadata=mock.ANY,
            proxy_server_request=mock.ANY,
        )
        assert response.status_code == 200
        result = response.json()
        print(len(result["data"][0]["embedding"]))
        assert len(result["data"][0]["embedding"]) > 10  # this usually has len==1536 so
    except Exception as e:
        pytest.fail(f"LiteLLM Proxy test failed. Exception - {str(e)}")
