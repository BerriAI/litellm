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
            secret_fields=mock.ANY,
        )
        assert response.status_code == 200
        result = response.json()
        print(len(result["data"][0]["embedding"]))
        assert len(result["data"][0]["embedding"]) > 10  # this usually has len==1536 so
    except Exception as e:
        pytest.fail(f"LiteLLM Proxy test failed. Exception - {str(e)}")


@pytest.mark.asyncio
async def test_get_all_team_models():
    """
    Test get_all_team_models function with both "*" and specific team IDs
    """
    from unittest.mock import AsyncMock, MagicMock

    from litellm.proxy._types import LiteLLM_TeamTable
    from litellm.proxy.proxy_server import get_all_team_models

    # Mock team data
    mock_team1 = MagicMock()
    mock_team1.model_dump.return_value = {
        "team_id": "team1",
        "models": ["gpt-4", "gpt-3.5-turbo"],
        "team_alias": "Team 1",
    }

    mock_team2 = MagicMock()
    mock_team2.model_dump.return_value = {
        "team_id": "team2",
        "models": ["claude-3", "gpt-4"],
        "team_alias": "Team 2",
    }

    # Mock model data returned by router
    mock_models_gpt4 = [
        {"model_info": {"id": "gpt-4-model-1"}},
        {"model_info": {"id": "gpt-4-model-2"}},
    ]
    mock_models_gpt35 = [
        {"model_info": {"id": "gpt-3.5-turbo-model-1"}},
    ]
    mock_models_claude = [
        {"model_info": {"id": "claude-3-model-1"}},
    ]

    # Mock prisma client
    mock_prisma_client = MagicMock()
    mock_db = MagicMock()
    mock_litellm_teamtable = MagicMock()

    mock_prisma_client.db = mock_db
    mock_db.litellm_teamtable = mock_litellm_teamtable

    # Make find_many async
    mock_litellm_teamtable.find_many = AsyncMock()

    # Mock router
    mock_router = MagicMock()

    def mock_get_model_list(model_name, team_id=None):
        if model_name == "gpt-4":
            return mock_models_gpt4
        elif model_name == "gpt-3.5-turbo":
            return mock_models_gpt35
        elif model_name == "claude-3":
            return mock_models_claude
        return None

    mock_router.get_model_list.side_effect = mock_get_model_list

    # Test Case 1: user_teams = "*" (all teams)
    mock_litellm_teamtable.find_many.return_value = [mock_team1, mock_team2]

    with patch("litellm.proxy.proxy_server.LiteLLM_TeamTable") as mock_team_table_class:
        # Configure the mock class to return proper instances
        def mock_team_table_constructor(**kwargs):
            mock_instance = MagicMock()
            mock_instance.team_id = kwargs["team_id"]
            mock_instance.models = kwargs["models"]
            return mock_instance

        mock_team_table_class.side_effect = mock_team_table_constructor

        result = await get_all_team_models(
            user_teams="*",
            prisma_client=mock_prisma_client,
            llm_router=mock_router,
        )

        # Verify find_many was called without where clause for "*"
        mock_litellm_teamtable.find_many.assert_called_with()

        # Verify router.get_model_list was called for each model
        expected_calls = [
            mock.call(model_name="gpt-4", team_id="team1"),
            mock.call(model_name="gpt-3.5-turbo", team_id="team1"),
            mock.call(model_name="claude-3", team_id="team2"),
            mock.call(model_name="gpt-4", team_id="team2"),
        ]
        mock_router.get_model_list.assert_has_calls(expected_calls, any_order=True)

    # Test Case 2: user_teams = specific list
    mock_litellm_teamtable.reset_mock()
    mock_router.reset_mock()
    mock_router.get_model_list.side_effect = mock_get_model_list

    # Only return team1 for specific team query
    mock_litellm_teamtable.find_many.return_value = [mock_team1]

    with patch("litellm.proxy.proxy_server.LiteLLM_TeamTable") as mock_team_table_class:
        mock_team_table_class.side_effect = mock_team_table_constructor

        result = await get_all_team_models(
            user_teams=["team1"],
            prisma_client=mock_prisma_client,
            llm_router=mock_router,
        )

        # Verify find_many was called with where clause for specific teams
        mock_litellm_teamtable.find_many.assert_called_with(
            where={"team_id": {"in": ["team1"]}}
        )

        # Verify router.get_model_list was called only for team1 models
        expected_calls = [
            mock.call(model_name="gpt-4", team_id="team1"),
            mock.call(model_name="gpt-3.5-turbo", team_id="team1"),
        ]
        mock_router.get_model_list.assert_has_calls(expected_calls, any_order=True)

    # Test Case 3: Empty teams list
    mock_litellm_teamtable.reset_mock()
    mock_router.reset_mock()
    mock_litellm_teamtable.find_many.return_value = []

    result = await get_all_team_models(
        user_teams=[],
        prisma_client=mock_prisma_client,
        llm_router=mock_router,
    )

    # Verify find_many was called with empty list
    mock_litellm_teamtable.find_many.assert_called_with(where={"team_id": {"in": []}})

    # Should return empty list when no teams
    assert result == {}

    # Test Case 4: Router returns None for some models
    mock_litellm_teamtable.reset_mock()
    mock_router.reset_mock()
    mock_litellm_teamtable.find_many.return_value = [mock_team1]

    def mock_get_model_list_with_none(model_name, team_id=None):
        if model_name == "gpt-4":
            return mock_models_gpt4
        # Return None for gpt-3.5-turbo to test None handling
        return None

    mock_router.get_model_list.side_effect = mock_get_model_list_with_none

    with patch("litellm.proxy.proxy_server.LiteLLM_TeamTable") as mock_team_table_class:
        mock_team_table_class.side_effect = mock_team_table_constructor

        result = await get_all_team_models(
            user_teams=["team1"],
            prisma_client=mock_prisma_client,
            llm_router=mock_router,
        )

        # Should handle None return gracefully
        assert isinstance(result, dict)
        print("result: ", result)
        assert result == {"gpt-4-model-1": ["team1"], "gpt-4-model-2": ["team1"]}


def test_add_team_models_to_all_models():
    """
    Test add_team_models_to_all_models function
    """
    from litellm.proxy._types import LiteLLM_TeamTable
    from litellm.proxy.proxy_server import _add_team_models_to_all_models

    team_db_objects_typed = MagicMock(spec=LiteLLM_TeamTable)
    team_db_objects_typed.team_id = "team1"
    team_db_objects_typed.models = ["all-proxy-models"]

    llm_router = MagicMock()
    llm_router.get_model_list.return_value = [
        {"model_info": {"id": "gpt-4-model-1", "team_id": "team2"}},
        {"model_info": {"id": "gpt-4-model-2"}},
    ]

    result = _add_team_models_to_all_models(
        team_db_objects_typed=[team_db_objects_typed],
        llm_router=llm_router,
    )
    assert result == {"gpt-4-model-2": {"team1"}}


@pytest.mark.asyncio
async def test_delete_deployment_type_mismatch():
    """
    Test that the _delete_deployment function handles type mismatches correctly.
    Specifically test that models 12345678 and 12345679 are NOT deleted when
    they exist in both combined_id_list (as integers) and router_model_ids (as strings).

    This test reproduces the bug where type mismatch causes valid models to be deleted.
    """
    from unittest.mock import MagicMock, patch

    from litellm.proxy.proxy_server import ProxyConfig

    # Create mock ProxyConfig instance
    pc = ProxyConfig()

    pc.get_config = MagicMock(
        return_value={
            "model_list": [
                {
                    "model_name": "openai-gpt-4o",
                    "litellm_params": {"model": "gpt-4o"},
                    "model_info": {"id": 12345678},
                },
                {
                    "model_name": "openai-gpt-4o",
                    "litellm_params": {"model": "gpt-4o"},
                    "model_info": {"id": 12345679},
                },
            ]
        }
    )

    # Mock llm_router with string IDs (this is the source of the type mismatch)
    mock_llm_router = MagicMock()
    mock_llm_router.get_model_ids.return_value = [
        "a96e12e76b36a57cfae57a41288eb41567629cac89b4828c6f7074afc3534695",
        "a40186dd0fdb9b7282380277d7f57044d29de95bfbfcd7f4322b3493702d5cd3",
        "12345678",  # String ID
        "12345679",  # String ID
    ]

    # Track which deployments were deleted
    deleted_ids = []

    def mock_delete_deployment(id):
        deleted_ids.append(id)
        return True  # Simulate successful deletion

    mock_llm_router.delete_deployment = MagicMock(side_effect=mock_delete_deployment)

    # Mock get_config to return empty config (no config models)
    async def mock_get_config(config_file_path):
        return {}

    pc.get_config = MagicMock(side_effect=mock_get_config)

    # Patch the global llm_router
    with patch("litellm.proxy.proxy_server.llm_router", mock_llm_router), patch(
        "litellm.proxy.proxy_server.user_config_file_path", "test_config.yaml"
    ):

        # Call the function under test
        deleted_count = await pc._delete_deployment(db_models=[])

        # Assertions: Models 12345678 and 12345679 should NOT be deleted
        # because they exist in combined_id_list (as integers) even though
        # router has them as strings

        # The function should delete the other 2 models that are not in combined_id_list
        assert deleted_count == 0, f"Expected 0 deletions, got {deleted_count}"

        # Verify that 12345678 and 12345679 were NOT deleted
        assert (
            "12345678" not in deleted_ids
        ), f"Model 12345678 should NOT be deleted. Deleted IDs: {deleted_ids}"
        assert (
            "12345679" not in deleted_ids
        ), f"Model 12345679 should NOT be deleted. Deleted IDs: {deleted_ids}"


@pytest.mark.asyncio
async def test_get_config_from_file(tmp_path, monkeypatch):
    """
    Test the _get_config_from_file method of ProxyConfig class.
    Tests various scenarios: valid file, non-existent file, no file path, None config.
    """
    import yaml

    from litellm.proxy.proxy_server import ProxyConfig

    # Create a ProxyConfig instance
    proxy_config = ProxyConfig()

    # Test Case 1: Valid YAML config file exists
    test_config = {
        "model_list": [{"model_name": "gpt-4", "litellm_params": {"model": "gpt-4"}}],
        "general_settings": {"master_key": "sk-test"},
        "router_settings": {"enable_pre_call_checks": True},
        "litellm_settings": {"drop_params": True},
    }

    config_file = tmp_path / "test_config.yaml"
    with open(config_file, "w") as f:
        yaml.dump(test_config, f)

    # Clear global user_config_file_path for this test
    monkeypatch.setattr("litellm.proxy.proxy_server.user_config_file_path", None)

    result = await proxy_config._get_config_from_file(str(config_file))
    assert result == test_config

    # Verify that user_config_file_path was set
    from litellm.proxy.proxy_server import user_config_file_path

    assert user_config_file_path == str(config_file)

    # Test Case 2: File path provided but file doesn't exist
    non_existent_file = tmp_path / "non_existent.yaml"

    with pytest.raises(Exception, match=f"Config file not found: {non_existent_file}"):
        await proxy_config._get_config_from_file(str(non_existent_file))

    # Test Case 3: No file path provided (should return default config)
    monkeypatch.setattr("litellm.proxy.proxy_server.user_config_file_path", None)

    expected_default = {
        "model_list": [],
        "general_settings": {},
        "router_settings": {},
        "litellm_settings": {},
    }

    result = await proxy_config._get_config_from_file(None)
    assert result == expected_default

    # Test Case 4: Empty YAML file (should raise exception for None config)
    empty_file = tmp_path / "empty_config.yaml"
    with open(empty_file, "w") as f:
        f.write("")  # Write empty content which will result in None when loaded

    with pytest.raises(Exception, match="Config cannot be None or Empty."):
        await proxy_config._get_config_from_file(str(empty_file))

    # Test Case 5: Using global user_config_file_path when no config_file_path provided
    monkeypatch.setattr(
        "litellm.proxy.proxy_server.user_config_file_path", str(config_file)
    )

    result = await proxy_config._get_config_from_file(None)
    assert result == test_config


@pytest.mark.asyncio
async def test_add_proxy_budget_to_db_only_creates_user_no_keys():
    """
    Test that _add_proxy_budget_to_db only creates a user and no keys are added.

    This validates that generate_key_helper_fn is called with table_name="user"
    which should prevent key creation in LiteLLM_VerificationToken table.
    """
    from unittest.mock import AsyncMock, patch

    import litellm
    from litellm.proxy.proxy_server import ProxyStartupEvent

    # Set up required litellm settings
    litellm.budget_duration = "30d"
    litellm.max_budget = 100.0

    litellm_proxy_budget_name = "litellm-proxy-budget"

    # Mock generate_key_helper_fn to capture its call arguments
    mock_generate_key_helper = AsyncMock(
        return_value={
            "user_id": litellm_proxy_budget_name,
            "max_budget": 100.0,
            "budget_duration": "30d",
            "spend": 0,
            "models": [],
        }
    )

    # Patch generate_key_helper_fn in proxy_server where it's being called from
    with patch(
        "litellm.proxy.proxy_server.generate_key_helper_fn", mock_generate_key_helper
    ):
        # Call the function under test
        ProxyStartupEvent._add_proxy_budget_to_db(litellm_proxy_budget_name)

        # Allow async task to complete
        import asyncio

        await asyncio.sleep(0.1)

        # Verify that generate_key_helper_fn was called
        mock_generate_key_helper.assert_called_once()
        call_args = mock_generate_key_helper.call_args

        # Verify critical parameters that prevent key creation
        assert call_args.kwargs["request_type"] == "user"
        assert call_args.kwargs["table_name"] == "user"
        assert call_args.kwargs["user_id"] == litellm_proxy_budget_name
        assert call_args.kwargs["max_budget"] == 100.0
        assert call_args.kwargs["budget_duration"] == "30d"
        assert call_args.kwargs["query_type"] == "update_data"


@pytest.mark.asyncio
async def test_custom_ui_sso_sign_in_handler_config_loading():
    """
    Test that custom_ui_sso_sign_in_handler from config gets properly loaded into the global variable
    """
    import tempfile
    from unittest.mock import MagicMock, patch

    import yaml

    from litellm.proxy.proxy_server import ProxyConfig

    # Create a test config with custom_ui_sso_sign_in_handler
    test_config = {
        "general_settings": {
            "custom_ui_sso_sign_in_handler": "custom_hooks.custom_ui_sso_hook.custom_ui_sso_sign_in_handler"
        },
        "model_list": [],
        "router_settings": {},
        "litellm_settings": {},
    }

    # Create temporary config file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(test_config, f)
        config_file_path = f.name

    # Mock the get_instance_fn to return a mock handler
    mock_custom_handler = MagicMock()

    try:
        with patch(
            "litellm.proxy.proxy_server.get_instance_fn",
            return_value=mock_custom_handler,
        ) as mock_get_instance:
            # Create ProxyConfig instance and load config
            proxy_config = ProxyConfig()
            # Create a mock router since load_config requires it
            mock_router = MagicMock()
            await proxy_config.load_config(
                router=mock_router, config_file_path=config_file_path
            )

            # Verify get_instance_fn was called with correct parameters
            mock_get_instance.assert_called_with(
                value="custom_hooks.custom_ui_sso_hook.custom_ui_sso_sign_in_handler",
                config_file_path=config_file_path,
            )

            # Verify the global variable was set
            from litellm.proxy.proxy_server import user_custom_ui_sso_sign_in_handler

            assert user_custom_ui_sso_sign_in_handler == mock_custom_handler

    finally:
        # Clean up temporary file
        import os

        os.unlink(config_file_path)


@pytest.mark.asyncio
async def test_load_environment_variables_direct_and_os_environ():
    """
    Test _load_environment_variables method with direct values and os.environ/ prefixed values
    """
    from unittest.mock import patch

    from litellm.proxy.proxy_server import ProxyConfig

    proxy_config = ProxyConfig()

    # Test config with both direct values and os.environ/ prefixed values
    test_config = {
        "environment_variables": {
            "DIRECT_VAR": "direct_value",
            "NUMERIC_VAR": 12345,
            "BOOL_VAR": True,
            "SECRET_VAR": "os.environ/ACTUAL_SECRET_VAR",
        }
    }

    # Mock get_secret_str to return a resolved value
    mock_secret_value = "resolved_secret_value"

    with patch(
        "litellm.proxy.proxy_server.get_secret_str", return_value=mock_secret_value
    ) as mock_get_secret:
        with patch.dict(
            os.environ, {}, clear=False
        ):  # Don't clear existing env vars, just track changes
            # Call the method under test
            proxy_config._load_environment_variables(test_config)

            # Verify direct environment variables were set correctly
            assert os.environ["DIRECT_VAR"] == "direct_value"
            assert os.environ["NUMERIC_VAR"] == "12345"  # Should be converted to string
            assert os.environ["BOOL_VAR"] == "True"  # Should be converted to string

            # Verify os.environ/ prefixed variable was resolved and set
            assert os.environ["SECRET_VAR"] == mock_secret_value

            # Verify get_secret_str was called with the correct value
            mock_get_secret.assert_called_once_with(
                secret_name="os.environ/ACTUAL_SECRET_VAR"
            )


@pytest.mark.asyncio
async def test_load_environment_variables_litellm_license_and_edge_cases():
    """
    Test _load_environment_variables method with LITELLM_LICENSE special handling and edge cases
    """
    from unittest.mock import MagicMock, patch

    from litellm.proxy.proxy_server import ProxyConfig

    proxy_config = ProxyConfig()

    # Test Case 1: LITELLM_LICENSE in environment_variables
    test_config_with_license = {
        "environment_variables": {
            "LITELLM_LICENSE": "test_license_key",
            "OTHER_VAR": "other_value",
        }
    }

    # Mock _license_check
    mock_license_check = MagicMock()
    mock_license_check.is_premium.return_value = True

    with patch("litellm.proxy.proxy_server._license_check", mock_license_check):
        with patch.dict(os.environ, {}, clear=False):
            # Call the method under test
            proxy_config._load_environment_variables(test_config_with_license)

            # Verify LITELLM_LICENSE was set in environment
            assert os.environ["LITELLM_LICENSE"] == "test_license_key"

            # Verify license check was updated
            assert mock_license_check.license_str == "test_license_key"
            mock_license_check.is_premium.assert_called_once()

    # Test Case 2: No environment_variables in config
    test_config_no_env_vars = {}

    # This should not raise any errors and should return without doing anything
    result = proxy_config._load_environment_variables(test_config_no_env_vars)
    assert result is None  # Method returns None

    # Test Case 3: environment_variables is None
    test_config_none_env_vars = {"environment_variables": None}

    # This should not raise any errors and should return without doing anything
    result = proxy_config._load_environment_variables(test_config_none_env_vars)
    assert result is None  # Method returns None

    # Test Case 4: os.environ/ prefix but get_secret_str returns None
    test_config_secret_none = {
        "environment_variables": {"FAILED_SECRET": "os.environ/NONEXISTENT_SECRET"}
    }

    with patch("litellm.proxy.proxy_server.get_secret_str", return_value=None):
        with patch.dict(os.environ, {}, clear=False):
            # Call the method under test
            proxy_config._load_environment_variables(test_config_secret_none)

            # Verify that the environment variable was not set when secret resolution fails
            assert "FAILED_SECRET" not in os.environ


@pytest.mark.asyncio
async def test_write_config_to_file(monkeypatch):
    """
    Do not write config to file if store_model_in_db is True
    """
    from unittest.mock import AsyncMock, MagicMock, mock_open, patch

    from litellm.proxy.proxy_server import ProxyConfig

    # Set store_model_in_db to True
    monkeypatch.setattr("litellm.proxy.proxy_server.store_model_in_db", True)

    # Mock prisma_client to not be None (so DB path is taken)
    mock_prisma_client = AsyncMock()
    mock_prisma_client.insert_data = AsyncMock()
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", mock_prisma_client)

    # Mock general_settings
    mock_general_settings = {"store_model_in_db": True}
    monkeypatch.setattr(
        "litellm.proxy.proxy_server.general_settings", mock_general_settings
    )

    # Mock user_config_file_path
    test_config_path = "/tmp/test_config.yaml"
    monkeypatch.setattr(
        "litellm.proxy.proxy_server.user_config_file_path", test_config_path
    )

    proxy_config = ProxyConfig()

    # Mock the open function to track if file writing is attempted
    mock_file_open = mock_open()

    with patch("builtins.open", mock_file_open), patch("yaml.dump") as mock_yaml_dump:
        # Call save_config with test data
        test_config = {"key": "value", "model_list": ["model1", "model2"]}
        await proxy_config.save_config(new_config=test_config)

        # Verify that file was NOT opened for writing (since store_model_in_db=True)
        mock_file_open.assert_not_called()
        mock_yaml_dump.assert_not_called()

        # Verify that database insert was called instead
        mock_prisma_client.insert_data.assert_called_once()

        # Verify the config passed to DB has model_list removed
        call_args = mock_prisma_client.insert_data.call_args
        assert call_args.kwargs["data"] == {
            "key": "value"
        }  # model_list should be popped
        assert call_args.kwargs["table_name"] == "config"


@pytest.mark.asyncio
async def test_write_config_to_file_when_store_model_in_db_false(monkeypatch):
    """
    Test that config IS written to file when store_model_in_db is False
    """
    from unittest.mock import AsyncMock, MagicMock, mock_open, patch

    from litellm.proxy.proxy_server import ProxyConfig

    # Set store_model_in_db to False
    monkeypatch.setattr("litellm.proxy.proxy_server.store_model_in_db", False)

    # Mock prisma_client to be None (so file path is taken)
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", None)

    # Mock general_settings
    mock_general_settings = {"store_model_in_db": False}
    monkeypatch.setattr(
        "litellm.proxy.proxy_server.general_settings", mock_general_settings
    )

    # Mock user_config_file_path
    test_config_path = "/tmp/test_config.yaml"
    monkeypatch.setattr(
        "litellm.proxy.proxy_server.user_config_file_path", test_config_path
    )

    proxy_config = ProxyConfig()

    # Mock the open function and yaml.dump
    mock_file_open = mock_open()

    with patch("builtins.open", mock_file_open), patch("yaml.dump") as mock_yaml_dump:
        # Call save_config with test data
        test_config = {"key": "value", "other_key": "other_value"}
        await proxy_config.save_config(new_config=test_config)

        # Verify that file WAS opened for writing (since store_model_in_db=False)
        mock_file_open.assert_called_once_with(f"{test_config_path}", "w")

        # Verify yaml.dump was called with the config
        mock_yaml_dump.assert_called_once_with(
            test_config,
            mock_file_open.return_value.__enter__.return_value,
            default_flow_style=False,
        )


@pytest.mark.asyncio
async def test_async_data_generator_midstream_error():
    """
    Test async_data_generator handles midstream error from async_post_call_streaming_hook
    Specifically testing the case where Azure Content Safety Guardrail returns an error
    """
    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.proxy.proxy_server import async_data_generator
    from litellm.proxy.utils import ProxyLogging

    # Create mock objects
    mock_user_api_key_dict = MagicMock(spec=UserAPIKeyAuth)
    mock_request_data = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": "test"}],
    }

    # Mock response chunks - simulating normal streaming that gets interrupted
    mock_chunks = [
        {"choices": [{"delta": {"content": "Hello"}}]},
        {"choices": [{"delta": {"content": " world"}}]},
        {"choices": [{"delta": {"content": " this"}}]},
    ]

    # Mock the proxy_logging_obj
    mock_proxy_logging_obj = MagicMock(spec=ProxyLogging)

    # Mock async_post_call_streaming_iterator_hook to yield chunks
    async def mock_streaming_iterator(*args, **kwargs):
        for chunk in mock_chunks:
            yield chunk

    mock_proxy_logging_obj.async_post_call_streaming_iterator_hook = (
        mock_streaming_iterator
    )

    # Mock async_post_call_streaming_hook to return error on third chunk
    def mock_streaming_hook(*args, **kwargs):
        chunk = kwargs.get("response")
        # Return error message for the third chunk (simulating guardrail trigger)
        if chunk == mock_chunks[2]:
            return 'data: {"error": {"error": "Azure Content Safety Guardrail: Hate crossed severity 2, Got severity: 2"}}'
        # Return normal chunks for first two
        return chunk

    mock_proxy_logging_obj.async_post_call_streaming_hook = AsyncMock(
        side_effect=mock_streaming_hook
    )
    mock_proxy_logging_obj.post_call_failure_hook = AsyncMock()

    # Mock the global proxy_logging_obj
    with patch("litellm.proxy.proxy_server.proxy_logging_obj", mock_proxy_logging_obj):
        # Create a mock response object
        mock_response = MagicMock()

        # Collect all yielded data from the generator
        yielded_data = []
        try:
            async for data in async_data_generator(
                mock_response, mock_user_api_key_dict, mock_request_data
            ):
                yielded_data.append(data)
        except Exception as e:
            # If there's an exception, that's also part of what we want to test
            pass

    # Verify the results
    assert (
        len(yielded_data) >= 3
    ), f"Expected at least 3 chunks, got {len(yielded_data)}: {yielded_data}"

    # First two chunks should be normal data
    assert yielded_data[0].startswith(
        "data: "
    ), f"First chunk should start with 'data: ', got: {yielded_data[0]}"
    assert yielded_data[1].startswith(
        "data: "
    ), f"Second chunk should start with 'data: ', got: {yielded_data[1]}"

    # The error message should be yielded
    error_found = False
    done_found = False

    for data in yielded_data:
        if "Azure Content Safety Guardrail: Hate crossed severity 2" in data:
            error_found = True
        if "data: [DONE]" in data:
            done_found = True

    assert (
        error_found
    ), f"Error message should be found in yielded data. Got: {yielded_data}"
    assert done_found, f"[DONE] message should be found at the end. Got: {yielded_data}"

    # Verify that the streaming hook was called for each chunk
    assert mock_proxy_logging_obj.async_post_call_streaming_hook.call_count == len(
        mock_chunks
    )

    # Verify that post_call_failure_hook was NOT called (since this is not an exception case)
    mock_proxy_logging_obj.post_call_failure_hook.assert_not_called()
