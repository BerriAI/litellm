import asyncio
import importlib
import json
import os
import socket
import subprocess
import sys
from datetime import datetime
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
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
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


def test_update_config_fields_deep_merge_db_wins():
    from litellm.proxy.proxy_server import ProxyConfig

    proxy_config = ProxyConfig()

    current_config = {
        "router_settings": {
            "routing_mode": "cost_optimized",
            "model_group_alias": {
                # Existing alias with older model + different hidden flag
                "claude-sonnet-4": {
                    "model": "claude-sonnet-4-20240219",
                    "hidden": True,
                },
                # An extra alias that should remain untouched unless DB overrides it
                "legacy-sonnet": {
                    "model": "claude-2.1",
                    "hidden": True,
                },
            },
        }
    }

    db_param_value = {
        "model_group_alias": {
            # Conflict: DB should win (both 'model' and 'hidden')
            "claude-sonnet-4": {
                "model": "claude-sonnet-4-20250514",
                "hidden": False,
            },
            # New alias to be added by the merge
            "claude-sonnet-latest": {
                "model": "claude-sonnet-4-20250514",
                "hidden": True,
            },
            # Demonstrate that None values from DB are skipped (preserve existing)
            "legacy-sonnet": {
                "hidden": None  # should not clobber current True
            },
        }
    }

    updated = proxy_config._update_config_fields(
        current_config=current_config,
        param_name="router_settings",
        db_param_value=db_param_value,
    )

    rs = updated["router_settings"]
    aliases = rs["model_group_alias"]

    # DB wins on conflicts (deep) for existing alias
    assert aliases["claude-sonnet-4"]["model"] == "claude-sonnet-4-20250514"
    assert aliases["claude-sonnet-4"]["hidden"] is False

    # New alias introduced by DB is present with its values
    assert "claude-sonnet-latest" in aliases
    assert aliases["claude-sonnet-latest"]["model"] == "claude-sonnet-4-20250514"
    assert aliases["claude-sonnet-latest"]["hidden"] is True

    # None in DB does not overwrite existing values
    assert aliases["legacy-sonnet"]["model"] == "claude-2.1"
    assert aliases["legacy-sonnet"]["hidden"] is True

    # Unrelated router_settings keys are preserved
    assert rs["routing_mode"] == "cost_optimized"


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

        # DEPRECATED - mock_aembedding.assert_called_once_with is too strict, and will fail when new kwargs are added to embeddings
        # mock_aembedding.assert_called_once_with(
        #     model="vllm_embed_model",
        #     input=[[2046, 13269, 158208]],
        #     metadata=mock.ANY,
        #     proxy_server_request=mock.ANY,
        #     secret_fields=mock.ANY,
        # )
        # Assert that aembedding was called, and that input was not modified
        mock_aembedding.assert_called_once()
        call_args, call_kwargs = mock_aembedding.call_args
        assert call_kwargs["model"] == "vllm_embed_model"
        assert call_kwargs["input"] == [[2046, 13269, 158208]]

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


def _has_nested_none_values(obj, path="root"):
    """
    Recursively check if an object contains nested None values.

    Args:
        obj: The object to check
        path: Current path in the object tree (for debugging)

    Returns:
        List of paths where None values were found
    """
    none_paths = []

    if obj is None:
        none_paths.append(path)
    elif isinstance(obj, dict):
        for key, value in obj.items():
            none_paths.extend(_has_nested_none_values(value, f"{path}.{key}"))
    elif isinstance(obj, (list, tuple)):
        for i, item in enumerate(obj):
            none_paths.extend(_has_nested_none_values(item, f"{path}[{i}]"))
    elif hasattr(obj, "__dict__"):
        # Handle object attributes
        for key, value in obj.__dict__.items():
            if not key.startswith("_"):  # Skip private attributes
                none_paths.extend(_has_nested_none_values(value, f"{path}.{key}"))

    return none_paths


@pytest.mark.asyncio
async def test_chat_completion_result_no_nested_none_values():
    """
    Test that chat_completion result doesn't have nested None values when using exclude_none=True
    """
    from unittest.mock import AsyncMock, MagicMock, patch

    from fastapi import Request, Response
    from pydantic import BaseModel

    import litellm
    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.proxy.proxy_server import chat_completion

    # Create a mock ModelResponse with nested None values
    mock_model_response = litellm.ModelResponse()
    mock_model_response.id = "test-id"
    mock_model_response.model = "gpt-3.5-turbo"
    mock_model_response.object = "chat.completion"
    mock_model_response.created = 1234567890

    # Create message with None values that should be excluded
    mock_message = litellm.Message(
        content="Hello, world!",
        role="assistant",
        function_call=None,  # This should be excluded
        tool_calls=None,  # This should be excluded
        audio=None,  # This should be excluded
        reasoning_content=None,  # This should be excluded
        thinking_blocks=None,  # This should be excluded
        annotations=None,  # This should be excluded
    )

    # Create choice with potential None values
    mock_choice = litellm.Choices(
        finish_reason="stop",
        index=0,
        message=mock_message,
        logprobs=None,  # This should be excluded when exclude_none=True
    )

    mock_model_response.choices = [mock_choice]
    setattr(
        mock_model_response,
        "usage",
        litellm.Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )

    # Verify the mock has None values before serialization
    raw_dict = mock_model_response.model_dump()
    none_paths_before = _has_nested_none_values(raw_dict)
    assert (
        len(none_paths_before) > 0
    ), "Mock should have None values before exclude_none=True"

    # Mock the request processing to return our mock response
    mock_base_processor = MagicMock()
    mock_base_processor.base_process_llm_request = AsyncMock(
        return_value=mock_model_response
    )

    # Mock other dependencies
    mock_request = MagicMock(spec=Request)
    mock_response = MagicMock(spec=Response)
    mock_user_api_key_dict = MagicMock(spec=UserAPIKeyAuth)

    with patch(
        "litellm.proxy.proxy_server._read_request_body",
        return_value={"model": "gpt-3.5-turbo", "messages": []},
    ), patch(
        "litellm.proxy.proxy_server.ProxyBaseLLMRequestProcessing",
        return_value=mock_base_processor,
    ):

        # Call the chat_completion function
        result = await chat_completion(
            request=mock_request,
            fastapi_response=mock_response,
            user_api_key_dict=mock_user_api_key_dict,
        )

    # Verify the result is a dict (since isinstance(result, BaseModel) was True)
    assert isinstance(result, dict), f"Expected dict result, got {type(result)}"

    # Check that there are no nested None values in the result
    none_paths_after = _has_nested_none_values(result)
    assert (
        len(none_paths_after) == 0
    ), f"Result should not contain nested None values. Found None at: {none_paths_after}"

    # Verify essential fields are present
    assert "id" in result
    assert "model" in result
    assert "object" in result
    assert "created" in result
    assert "choices" in result
    assert "usage" in result

    # Verify that the choices contain the expected message content
    assert len(result["choices"]) == 1
    assert result["choices"][0]["message"]["content"] == "Hello, world!"
    assert result["choices"][0]["message"]["role"] == "assistant"

    # Verify that None fields were excluded (should not be present in the dict)
    message = result["choices"][0]["message"]
    excluded_fields = [
        "function_call",
        "tool_calls",
        "audio",
        "reasoning_content",
        "thinking_blocks",
        "annotations",
    ]
    for field in excluded_fields:
        assert (
            field not in message
        ), f"Field '{field}' should be excluded when it's None"


# ============================================================================
# Price Data Reload Tests
# ============================================================================


class TestPriceDataReloadAPI:
    """Test cases for price data reload API endpoints"""

    @pytest.fixture
    def client_with_auth(self):
        """Create a test client with authentication"""
        from litellm.proxy._types import LitellmUserRoles
        from litellm.proxy.proxy_server import cleanup_router_config_variables

        cleanup_router_config_variables()
        filepath = os.path.dirname(os.path.abspath(__file__))
        config_fp = f"{filepath}/test_configs/test_config_no_auth.yaml"
        asyncio.run(initialize(config=config_fp, debug=True))

        # Mock admin user authentication
        mock_auth = MagicMock()
        mock_auth.user_role = LitellmUserRoles.PROXY_ADMIN
        app.dependency_overrides[user_api_key_auth] = lambda: mock_auth

        return TestClient(app)

    def test_reload_model_cost_map_admin_access(self, client_with_auth):
        """Test that admin users can access the reload endpoint"""
        with patch(
            "litellm.litellm_core_utils.get_model_cost_map.get_model_cost_map"
        ) as mock_get_map:
            mock_get_map.return_value = {
                "gpt-3.5-turbo": {"input_cost_per_token": 0.001}
            }
            # Mock the database connection
            with patch("litellm.proxy.proxy_server.prisma_client") as mock_prisma:
                mock_prisma.db.litellm_config.upsert = AsyncMock(return_value=None)

                response = client_with_auth.post("/reload/model_cost_map")

                assert response.status_code == 200
                data = response.json()
                assert data["status"] == "success"
                assert "message" in data
                assert "timestamp" in data
                assert "models_count" in data
                # The new implementation immediately reloads and returns the count
                assert (
                    "Price data reloaded successfully! 1 models updated."
                    in data["message"]
                )
                assert data["models_count"] == 1

    def test_reload_model_cost_map_non_admin_access(self, client_with_auth):
        """Test that non-admin users cannot access the reload endpoint"""
        # Mock non-admin user
        mock_auth = MagicMock()
        mock_auth.user_role = "user"  # Non-admin role
        app.dependency_overrides[user_api_key_auth] = lambda: mock_auth

        response = client_with_auth.post("/reload/model_cost_map")

        assert response.status_code == 403
        data = response.json()
        assert "Access denied" in data["detail"]
        assert "Admin role required" in data["detail"]

    def test_get_model_cost_map_admin_access(self, client_with_auth):
        """Test that admin users can access the get model cost map endpoint"""
        with patch(
            "litellm.model_cost", {"gpt-3.5-turbo": {"input_cost_per_token": 0.001}}
        ):
            response = client_with_auth.get("/get/litellm_model_cost_map")

            assert response.status_code == 200
            data = response.json()
            assert "gpt-3.5-turbo" in data

    def test_get_model_cost_map_non_admin_access(self, client_with_auth):
        """Test that non-admin users cannot access the get model cost map endpoint"""
        # Mock non-admin user
        mock_auth = MagicMock()
        mock_auth.user_role = "user"  # Non-admin role
        app.dependency_overrides[user_api_key_auth] = lambda: mock_auth

        response = client_with_auth.get("/get/litellm_model_cost_map")

        assert response.status_code == 403
        data = response.json()
        assert "Access denied" in data["detail"]
        assert "Admin role required" in data["detail"]

    def test_reload_model_cost_map_error_handling(self, client_with_auth):
        """Test error handling in the reload endpoint"""
        with patch(
            "litellm.litellm_core_utils.get_model_cost_map.get_model_cost_map"
        ) as mock_get_map:
            mock_get_map.side_effect = Exception("Network error")

            # Mock the database connection
            with patch("litellm.proxy.proxy_server.prisma_client") as mock_prisma:
                mock_prisma.db.litellm_config.upsert = AsyncMock(return_value=None)

                response = client_with_auth.post("/reload/model_cost_map")

                assert (
                    response.status_code == 500
                )  # The new implementation immediately reloads and fails on error
                data = response.json()
                assert "Failed to reload model cost map" in data["detail"]

    def test_schedule_model_cost_map_reload_admin_access(self, client_with_auth):
        """Test that admin users can schedule periodic reload"""
        with patch("litellm.proxy.proxy_server.prisma_client") as mock_prisma:
            # Mock database upsert
            mock_prisma.db.litellm_config.upsert = AsyncMock(return_value=None)

            response = client_with_auth.post("/schedule/model_cost_map_reload?hours=6")

            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "success"
            assert data["interval_hours"] == 6
            assert "message" in data
            assert "timestamp" in data

    def test_schedule_model_cost_map_reload_non_admin_access(self, client_with_auth):
        """Test that non-admin users cannot schedule periodic reload"""
        # Mock non-admin user
        mock_auth = MagicMock()
        mock_auth.user_role = "user"  # Non-admin role
        app.dependency_overrides[user_api_key_auth] = lambda: mock_auth

        response = client_with_auth.post("/schedule/model_cost_map_reload?hours=6")

        assert response.status_code == 403
        data = response.json()
        assert "Access denied" in data["detail"]
        assert "Admin role required" in data["detail"]

    def test_schedule_model_cost_map_reload_invalid_hours(self, client_with_auth):
        """Test that invalid hours parameter is rejected"""
        response = client_with_auth.post("/schedule/model_cost_map_reload?hours=0")

        assert response.status_code == 400
        data = response.json()
        assert "Hours must be greater than 0" in data["detail"]

    def test_cancel_model_cost_map_reload_admin_access(self, client_with_auth):
        """Test that admin users can cancel periodic reload"""
        with patch("litellm.proxy.proxy_server.prisma_client") as mock_prisma:
            # Mock database delete
            mock_prisma.db.litellm_config.delete = AsyncMock(return_value=None)

            response = client_with_auth.delete("/schedule/model_cost_map_reload")

            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "success"
            assert "message" in data
            assert "timestamp" in data

    def test_cancel_model_cost_map_reload_non_admin_access(self, client_with_auth):
        """Test that non-admin users cannot cancel periodic reload"""
        # Mock non-admin user
        mock_auth = MagicMock()
        mock_auth.user_role = "user"  # Non-admin role
        app.dependency_overrides[user_api_key_auth] = lambda: mock_auth

        response = client_with_auth.delete("/schedule/model_cost_map_reload")

        assert response.status_code == 403
        data = response.json()
        assert "Access denied" in data["detail"]
        assert "Admin role required" in data["detail"]

    def test_get_model_cost_map_reload_status_admin_access(self, client_with_auth):
        """Test that admin users can get reload status"""
        with patch("litellm.proxy.proxy_server.prisma_client") as mock_prisma:
            # Mock database config record
            mock_config = MagicMock()
            mock_config.param_value = {"interval_hours": 6, "force_reload": False}
            mock_prisma.db.litellm_config.find_unique = AsyncMock(
                return_value=mock_config
            )

            # Mock the last reload time and current time
            with patch(
                "litellm.proxy.proxy_server.last_model_cost_map_reload",
                "2024-01-01T06:00:00",
            ):
                with patch("litellm.proxy.proxy_server.datetime") as mock_datetime:
                    # Mock current time to be 1 hour after last reload
                    mock_datetime.utcnow.return_value = datetime(2024, 1, 1, 7, 0, 0)
                    mock_datetime.fromisoformat = datetime.fromisoformat

                    response = client_with_auth.get(
                        "/schedule/model_cost_map_reload/status"
                    )

                    assert response.status_code == 200
                    data = response.json()
                    assert data["scheduled"] == True
                    assert data["interval_hours"] == 6
                    assert data["last_run"] == "2024-01-01T06:00:00"
                    assert data["next_run"] == "2024-01-01T12:00:00"

    def test_get_model_cost_map_reload_status_non_admin_access(self, client_with_auth):
        """Test that non-admin users cannot get reload status"""
        # Mock non-admin user
        mock_auth = MagicMock()
        mock_auth.user_role = "user"  # Non-admin role
        app.dependency_overrides[user_api_key_auth] = lambda: mock_auth

        response = client_with_auth.get("/schedule/model_cost_map_reload/status")

        assert response.status_code == 403
        data = response.json()
        assert "Access denied" in data["detail"]
        assert "Admin role required" in data["detail"]

    def test_get_model_cost_map_reload_status_no_config(self, client_with_auth):
        """Test that status returns not scheduled when no config exists"""
        with patch("litellm.proxy.proxy_server.prisma_client") as mock_prisma:
            mock_prisma.db.litellm_config.find_unique = AsyncMock(return_value=None)

            response = client_with_auth.get("/schedule/model_cost_map_reload/status")

            assert response.status_code == 200
            data = response.json()
            assert data["scheduled"] == False
            assert data["interval_hours"] == None
            assert data["last_run"] == None
            assert data["next_run"] == None

    def test_get_model_cost_map_reload_status_no_interval(self, client_with_auth):
        """Test that status returns not scheduled when no interval is configured"""
        with patch("litellm.proxy.proxy_server.prisma_client") as mock_prisma:
            # Mock config with no interval
            mock_config = MagicMock()
            mock_config.param_value = {"interval_hours": None, "force_reload": False}
            mock_prisma.db.litellm_config.find_unique = AsyncMock(
                return_value=mock_config
            )

            response = client_with_auth.get("/schedule/model_cost_map_reload/status")

            assert response.status_code == 200
            data = response.json()
            assert data["scheduled"] == False
            assert data["interval_hours"] == None
            assert data["last_run"] == None
            assert data["next_run"] == None


class TestPriceDataReloadIntegration:
    """Integration tests for the complete price data reload feature"""

    @pytest.fixture
    def client_with_auth(self):
        """Create a test client with authentication"""
        from litellm.proxy._types import LitellmUserRoles
        from litellm.proxy.proxy_server import cleanup_router_config_variables

        cleanup_router_config_variables()
        filepath = os.path.dirname(os.path.abspath(__file__))
        config_fp = f"{filepath}/test_configs/test_config_no_auth.yaml"
        asyncio.run(initialize(config=config_fp, debug=True))

        # Mock admin user authentication
        mock_auth = MagicMock()
        mock_auth.user_role = LitellmUserRoles.PROXY_ADMIN
        app.dependency_overrides[user_api_key_auth] = lambda: mock_auth

        return TestClient(app)

    def test_complete_reload_flow(self, client_with_auth):
        """Test the complete reload flow from API to model cost update"""
        # Mock the model cost map
        mock_cost_map = {
            "gpt-3.5-turbo": {
                "input_cost_per_token": 0.001,
                "output_cost_per_token": 0.002,
            },
            "gpt-4": {"input_cost_per_token": 0.03, "output_cost_per_token": 0.06},
        }

        with patch(
            "litellm.litellm_core_utils.get_model_cost_map.get_model_cost_map"
        ) as mock_get_map:
            mock_get_map.return_value = mock_cost_map

            # Mock the database connection
            with patch("litellm.proxy.proxy_server.prisma_client") as mock_prisma:
                mock_prisma.db.litellm_config.upsert = AsyncMock(return_value=None)

                # Test reload endpoint
                response = client_with_auth.post("/reload/model_cost_map")
                assert response.status_code == 200

                # Test get endpoint
                response = client_with_auth.get("/get/litellm_model_cost_map")
                assert response.status_code == 200

    def test_distributed_reload_check_function(self):
        """Test the _check_and_reload_model_cost_map function"""
        from litellm.proxy.proxy_server import ProxyConfig

        proxy_config = ProxyConfig()

        # Mock prisma client
        mock_prisma = MagicMock()

        # Test case 1: No config in database
        mock_prisma.db.litellm_config.find_unique = AsyncMock(return_value=None)

        # Should return early without reloading
        asyncio.run(proxy_config._check_and_reload_model_cost_map(mock_prisma))

        # Test case 2: Config with interval but not time to reload
        mock_config = MagicMock()
        mock_config.param_value = {"interval_hours": 6, "force_reload": False}
        mock_prisma.db.litellm_config.find_unique = AsyncMock(return_value=mock_config)

        # Mock current time and last reload time
        with patch(
            "litellm.proxy.proxy_server.last_model_cost_map_reload",
            "2024-01-01T06:00:00",
        ):
            with patch("litellm.proxy.proxy_server.datetime") as mock_datetime:
                mock_datetime.utcnow.return_value = datetime(
                    2024, 1, 1, 7, 0, 0
                )  # 1 hour later

                # Should not reload (only 1 hour passed, need 6)
                asyncio.run(proxy_config._check_and_reload_model_cost_map(mock_prisma))

        # Test case 3: Config with force reload
        mock_config.param_value = {"interval_hours": 6, "force_reload": True}
        mock_prisma.db.litellm_config.find_unique = AsyncMock(return_value=mock_config)
        mock_prisma.db.litellm_config.upsert = AsyncMock(return_value=None)

        with patch(
            "litellm.litellm_core_utils.get_model_cost_map.get_model_cost_map"
        ) as mock_get_map:
            mock_get_map.return_value = {
                "gpt-3.5-turbo": {"input_cost_per_token": 0.001}
            }

            # Should reload due to force flag
            asyncio.run(proxy_config._check_and_reload_model_cost_map(mock_prisma))

            # Verify force_reload was reset to False
            mock_prisma.db.litellm_config.upsert.assert_called()
            call_args = mock_prisma.db.litellm_config.upsert.call_args
            # The param_value is now a JSON string, so we need to parse it
            param_value_json = call_args[1]["data"]["update"]["param_value"]
            param_value_dict = json.loads(param_value_json)
            assert param_value_dict["force_reload"] == False

    def test_config_file_parsing(self):
        """Test parsing of config file with reload settings"""
        config_content = """
general_settings:
  master_key: sk-1234
  model_cost_map_reload_interval: 21600

model_list:
  - model_name: gpt-3.5-turbo
    litellm_params:
      model: gpt-3.5-turbo
  - model_name: gpt-4
    litellm_params:
      model: gpt-4
"""

        # Parse the config
        config = yaml.safe_load(config_content)

        # Verify the reload setting is present
        assert "general_settings" in config
        assert "model_cost_map_reload_interval" in config["general_settings"]
        assert config["general_settings"]["model_cost_map_reload_interval"] == 21600

        # Verify models are present
        assert "model_list" in config
        assert len(config["model_list"]) == 2

    def test_database_config_storage(self):
        """Test that configuration is properly stored in database"""
        # Mock prisma client
        mock_prisma = MagicMock()

        # Test the database upsert call that would be made by the schedule endpoint
        mock_prisma.db.litellm_config.upsert = AsyncMock(return_value=None)

        # Simulate the database call that the schedule endpoint would make
        asyncio.run(
            mock_prisma.db.litellm_config.upsert(
                where={"param_name": "model_cost_map_reload_config"},
                data={
                    "create": {
                        "param_name": "model_cost_map_reload_config",
                        "param_value": {"interval_hours": 6, "force_reload": False},
                    },
                    "update": {
                        "param_value": {"interval_hours": 6, "force_reload": False}
                    },
                },
            )
        )

        # Verify database upsert was called with correct data
        mock_prisma.db.litellm_config.upsert.assert_called_once()
        call_args = mock_prisma.db.litellm_config.upsert.call_args
        assert call_args[1]["where"]["param_name"] == "model_cost_map_reload_config"
        assert call_args[1]["data"]["create"]["param_value"]["interval_hours"] == 6
        assert call_args[1]["data"]["create"]["param_value"]["force_reload"] == False

    def test_manual_reload_force_flag(self):
        """Test that manual reload sets force flag correctly"""
        # Mock prisma client
        mock_prisma = MagicMock()

        # Test the database upsert call that would be made by the manual reload endpoint
        mock_prisma.db.litellm_config.upsert = AsyncMock(return_value=None)

        # Simulate the database call that the manual reload endpoint would make
        asyncio.run(
            mock_prisma.db.litellm_config.upsert(
                where={"param_name": "model_cost_map_reload_config"},
                data={
                    "create": {
                        "param_name": "model_cost_map_reload_config",
                        "param_value": {"interval_hours": None, "force_reload": True},
                    },
                    "update": {"param_value": {"force_reload": True}},
                },
            )
        )

        # Verify force_reload flag was set
        mock_prisma.db.litellm_config.upsert.assert_called_once()
        call_args = mock_prisma.db.litellm_config.upsert.call_args
        assert call_args[1]["data"]["update"]["param_value"]["force_reload"] == True


@pytest.mark.asyncio
async def test_add_router_settings_from_db_config_merge_logic():
    """
    Test the _add_router_settings_from_db_config method's merge logic.

    This tests how router settings from config file and database are combined,
    including scenarios where nested dictionaries should be properly merged.
    """
    from unittest.mock import AsyncMock, MagicMock, patch

    from litellm.proxy.proxy_server import ProxyConfig

    # Create ProxyConfig instance
    proxy_config = ProxyConfig()

    # Mock router
    mock_router = MagicMock()
    mock_router.update_settings = MagicMock()

    # Test Case 1: Both config and DB settings exist - should merge them
    config_data = {
        "router_settings": {
            "routing_strategy": "usage-based-routing",
            "model_group_alias": {"gpt-4": "openai-gpt-4"},
            "enable_pre_call_checks": True,
            "timeout": 30,
            "nested_config": {"setting1": "config_value1", "setting2": "config_value2"},
        }
    }

    # Mock database config record
    mock_db_config = MagicMock()
    mock_db_config.param_value = {
        "routing_strategy": "least-busy",  # This should override config value
        "retry_delay": 2,  # This is new, should be added
        "nested_config": {
            "setting2": "db_value2",  # This should override config value
            "setting3": "db_value3",  # This is new, should be added
        },
    }

    # Mock prisma client
    mock_prisma_client = MagicMock()
    mock_prisma_client.db.litellm_config.find_first = AsyncMock(
        return_value=mock_db_config
    )

    # Call the method under test
    await proxy_config._add_router_settings_from_db_config(
        config_data=config_data,
        llm_router=mock_router,
        prisma_client=mock_prisma_client,
    )

    # Verify find_first was called with correct parameters
    mock_prisma_client.db.litellm_config.find_first.assert_called_once_with(
        where={"param_name": "router_settings"}
    )

    # Verify update_settings was called
    mock_router.update_settings.assert_called_once()

    # Get the actual settings passed to update_settings
    call_args = mock_router.update_settings.call_args
    combined_settings = call_args[1]  # kwargs

    # Verify the merge results
    # DB values should override config values
    assert combined_settings["routing_strategy"] == "least-busy"

    # Config-only values should be preserved
    assert combined_settings["model_group_alias"] == {"gpt-4": "openai-gpt-4"}
    assert combined_settings["enable_pre_call_checks"] == True
    assert combined_settings["timeout"] == 30

    # DB-only values should be added
    assert combined_settings["retry_delay"] == 2

    # Nested dictionaries should be merged (but this is shallow merge)
    expected_nested = {
        "setting1": "config_value1",
        "setting2": "db_value2",
        "setting3": "db_value3",
    }
    assert combined_settings["nested_config"] == expected_nested


@pytest.mark.asyncio
async def test_add_router_settings_from_db_config_edge_cases():
    """
    Test edge cases for _add_router_settings_from_db_config method.
    """
    from unittest.mock import AsyncMock, MagicMock

    from litellm.proxy.proxy_server import ProxyConfig

    proxy_config = ProxyConfig()
    mock_router = MagicMock()
    mock_router.update_settings = MagicMock()

    # Test Case 1: No router provided
    await proxy_config._add_router_settings_from_db_config(
        config_data={"router_settings": {"test": "value"}},
        llm_router=None,
        prisma_client=MagicMock(),
    )
    # Should not call anything when router is None
    mock_router.update_settings.assert_not_called()

    # Test Case 2: No prisma client provided
    await proxy_config._add_router_settings_from_db_config(
        config_data={"router_settings": {"test": "value"}},
        llm_router=mock_router,
        prisma_client=None,
    )
    # Should not call anything when prisma_client is None
    mock_router.update_settings.assert_not_called()

    # Test Case 3: DB returns None (no router_settings in DB)
    mock_prisma_client = MagicMock()
    mock_prisma_client.db.litellm_config.find_first = AsyncMock(return_value=None)

    config_data = {"router_settings": {"routing_strategy": "usage-based"}}

    await proxy_config._add_router_settings_from_db_config(
        config_data=config_data,
        llm_router=mock_router,
        prisma_client=mock_prisma_client,
    )

    # Should use only config settings
    mock_router.update_settings.assert_called_once_with(routing_strategy="usage-based")
    mock_router.reset_mock()

    # Test Case 4: Config has no router_settings
    mock_db_config = MagicMock()
    mock_db_config.param_value = {"db_setting": "db_value"}
    mock_prisma_client.db.litellm_config.find_first = AsyncMock(
        return_value=mock_db_config
    )

    await proxy_config._add_router_settings_from_db_config(
        config_data={},  # No router_settings in config
        llm_router=mock_router,
        prisma_client=mock_prisma_client,
    )

    # Should use only DB settings
    mock_router.update_settings.assert_called_once_with(db_setting="db_value")
    mock_router.reset_mock()

    # Test Case 5: Both config and DB router_settings are None/empty
    mock_prisma_client.db.litellm_config.find_first = AsyncMock(return_value=None)

    await proxy_config._add_router_settings_from_db_config(
        config_data={}, llm_router=mock_router, prisma_client=mock_prisma_client
    )

    # Should not call update_settings when no settings exist
    mock_router.update_settings.assert_not_called()

    # Test Case 6: DB config exists but param_value is not a dict
    mock_db_config_invalid = MagicMock()
    mock_db_config_invalid.param_value = "not_a_dict"
    mock_prisma_client.db.litellm_config.find_first = AsyncMock(
        return_value=mock_db_config_invalid
    )

    config_data = {"router_settings": {"config_setting": "config_value"}}

    await proxy_config._add_router_settings_from_db_config(
        config_data=config_data,
        llm_router=mock_router,
        prisma_client=mock_prisma_client,
    )

    # Should use only config settings when DB param_value is invalid
    mock_router.update_settings.assert_called_once_with(config_setting="config_value")


@pytest.mark.asyncio
async def test_add_router_settings_shallow_merge_behavior():
    """
    Test that the merge behavior is shallow (nested dicts get replaced, not merged).
    This documents the current behavior using _update_dictionary.
    """
    from unittest.mock import AsyncMock, MagicMock

    from litellm.proxy.proxy_server import ProxyConfig

    proxy_config = ProxyConfig()
    mock_router = MagicMock()
    mock_router.update_settings = MagicMock()

    # Config with nested dictionary
    config_data = {
        "router_settings": {
            "nested_setting": {
                "key1": "config_value1",
                "key2": "config_value2",
                "key3": "config_value3",
            },
            "top_level": "config_top",
        }
    }

    # DB config that partially overlaps the nested dictionary
    mock_db_config = MagicMock()
    mock_db_config.param_value = {
        "nested_setting": {
            "key2": "db_value2",  # Override existing key
            "key4": "db_value4",  # Add new key
            # Note: key1 and key3 from config will be lost due to shallow merge
        },
        "top_level": "db_top",  # Override top level
    }

    mock_prisma_client = MagicMock()
    mock_prisma_client.db.litellm_config.find_first = AsyncMock(
        return_value=mock_db_config
    )

    await proxy_config._add_router_settings_from_db_config(
        config_data=config_data,
        llm_router=mock_router,
        prisma_client=mock_prisma_client,
    )

    # Get the merged settings
    call_args = mock_router.update_settings.call_args
    merged_settings = call_args[1]

    # Verify shallow merge behavior:
    # The entire nested_setting dict from config is replaced by the DB version
    expected_nested = {
        "key1": "config_value1",
        "key3": "config_value3",
        "key2": "db_value2",
        "key4": "db_value4",
    }

    assert merged_settings["nested_setting"] == expected_nested
    assert merged_settings["top_level"] == "db_top"


@pytest.mark.asyncio
async def test_model_info_v1_oci_secrets_not_leaked():
    """
    Test that model_info_v1 endpoint properly masks OCI sensitive parameters and does not leak secrets.
    """
    from unittest.mock import MagicMock, patch

    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.proxy.proxy_server import model_info_v1

    # Mock user authentication
    mock_user_api_key_dict = MagicMock(spec=UserAPIKeyAuth)
    mock_user_api_key_dict.user_id = "test-user"
    mock_user_api_key_dict.api_key = "test-key"
    mock_user_api_key_dict.team_models = []
    mock_user_api_key_dict.models = ["oci-grok-test"]
    
    # Mock model data with OCI sensitive information
    mock_model_data = {
        "model_name": "oci-grok-test",
        "litellm_params": {
            "model": "oci/xai.grok-4",
            "oci_key": "ocid1.api_key.oc1..aaaaaaaa7kbkbkbkbkbkbkbkbkbkbkbkbkbkbkbkbkbkbkbkbkbkbkbkbkbk",
            "oci_region": "us-phoenix-1",
            "oci_user": "ocid1.user.oc1..aaaaaaaa7kbkbkbkbkbkbkbkbkbkbkbkbkbkbkbkbkbkbkbkbkbkbkbkbkbk",
            "oci_fingerprint": "aa:bb:cc:dd:ee:ff:11:22:33:44:55:66:77:88:99:00",
            "oci_tenancy": "ocid1.tenancy.oc1..aaaaaaaa7kbkbkbkbkbkbkbkbkbkbkbkbkbkbkbkbkbkbkbkbkbkbkbkbkbk",
            "oci_key_file": "/path/to/oci_api_key.pem",
            "oci_compartment_id": "ocid1.compartment.oc1..aaaaaaaa7kbkbkbkbkbkbkbkbkbkbkbkbkbkbkbkbkbkbkbkbkbkbkbkbkbk",
            "drop_params": True
        },
        "model_info": {
            "mode": "completion",
            "id": "test-model-id"
        }
    }
    
    # Mock the llm_router to return our test data
    mock_router = MagicMock()
    mock_router.get_model_names.return_value = ["oci-grok-test"]
    mock_router.get_model_access_groups.return_value = {}
    mock_router.get_model_list.return_value = [mock_model_data]
    
    # Mock global variables
    with patch("litellm.proxy.proxy_server.llm_router", mock_router), \
         patch("litellm.proxy.proxy_server.llm_model_list", [mock_model_data]), \
         patch("litellm.proxy.proxy_server.general_settings", {"infer_model_from_keys": False}), \
         patch("litellm.proxy.proxy_server.user_model", None):
        
        # Call the model_info_v1 endpoint
        result = await model_info_v1(
            user_api_key_dict=mock_user_api_key_dict,
            litellm_model_id=None
        )
        
        # Verify the result structure
        assert "data" in result
        assert len(result["data"]) == 1
        
        model_info = result["data"][0]
        litellm_params = model_info["litellm_params"]
        
        # Verify that sensitive OCI fields are masked
        assert "****" in litellm_params["oci_key"], "oci_key should be masked"
        assert "****" in litellm_params["oci_fingerprint"], "oci_fingerprint should be masked"
        assert "****" in litellm_params["oci_tenancy"], "oci_tenancy should be masked"
        assert "****" in litellm_params["oci_key_file"], "oci_key_file should be masked"
        
        # Verify that non-sensitive fields are NOT masked
        assert litellm_params["model"] == "oci/xai.grok-4", "model field should not be masked"
        assert litellm_params["oci_region"] == "us-phoenix-1", "oci_region should not be masked"
        assert litellm_params["drop_params"] is True, "drop_params should not be masked"
        
        # Verify the model field specifically is not masked (this was the original issue)
        assert "****" not in litellm_params["model"], "model field should never be masked"
        assert litellm_params["model"].startswith("oci/"), "model should retain its full value"
        
        # Verify that actual secret values are not present in the response
        result_str = str(result)
        assert "ocid1.api_key.oc1..aaaaaaaa7kbkbkbkbkbkbkbkbkbkbkbkbkbkbkbkbkbkbkbkbkbkbkbkbkbk" not in result_str
        assert "aa:bb:cc:dd:ee:ff:11:22:33:44:55:66:77:88:99:00" not in result_str
        assert "ocid1.tenancy.oc1..aaaaaaaa7kbkbkbkbkbkbkbkbkbkbkbkbkbkbkbkbkbkbkbkbkbkbkbkbkbk" not in result_str
        assert "/path/to/oci_api_key.pem" not in result_str


def test_add_callback_from_db_to_in_memory_litellm_callbacks():
    """
    Test that _add_callback_from_db_to_in_memory_litellm_callbacks correctly adds callbacks
    for success, failure, and combined event types.
    """
    from unittest.mock import MagicMock, patch

    from litellm.proxy.proxy_server import ProxyConfig
    
    proxy_config = ProxyConfig()
    
    # Mock the callback manager
    mock_callback_manager = MagicMock()
    
    with patch("litellm.proxy.proxy_server.litellm") as mock_litellm:
        # Set up mock litellm attributes
        mock_litellm._known_custom_logger_compatible_callbacks = []
        mock_litellm.logging_callback_manager = mock_callback_manager
        
        # Test Case 1: Add success callback
        mock_success_callbacks = []
        proxy_config._add_callback_from_db_to_in_memory_litellm_callbacks(
            callback="prometheus",
            event_types=["success"],
            existing_callbacks=mock_success_callbacks,
        )
        mock_callback_manager.add_litellm_success_callback.assert_called_once_with("prometheus")
        mock_callback_manager.reset_mock()
        
        # Test Case 2: Add failure callback
        mock_failure_callbacks = []
        proxy_config._add_callback_from_db_to_in_memory_litellm_callbacks(
            callback="langfuse",
            event_types=["failure"],
            existing_callbacks=mock_failure_callbacks,
        )
        mock_callback_manager.add_litellm_failure_callback.assert_called_once_with("langfuse")
        mock_callback_manager.reset_mock()
        
        # Test Case 3: Add callback for both success and failure
        mock_callbacks = []
        proxy_config._add_callback_from_db_to_in_memory_litellm_callbacks(
            callback="s3",
            event_types=["success", "failure"],
            existing_callbacks=mock_callbacks,
        )
        mock_callback_manager.add_litellm_callback.assert_called_once_with("s3")
        mock_callback_manager.reset_mock()
        
        # Test Case 4: Don't add callback if it already exists
        existing_callbacks_with_item = ["prometheus"]
        proxy_config._add_callback_from_db_to_in_memory_litellm_callbacks(
            callback="prometheus",
            event_types=["success"],
            existing_callbacks=existing_callbacks_with_item,
        )
        mock_callback_manager.add_litellm_success_callback.assert_not_called()


def test_should_load_db_object_with_supported_db_objects():
    """
    Test _should_load_db_object method with supported_db_objects configuration.
    
    Verifies that when supported_db_objects is set, only specified object types
    are loaded from the database.
    """
    from unittest.mock import patch

    from litellm.proxy.proxy_server import ProxyConfig

    proxy_config = ProxyConfig()

    # Test Case 1: supported_db_objects not set - all objects should be loaded
    with patch("litellm.proxy.proxy_server.general_settings", {}):
        assert proxy_config._should_load_db_object(object_type="models") is True
        assert proxy_config._should_load_db_object(object_type="mcp") is True
        assert proxy_config._should_load_db_object(object_type="guardrails") is True
        assert proxy_config._should_load_db_object(object_type="vector_stores") is True

    # Test Case 2: supported_db_objects set to only load MCP
    with patch(
        "litellm.proxy.proxy_server.general_settings",
        {"supported_db_objects": ["mcp"]},
    ):
        assert proxy_config._should_load_db_object(object_type="models") is False
        assert proxy_config._should_load_db_object(object_type="mcp") is True
        assert proxy_config._should_load_db_object(object_type="guardrails") is False
        assert proxy_config._should_load_db_object(object_type="vector_stores") is False
        assert proxy_config._should_load_db_object(object_type="prompts") is False

    # Test Case 3: supported_db_objects set to load multiple types
    with patch(
        "litellm.proxy.proxy_server.general_settings",
        {"supported_db_objects": ["mcp", "guardrails", "vector_stores"]},
    ):
        assert proxy_config._should_load_db_object(object_type="models") is False
        assert proxy_config._should_load_db_object(object_type="mcp") is True
        assert proxy_config._should_load_db_object(object_type="guardrails") is True
        assert proxy_config._should_load_db_object(object_type="vector_stores") is True
        assert proxy_config._should_load_db_object(object_type="prompts") is False

    # Test Case 4: supported_db_objects is not a list (should default to loading all)
    with patch(
        "litellm.proxy.proxy_server.general_settings",
        {"supported_db_objects": "invalid_type"},
    ):
        assert proxy_config._should_load_db_object(object_type="models") is True
        assert proxy_config._should_load_db_object(object_type="mcp") is True

    # Test Case 5: supported_db_objects is an empty list (nothing should be loaded)
    with patch(
        "litellm.proxy.proxy_server.general_settings",
        {"supported_db_objects": []},
    ):
        assert proxy_config._should_load_db_object(object_type="models") is False
        assert proxy_config._should_load_db_object(object_type="mcp") is False
        assert proxy_config._should_load_db_object(object_type="guardrails") is False

    # Test Case 6: Test all available object types
    with patch(
        "litellm.proxy.proxy_server.general_settings",
        {
            "supported_db_objects": [
                "models",
                "mcp",
                "guardrails",
                "vector_stores",
                "pass_through_endpoints",
                "prompts",
                "model_cost_map",
            ]
        },
    ):
        assert proxy_config._should_load_db_object(object_type="models") is True
        assert proxy_config._should_load_db_object(object_type="mcp") is True
        assert proxy_config._should_load_db_object(object_type="guardrails") is True
        assert proxy_config._should_load_db_object(object_type="vector_stores") is True
        assert (
            proxy_config._should_load_db_object(object_type="pass_through_endpoints")
            is True
        )
        assert proxy_config._should_load_db_object(object_type="prompts") is True
        assert proxy_config._should_load_db_object(object_type="model_cost_map") is True


@pytest.mark.asyncio
async def test_tag_cache_update_called():
    """
    Test that update_cache updates tag cache when tags are provided.
    """
    from litellm.caching.caching import DualCache
    from litellm.proxy.proxy_server import user_api_key_cache

    cache = DualCache()

    setattr(
        litellm.proxy.proxy_server,
        "user_api_key_cache",
        cache,
    )

    mock_tag_obj = {
        "tag_name": "test-tag",
        "spend": 10.0,
    }

    with patch.object(cache, "async_get_cache", new=AsyncMock(return_value=mock_tag_obj)) as mock_get_cache:
        with patch.object(cache, "async_set_cache_pipeline", new=AsyncMock()) as mock_set_cache:
            await litellm.proxy.proxy_server.update_cache(
                token=None,
                user_id=None,
                end_user_id=None,
                team_id=None,
                response_cost=5.0,
                parent_otel_span=None,
                tags=["test-tag"],
            )

            await asyncio.sleep(0.1)

            mock_get_cache.assert_awaited_once_with(key="tag:test-tag")
            mock_set_cache.assert_awaited_once()

            call_args = mock_set_cache.call_args
            cache_list = call_args.kwargs["cache_list"]

            assert len(cache_list) == 1
            cache_key, cache_value = cache_list[0]
            assert cache_key == "tag:test-tag"
            assert cache_value["spend"] == 15.0


@pytest.mark.asyncio
async def test_tag_cache_update_multiple_tags():
    """
    Test that multiple tags are updated in cache.
    """
    from litellm.caching.caching import DualCache
    from litellm.proxy.proxy_server import user_api_key_cache

    cache = DualCache()

    setattr(
        litellm.proxy.proxy_server,
        "user_api_key_cache",
        cache,
    )

    mock_tag1_obj = {"tag_name": "tag1", "spend": 10.0}
    mock_tag2_obj = {"tag_name": "tag2", "spend": 20.0}

    async def mock_get_cache_side_effect(key):
        if key == "tag:tag1":
            return mock_tag1_obj
        elif key == "tag:tag2":
            return mock_tag2_obj
        return None

    with patch.object(cache, "async_get_cache", new=AsyncMock(side_effect=mock_get_cache_side_effect)) as mock_get_cache:
        with patch.object(cache, "async_set_cache_pipeline", new=AsyncMock()) as mock_set_cache:
            await litellm.proxy.proxy_server.update_cache(
                token=None,
                user_id=None,
                end_user_id=None,
                team_id=None,
                response_cost=5.0,
                parent_otel_span=None,
                tags=["tag1", "tag2"],
            )

            await asyncio.sleep(0.1)

            assert mock_get_cache.call_count == 2
            mock_set_cache.assert_awaited_once()

            call_args = mock_set_cache.call_args
            cache_list = call_args.kwargs["cache_list"]

            assert len(cache_list) == 2

            tag_updates = {cache_key: cache_value for cache_key, cache_value in cache_list}
            assert "tag:tag1" in tag_updates
            assert "tag:tag2" in tag_updates
            assert tag_updates["tag:tag1"]["spend"] == 15.0
            assert tag_updates["tag:tag2"]["spend"] == 25.0


@pytest.mark.asyncio
async def test_init_sso_settings_in_db():
    """
    Test that _init_sso_settings_in_db properly loads SSO settings from database,
    uppercases keys, and calls _decrypt_and_set_db_env_variables.
    """
    from unittest.mock import AsyncMock, MagicMock, patch

    from litellm.proxy.proxy_server import ProxyConfig

    proxy_config = ProxyConfig()

    # Test Case 1: SSO settings exist in database
    mock_sso_config = MagicMock()
    mock_sso_config.sso_settings = {
        "google_client_id": "test-client-id",
        "google_client_secret": "test-client-secret",
        "microsoft_client_id": "ms-client-id",
        "microsoft_client_secret": "ms-client-secret",
    }

    mock_prisma_client = MagicMock()
    mock_prisma_client.db.litellm_ssoconfig.find_unique = AsyncMock(
        return_value=mock_sso_config
    )

    # Mock _decrypt_and_set_db_env_variables
    with patch.object(
        proxy_config, "_decrypt_and_set_db_env_variables"
    ) as mock_decrypt_and_set:
        await proxy_config._init_sso_settings_in_db(prisma_client=mock_prisma_client)

        # Verify find_unique was called with correct parameters
        mock_prisma_client.db.litellm_ssoconfig.find_unique.assert_awaited_once_with(
            where={"id": "sso_config"}
        )

        # Verify _decrypt_and_set_db_env_variables was called with uppercased keys
        mock_decrypt_and_set.assert_called_once()
        call_args = mock_decrypt_and_set.call_args
        uppercased_settings = call_args.kwargs["environment_variables"]

        # Verify all keys are uppercased
        assert "GOOGLE_CLIENT_ID" in uppercased_settings
        assert "GOOGLE_CLIENT_SECRET" in uppercased_settings
        assert "MICROSOFT_CLIENT_ID" in uppercased_settings
        assert "MICROSOFT_CLIENT_SECRET" in uppercased_settings

        # Verify values are preserved
        assert uppercased_settings["GOOGLE_CLIENT_ID"] == "test-client-id"
        assert uppercased_settings["GOOGLE_CLIENT_SECRET"] == "test-client-secret"
        assert uppercased_settings["MICROSOFT_CLIENT_ID"] == "ms-client-id"
        assert uppercased_settings["MICROSOFT_CLIENT_SECRET"] == "ms-client-secret"

        # Verify original lowercase keys are not present
        assert "google_client_id" not in uppercased_settings
        assert "microsoft_client_id" not in uppercased_settings


@pytest.mark.asyncio
async def test_init_sso_settings_in_db_no_settings():
    """
    Test that _init_sso_settings_in_db handles the case when no SSO settings exist in database.
    """
    from unittest.mock import AsyncMock, MagicMock, patch

    from litellm.proxy.proxy_server import ProxyConfig

    proxy_config = ProxyConfig()

    # Mock prisma client to return None (no SSO settings)
    mock_prisma_client = MagicMock()
    mock_prisma_client.db.litellm_ssoconfig.find_unique = AsyncMock(return_value=None)

    # Mock _decrypt_and_set_db_env_variables
    with patch.object(
        proxy_config, "_decrypt_and_set_db_env_variables"
    ) as mock_decrypt_and_set:
        await proxy_config._init_sso_settings_in_db(prisma_client=mock_prisma_client)

        # Verify find_unique was called
        mock_prisma_client.db.litellm_ssoconfig.find_unique.assert_awaited_once_with(
            where={"id": "sso_config"}
        )

        # Verify _decrypt_and_set_db_env_variables was NOT called when no settings exist
        mock_decrypt_and_set.assert_not_called()


@pytest.mark.asyncio
async def test_init_sso_settings_in_db_error_handling():
    """
    Test that _init_sso_settings_in_db handles database errors gracefully.
    """
    from unittest.mock import AsyncMock, MagicMock, patch

    from litellm.proxy.proxy_server import ProxyConfig

    proxy_config = ProxyConfig()

    # Mock prisma client to raise an exception
    mock_prisma_client = MagicMock()
    mock_prisma_client.db.litellm_ssoconfig.find_unique = AsyncMock(
        side_effect=Exception("Database connection error")
    )

    # The method should not raise an exception, it should log it instead
    try:
        await proxy_config._init_sso_settings_in_db(prisma_client=mock_prisma_client)
        # If we get here, the exception was handled properly
        assert True
    except Exception as e:
        # The exception should be caught and logged, not propagated
        pytest.fail(f"Exception should have been caught and logged, but was raised: {e}")


@pytest.mark.asyncio
async def test_init_sso_settings_in_db_empty_settings():
    """
    Test that _init_sso_settings_in_db handles empty SSO settings dictionary.
    """
    from unittest.mock import AsyncMock, MagicMock, patch

    from litellm.proxy.proxy_server import ProxyConfig

    proxy_config = ProxyConfig()

    # Mock SSO config with empty settings dictionary
    mock_sso_config = MagicMock()
    mock_sso_config.sso_settings = {}

    mock_prisma_client = MagicMock()
    mock_prisma_client.db.litellm_ssoconfig.find_unique = AsyncMock(
        return_value=mock_sso_config
    )

    # Mock _decrypt_and_set_db_env_variables
    with patch.object(
        proxy_config, "_decrypt_and_set_db_env_variables"
    ) as mock_decrypt_and_set:
        await proxy_config._init_sso_settings_in_db(prisma_client=mock_prisma_client)

        # Verify find_unique was called
        mock_prisma_client.db.litellm_ssoconfig.find_unique.assert_awaited_once_with(
            where={"id": "sso_config"}
        )

        # Verify _decrypt_and_set_db_env_variables was called with empty dict
        mock_decrypt_and_set.assert_called_once()
        call_args = mock_decrypt_and_set.call_args
        uppercased_settings = call_args.kwargs["environment_variables"]

        # Verify empty dictionary
        assert uppercased_settings == {}


def test_get_prompt_spec_for_db_prompt_with_versions():
    """
    Test that _get_prompt_spec_for_db_prompt correctly converts database prompts
    to PromptSpec with versioned naming convention.
    """
    from unittest.mock import MagicMock

    from litellm.proxy.proxy_server import ProxyConfig

    proxy_config = ProxyConfig()

    # Mock database prompt version 1
    mock_prompt_v1 = MagicMock()
    mock_prompt_v1.model_dump.return_value = {
        "id": "uuid-1",
        "prompt_id": "chat_prompt",
        "version": 1,
        "litellm_params": '{"prompt_id": "chat_prompt", "prompt_integration": "dotprompt", "model": "gpt-3.5-turbo", "messages": [{"role": "user", "content": "v1 content"}]}',
        "prompt_info": '{"prompt_type": "db"}',
        "created_at": "2024-01-01T00:00:00",
        "updated_at": "2024-01-01T00:00:00",
    }

    # Mock database prompt version 2
    mock_prompt_v2 = MagicMock()
    mock_prompt_v2.model_dump.return_value = {
        "id": "uuid-2",
        "prompt_id": "chat_prompt",
        "version": 2,
        "litellm_params": '{"prompt_id": "chat_prompt", "prompt_integration": "dotprompt", "model": "gpt-4", "messages": [{"role": "user", "content": "v2 content"}]}',
        "prompt_info": '{"prompt_type": "db"}',
        "created_at": "2024-01-02T00:00:00",
        "updated_at": "2024-01-02T00:00:00",
    }

    # Test version 1
    prompt_spec_v1 = proxy_config._get_prompt_spec_for_db_prompt(db_prompt=mock_prompt_v1)
    assert prompt_spec_v1.prompt_id == "chat_prompt.v1"

    # Test version 2
    prompt_spec_v2 = proxy_config._get_prompt_spec_for_db_prompt(db_prompt=mock_prompt_v2)
    assert prompt_spec_v2.prompt_id == "chat_prompt.v2"

