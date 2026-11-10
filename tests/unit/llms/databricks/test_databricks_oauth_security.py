"""
Tests for security hardening of Databricks OAuth M2M credentials:
1. Alias normalization & scrub on api_base override in get_dynamic_litellm_params
2. is_request_body_safe rejection of secret references (os.environ/, oidc/) in OAuth credential fields
3. Transformation-level rejection of secret references in request optional_params
"""

import pytest
import re
from unittest.mock import MagicMock, patch

from litellm.litellm_core_utils.clientside_credential_handler import (
    get_dynamic_litellm_params,
)
from litellm.llms.databricks.chat.transformation import DatabricksConfig
from litellm.llms.databricks.common_utils import DatabricksException
from litellm.llms.databricks.embed.handler import DatabricksEmbeddingHandler
from litellm.llms.databricks.messages.transformation import (
    DatabricksAnthropicMessagesConfig,
)
from litellm.proxy.auth.auth_utils import is_request_body_safe


class TestDatabricksOAuthAliasScrubbing:
    """Security tests: OAuth aliases must be normalized and cleared on base override."""

    def test_clears_databricks_oauth_credentials_and_aliases_on_base_override(self):
        admin_params = {
            "model": "databricks/dbrx",
            "api_base": "https://admin-workspace.databricks.net",
            "databricks_client_id": "admin-cid",
            "databricks_client_secret": "admin-csec",
        }
        out = get_dynamic_litellm_params(
            litellm_params=dict(admin_params),
            request_kwargs={
                "api_base": "https://attacker.example/serving-endpoints",
                "api_key": "sk-caller",
            },
        )
        assert out["api_base"] == "https://attacker.example/serving-endpoints"
        assert "databricks_client_id" not in out
        assert "databricks_client_secret" not in out
        assert "client_id" not in out
        assert "client_secret" not in out

    def test_normalizes_databricks_oauth_aliases_without_base_override(self):
        admin_params = {
            "model": "databricks/dbrx",
            "api_base": "https://admin-workspace.databricks.net",
            "databricks_client_id": "admin-cid",
            "databricks_client_secret": "admin-csec",
        }
        out = get_dynamic_litellm_params(
            litellm_params=dict(admin_params),
            request_kwargs={},
        )
        assert out["client_id"] == "admin-cid"
        assert out["client_secret"] == "admin-csec"
        assert "databricks_client_id" not in out
        assert "databricks_client_secret" not in out


class TestDatabricksOAuthSecretReferenceRejection:
    """Security tests: request bodies and optional_params cannot supply secret references."""

    @pytest.mark.parametrize(
        "param",
        ["client_id", "client_secret", "databricks_client_id", "databricks_client_secret"],
    )
    @pytest.mark.parametrize(
        "secret_val",
        ["os.environ/MY_SECRET", "oidc/env/MY_SECRET"],
    )
    def test_is_request_body_safe_rejects_top_level(self, param: str, secret_val: str):
        with pytest.raises(ValueError, match=re.escape("holds an os.environ/ or oidc/ secret reference")):
            is_request_body_safe(
                request_body={"model": "databricks/dbrx", param: secret_val},
                general_settings={"allow_client_side_credentials": True},
                llm_router=None,
                model="databricks/dbrx",
            )

    @pytest.mark.parametrize(
        "param",
        ["client_id", "client_secret", "databricks_client_id", "databricks_client_secret"],
    )
    @pytest.mark.parametrize(
        "secret_val",
        ["os.environ/MY_SECRET", "oidc/env/MY_SECRET"],
    )
    def test_is_request_body_safe_rejects_extra_body(self, param: str, secret_val: str):
        with pytest.raises(ValueError, match=re.escape("holds an os.environ/ or oidc/ secret reference")):
            is_request_body_safe(
                request_body={"model": "databricks/dbrx", "extra_body": {param: secret_val}},
                general_settings={"allow_client_side_credentials": True},
                llm_router=None,
                model="databricks/dbrx",
            )

    @pytest.mark.parametrize(
        "param",
        ["client_id", "client_secret", "databricks_client_id", "databricks_client_secret"],
    )
    @pytest.mark.parametrize(
        "secret_val",
        ["os.environ/MY_SECRET", "oidc/env/MY_SECRET"],
    )
    def test_is_request_body_safe_rejects_litellm_params(self, param: str, secret_val: str):
        with pytest.raises(ValueError, match=re.escape("holds an os.environ/ or oidc/ secret reference")):
            is_request_body_safe(
                request_body={"model": "databricks/dbrx", "litellm_params": {param: secret_val}},
                general_settings={"allow_client_side_credentials": True},
                llm_router=None,
                model="databricks/dbrx",
            )

    def test_chat_transformation_rejects_request_secret_reference(self):
        config = DatabricksConfig()
        with pytest.raises(DatabricksException, match=re.escape("cannot be an os.environ/ or oidc/ secret reference")):
            config.validate_environment(
                headers={},
                model="databricks/dbrx",
                messages=[{"role": "user", "content": "hi"}],
                optional_params={"client_id": "os.environ/LEAK_KEY"},
                litellm_params={},
            )

    def test_embed_handler_rejects_request_secret_reference(self):
        handler = DatabricksEmbeddingHandler()
        with pytest.raises(DatabricksException, match=re.escape("cannot be an os.environ/ or oidc/ secret reference")):
            handler.embedding(
                model="databricks/bge-large-en",
                input=["hello"],
                timeout=10,
                logging_obj=MagicMock(),
                api_key=None,
                api_base="https://x.databricks.net",
                optional_params={"client_secret": "os.environ/LEAK_KEY"},
            )

    def test_messages_transformation_rejects_request_secret_reference(self):
        config = DatabricksAnthropicMessagesConfig()
        with pytest.raises(DatabricksException, match=re.escape("cannot be an os.environ/ or oidc/ secret reference")):
            config.validate_anthropic_messages_environment(
                headers={},
                model="databricks/claude-3-5-sonnet",
                messages=[{"role": "user", "content": "hi"}],
                optional_params={"databricks_client_id": "os.environ/LEAK_KEY"},
                litellm_params={},
            )

    @patch("litellm.llms.databricks.common_utils.DatabricksBase.get_oauth_m2m_token", return_value="mock_token")
    def test_chat_transformation_unconditionally_pops_credentials(self, mock_get_token):
        config = DatabricksConfig()
        optional_params = {
            "client_id": "req-id",
            "databricks_client_id": "req-db-id",
            "client_secret": "req-secret",
            "databricks_client_secret": "req-db-secret",
            "temperature": 0.7,
        }
        config.validate_environment(
            headers={},
            model="databricks/dbrx",
            messages=[{"role": "user", "content": "hi"}],
            optional_params=optional_params,
            litellm_params={"client_id": "dep-id", "client_secret": "dep-secret"},
            api_base="https://workspace.databricks.net",
        )
        assert "client_id" not in optional_params
        assert "databricks_client_id" not in optional_params
        assert "client_secret" not in optional_params
        assert "databricks_client_secret" not in optional_params
        assert optional_params["temperature"] == 0.7

    @patch("litellm.llms.databricks.common_utils.DatabricksBase.get_oauth_m2m_token", return_value="mock_token")
    def test_embed_handler_unconditionally_pops_credentials(self, mock_get_token):
        handler = DatabricksEmbeddingHandler()
        optional_params = {
            "client_id": "opt-id",
            "databricks_client_id": "opt-db-id",
            "client_secret": "opt-secret",
            "databricks_client_secret": "opt-db-secret",
            "dimensions": 512,
        }
        kwargs = {
            "client_id": "kw-id",
            "databricks_client_id": "kw-db-id",
            "client_secret": "kw-secret",
            "databricks_client_secret": "kw-db-secret",
        }
        with patch(
            "litellm.llms.openai_like.embedding.handler.OpenAILikeEmbeddingHandler.embedding",
            return_value=MagicMock(),
        ) as mock_super_embed:
            handler.embedding(
                model="databricks/bge-large-en",
                input=["hello"],
                timeout=10,
                logging_obj=MagicMock(),
                api_key=None,
                api_base="https://x.databricks.net",
                optional_params=optional_params,
                litellm_params={"client_id": "dep-id", "client_secret": "dep-secret"},
                **kwargs,
            )
        assert "client_id" not in optional_params
        assert "databricks_client_id" not in optional_params
        assert "client_secret" not in optional_params
        assert "databricks_client_secret" not in optional_params
        assert optional_params["dimensions"] == 512
        passed_kwargs = mock_super_embed.call_args.kwargs
        assert "client_id" not in passed_kwargs
        assert "databricks_client_id" not in passed_kwargs
        assert "client_secret" not in passed_kwargs
        assert "databricks_client_secret" not in passed_kwargs

    @patch("litellm.llms.databricks.common_utils.DatabricksBase.get_oauth_m2m_token", return_value="mock_token")
    def test_messages_transformation_unconditionally_pops_credentials(self, mock_get_token):
        config = DatabricksAnthropicMessagesConfig()
        optional_params = {
            "client_id": "req-id",
            "databricks_client_id": "req-db-id",
            "client_secret": "req-secret",
            "databricks_client_secret": "req-db-secret",
            "max_tokens": 100,
        }
        config.validate_anthropic_messages_environment(
            headers={},
            model="databricks/claude-3-5-sonnet",
            messages=[{"role": "user", "content": "hi"}],
            optional_params=optional_params,
            litellm_params={"client_id": "dep-id", "client_secret": "dep-secret"},
            api_base="https://workspace.databricks.net",
        )
        assert "client_id" not in optional_params
        assert "databricks_client_id" not in optional_params
        assert "client_secret" not in optional_params
        assert "databricks_client_secret" not in optional_params
        assert optional_params["max_tokens"] == 100
