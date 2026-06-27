import os
import sys
from unittest.mock import patch

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../.."))
)

import litellm
from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider
from litellm.llms.foundry_local.chat.transformation import FoundryLocalChatConfig


class TestFoundryLocalChatConfig:
    def test_should_resolve_provider_from_model_prefix(self):
        """get_llm_provider should return foundry_local for 'foundry_local/model' strings."""
        _, provider, _, _ = get_llm_provider("foundry_local/phi-3.5-mini")
        assert provider == "foundry_local"

    def test_should_return_fake_api_key_when_none_provided(self):
        """Foundry Local does not require auth; a fake key is returned for the OpenAI client."""
        config = FoundryLocalChatConfig()
        _, api_key = config._get_openai_compatible_provider_info(None, None)
        assert api_key == "fake-api-key"

    def test_should_use_explicit_api_key_when_provided(self):
        config = FoundryLocalChatConfig()
        _, api_key = config._get_openai_compatible_provider_info(None, "my-key")
        assert api_key == "my-key"

    def test_should_use_explicit_api_base_when_provided(self):
        config = FoundryLocalChatConfig()
        api_base, _ = config._get_openai_compatible_provider_info(
            "http://localhost:5272/v1", None
        )
        assert api_base == "http://localhost:5272/v1"

    def test_should_read_env_vars_for_api_base_and_key(self):
        config = FoundryLocalChatConfig()
        with patch.dict(
            "os.environ",
            {
                "FOUNDRY_LOCAL_API_BASE": "http://localhost:5272/v1",
                "FOUNDRY_LOCAL_API_KEY": "env-key",
            },
        ):
            api_base, api_key = config._get_openai_compatible_provider_info(None, None)
            assert api_base == "http://localhost:5272/v1"
            assert api_key == "env-key"

    def test_should_prefer_explicit_over_env_vars(self):
        config = FoundryLocalChatConfig()
        with patch.dict(
            "os.environ",
            {
                "FOUNDRY_LOCAL_API_BASE": "http://env-base:5272/v1",
                "FOUNDRY_LOCAL_API_KEY": "env-key",
            },
        ):
            api_base, api_key = config._get_openai_compatible_provider_info(
                "http://explicit:9999/v1", "explicit-key"
            )
            assert api_base == "http://explicit:9999/v1"
            assert api_key == "explicit-key"

    def test_should_be_in_openai_compatible_providers(self):
        """foundry_local must be listed as an OpenAI-compatible provider."""
        assert "foundry_local" in litellm.openai_compatible_providers

    def test_should_be_in_provider_list(self):
        """foundry_local must appear in the global provider list."""
        assert "foundry_local" in litellm.provider_list

    def test_should_resolve_provider_info_in_get_llm_provider(self):
        """get_llm_provider should resolve api_base and api_key via env vars."""
        with patch.dict(
            "os.environ",
            {
                "FOUNDRY_LOCAL_API_BASE": "http://localhost:5272/v1",
            },
        ):
            model, provider, dynamic_api_key, api_base = get_llm_provider(
                "foundry_local/phi-3.5-mini"
            )
            assert model == "phi-3.5-mini"
            assert provider == "foundry_local"
            assert api_base == "http://localhost:5272/v1"
            assert dynamic_api_key == "fake-api-key"
