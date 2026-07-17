"""
Unit tests for Amazon Bedrock Mantle Responses API configuration.

Mantle serves Responses on two paths: gpt frontier models on
`/openai/v1/responses` and other Responses-capable models (e.g. gpt-oss) on the
standard `/v1/responses`. These tests lock the per-model path selection in the
gate, the URL construction for both paths, and the shared Bearer auth.
"""

import copy
import os
import sys

sys.path.insert(0, os.path.abspath("../../../../.."))

import pytest
from botocore.exceptions import (
    ConnectTimeoutError,
    PartialCredentialsError,
    ProfileNotFound,
)

import litellm
from litellm.llms.bedrock_mantle.responses.transformation import (
    BedrockMantleResponsesAPIConfig,
)
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import LlmProviders


class TestBedrockMantleResponsesURL:
    def test_url_uses_region_from_env(self, monkeypatch):
        monkeypatch.setenv("BEDROCK_MANTLE_REGION", "us-east-2")
        monkeypatch.delenv("BEDROCK_MANTLE_API_BASE", raising=False)
        cfg = BedrockMantleResponsesAPIConfig()
        url = cfg.get_complete_url(api_base=None, litellm_params={})
        assert url == "https://bedrock-mantle.us-east-2.api.aws/openai/v1/responses"

    def test_url_normalizes_v1_suffix(self, monkeypatch):
        monkeypatch.delenv("BEDROCK_MANTLE_API_BASE", raising=False)
        cfg = BedrockMantleResponsesAPIConfig()
        url = cfg.get_complete_url(
            api_base="https://bedrock-mantle.us-east-2.api.aws/v1",
            litellm_params={},
        )
        assert url == "https://bedrock-mantle.us-east-2.api.aws/openai/v1/responses"
        assert "/v1/openai/v1/responses" not in url
        url_trailing = cfg.get_complete_url(
            api_base="https://bedrock-mantle.us-east-2.api.aws/v1/",
            litellm_params={},
        )
        assert (
            url_trailing
            == "https://bedrock-mantle.us-east-2.api.aws/openai/v1/responses"
        )

    def test_url_does_not_double_openai_v1(self, monkeypatch):
        monkeypatch.delenv("BEDROCK_MANTLE_API_BASE", raising=False)
        cfg = BedrockMantleResponsesAPIConfig()
        url = cfg.get_complete_url(
            api_base="https://bedrock-mantle.us-east-2.api.aws/openai/v1",
            litellm_params={},
        )
        assert url == "https://bedrock-mantle.us-east-2.api.aws/openai/v1/responses"

    def test_url_full_endpoint_base_not_doubled(self, monkeypatch):
        # AWS model card tells users to set OPENAI_BASE_URL to the full endpoint.
        # If copied into api_base, it must not be doubled.
        monkeypatch.delenv("BEDROCK_MANTLE_API_BASE", raising=False)
        cfg = BedrockMantleResponsesAPIConfig()
        url = cfg.get_complete_url(
            api_base="https://bedrock-mantle.us-east-2.api.aws/openai/v1/responses",
            litellm_params={},
        )
        assert url == "https://bedrock-mantle.us-east-2.api.aws/openai/v1/responses"
        assert url.count("/responses") == 1

    def test_url_region_fallback_to_aws_region(self, monkeypatch):
        monkeypatch.delenv("BEDROCK_MANTLE_REGION", raising=False)
        monkeypatch.delenv("BEDROCK_MANTLE_API_BASE", raising=False)
        monkeypatch.setenv("AWS_REGION", "us-west-2")
        cfg = BedrockMantleResponsesAPIConfig()
        url = cfg.get_complete_url(api_base=None, litellm_params={})
        assert url == "https://bedrock-mantle.us-west-2.api.aws/openai/v1/responses"

    def test_url_region_from_aws_region_name_litellm_params(self, monkeypatch):
        monkeypatch.delenv("BEDROCK_MANTLE_REGION", raising=False)
        monkeypatch.delenv("BEDROCK_MANTLE_API_BASE", raising=False)
        monkeypatch.delenv("AWS_REGION", raising=False)
        monkeypatch.setenv("AWS_REGION", "us-west-2")
        cfg = BedrockMantleResponsesAPIConfig()
        url = cfg.get_complete_url(
            api_base=None,
            litellm_params={"aws_region_name": "us-east-2"},
        )
        assert url == "https://bedrock-mantle.us-east-2.api.aws/openai/v1/responses"

    def test_url_aws_region_name_overrides_env_region(self, monkeypatch):
        monkeypatch.setenv("BEDROCK_MANTLE_REGION", "us-west-2")
        cfg = BedrockMantleResponsesAPIConfig()
        url = cfg.get_complete_url(
            api_base=None,
            litellm_params={"aws_region_name": "us-east-2"},
        )
        assert url == "https://bedrock-mantle.us-east-2.api.aws/openai/v1/responses"

    def test_url_rejects_malicious_aws_region_name(self, monkeypatch):
        monkeypatch.delenv("BEDROCK_MANTLE_REGION", raising=False)
        monkeypatch.delenv("BEDROCK_MANTLE_API_BASE", raising=False)
        monkeypatch.delenv("AWS_REGION", raising=False)
        cfg = BedrockMantleResponsesAPIConfig()
        with pytest.raises(ValueError):
            cfg.get_complete_url(
                api_base=None,
                litellm_params={
                    "aws_region_name": "us-east-1.api.aws.attacker.example/"
                },
            )

    def test_url_region_default_us_east_1(self, monkeypatch):
        monkeypatch.delenv("BEDROCK_MANTLE_REGION", raising=False)
        monkeypatch.delenv("BEDROCK_MANTLE_API_BASE", raising=False)
        monkeypatch.delenv("AWS_REGION", raising=False)
        cfg = BedrockMantleResponsesAPIConfig()
        url = cfg.get_complete_url(api_base=None, litellm_params={})
        assert url == "https://bedrock-mantle.us-east-1.api.aws/openai/v1/responses"

    def test_standard_path_uses_region_from_env(self, monkeypatch):
        monkeypatch.setenv("BEDROCK_MANTLE_REGION", "us-east-2")
        monkeypatch.delenv("BEDROCK_MANTLE_API_BASE", raising=False)
        cfg = BedrockMantleResponsesAPIConfig(use_openai_path=False)
        url = cfg.get_complete_url(api_base=None, litellm_params={})
        assert url == "https://bedrock-mantle.us-east-2.api.aws/v1/responses"
        assert "/openai/v1/responses" not in url

    def test_standard_path_normalizes_v1_base(self, monkeypatch):
        monkeypatch.delenv("BEDROCK_MANTLE_API_BASE", raising=False)
        cfg = BedrockMantleResponsesAPIConfig(use_openai_path=False)
        url = cfg.get_complete_url(
            api_base="https://bedrock-mantle.us-east-2.api.aws/v1",
            litellm_params={},
        )
        assert url == "https://bedrock-mantle.us-east-2.api.aws/v1/responses"
        assert url.count("/responses") == 1
        assert "/v1/v1/responses" not in url

    def test_standard_path_full_endpoint_base_not_doubled(self, monkeypatch):
        monkeypatch.delenv("BEDROCK_MANTLE_API_BASE", raising=False)
        cfg = BedrockMantleResponsesAPIConfig(use_openai_path=False)
        url = cfg.get_complete_url(
            api_base="https://bedrock-mantle.us-east-2.api.aws/v1/responses",
            litellm_params={},
        )
        assert url == "https://bedrock-mantle.us-east-2.api.aws/v1/responses"
        assert url.count("/responses") == 1

    def test_default_construction_keeps_openai_path(self, monkeypatch):
        monkeypatch.setenv("BEDROCK_MANTLE_REGION", "us-east-2")
        monkeypatch.delenv("BEDROCK_MANTLE_API_BASE", raising=False)
        cfg = BedrockMantleResponsesAPIConfig()
        url = cfg.get_complete_url(api_base=None, litellm_params={})
        assert url == "https://bedrock-mantle.us-east-2.api.aws/openai/v1/responses"

    def test_url_aws_region_name_overrides_stale_api_base(self, monkeypatch):
        monkeypatch.delenv("BEDROCK_MANTLE_REGION", raising=False)
        monkeypatch.delenv("BEDROCK_MANTLE_API_BASE", raising=False)
        monkeypatch.delenv("AWS_REGION", raising=False)
        cfg = BedrockMantleResponsesAPIConfig()
        url = cfg.get_complete_url(
            api_base="https://bedrock-mantle.us-east-1.api.aws/v1",
            litellm_params={"aws_region_name": "us-east-2"},
        )
        assert url == "https://bedrock-mantle.us-east-2.api.aws/openai/v1/responses"


class TestBedrockMantleGetLlmProviderRegion:
    def test_get_llm_provider_uses_supplemental_litellm_params(
        self, monkeypatch, local_cost_map
    ):
        monkeypatch.delenv("BEDROCK_MANTLE_REGION", raising=False)
        monkeypatch.delenv("BEDROCK_MANTLE_API_BASE", raising=False)
        monkeypatch.delenv("AWS_REGION", raising=False)
        from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider
        from litellm.types.router import GenericLiteLLMParams

        _, provider, _, api_base = get_llm_provider(
            model="bedrock_mantle/openai.gpt-5.5",
            api_key="test-key",
            litellm_params=GenericLiteLLMParams(aws_region_name="us-east-2"),
        )
        assert provider == "bedrock_mantle"
        # gpt-5.x carries use_openai_responses_path, so its whole surface (incl.
        # the resolved chat base) is on the /openai/v1 base per the AWS card.
        assert api_base == "https://bedrock-mantle.us-east-2.api.aws/openai/v1"

    def test_get_llm_provider_uses_aws_region_from_litellm_params(
        self, monkeypatch, local_cost_map
    ):
        monkeypatch.delenv("BEDROCK_MANTLE_REGION", raising=False)
        monkeypatch.delenv("BEDROCK_MANTLE_API_BASE", raising=False)
        monkeypatch.delenv("AWS_REGION", raising=False)
        from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider
        from litellm.types.router import GenericLiteLLMParams

        params = GenericLiteLLMParams(
            custom_llm_provider="bedrock_mantle",
            aws_region_name="us-east-2",
        )
        _, provider, _, api_base = get_llm_provider(
            model="bedrock_mantle/openai.gpt-5.5",
            litellm_params=params,
        )
        assert provider == "bedrock_mantle"
        assert api_base == "https://bedrock-mantle.us-east-2.api.aws/openai/v1"


class TestBedrockMantleResponsesAuth:
    def test_config_api_key_takes_priority(self, monkeypatch):
        monkeypatch.setenv("BEDROCK_MANTLE_API_KEY", "env-key")
        cfg = BedrockMantleResponsesAPIConfig()
        headers = cfg.validate_environment(
            headers={},
            model="openai.gpt-5.5",
            litellm_params=GenericLiteLLMParams(api_key="config-key"),
        )
        assert headers["Authorization"] == "Bearer config-key"

    def test_env_key_fallback(self, monkeypatch):
        monkeypatch.setenv("BEDROCK_MANTLE_API_KEY", "env-key")
        monkeypatch.delenv("AWS_BEARER_TOKEN_BEDROCK", raising=False)
        cfg = BedrockMantleResponsesAPIConfig()
        headers = cfg.validate_environment(
            headers={}, model="openai.gpt-5.5", litellm_params=GenericLiteLLMParams()
        )
        assert headers["Authorization"] == "Bearer env-key"

    def test_bedrock_bearer_token_fallback(self, monkeypatch):
        monkeypatch.delenv("BEDROCK_MANTLE_API_KEY", raising=False)
        monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "bearer-key")
        cfg = BedrockMantleResponsesAPIConfig()
        headers = cfg.validate_environment(
            headers={}, model="openai.gpt-5.5", litellm_params=GenericLiteLLMParams()
        )
        assert headers["Authorization"] == "Bearer bearer-key"

    def test_missing_bearer_does_not_raise_in_validate_environment(self, monkeypatch):
        # SigV4 may still apply, so validate_environment must defer instead of raising.
        monkeypatch.delenv("BEDROCK_MANTLE_API_KEY", raising=False)
        monkeypatch.delenv("AWS_BEARER_TOKEN_BEDROCK", raising=False)
        cfg = BedrockMantleResponsesAPIConfig()
        headers = cfg.validate_environment(
            headers={}, model="openai.gpt-5.5", litellm_params=GenericLiteLLMParams()
        )
        assert "Authorization" not in headers

    def test_project_id_sets_openai_project_header(self):
        cfg = BedrockMantleResponsesAPIConfig()
        headers = cfg.validate_environment(
            headers={},
            model="openai.gpt-5.5",
            litellm_params=GenericLiteLLMParams(
                api_key="fake-key", aws_bedrock_project_id="proj_abc123def456"
            ),
        )
        assert headers["OpenAI-Project"] == "proj_abc123def456"

    def test_no_project_id_no_openai_project_header(self):
        cfg = BedrockMantleResponsesAPIConfig()
        headers = cfg.validate_environment(
            headers={},
            model="openai.gpt-5.5",
            litellm_params=GenericLiteLLMParams(api_key="fake-key"),
        )
        assert "OpenAI-Project" not in headers

    def test_custom_llm_provider(self):
        cfg = BedrockMantleResponsesAPIConfig()
        assert cfg.custom_llm_provider == LlmProviders.BEDROCK_MANTLE

    def test_native_websocket_disabled(self):
        # Mantle Responses has no realtime/websocket transport, so the config
        # must opt out; otherwise realtime routing would try a socket Mantle
        # does not serve.
        cfg = BedrockMantleResponsesAPIConfig()
        assert cfg.supports_native_websocket() is False

    def test_file_search_routes_to_emulation(self):
        # Mantle cannot reach OpenAI's vector stores, so a native file_search
        # tool forwarded as-is gets a 400. The config must opt out of native
        # file_search so LiteLLM's emulation handles it instead of forwarding.
        from litellm.responses.file_search.emulated_handler import (
            should_use_emulated_file_search,
        )

        cfg = BedrockMantleResponsesAPIConfig()
        assert cfg.supports_native_file_search() is False
        assert (
            should_use_emulated_file_search(
                tools=[{"type": "file_search", "vector_store_ids": ["vs_1"]}],
                provider_config=cfg,
            )
            is True
        )

    def test_standard_path_still_uses_bearer_auth(self, monkeypatch):
        monkeypatch.setenv("BEDROCK_MANTLE_API_KEY", "env-key")
        monkeypatch.delenv("AWS_BEARER_TOKEN_BEDROCK", raising=False)
        cfg = BedrockMantleResponsesAPIConfig(use_openai_path=False)
        headers = cfg.validate_environment(
            headers={},
            model="openai.gpt-oss-120b",
            litellm_params=GenericLiteLLMParams(),
        )
        assert headers["Authorization"] == "Bearer env-key"

    def test_standard_path_opts_out_of_native_features(self):
        cfg = BedrockMantleResponsesAPIConfig(use_openai_path=False)
        assert cfg.supports_native_file_search() is False
        assert cfg.supports_native_websocket() is False


class TestBedrockMantleResponsesRequestBody:
    def test_standard_path_outbound_body_carries_bare_model(self):
        cfg = BedrockMantleResponsesAPIConfig(use_openai_path=False)
        body = cfg.transform_responses_api_request(
            model="openai.gpt-oss-120b",
            input="hello",
            response_api_optional_request_params={},
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )
        assert body["model"] == "openai.gpt-oss-120b"
        assert "input" in body


class TestBedrockMantleResponsesTools:
    def test_map_openai_params_drops_unsupported_tools(self):
        cfg = BedrockMantleResponsesAPIConfig()
        params = cfg.map_openai_params(
            response_api_optional_params={
                "tools": [
                    {"type": "web_search"},
                    {"type": "function", "name": "exec_command"},
                ]
            },
            model="openai.gpt-5.5",
            drop_params=False,
        )
        assert params["tools"] == [{"type": "function", "name": "exec_command"}]

    def test_map_openai_params_removes_tools_when_all_unsupported(self):
        cfg = BedrockMantleResponsesAPIConfig()
        params = cfg.map_openai_params(
            response_api_optional_params={"tools": [{"type": "web_search"}]},
            model="openai.gpt-5.5",
            drop_params=False,
        )
        assert "tools" not in params

    def test_dropped_tools_are_logged_at_warning_level(self):
        from unittest.mock import patch

        cfg = BedrockMantleResponsesAPIConfig()
        with patch(
            "litellm.llms.bedrock_mantle.responses.transformation.verbose_logger.warning"
        ) as mock_warning:
            cfg.map_openai_params(
                response_api_optional_params={"tools": [{"type": "web_search"}]},
                model="openai.gpt-5.5",
                drop_params=False,
            )
        assert mock_warning.call_count == 1
        assert "web_search" in str(mock_warning.call_args)


class TestBedrockMantleResponsesRegistry:
    def test_registry_returns_config_for_gpt_5_5(self, local_cost_map):
        # gpt-5.x advertises /v1/responses in supported_endpoints (capability)
        # and use_openai_responses_path (wire path), so it gets the native config
        # on the /openai/v1/responses path. local_cost_map loads the entry.
        from litellm.utils import ProviderConfigManager

        cfg = ProviderConfigManager.get_provider_responses_api_config(
            provider="bedrock_mantle",
            model="openai.gpt-5.5",
        )
        assert isinstance(cfg, BedrockMantleResponsesAPIConfig)
        assert cfg.use_openai_path is True

    def test_registry_returns_config_for_gpt_5_4_enum(self, local_cost_map):
        from litellm.utils import ProviderConfigManager

        cfg = ProviderConfigManager.get_provider_responses_api_config(
            provider=LlmProviders.BEDROCK_MANTLE,
            model="openai.gpt-5.4",
        )
        assert isinstance(cfg, BedrockMantleResponsesAPIConfig)
        assert cfg.use_openai_path is True

    @pytest.mark.parametrize(
        "model",
        ["openai.gpt-5.6-sol", "openai.gpt-5.6-terra", "openai.gpt-5.6-luna"],
    )
    def test_registry_returns_config_for_gpt_5_6_family(self, local_cost_map, model):
        from litellm.utils import ProviderConfigManager

        cfg = ProviderConfigManager.get_provider_responses_api_config(
            provider="bedrock_mantle",
            model=model,
        )
        assert isinstance(cfg, BedrockMantleResponsesAPIConfig)
        assert cfg.use_openai_path is True

    def test_registry_returns_native_config_for_gpt_oss(self, local_cost_map):
        # Core regression: gpt-oss-120b supports the native Responses API (AWS
        # model card), so it must get a BedrockMantleResponsesAPIConfig on the
        # STANDARD /v1/responses path -- NOT fall through to None / chat-completions
        # emulation. Driven by /v1/responses in its price-map supported_endpoints.
        # Fails on the old gate, which had no responses entry for gpt-oss.
        from litellm.utils import ProviderConfigManager

        cfg = ProviderConfigManager.get_provider_responses_api_config(
            provider="bedrock_mantle",
            model="openai.gpt-oss-120b",
        )
        assert isinstance(cfg, BedrockMantleResponsesAPIConfig)
        assert cfg.use_openai_path is False

    def test_registry_returns_native_config_for_gpt_oss_20b(self, local_cost_map):
        from litellm.utils import ProviderConfigManager

        cfg = ProviderConfigManager.get_provider_responses_api_config(
            provider="bedrock_mantle",
            model="openai.gpt-oss-20b",
        )
        assert isinstance(cfg, BedrockMantleResponsesAPIConfig)
        assert cfg.use_openai_path is False

    def test_registry_returns_none_for_gpt_oss_safeguard(self, local_cost_map):
        # Key discriminator: gpt-oss-safeguard shares the "gpt-oss" substring with
        # gpt-oss-120b but does NOT support Responses (AWS card), so it must return
        # None. Proves the gate is per-model (supported_endpoints) and not a naive
        # gpt-oss substring match. local_cost_map loads the chat-only entry.
        from litellm.utils import ProviderConfigManager

        for model in ("openai.gpt-oss-safeguard-120b", "openai.gpt-oss-safeguard-20b"):
            cfg = ProviderConfigManager.get_provider_responses_api_config(
                provider="bedrock_mantle",
                model=model,
            )
            assert cfg is None, model

    @pytest.mark.parametrize(
        "model",
        ["google.gemma-4-31b", "google.gemma-4-26b-a4b", "google.gemma-4-e2b"],
    )
    def test_registry_returns_native_config_for_gemma_4(self, local_cost_map, model):
        # All three gemma-4 models support Responses (AWS cards) on the /openai/v1
        # base, so each must get the native config with the openai path.
        from litellm.utils import ProviderConfigManager

        cfg = ProviderConfigManager.get_provider_responses_api_config(
            provider="bedrock_mantle",
            model=model,
        )
        assert isinstance(cfg, BedrockMantleResponsesAPIConfig)
        assert cfg.use_openai_path is True

    def test_registry_returns_native_config_for_xai_grok(self, local_cost_map):
        from litellm.utils import ProviderConfigManager

        cfg = ProviderConfigManager.get_provider_responses_api_config(
            provider="bedrock_mantle",
            model="xai.grok-4.3",
        )
        assert isinstance(cfg, BedrockMantleResponsesAPIConfig)
        # grok-4.3 is a third-party frontier model on Bedrock Mantle, served on the
        # /openai/v1 base (like gpt-5.x / gemma-4), not the standard /v1 path used by
        # open-weights models such as gpt-oss. The standard /v1 base returns
        # "Berm is not enabled for this account", so the price-map entry carries
        # use_openai_responses_path=true.
        assert cfg.use_openai_path is True

    def test_unmapped_frontier_model_falls_through_to_none(self, restore_model_cost):
        # The gate is data-driven, not name-based: an unseen model not yet in the
        # price map (e.g. a future gpt-6) has no capability signal, so it falls
        # through to None (chat-completions emulation) rather than being routed
        # natively by a model-name guess. Onboarding it is a JSON / register_model
        # change, never a code change (see the register_model tests below).
        from litellm.utils import ProviderConfigManager

        litellm.model_cost.pop("bedrock_mantle/openai.gpt-6", None)
        litellm.get_model_info.cache_clear()
        cfg = ProviderConfigManager.get_provider_responses_api_config(
            provider="bedrock_mantle",
            model="openai.gpt-6",
        )
        assert cfg is None

    def test_price_map_flag_routes_non_gpt_name_to_openai_path(
        self, restore_model_cost
    ):
        # Data-driven onboarding: a frontier model whose name does NOT match the
        # openai.gpt- convention can still be routed to /openai/v1/responses by
        # declaring use_openai_responses_path in its price-map entry, with no code
        # change. The string fallback alone could never catch this name.
        from litellm.utils import ProviderConfigManager, register_model

        register_model(
            {
                "bedrock_mantle/somelab.frontier-x": {
                    "litellm_provider": "bedrock_mantle",
                    "mode": "responses",
                    "use_openai_responses_path": True,
                }
            }
        )
        cfg = ProviderConfigManager.get_provider_responses_api_config(
            provider="bedrock_mantle",
            model="somelab.frontier-x",
        )
        assert isinstance(cfg, BedrockMantleResponsesAPIConfig)
        assert cfg.use_openai_path is True

    def test_gpt_5_5_price_map_declares_openai_responses_path(self, local_cost_map):
        # The gpt-5.x entries must carry the data-driven flag so frontier routing
        # does not rely on the name-string fallback alone.
        assert (
            litellm.model_cost["bedrock_mantle/openai.gpt-5.5"].get(
                "use_openai_responses_path"
            )
            is True
        )
        assert (
            litellm.model_cost["bedrock_mantle/openai.gpt-5.4"].get(
                "use_openai_responses_path"
            )
            is True
        )

    @pytest.mark.parametrize(
        "model",
        [
            "nvidia.nemotron-nano-9b-v2",
            "mistral.ministral-3-3b-instruct",
            "google.gemma-3-27b-it",
            "zai.glm-4.6",
        ],
    )
    def test_registry_returns_none_for_non_openai_models(self, model):
        # Regression for the chat-only families on Mantle. These models 400 on
        # /openai/v1/responses and are served on /v1/chat/completions, so the
        # registry must NOT hand them the Responses config; they fall through to
        # None and keep the chat-completions emulation.
        from litellm.utils import ProviderConfigManager

        cfg = ProviderConfigManager.get_provider_responses_api_config(
            provider="bedrock_mantle",
            model=model,
        )
        assert cfg is None

    def test_registry_returns_none_when_model_is_none(self):
        # By-id operations (delete/get/cancel) call with model=None; keep returning
        # None so those paths are unchanged.
        from litellm.utils import ProviderConfigManager

        cfg = ProviderConfigManager.get_provider_responses_api_config(
            provider="bedrock_mantle",
            model=None,
        )
        assert cfg is None

    def test_declared_responses_non_openai_routes_to_standard_path(
        self, restore_model_cost
    ):
        # New feature: a non-OpenAI model declared mode=responses (e.g. via a
        # user's proxy model_info block) must route to the STANDARD /v1/responses
        # path, not the frontier /openai/v1/responses path. Fails before the
        # path-aware gate exists (old gate returned None for non-gpt models).
        from litellm.utils import ProviderConfigManager, register_model

        register_model(
            {
                "bedrock_mantle/somelab.future-model": {
                    "litellm_provider": "bedrock_mantle",
                    "mode": "responses",
                }
            }
        )
        cfg = ProviderConfigManager.get_provider_responses_api_config(
            provider="bedrock_mantle",
            model="somelab.future-model",
        )
        assert isinstance(cfg, BedrockMantleResponsesAPIConfig)
        assert cfg.use_openai_path is False

    def test_gpt_oss_opt_in_routes_to_standard_path(self, restore_model_cost):
        # When a user opts gpt-oss into native Responses via model_info mode,
        # it must take the STANDARD /v1/responses path (gpt-oss Responses is on
        # /v1/responses, NOT the frontier /openai/v1/responses path).
        from litellm.utils import ProviderConfigManager, register_model

        register_model(
            {
                "bedrock_mantle/openai.gpt-oss-120b": {
                    "litellm_provider": "bedrock_mantle",
                    "mode": "responses",
                }
            }
        )
        cfg = ProviderConfigManager.get_provider_responses_api_config(
            provider="bedrock_mantle",
            model="openai.gpt-oss-120b",
        )
        assert isinstance(cfg, BedrockMantleResponsesAPIConfig)
        assert cfg.use_openai_path is False

    def test_unmapped_model_degrades_to_none_without_crashing(self, restore_model_cost):
        # A model absent from model_cost has no capability signal, so the gate
        # returns None (chat-completions emulation) rather than crashing.
        from litellm.utils import ProviderConfigManager

        litellm.model_cost.pop("bedrock_mantle/somelab.unmapped-model", None)
        litellm.get_model_info.cache_clear()
        cfg = ProviderConfigManager.get_provider_responses_api_config(
            provider="bedrock_mantle",
            model="somelab.unmapped-model",
        )
        assert cfg is None

    def test_register_model_restore_undoes_existing_key_overwrite(self):
        # Self-contained guard for the deepcopy requirement of restore_model_cost.
        # register_model overwrites an existing key by mutating its nested dict in
        # place, so the snapshot must be a deepcopy: a shallow dict() copy would
        # share that nested dict and leave mode=responses after restore, making
        # the final assertion fail. The in-place clear+update mirrors the fixture.
        # gpt-oss-safeguard is the right vehicle here: it is chat-only, so without
        # the registered mode=responses it resolves to None, isolating the effect
        # of the register/restore from the model's own (lack of) capability.
        from litellm.utils import ProviderConfigManager, register_model

        snapshot = copy.deepcopy(litellm.model_cost)
        litellm.get_model_info.cache_clear()
        try:
            register_model(
                {
                    "bedrock_mantle/openai.gpt-oss-safeguard-120b": {
                        "litellm_provider": "bedrock_mantle",
                        "mode": "responses",
                    }
                }
            )
            during = ProviderConfigManager.get_provider_responses_api_config(
                provider="bedrock_mantle", model="openai.gpt-oss-safeguard-120b"
            )
            assert isinstance(during, BedrockMantleResponsesAPIConfig)
        finally:
            litellm.model_cost.clear()
            litellm.model_cost.update(snapshot)
            litellm.get_model_info.cache_clear()
        after = ProviderConfigManager.get_provider_responses_api_config(
            provider="bedrock_mantle", model="openai.gpt-oss-safeguard-120b"
        )
        assert after is None


class TestMantleBaseSegment:
    """The wire-path helper is data-driven from the price-map
    use_openai_responses_path flag (NOT a model-name match): flagged models are on
    the /openai/v1 base, everything else on /v1. An unmapped model defaults to /v1.
    """

    @pytest.mark.parametrize(
        "model,model_cost,expected",
        [
            (
                "openai.gpt-5.5",
                {"bedrock_mantle/openai.gpt-5.5": {"use_openai_responses_path": True}},
                "openai/v1",
            ),
            (
                "google.gemma-4-31b",
                {
                    "bedrock_mantle/google.gemma-4-31b": {
                        "use_openai_responses_path": True
                    }
                },
                "openai/v1",
            ),
            (
                "openai.gpt-oss-120b",
                {"bedrock_mantle/openai.gpt-oss-120b": {}},
                "v1",
            ),
            ("openai.gpt-oss-120b", {}, "v1"),
            (None, {}, "v1"),
        ],
    )
    def test_base_segment(self, model, model_cost, expected):
        from litellm.llms.bedrock_mantle.common_utils import mantle_base_segment

        assert mantle_base_segment(model, model_cost) == expected


class TestMantleSupportsResponses:
    """The capability helper is data-driven (supported_endpoints / mode), with no
    model-name match: per-model, so gpt-oss-120b is supported but the safeguard
    variant is not despite the shared substring."""

    @pytest.mark.parametrize(
        "model,model_cost,expected",
        [
            # supported_endpoints lists responses -> supported
            (
                "openai.gpt-oss-120b",
                {
                    "bedrock_mantle/openai.gpt-oss-120b": {
                        "supported_endpoints": ["/v1/chat/completions", "/v1/responses"]
                    }
                },
                True,
            ),
            # chat-only supported_endpoints -> not supported (the discriminator)
            (
                "openai.gpt-oss-safeguard-120b",
                {
                    "bedrock_mantle/openai.gpt-oss-safeguard-120b": {
                        "supported_endpoints": ["/v1/chat/completions"]
                    }
                },
                False,
            ),
            # mode=responses (no supported_endpoints) -> supported
            (
                "somelab.future-model",
                {"bedrock_mantle/somelab.future-model": {"mode": "responses"}},
                True,
            ),
            # mode=chat, no responses endpoint -> not supported
            (
                "google.gemma-3-27b-it",
                {"bedrock_mantle/google.gemma-3-27b-it": {"mode": "chat"}},
                False,
            ),
            # absent from model_cost -> no signal -> not supported
            ("somelab.unmapped", {}, False),
            (None, {}, False),
        ],
    )
    def test_supports_responses(self, model, model_cost, expected):
        from litellm.llms.bedrock_mantle.common_utils import mantle_supports_responses

        assert mantle_supports_responses(model, model_cost) is expected


class TestBedrockMantlePerModelResponsesURL:
    """End-to-end: the registry-selected config must build the correct wire URL
    per model. gpt-oss on /v1/responses, gpt-5.x and gemma-4 on
    /openai/v1/responses."""

    def _url_for(self, model, region="us-east-2"):
        from litellm.utils import ProviderConfigManager

        cfg = ProviderConfigManager.get_provider_responses_api_config(
            provider="bedrock_mantle",
            model=model,
        )
        assert isinstance(cfg, BedrockMantleResponsesAPIConfig)
        return cfg.get_complete_url(
            api_base=None, litellm_params={"aws_region_name": region}
        )

    def test_gpt_oss_uses_standard_responses_path(self, local_cost_map):
        url = self._url_for("openai.gpt-oss-120b")
        assert url == "https://bedrock-mantle.us-east-2.api.aws/v1/responses"
        assert "/openai/v1/responses" not in url

    def test_gpt_5_5_uses_openai_responses_path(self, local_cost_map):
        url = self._url_for("openai.gpt-5.5")
        assert url == "https://bedrock-mantle.us-east-2.api.aws/openai/v1/responses"

    @pytest.mark.parametrize(
        "model",
        ["google.gemma-4-31b", "google.gemma-4-26b-a4b", "google.gemma-4-e2b"],
    )
    def test_gemma_4_uses_openai_responses_path(self, local_cost_map, model):
        url = self._url_for(model)
        assert url == "https://bedrock-mantle.us-east-2.api.aws/openai/v1/responses"


class TestBedrockMantleEndpointHonoring:
    def test_plain_chat_call_to_gpt_oss_is_not_bridged(self, local_cost_map):
        # Adding native Responses support to gpt-oss must NOT reroute its plain
        # chat-completions traffic. responses_api_bridge_check keys off mode, and
        # gpt-oss stays mode=chat, so a completion() call is not flipped to the
        # Responses API. Guards the dual-capability contract.
        from litellm.main import responses_api_bridge_check

        model_info, resolved_model = responses_api_bridge_check(
            model="openai.gpt-oss-120b",
            custom_llm_provider="bedrock_mantle",
        )
        assert model_info.get("mode") != "responses"
        assert resolved_model == "openai.gpt-oss-120b"


@pytest.fixture
def restore_model_cost():
    """Snapshot litellm.model_cost so register_model edits don't leak across tests.

    register_model mutates the global litellm.model_cost, and get_model_info is
    lru_cached, so without restore + cache_clear a registered model would bleed
    into sibling tests in the same process.

    Two subtleties make this fixture non-obvious:

    1. The snapshot must be a deepcopy. register_model overwrites an existing key
       via `litellm.model_cost.setdefault(key, {}).update(...)`, mutating the
       nested dict in place; a shallow copy would share those nested dicts and
       could not capture the pre-mutation values of an existing entry.
    2. The restore must be in place (clear + update the SAME dict object), not a
       reassignment. The conftest autouse `isolate_litellm_state` fixture
       snapshots `litellm.model_cost` by reference and restores that reference on
       its teardown, which runs after this one. Reassigning `litellm.model_cost`
       to a fresh dict here is undone when conftest reinstalls its (in-place
       mutated) reference, so the registered mode would leak and poison
       TestBedrockMantleResponsesPricing. Mutating the original object in place
       restores the contents conftest's reference points at.
    """
    original_model_cost = copy.deepcopy(litellm.model_cost)
    litellm.get_model_info.cache_clear()
    try:
        yield
    finally:
        litellm.model_cost.clear()
        litellm.model_cost.update(original_model_cost)
        litellm.get_model_info.cache_clear()


@pytest.fixture
def local_cost_map(monkeypatch):
    """Force the bundled backup cost map and re-derive the provider model sets.

    ``litellm.model_cost`` is populated once at import time (here, from the
    network-fetched ``main`` copy, which lags this branch). ``add_known_models``
    only re-buckets whatever is already in ``model_cost``, so the cost map must
    first be reloaded from the local backup before the new keys appear.
    """
    original_model_cost = litellm.model_cost
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "true")
    litellm.model_cost = litellm.get_model_cost_map(url="")
    litellm.get_model_info.cache_clear()
    litellm.add_known_models()
    try:
        yield
    finally:
        litellm.model_cost = original_model_cost
        litellm.get_model_info.cache_clear()


class TestBedrockMantleResponsesSigV4:
    def test_bearer_short_circuits_without_credentials(self, monkeypatch):
        from unittest.mock import MagicMock
        from litellm.llms.bedrock.base_aws_llm import BaseAWSLLM

        monkeypatch.delenv("AWS_BEARER_TOKEN_BEDROCK", raising=False)
        monkeypatch.delenv("BEDROCK_MANTLE_API_KEY", raising=False)

        signer = BaseAWSLLM()
        signer.get_credentials = MagicMock(
            side_effect=AssertionError("get_credentials must not run for bearer auth")
        )
        cfg = BedrockMantleResponsesAPIConfig(aws_signer=signer)

        headers, signed_body = cfg.sign_request(
            headers={},
            optional_params={},
            request_data={"input": "hi"},
            api_base="https://bedrock-mantle.us-east-2.api.aws/openai/v1/responses",
            api_key="bearer-from-config",
        )
        assert headers["Authorization"] == "Bearer bearer-from-config"
        assert signed_body == b'{"input": "hi"}'
        signer.get_credentials.assert_not_called()

    def test_bearer_resolved_from_mantle_env_key(self, monkeypatch):
        from unittest.mock import MagicMock
        from litellm.llms.bedrock.base_aws_llm import BaseAWSLLM

        monkeypatch.delenv("AWS_BEARER_TOKEN_BEDROCK", raising=False)
        monkeypatch.setenv("BEDROCK_MANTLE_API_KEY", "env-bearer")

        signer = BaseAWSLLM()
        signer.get_credentials = MagicMock(
            side_effect=AssertionError("get_credentials must not run for bearer auth")
        )
        cfg = BedrockMantleResponsesAPIConfig(aws_signer=signer)

        headers, _ = cfg.sign_request(
            headers={},
            optional_params={},
            request_data={"input": "hi"},
            api_base="https://bedrock-mantle.us-east-2.api.aws/openai/v1/responses",
            api_key=None,
        )
        assert headers["Authorization"] == "Bearer env-bearer"

    def test_bearer_arg_takes_priority_over_mantle_env_key(self, monkeypatch):
        # The passed api_key (e.g. litellm_params.api_key) must win over the env
        # bearer; a reordered precedence chain would silently use the wrong token.
        from unittest.mock import MagicMock
        from litellm.llms.bedrock.base_aws_llm import BaseAWSLLM

        monkeypatch.delenv("AWS_BEARER_TOKEN_BEDROCK", raising=False)
        monkeypatch.setenv("BEDROCK_MANTLE_API_KEY", "env-bearer")

        signer = BaseAWSLLM()
        signer.get_credentials = MagicMock(
            side_effect=AssertionError("get_credentials must not run for bearer auth")
        )
        cfg = BedrockMantleResponsesAPIConfig(aws_signer=signer)

        headers, _ = cfg.sign_request(
            headers={},
            optional_params={},
            request_data={"input": "hi"},
            api_base="https://bedrock-mantle.us-east-2.api.aws/openai/v1/responses",
            api_key="arg-bearer",
        )
        assert headers["Authorization"] == "Bearer arg-bearer"
        signer.get_credentials.assert_not_called()

    def test_access_key_produces_sigv4_headers(self, monkeypatch):
        from litellm.llms.bedrock.base_aws_llm import BaseAWSLLM

        monkeypatch.delenv("AWS_BEARER_TOKEN_BEDROCK", raising=False)
        monkeypatch.delenv("BEDROCK_MANTLE_API_KEY", raising=False)

        cfg = BedrockMantleResponsesAPIConfig(aws_signer=BaseAWSLLM())
        headers, signed_body = cfg.sign_request(
            headers={},
            optional_params={
                "aws_access_key_id": "AKIAEXAMPLE",
                "aws_secret_access_key": "c2VjcmV0LXRlc3Qtc2VjcmV0LXRlc3Qtc2VjcmV0",
                "aws_session_token": "session-token-test",
                "aws_region_name": "us-east-2",
            },
            request_data={"input": "hi"},
            api_base="https://bedrock-mantle.us-east-2.api.aws/openai/v1/responses",
            api_key=None,
        )
        assert headers["Authorization"].startswith("AWS4-HMAC-SHA256")
        assert "Credential=AKIAEXAMPLE/" in headers["Authorization"]
        assert "/us-east-2/bedrock/aws4_request" in headers["Authorization"]
        assert "X-Amz-Date" in headers
        assert headers["X-Amz-Security-Token"] == "session-token-test"
        assert signed_body == b'{"input": "hi"}'

    def test_assume_role_path_produces_sigv4_headers(self, monkeypatch):
        from unittest.mock import MagicMock
        from botocore.credentials import Credentials
        from litellm.llms.bedrock.base_aws_llm import BaseAWSLLM

        monkeypatch.delenv("AWS_BEARER_TOKEN_BEDROCK", raising=False)
        monkeypatch.delenv("BEDROCK_MANTLE_API_KEY", raising=False)

        signer = BaseAWSLLM()
        signer.get_credentials = MagicMock(
            return_value=Credentials(
                access_key="ASIAEXAMPLE",
                secret_key="YXNzdW1lZC1yb2xlLXNlY3JldC1hc3N1bWVk",
                token="assumed-session-token",
            )
        )
        cfg = BedrockMantleResponsesAPIConfig(aws_signer=signer)

        headers, _ = cfg.sign_request(
            headers={},
            optional_params={
                "aws_role_name": "arn:aws:iam::000000000000:role/test-role",
                "aws_session_name": "litellm-test",
                "aws_region_name": "us-east-2",
            },
            request_data={"input": "hi"},
            api_base="https://bedrock-mantle.us-east-2.api.aws/openai/v1/responses",
            api_key=None,
        )
        signer.get_credentials.assert_called_once()
        call = signer.get_credentials.call_args.kwargs
        assert call["aws_role_name"] == "arn:aws:iam::000000000000:role/test-role"
        assert call["aws_session_name"] == "litellm-test"
        assert headers["Authorization"].startswith("AWS4-HMAC-SHA256")
        assert "/us-east-2/bedrock/aws4_request" in headers["Authorization"]

    def test_signed_body_matches_final_data_after_normalize(self, monkeypatch):
        """Core regression: the signed bytes must equal the bytes actually sent.

        Sign the *final* data dict and assert the returned signed_body decodes to
        exactly that dict, so a later change to the data would break the SigV4 hash.
        """
        import json
        from litellm.llms.bedrock.base_aws_llm import BaseAWSLLM

        monkeypatch.delenv("AWS_BEARER_TOKEN_BEDROCK", raising=False)
        monkeypatch.delenv("BEDROCK_MANTLE_API_KEY", raising=False)

        final_data = {"model": "openai.gpt-5.5", "input": "hi", "max_output_tokens": 16}
        cfg = BedrockMantleResponsesAPIConfig(aws_signer=BaseAWSLLM())
        _, signed_body = cfg.sign_request(
            headers={},
            optional_params={
                "aws_access_key_id": "AKIAEXAMPLE",
                "aws_secret_access_key": "c2VjcmV0LXRlc3Qtc2VjcmV0LXRlc3Qtc2VjcmV0",
                "aws_region_name": "us-east-2",
            },
            request_data=final_data,
            api_base="https://bedrock-mantle.us-east-2.api.aws/openai/v1/responses",
            api_key=None,
        )
        assert signed_body is not None
        assert json.loads(signed_body) == final_data

    def test_region_comes_from_optional_params(self, monkeypatch):
        from litellm.llms.bedrock.base_aws_llm import BaseAWSLLM

        monkeypatch.delenv("AWS_BEARER_TOKEN_BEDROCK", raising=False)
        monkeypatch.delenv("BEDROCK_MANTLE_API_KEY", raising=False)
        monkeypatch.delenv("AWS_REGION", raising=False)
        monkeypatch.delenv("AWS_REGION_NAME", raising=False)

        cfg = BedrockMantleResponsesAPIConfig(aws_signer=BaseAWSLLM())
        headers, _ = cfg.sign_request(
            headers={},
            optional_params={
                "aws_access_key_id": "AKIAEXAMPLE",
                "aws_secret_access_key": "c2VjcmV0LXRlc3Qtc2VjcmV0LXRlc3Qtc2VjcmV0",
                "aws_region_name": "eu-west-1",
            },
            request_data={"input": "hi"},
            api_base="https://bedrock-mantle.eu-west-1.api.aws/openai/v1/responses",
            api_key=None,
        )
        assert "/eu-west-1/bedrock/aws4_request" in headers["Authorization"]

    def test_url_region_and_sigv4_region_agree_from_litellm_params(self, monkeypatch):
        """Adversarial-review regression: a caller-supplied aws_region_name (no region
        env set) must shape BOTH the URL host and the SigV4 credential scope, or the
        request is signed for one region and sent to another -> 401.
        """
        monkeypatch.delenv("BEDROCK_MANTLE_REGION", raising=False)
        monkeypatch.delenv("BEDROCK_MANTLE_API_BASE", raising=False)
        monkeypatch.delenv("AWS_REGION", raising=False)
        monkeypatch.delenv("AWS_REGION_NAME", raising=False)
        monkeypatch.delenv("AWS_BEARER_TOKEN_BEDROCK", raising=False)
        monkeypatch.delenv("BEDROCK_MANTLE_API_KEY", raising=False)

        from litellm.llms.bedrock.base_aws_llm import BaseAWSLLM

        params = {
            "aws_region_name": "ap-southeast-2",
            "aws_access_key_id": "AKIAEXAMPLE",
            "aws_secret_access_key": "c2VjcmV0LXRlc3Qtc2VjcmV0LXRlc3Qtc2VjcmV0",
        }
        cfg = BedrockMantleResponsesAPIConfig(aws_signer=BaseAWSLLM())
        url = cfg.get_complete_url(api_base=None, litellm_params=params)
        assert (
            url == "https://bedrock-mantle.ap-southeast-2.api.aws/openai/v1/responses"
        )

        headers, _ = cfg.sign_request(
            headers={},
            optional_params=params,
            request_data={"input": "hi"},
            api_base=url,
            api_key=None,
        )
        assert "/ap-southeast-2/bedrock/aws4_request" in headers["Authorization"]

    def test_injected_default_region_base_does_not_override_aws_region_name(
        self, monkeypatch
    ):
        """2nd-round adversarial regression: responses/main.py auto-injects
        litellm_params.api_base = https://bedrock-mantle.<DEFAULT>.api.aws/v1 (default
        region, ignoring aws_region_name). The config must still pin BOTH the URL host
        and the SigV4 scope to aws_region_name, or the IAM deployment 401s. A naive
        'resolve region only when api_base is None' fix would fail this test.
        """
        monkeypatch.delenv("BEDROCK_MANTLE_REGION", raising=False)
        monkeypatch.delenv("BEDROCK_MANTLE_API_BASE", raising=False)
        monkeypatch.delenv("AWS_REGION", raising=False)
        monkeypatch.delenv("AWS_REGION_NAME", raising=False)
        monkeypatch.delenv("AWS_BEARER_TOKEN_BEDROCK", raising=False)
        monkeypatch.delenv("BEDROCK_MANTLE_API_KEY", raising=False)

        from litellm.llms.bedrock.base_aws_llm import BaseAWSLLM

        injected_base = "https://bedrock-mantle.us-east-1.api.aws/v1"  # default region
        params = {
            "aws_region_name": "us-east-2",  # what the caller actually wants
            "api_base": injected_base,
            "aws_access_key_id": "AKIAEXAMPLE",
            "aws_secret_access_key": "c2VjcmV0LXRlc3Qtc2VjcmV0LXRlc3Qtc2VjcmV0",
        }
        cfg = BedrockMantleResponsesAPIConfig(aws_signer=BaseAWSLLM())
        url = cfg.get_complete_url(api_base=injected_base, litellm_params=params)
        assert url == "https://bedrock-mantle.us-east-2.api.aws/openai/v1/responses"

        headers, _ = cfg.sign_request(
            headers={},
            optional_params=params,
            request_data={"input": "hi"},
            api_base=url,
            api_key=None,
        )
        assert "/us-east-2/bedrock/aws4_request" in headers["Authorization"]
        assert "us-east-1" not in headers["Authorization"]

    def test_custom_proxy_host_is_preserved(self, monkeypatch):
        """A genuinely custom (non-Mantle) api_base host must be preserved, not rewritten
        to a bedrock-mantle host. Only standard Mantle hosts are region-pinned.
        """
        monkeypatch.delenv("BEDROCK_MANTLE_API_BASE", raising=False)
        cfg = BedrockMantleResponsesAPIConfig()
        url = cfg.get_complete_url(
            api_base="https://mantle-proxy.internal.example/openai/v1",
            litellm_params={"aws_region_name": "us-east-2"},
        )
        assert url == "https://mantle-proxy.internal.example/openai/v1/responses"

    def test_caller_authorization_does_not_override_sigv4(self, monkeypatch):
        """Adversarial-review regression: a caller-supplied Authorization header (e.g.
        from extra_headers, surviving the relaxed validate_environment) must not clobber
        the SigV4 Authorization that _sign_request would otherwise restore.
        """
        monkeypatch.delenv("AWS_BEARER_TOKEN_BEDROCK", raising=False)
        monkeypatch.delenv("BEDROCK_MANTLE_API_KEY", raising=False)

        from litellm.llms.bedrock.base_aws_llm import BaseAWSLLM

        cfg = BedrockMantleResponsesAPIConfig(aws_signer=BaseAWSLLM())
        headers, _ = cfg.sign_request(
            headers={"Authorization": "Bearer stale-caller-token"},
            optional_params={
                "aws_access_key_id": "AKIAEXAMPLE",
                "aws_secret_access_key": "c2VjcmV0LXRlc3Qtc2VjcmV0LXRlc3Qtc2VjcmV0",
                "aws_region_name": "us-east-2",
            },
            request_data={"input": "hi"},
            api_base="https://bedrock-mantle.us-east-2.api.aws/openai/v1/responses",
            api_key=None,
        )
        assert headers["Authorization"].startswith("AWS4-HMAC-SHA256")
        assert "Bearer stale-caller-token" not in headers["Authorization"]

    def test_no_bearer_and_no_credentials_raises_both_paths(self, monkeypatch):
        from unittest.mock import MagicMock
        from botocore.exceptions import NoCredentialsError
        from litellm.llms.bedrock.base_aws_llm import BaseAWSLLM

        monkeypatch.delenv("BEDROCK_MANTLE_API_KEY", raising=False)
        monkeypatch.delenv("AWS_BEARER_TOKEN_BEDROCK", raising=False)

        signer = BaseAWSLLM()
        signer.get_credentials = MagicMock(side_effect=NoCredentialsError())
        cfg = BedrockMantleResponsesAPIConfig(aws_signer=signer)

        with pytest.raises(ValueError) as exc:
            cfg.sign_request(
                headers={},
                optional_params={"aws_region_name": "us-east-2"},
                request_data={"input": "hi"},
                api_base="https://bedrock-mantle.us-east-2.api.aws/openai/v1/responses",
                api_key=None,
            )
        msg = str(exc.value)
        assert "Bearer" in msg
        assert "SigV4" in msg or "IAM" in msg

    @pytest.mark.parametrize(
        "cred_error",
        [
            PartialCredentialsError(provider="env", cred_var="aws_secret_access_key"),
            ProfileNotFound(profile="missing-profile"),
        ],
    )
    def test_partial_credentials_raises_both_paths(self, monkeypatch, cred_error):
        from unittest.mock import MagicMock
        from litellm.llms.bedrock.base_aws_llm import BaseAWSLLM

        monkeypatch.delenv("BEDROCK_MANTLE_API_KEY", raising=False)
        monkeypatch.delenv("AWS_BEARER_TOKEN_BEDROCK", raising=False)

        signer = BaseAWSLLM()
        signer.get_credentials = MagicMock(side_effect=cred_error)
        cfg = BedrockMantleResponsesAPIConfig(aws_signer=signer)

        with pytest.raises(ValueError) as exc:
            cfg.sign_request(
                headers={},
                optional_params={"aws_region_name": "us-east-2"},
                request_data={"input": "hi"},
                api_base="https://bedrock-mantle.us-east-2.api.aws/openai/v1/responses",
                api_key=None,
            )
        msg = str(exc.value)
        assert "Bearer" in msg
        assert "SigV4" in msg or "IAM" in msg

    def test_sts_transport_error_is_not_masked_as_credentials(self, monkeypatch):
        # An AssumeRole / web-identity flow hits STS over the network, so a transient
        # connection error must surface as itself, not be rewritten into the
        # "no usable AWS credentials" message that would send the user to fix the
        # wrong thing.
        from unittest.mock import MagicMock
        from litellm.llms.bedrock.base_aws_llm import BaseAWSLLM

        monkeypatch.delenv("BEDROCK_MANTLE_API_KEY", raising=False)
        monkeypatch.delenv("AWS_BEARER_TOKEN_BEDROCK", raising=False)

        signer = BaseAWSLLM()
        signer.get_credentials = MagicMock(
            side_effect=ConnectTimeoutError(
                endpoint_url="https://sts.us-east-2.amazonaws.com"
            )
        )
        cfg = BedrockMantleResponsesAPIConfig(aws_signer=signer)

        with pytest.raises(ConnectTimeoutError):
            cfg.sign_request(
                headers={},
                optional_params={
                    "aws_role_name": "arn:aws:iam::000000000000:role/test-role",
                    "aws_region_name": "us-east-2",
                },
                request_data={"input": "hi"},
                api_base="https://bedrock-mantle.us-east-2.api.aws/openai/v1/responses",
                api_key=None,
            )


class TestBedrockMantleResponsesPricing:
    def test_gpt_5_5_pricing_and_mode(self, local_cost_map):
        info = litellm.get_model_info("bedrock_mantle/openai.gpt-5.5")
        assert info["mode"] == "responses"
        assert info["input_cost_per_token"] == pytest.approx(5.5e-06)
        assert info["output_cost_per_token"] == pytest.approx(3.3e-05)
        assert info["cache_read_input_token_cost"] == pytest.approx(5.5e-07)
        assert info["max_input_tokens"] == 272000

    def test_gpt_5_4_pricing_and_mode(self, local_cost_map):
        info = litellm.get_model_info("bedrock_mantle/openai.gpt-5.4")
        assert info["mode"] == "responses"
        assert info["input_cost_per_token"] == pytest.approx(2.75e-06)
        assert info["output_cost_per_token"] == pytest.approx(1.65e-05)
        assert info["cache_read_input_token_cost"] == pytest.approx(2.75e-07)
        assert info["max_input_tokens"] == 272000

    @pytest.mark.parametrize(
        "model, input_cost, cache_creation_cost, cache_read_cost, output_cost",
        [
            ("openai.gpt-5.6-sol", 5.5e-06, 6.875e-06, 5.5e-07, 3.3e-05),
            ("openai.gpt-5.6-terra", 2.75e-06, 3.4375e-06, 2.75e-07, 1.65e-05),
            ("openai.gpt-5.6-luna", 1.1e-06, 1.375e-06, 1.1e-07, 6.6e-06),
        ],
    )
    def test_gpt_5_6_pricing_and_mode(
        self, local_cost_map, model, input_cost, cache_creation_cost, cache_read_cost, output_cost
    ):
        info = litellm.get_model_info(f"bedrock_mantle/{model}")
        assert info["mode"] == "responses"
        assert info["input_cost_per_token"] == pytest.approx(input_cost)
        assert info["cache_creation_input_token_cost"] == pytest.approx(cache_creation_cost)
        assert info["cache_read_input_token_cost"] == pytest.approx(cache_read_cost)
        assert info["output_cost_per_token"] == pytest.approx(output_cost)
        assert info["max_input_tokens"] == 272000

    def test_models_registered(self, local_cost_map):
        assert "bedrock_mantle/openai.gpt-5.5" in litellm.bedrock_mantle_models
        assert "bedrock_mantle/openai.gpt-5.4" in litellm.bedrock_mantle_models
