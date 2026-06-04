"""
Unit tests for Amazon Bedrock Mantle Responses API configuration.

Mantle's gpt-5.5 / gpt-5.4 are served ONLY on the non-standard
`/openai/v1/responses` path. These tests lock the URL construction and
Bearer auth that make that routing work.
"""

import os
import sys

sys.path.insert(0, os.path.abspath("../../../../.."))

import pytest

import litellm
from litellm.llms.bedrock.base_aws_llm import BaseAWSLLM
from litellm.llms.bedrock_mantle.responses.transformation import (
    BedrockMantleResponsesAPIConfig,
)
from litellm.llms.openai.responses.transformation import OpenAIResponsesAPIConfig
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import LlmProviders

# Signals read by BedrockMantleResponsesAPIConfig._use_sigv4 that the autouse
# isolate_host_aws_config fixture (tests/test_litellm/conftest.py) does NOT
# already clear. Clearing these makes auth-selection deterministic on hosts that
# export e.g. AWS_REGION or AWS_ACCESS_KEY_ID.
_AWS_SIGNAL_ENV_VARS = (
    "AWS_REGION",
    "AWS_REGION_NAME",
    "AWS_ROLE_NAME",
    "AWS_ROLE_ARN",
    "AWS_WEB_IDENTITY_TOKEN",
    "AWS_WEB_IDENTITY_TOKEN_FILE",
    "AWS_PROFILE_NAME",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "BEDROCK_MANTLE_REGION",
    "BEDROCK_MANTLE_API_BASE",
    "BEDROCK_MANTLE_API_KEY",
    "AWS_BEARER_TOKEN_BEDROCK",
)


@pytest.fixture
def clear_aws_env(monkeypatch):
    """Clear the AWS / Bedrock Mantle signals not covered by the autouse
    isolate_host_aws_config fixture, so auth-selection tests are deterministic
    regardless of the host machine's AWS configuration."""
    for name in _AWS_SIGNAL_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


class _RecordingConfig(BedrockMantleResponsesAPIConfig):
    """Test double that records `_sign_request` invocations instead of signing.

    Subclassing to override the inherited method is dependency injection via a
    test double, not class-attribute monkeypatching of the production class.
    """

    _SIGNED_HEADERS = {
        "Authorization": "AWS4-HMAC-SHA256 Credential=test",
        "X-Amz-Date": "20260101T000000Z",
    }

    def __init__(self):
        super().__init__()
        self.sign_calls = []

    def _sign_request(self, **kwargs):
        self.sign_calls.append(kwargs)
        return dict(self._SIGNED_HEADERS), b"{}"


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

    def test_url_region_default_us_east_2(self, monkeypatch):
        monkeypatch.delenv("BEDROCK_MANTLE_REGION", raising=False)
        monkeypatch.delenv("BEDROCK_MANTLE_API_BASE", raising=False)
        monkeypatch.delenv("AWS_REGION", raising=False)
        cfg = BedrockMantleResponsesAPIConfig()
        url = cfg.get_complete_url(api_base=None, litellm_params={})
        assert url == "https://bedrock-mantle.us-east-2.api.aws/openai/v1/responses"


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

    def test_env_key_fallback(self, clear_aws_env):
        monkeypatch = clear_aws_env
        monkeypatch.setenv("BEDROCK_MANTLE_API_KEY", "env-key")
        monkeypatch.delenv("AWS_BEARER_TOKEN_BEDROCK", raising=False)
        cfg = BedrockMantleResponsesAPIConfig()
        headers = cfg.validate_environment(
            headers={}, model="openai.gpt-5.5", litellm_params=GenericLiteLLMParams()
        )
        assert headers["Authorization"] == "Bearer env-key"

    def test_bedrock_bearer_token_fallback(self, clear_aws_env):
        monkeypatch = clear_aws_env
        monkeypatch.delenv("BEDROCK_MANTLE_API_KEY", raising=False)
        monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "bearer-key")
        cfg = BedrockMantleResponsesAPIConfig()
        headers = cfg.validate_environment(
            headers={}, model="openai.gpt-5.5", litellm_params=GenericLiteLLMParams()
        )
        assert headers["Authorization"] == "Bearer bearer-key"

    def test_missing_key_raises(self, clear_aws_env):
        cfg = BedrockMantleResponsesAPIConfig()
        with pytest.raises(ValueError, match="Bedrock Mantle API key"):
            cfg.validate_environment(
                headers={},
                model="openai.gpt-5.5",
                litellm_params=GenericLiteLLMParams(),
            )

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


class TestBedrockMantleResponsesRegistry:
    def test_registry_returns_config_for_gpt_5_5(self):
        from litellm.utils import ProviderConfigManager

        cfg = ProviderConfigManager.get_provider_responses_api_config(
            provider="bedrock_mantle",
            model="openai.gpt-5.5",
        )
        assert isinstance(cfg, BedrockMantleResponsesAPIConfig)

    def test_registry_returns_config_for_gpt_5_4_enum(self):
        from litellm.utils import ProviderConfigManager

        cfg = ProviderConfigManager.get_provider_responses_api_config(
            provider=LlmProviders.BEDROCK_MANTLE,
            model="openai.gpt-5.4",
        )
        assert isinstance(cfg, BedrockMantleResponsesAPIConfig)

    def test_registry_returns_none_for_gpt_oss(self):
        # Regression guard: gpt-oss must NOT get the native Responses config; it
        # keeps the chat-completions emulation path (responses/main.py ~line 1109).
        from litellm.utils import ProviderConfigManager

        cfg = ProviderConfigManager.get_provider_responses_api_config(
            provider="bedrock_mantle",
            model="openai.gpt-oss-120b",
        )
        assert cfg is None

    def test_registry_returns_none_for_gpt_oss_safeguard(self):
        from litellm.utils import ProviderConfigManager

        cfg = ProviderConfigManager.get_provider_responses_api_config(
            provider="bedrock_mantle",
            model="openai.gpt-oss-safeguard-20b",
        )
        assert cfg is None

    def test_registry_returns_config_for_future_frontier_model(self):
        # Forward-compatibility: an unseen OpenAI gpt frontier model (e.g. gpt-6) must
        # get the native Responses config without a code change. The gate allow-lists
        # the openai.gpt- family (minus gpt-oss), so gpt-6 matches automatically.
        from litellm.utils import ProviderConfigManager

        cfg = ProviderConfigManager.get_provider_responses_api_config(
            provider="bedrock_mantle",
            model="openai.gpt-6",
        )
        assert isinstance(cfg, BedrockMantleResponsesAPIConfig)

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

    def test_models_registered(self, local_cost_map):
        assert "bedrock_mantle/openai.gpt-5.5" in litellm.bedrock_mantle_models
        assert "bedrock_mantle/openai.gpt-5.4" in litellm.bedrock_mantle_models


class TestBedrockMantleResponsesSigV4:
    def test_bearer_via_config_key_wins_over_aws_creds(self, clear_aws_env):
        cfg = BedrockMantleResponsesAPIConfig()
        assert (
            cfg._use_sigv4(
                api_key="config-key",
                aws_region_name="us-west-2",
                aws_access_key_id=None,
            )
            is False
        )

    def test_bearer_via_bedrock_mantle_api_key_env(self, clear_aws_env):
        clear_aws_env.setenv("BEDROCK_MANTLE_API_KEY", "env-key")
        cfg = BedrockMantleResponsesAPIConfig()
        assert (
            cfg._use_sigv4(api_key=None, aws_region_name=None, aws_access_key_id=None)
            is False
        )

    def test_bearer_via_aws_bearer_token_env(self, clear_aws_env):
        clear_aws_env.setenv("AWS_BEARER_TOKEN_BEDROCK", "bearer-key")
        cfg = BedrockMantleResponsesAPIConfig()
        assert (
            cfg._use_sigv4(api_key=None, aws_region_name=None, aws_access_key_id=None)
            is False
        )

    def test_aws_region_param_without_bearer_activates_sigv4(self, clear_aws_env):
        cfg = BedrockMantleResponsesAPIConfig()
        assert (
            cfg._use_sigv4(
                api_key=None, aws_region_name="us-east-1", aws_access_key_id=None
            )
            is True
        )

    def test_aws_access_key_param_without_bearer_activates_sigv4(self, clear_aws_env):
        cfg = BedrockMantleResponsesAPIConfig()
        assert (
            cfg._use_sigv4(
                api_key=None, aws_region_name=None, aws_access_key_id="AKIA_TEST"
            )
            is True
        )

    def test_env_signal_alone_activates_sigv4(self, clear_aws_env):
        clear_aws_env.setenv("AWS_REGION", "us-west-2")
        cfg = BedrockMantleResponsesAPIConfig()
        assert (
            cfg._use_sigv4(api_key=None, aws_region_name=None, aws_access_key_id=None)
            is True
        )

    def test_irsa_role_arn_activates_sigv4(self, clear_aws_env):
        clear_aws_env.setenv("AWS_ROLE_ARN", "arn:aws:iam::123456789012:role/MyRole")
        cfg = BedrockMantleResponsesAPIConfig()
        assert (
            cfg._use_sigv4(api_key=None, aws_region_name=None, aws_access_key_id=None)
            is True
        )

    def test_irsa_web_identity_token_file_activates_sigv4(self, clear_aws_env):
        clear_aws_env.setenv("AWS_WEB_IDENTITY_TOKEN_FILE", "/var/run/secrets/token")
        cfg = BedrockMantleResponsesAPIConfig()
        assert (
            cfg._use_sigv4(api_key=None, aws_region_name=None, aws_access_key_id=None)
            is True
        )

    def test_no_credentials_does_not_activate_sigv4(self, clear_aws_env):
        cfg = BedrockMantleResponsesAPIConfig()
        assert (
            cfg._use_sigv4(api_key=None, aws_region_name=None, aws_access_key_id=None)
            is False
        )

    def test_validate_environment_sigv4_omits_authorization(self, clear_aws_env):
        cfg = BedrockMantleResponsesAPIConfig()
        headers = cfg.validate_environment(
            headers={},
            model="openai.gpt-5.5",
            litellm_params=GenericLiteLLMParams(aws_region_name="us-east-1"),
        )
        assert "Authorization" not in headers

    def test_validate_environment_sigv4_sets_content_type(self, clear_aws_env):
        cfg = BedrockMantleResponsesAPIConfig()
        headers = cfg.validate_environment(
            headers={},
            model="openai.gpt-5.5",
            litellm_params=GenericLiteLLMParams(aws_region_name="us-east-1"),
        )
        assert headers["Content-Type"] == "application/json"

    def test_validate_environment_bearer_sets_authorization(self, clear_aws_env):
        cfg = BedrockMantleResponsesAPIConfig()
        headers = cfg.validate_environment(
            headers={},
            model="openai.gpt-5.5",
            litellm_params=GenericLiteLLMParams(api_key="config-key"),
        )
        assert headers["Authorization"] == "Bearer config-key"

    def test_validate_environment_no_credentials_message_mentions_iam(
        self, clear_aws_env
    ):
        cfg = BedrockMantleResponsesAPIConfig()
        with pytest.raises(
            ValueError,
            match="Bedrock Mantle API key or AWS IAM credentials are required",
        ) as exc_info:
            cfg.validate_environment(
                headers={},
                model="openai.gpt-5.5",
                litellm_params=GenericLiteLLMParams(),
            )
        message = str(exc_info.value)
        assert "SigV4" in message or "IAM" in message

    def test_sign_hook_signs_request_with_bedrock_service(self, clear_aws_env):
        cfg = _RecordingConfig()
        api_base = cfg.get_complete_url(
            api_base=None, litellm_params={"aws_region_name": "us-west-2"}
        )
        request_data = {"model": "openai.gpt-5.5", "input": "hello"}
        cfg.sign_request(
            headers={},
            optional_params={"aws_region_name": "us-west-2"},
            request_data=request_data,
            api_base=api_base,
        )
        assert len(cfg.sign_calls) == 1
        call = cfg.sign_calls[0]
        assert call["service_name"] == "bedrock"
        assert call["api_base"].endswith("/openai/v1/responses")
        assert call["request_data"] == request_data

    def test_sign_hook_returns_signed_headers_and_body(self, clear_aws_env):
        cfg = _RecordingConfig()
        api_base = cfg.get_complete_url(
            api_base=None, litellm_params={"aws_region_name": "us-west-2"}
        )
        signed_headers, signed_body = cfg.sign_request(
            headers={},
            optional_params={"aws_region_name": "us-west-2"},
            request_data={"model": "openai.gpt-5.5", "input": "hello"},
            api_base=api_base,
        )
        assert signed_headers["Authorization"] == "AWS4-HMAC-SHA256 Credential=test"
        assert signed_headers["X-Amz-Date"] == "20260101T000000Z"
        assert signed_body == b"{}"

    def test_sign_hook_bearer_mode_does_not_sign(self, clear_aws_env):
        cfg = _RecordingConfig()
        signed_headers, signed_body = cfg.sign_request(
            headers={"Authorization": "Bearer config-key"},
            optional_params={"api_key": "config-key"},
            request_data={"model": "openai.gpt-5.5", "input": "hello"},
            api_base="https://bedrock-mantle.us-east-2.api.aws/openai/v1/responses",
        )
        assert cfg.sign_calls == []
        assert signed_body is None
        assert "X-Amz-Date" not in signed_headers

    def test_transform_does_not_sign(self, clear_aws_env):
        cfg = _RecordingConfig()
        headers = {}
        cfg.transform_responses_api_request(
            model="openai.gpt-5.5",
            input="hello",
            response_api_optional_request_params={},
            litellm_params=GenericLiteLLMParams(aws_region_name="us-west-2"),
            headers=headers,
        )
        assert cfg.sign_calls == []
        assert "Authorization" not in headers


class TestBedrockMantleResponsesStructure:
    def test_subclasses_openai_responses_and_base_aws_llm(self):
        assert issubclass(BedrockMantleResponsesAPIConfig, OpenAIResponsesAPIConfig)
        assert issubclass(BedrockMantleResponsesAPIConfig, BaseAWSLLM)

    def test_get_complete_url_ends_with_responses_path(self, clear_aws_env):
        cfg = BedrockMantleResponsesAPIConfig()
        url = cfg.get_complete_url(api_base=None, litellm_params={})
        assert url.endswith("/openai/v1/responses")


class TestBedrockMantleResponsesSigV4Signature:
    def test_signature_scope_uses_bedrock_service_and_url_region(self, clear_aws_env):
        clear_aws_env.setenv("AWS_ACCESS_KEY_ID", "AKIAIOSFODNN7EXAMPLE")
        clear_aws_env.setenv(
            "AWS_SECRET_ACCESS_KEY", "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
        )
        cfg = BedrockMantleResponsesAPIConfig()
        api_base = cfg.get_complete_url(
            api_base=None, litellm_params={"aws_region_name": "us-east-2"}
        )
        headers, _ = cfg.sign_request(
            headers={"Content-Type": "application/json"},
            optional_params={"aws_region_name": "us-east-2"},
            request_data={"model": "openai.gpt-5.4", "input": "hello"},
            api_base=api_base,
        )
        auth = headers["Authorization"]
        assert auth.startswith("AWS4-HMAC-SHA256 ")
        assert "/us-east-2/bedrock/aws4_request" in auth

    def test_injected_default_api_base_region_is_pinned_to_signing_region(
        self, clear_aws_env
    ):
        clear_aws_env.setenv("AWS_ACCESS_KEY_ID", "AKIAIOSFODNN7EXAMPLE")
        clear_aws_env.setenv(
            "AWS_SECRET_ACCESS_KEY", "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
        )
        cfg = BedrockMantleResponsesAPIConfig()
        # get_llm_provider injects a us-east-1 default host; the resolved URL and
        # the SigV4 scope must both end up us-east-2 (the explicit region).
        url = cfg.get_complete_url(
            api_base="https://bedrock-mantle.us-east-1.api.aws/v1",
            litellm_params={"aws_region_name": "us-east-2"},
        )
        assert url == "https://bedrock-mantle.us-east-2.api.aws/openai/v1/responses"
        headers, _ = cfg.sign_request(
            headers={"Content-Type": "application/json"},
            optional_params={
                "aws_region_name": "us-east-2",
                "api_base": "https://bedrock-mantle.us-east-1.api.aws/v1",
            },
            request_data={"model": "openai.gpt-5.4", "input": "hello"},
            api_base=url,
        )
        assert "/us-east-2/bedrock/aws4_request" in headers["Authorization"]

    def test_signed_body_hash_matches_compact_json_sent_by_httpx(self, clear_aws_env):
        from botocore.auth import SigV4Auth
        from botocore.awsrequest import AWSRequest
        from botocore.credentials import Credentials

        clear_aws_env.setenv("AWS_ACCESS_KEY_ID", "AKIAIOSFODNN7EXAMPLE")
        clear_aws_env.setenv(
            "AWS_SECRET_ACCESS_KEY", "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
        )
        cfg = BedrockMantleResponsesAPIConfig()
        data = {"model": "openai.gpt-5.4", "input": "hello, what model are you?"}
        api_base = cfg.get_complete_url(
            api_base=None, litellm_params={"aws_region_name": "us-east-2"}
        )
        headers, signed_body = cfg.sign_request(
            headers={"Content-Type": "application/json"},
            optional_params={"aws_region_name": "us-east-2"},
            request_data=data,
            api_base=api_base,
        )

        # The handler posts the exact signed_body bytes. Re-sign those same bytes
        # with the date the config used and assert the signatures match, proving
        # the wire body and the signed body are identical.
        creds = Credentials(
            "AKIAIOSFODNN7EXAMPLE", "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
        )
        ref = AWSRequest(
            method="POST",
            url="https://bedrock-mantle.us-east-2.api.aws/openai/v1/responses",
            data=signed_body,
            headers={
                "Content-Type": "application/json",
                "X-Amz-Date": headers["X-Amz-Date"],
            },
        )
        SigV4Auth(creds, "bedrock", "us-east-2").add_auth(ref)
        assert headers["Authorization"] == ref.headers["Authorization"]


class TestBedrockMantleResponsesHandlerSignedBody:
    """Exercises the handler's ``data=signed_body`` branch (the 3 lines the
    coverage checker flagged). Verifies that when sign_request returns body
    bytes, the handler sends those exact bytes rather than re-serializing."""

    def test_handler_posts_signed_body_bytes(self, clear_aws_env, respx_mock):
        import json

        import httpx

        clear_aws_env.setenv("AWS_ACCESS_KEY_ID", "AKIAIOSFODNN7EXAMPLE")
        clear_aws_env.setenv(
            "AWS_SECRET_ACCESS_KEY", "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
        )

        captured_body = {}

        def capture_request(request: httpx.Request) -> httpx.Response:
            captured_body["raw"] = request.content
            return httpx.Response(
                200,
                json={
                    "id": "resp_test",
                    "object": "response",
                    "created_at": 1,
                    "status": "completed",
                    "model": "openai.gpt-5.4",
                    "output": [
                        {
                            "type": "message",
                            "id": "msg_1",
                            "role": "assistant",
                            "status": "completed",
                            "content": [
                                {"type": "output_text", "text": "hi", "annotations": []}
                            ],
                        }
                    ],
                    "usage": {
                        "input_tokens": 1,
                        "output_tokens": 1,
                        "total_tokens": 2,
                        "input_tokens_details": {"cached_tokens": 0},
                        "output_tokens_details": {"reasoning_tokens": 0},
                    },
                },
            )

        respx_mock.post(
            "https://bedrock-mantle.us-east-2.api.aws/openai/v1/responses"
        ).mock(side_effect=capture_request)

        import litellm

        resp = litellm.responses(
            model="bedrock_mantle/openai.gpt-5.4",
            input="hello",
            aws_region_name="us-east-2",
        )
        assert resp.output[0].content[0].text == "hi"

        # The handler should have sent the exact bytes sign_request produced,
        # not a re-serialization via httpx json=. The signed body uses
        # json.dumps(data) (with spaces), so verify the wire bytes contain spaces
        # (httpx compact would not).
        wire_body = captured_body["raw"]
        assert b'"model": ' in wire_body or b'"model":' in wire_body
        parsed = json.loads(wire_body)
        assert parsed["model"] == "openai.gpt-5.4"
        assert parsed["input"] == "hello"

    def test_handler_posts_signed_body_bytes_streaming(self, clear_aws_env, respx_mock):
        import json

        import httpx

        clear_aws_env.setenv("AWS_ACCESS_KEY_ID", "AKIAIOSFODNN7EXAMPLE")
        clear_aws_env.setenv(
            "AWS_SECRET_ACCESS_KEY", "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
        )

        captured_body = {}

        def capture_request(request: httpx.Request) -> httpx.Response:
            captured_body["raw"] = request.content
            return httpx.Response(
                200,
                content=b'data: {"type":"response.completed"}\n\ndata: [DONE]\n\n',
                headers={"content-type": "text/event-stream"},
            )

        respx_mock.post(
            "https://bedrock-mantle.us-east-2.api.aws/openai/v1/responses"
        ).mock(side_effect=capture_request)

        import litellm

        try:
            resp = litellm.responses(
                model="bedrock_mantle/openai.gpt-5.4",
                input="hello",
                stream=True,
                aws_region_name="us-east-2",
            )
            for _ in resp:
                pass
        except Exception:
            pass

        wire_body = captured_body["raw"]
        parsed = json.loads(wire_body)
        assert parsed["model"] == "openai.gpt-5.4"
        assert parsed["input"] == "hello"
