from typing import Iterator

import pytest

import litellm
from litellm.llms.azure.chat.gpt_5_transformation import AzureOpenAIGPT5Config


@pytest.fixture()
def config() -> AzureOpenAIGPT5Config:
    return AzureOpenAIGPT5Config()


def test_azure_gpt5_supports_reasoning_effort(config: AzureOpenAIGPT5Config):
    assert "reasoning_effort" in config.get_supported_openai_params(model="gpt-5")
    assert "reasoning_effort" in config.get_supported_openai_params(model="gpt5_series/my-deployment")


def test_azure_gpt5_allows_tool_choice_for_deployment_names():
    supported_params = litellm.get_supported_openai_params(model="gpt-5-chat-2025-08-07", custom_llm_provider="azure")
    assert supported_params is not None
    assert "tool_choice" in supported_params
    # gpt-5-chat* should not be treated as a GPT-5 reasoning model
    assert "reasoning_effort" not in supported_params
    assert "temperature" in supported_params


def test_azure_gpt5_maps_max_tokens(config: AzureOpenAIGPT5Config):
    params = config.map_openai_params(
        non_default_params={"max_tokens": 5},
        optional_params={},
        model="gpt5_series/gpt-5",
        drop_params=False,
        api_version="2024-05-01-preview",
    )
    assert params["max_completion_tokens"] == 5
    assert "max_tokens" not in params


def test_azure_gpt5_temperature_error(config: AzureOpenAIGPT5Config):
    with pytest.raises(litellm.utils.UnsupportedParamsError):
        config.map_openai_params(
            non_default_params={"temperature": 0.2},
            optional_params={},
            model="gpt-5",
            drop_params=False,
            api_version="2024-05-01-preview",
        )


def test_azure_gpt5_series_transform_request(config: AzureOpenAIGPT5Config):
    request = config.transform_request(
        model="gpt5_series/gpt-5",
        messages=[],
        optional_params={},
        litellm_params={},
        headers={},
    )
    assert request["model"] == "gpt-5"


# GPT-5-Codex specific tests for Azure
def test_azure_gpt5_codex_model_detection(config: AzureOpenAIGPT5Config):
    """Test that Azure GPT-5-Codex models are correctly detected."""
    assert config.is_model_gpt_5_model("gpt-5-codex")
    assert config.is_model_gpt_5_model("gpt5_series/gpt-5-codex")


def test_azure_gpt5_codex_supports_reasoning_effort(config: AzureOpenAIGPT5Config):
    """Test that Azure GPT-5-Codex supports reasoning_effort parameter."""
    assert "reasoning_effort" in config.get_supported_openai_params(model="gpt-5-codex")
    assert "reasoning_effort" in config.get_supported_openai_params(model="gpt5_series/gpt-5-codex")


def test_azure_gpt5_codex_maps_max_tokens(config: AzureOpenAIGPT5Config):
    """Test that Azure GPT-5-Codex correctly maps max_tokens to max_completion_tokens."""
    params = config.map_openai_params(
        non_default_params={"max_tokens": 150},
        optional_params={},
        model="gpt-5-codex",
        drop_params=False,
        api_version="2024-05-01-preview",
    )
    assert params["max_completion_tokens"] == 150
    assert "max_tokens" not in params


def test_azure_gpt5_codex_temperature_error(config: AzureOpenAIGPT5Config):
    """Test that Azure GPT-5-Codex raises error for unsupported temperature."""
    with pytest.raises(litellm.utils.UnsupportedParamsError):
        config.map_openai_params(
            non_default_params={"temperature": 0.8},
            optional_params={},
            model="gpt-5-codex",
            drop_params=False,
            api_version="2024-05-01-preview",
        )


def test_azure_gpt5_codex_series_transform_request(config: AzureOpenAIGPT5Config):
    """Test that Azure GPT-5-Codex series routing works correctly."""
    request = config.transform_request(
        model="gpt5_series/gpt-5-codex",
        messages=[],
        optional_params={},
        litellm_params={},
        headers={},
    )
    assert request["model"] == "gpt-5-codex"


# GPT-5.1 temperature handling tests for Azure
def test_azure_gpt5_1_temperature_with_reasoning_effort_none(
    config: AzureOpenAIGPT5Config,
):
    """Test that Azure GPT-5.1 supports any temperature when reasoning_effort='none'.

    Azure OpenAI supports reasoning_effort='none' for gpt-5.1 models.
    See: https://learn.microsoft.com/en-us/azure/ai-foundry/openai/how-to/reasoning
    """
    params = config.map_openai_params(
        non_default_params={"temperature": 0.5, "reasoning_effort": "none"},
        optional_params={},
        model="azure/gpt-5.1",
        drop_params=False,
        api_version="2024-05-01-preview",
    )
    assert params["temperature"] == 0.5
    # Azure supports reasoning_effort="none" for gpt-5.1
    assert params.get("reasoning_effort") == "none"


def test_azure_gpt5_1_reasoning_effort_none_supported(config: AzureOpenAIGPT5Config):
    """Test that Azure GPT-5.1 supports reasoning_effort='none' without error."""
    params = config.map_openai_params(
        non_default_params={"reasoning_effort": "none"},
        optional_params={},
        model="azure/gpt-5.1",
        drop_params=False,
        api_version="2024-05-01-preview",
    )
    assert params.get("reasoning_effort") == "none"


def test_azure_gpt5_1_temperature_without_reasoning_effort(
    config: AzureOpenAIGPT5Config,
):
    """Test that Azure GPT-5.1 supports any temperature when reasoning_effort is not specified."""
    params = config.map_openai_params(
        non_default_params={"temperature": 0.7},
        optional_params={},
        model="azure/gpt-5.1",
        drop_params=False,
        api_version="2024-05-01-preview",
    )
    assert params["temperature"] == 0.7


def test_azure_gpt5_1_temperature_with_reasoning_effort_other_values(
    config: AzureOpenAIGPT5Config,
):
    """Test that Azure GPT-5.1 only allows temperature=1 when reasoning_effort is not 'none'."""
    # Test that temperature != 1 raises error when reasoning_effort is set to other values
    with pytest.raises(litellm.utils.UnsupportedParamsError):
        config.map_openai_params(
            non_default_params={"temperature": 0.7, "reasoning_effort": "low"},
            optional_params={},
            model="azure/gpt-5.1",
            drop_params=False,
            api_version="2024-05-01-preview",
        )

    # Test that temperature=1 is allowed with other reasoning_effort values
    params = config.map_openai_params(
        non_default_params={"temperature": 1.0, "reasoning_effort": "medium"},
        optional_params={},
        model="azure/gpt-5.1",
        drop_params=False,
        api_version="2024-05-01-preview",
    )
    assert params["temperature"] == 1.0
    assert params["reasoning_effort"] == "medium"


def test_azure_gpt5_1_series_temperature_handling(config: AzureOpenAIGPT5Config):
    """Test that Azure GPT-5.1 with gpt5_series prefix supports temperature with reasoning_effort='none'."""
    params = config.map_openai_params(
        non_default_params={"temperature": 0.6},
        optional_params={},
        model="gpt5_series/gpt-5.1",
        drop_params=False,
        api_version="2024-05-01-preview",
    )
    assert params["temperature"] == 0.6


def test_azure_gpt5_4_preserves_reasoning_effort_when_tools_present(
    config: AzureOpenAIGPT5Config,
):
    """Azure GPT-5.4+ no longer drops reasoning_effort when tools are present.

    Both OpenAI and Azure now route tools+reasoning to the Responses API bridge,
    so reasoning_effort must be preserved in map_openai_params.
    """
    tools = [{"type": "function", "function": {"name": "test", "description": "test"}}]
    params = config.map_openai_params(
        non_default_params={"reasoning_effort": "high", "tools": tools},
        optional_params={},
        model="gpt5_series/gpt-5.4",
        drop_params=False,
        api_version="2024-05-01-preview",
    )
    assert params.get("reasoning_effort") == "high"
    assert params["tools"] == tools


def test_azure_gpt5_reasoning_effort_none_error(config: AzureOpenAIGPT5Config):
    """Test that Azure GPT-5 (non-5.1) raises error for reasoning_effort='none' when drop_params=False."""
    with pytest.raises(litellm.utils.UnsupportedParamsError):
        config.map_openai_params(
            non_default_params={"reasoning_effort": "none"},
            optional_params={},
            model="azure/gpt-5",
            drop_params=False,
            api_version="2024-05-01-preview",
        )


def test_azure_gpt5_reasoning_effort_none_dropped(config: AzureOpenAIGPT5Config):
    """Test that Azure GPT-5 (non-5.1) drops reasoning_effort='none' when drop_params=True."""
    params = config.map_openai_params(
        non_default_params={"reasoning_effort": "none"},
        optional_params={},
        model="azure/gpt-5",
        drop_params=True,
        api_version="2024-05-01-preview",
    )
    assert "reasoning_effort" not in params or params.get("reasoning_effort") != "none"


# Logprobs support tests for Azure GPT-5.2
def test_azure_gpt5_2_supports_logprobs(config: AzureOpenAIGPT5Config):
    """Test that Azure GPT-5.2 models support logprobs parameters.

    Only Azure OpenAI GPT-5.2 supports logprobs, unlike OpenAI's GPT-5 or Azure's gpt-5/gpt-5.1.
    Tested with gpt-5.2 on api-version 2025-01-01-preview.
    """
    supported_params = config.get_supported_openai_params(model="gpt-5.2")
    assert "logprobs" in supported_params
    assert "top_logprobs" in supported_params


def test_azure_gpt5_2_with_prefix_supports_logprobs(config: AzureOpenAIGPT5Config):
    """Test that Azure GPT-5.2 with azure/ prefix supports logprobs parameters."""
    supported_params = config.get_supported_openai_params(model="azure/gpt-5.2")
    assert "logprobs" in supported_params
    assert "top_logprobs" in supported_params


def test_azure_gpt5_2_series_supports_logprobs(config: AzureOpenAIGPT5Config):
    """Test that Azure GPT-5.2 with gpt5_series prefix supports logprobs."""
    supported_params = config.get_supported_openai_params(model="gpt5_series/gpt-5.2")
    assert "logprobs" in supported_params
    assert "top_logprobs" in supported_params


def test_azure_gpt5_2_logprobs_params_passed_through(config: AzureOpenAIGPT5Config):
    """Test that logprobs parameters are correctly passed through to the API for gpt-5.2."""
    params = config.map_openai_params(
        non_default_params={"logprobs": True, "top_logprobs": 5},
        optional_params={},
        model="azure/gpt-5.2",
        drop_params=False,
        api_version="2025-01-01-preview",
    )
    assert params["logprobs"] is True
    assert params["top_logprobs"] == 5


def test_azure_gpt5_base_does_not_support_logprobs(config: AzureOpenAIGPT5Config):
    """Test that Azure GPT-5 (non-5.2) does not support logprobs parameters.

    Only gpt-5.2 has been verified to support logprobs on Azure.
    """
    supported_params = config.get_supported_openai_params(model="gpt-5")
    assert "logprobs" not in supported_params
    assert "top_logprobs" not in supported_params


def test_azure_gpt5_1_does_not_support_logprobs(config: AzureOpenAIGPT5Config):
    """Test that Azure GPT-5.1 does not support logprobs parameters.

    Only gpt-5.2 has been verified to support logprobs on Azure.
    """
    supported_params = config.get_supported_openai_params(model="gpt-5.1")
    assert "logprobs" not in supported_params
    assert "top_logprobs" not in supported_params


@pytest.fixture()
def deployment_capability_override() -> Iterator[None]:
    """Register a deployment-name-keyed capability override, as the router does.

    Router._create_deployment registers per-deployment ``model_info`` under the
    backend model name (``azure/<deployment>``), while Azure capability lookups
    resolve against ``base_model``.  This fixture reproduces that registration so
    tests can assert the override is honoured.
    """
    original_model_cost = litellm.model_cost.copy()
    litellm.register_model(
        model_cost={
            "azure/gpt-5.6-luna-dz": {"supports_none_reasoning_effort": False},
            "azure/gpt-5-none-enabled-dz": {"supports_none_reasoning_effort": True},
        }
    )
    yield
    litellm.model_cost = original_model_cost


def test_azure_gpt5_deployment_override_disables_flexible_temperature(
    config: AzureOpenAIGPT5Config, deployment_capability_override: None
):
    """A per-deployment supports_none_reasoning_effort=false must drop temperature.

    The registry entry for the base_model (azure/gpt-5.6-luna) sets the capability
    to true, so without honouring the deployment override the temperature is
    forwarded and Azure rejects the request with a 400.
    """
    assert litellm.model_cost["azure/gpt-5.6-luna"]["supports_none_reasoning_effort"] is True

    params = config.map_openai_params(
        non_default_params={"temperature": 0.2},
        optional_params={},
        model="azure/gpt-5.6-luna",
        drop_params=True,
        api_version="2025-01-01-preview",
        deployment_model="gpt-5.6-luna-dz",
    )
    assert "temperature" not in params


def test_azure_gpt5_deployment_override_raises_without_drop_params(
    config: AzureOpenAIGPT5Config, deployment_capability_override: None
):
    """With drop_params disabled the override must surface as an UnsupportedParamsError."""
    with pytest.raises(litellm.utils.UnsupportedParamsError):
        config.map_openai_params(
            non_default_params={"temperature": 0.2},
            optional_params={},
            model="azure/gpt-5.6-luna",
            drop_params=False,
            api_version="2025-01-01-preview",
            deployment_model="gpt-5.6-luna-dz",
        )


def test_azure_gpt5_deployment_override_enables_flexible_temperature(
    config: AzureOpenAIGPT5Config, deployment_capability_override: None
):
    """A per-deployment supports_none_reasoning_effort=true must allow temperature.

    The base_model here (azure/gpt-5) does not support reasoning_effort='none', so
    the temperature is only preserved when the deployment override is honoured.
    """
    assert not litellm.model_cost["azure/gpt-5"].get("supports_none_reasoning_effort")

    params = config.map_openai_params(
        non_default_params={"temperature": 0.3},
        optional_params={},
        model="azure/gpt-5",
        drop_params=True,
        api_version="2025-01-01-preview",
        deployment_model="gpt-5-none-enabled-dz",
    )
    assert params["temperature"] == 0.3


def test_azure_gpt5_without_deployment_override_uses_base_model(
    config: AzureOpenAIGPT5Config, deployment_capability_override: None
):
    """A deployment without an explicit override must keep using base_model capabilities.

    Guards the base_model resolution added in #31243 against regression.
    """
    params = config.map_openai_params(
        non_default_params={"temperature": 0.2},
        optional_params={},
        model="azure/gpt-5.6-luna",
        drop_params=True,
        api_version="2025-01-01-preview",
        deployment_model="gpt-5.6-luna-no-override-dz",
    )
    assert params["temperature"] == 0.2


def test_azure_gpt5_temperature_one_survives_deployment_override(
    config: AzureOpenAIGPT5Config, deployment_capability_override: None
):
    """temperature=1 is the Azure default and must never be dropped."""
    params = config.map_openai_params(
        non_default_params={"temperature": 1},
        optional_params={},
        model="azure/gpt-5.6-luna",
        drop_params=True,
        api_version="2025-01-01-preview",
        deployment_model="gpt-5.6-luna-dz",
    )
    assert params["temperature"] == 1


def test_get_optional_params_honours_deployment_override(
    deployment_capability_override: None,
):
    """End-to-end through get_optional_params, which is what the proxy calls.

    Covers the deployment identity being threaded through the azure branch rather
    than collapsed into ``base_model or model``.
    """
    params = litellm.utils.get_optional_params(
        model="gpt-5.6-luna-dz",
        custom_llm_provider="azure",
        base_model="azure/gpt-5.6-luna",
        temperature=0.2,
        drop_params=True,
    )
    assert "temperature" not in params


def test_get_optional_params_without_override_keeps_temperature(
    deployment_capability_override: None,
):
    """Sibling control: an unoverridden deployment still resolves via base_model."""
    params = litellm.utils.get_optional_params(
        model="gpt-5.6-terra-dz",
        custom_llm_provider="azure",
        base_model="azure/gpt-5.6-terra",
        temperature=0.2,
        drop_params=True,
    )
    assert params["temperature"] == 0.2


@pytest.fixture()
def sampling_capability_override() -> Iterator[None]:
    """Register a deployment that enables reasoning_effort='none' over a base model without it."""
    original_model_cost = litellm.model_cost.copy()
    litellm.register_model(model_cost={"azure/gpt-5-sampling-dz": {"supports_none_reasoning_effort": True}})
    yield
    litellm.model_cost = original_model_cost


def test_azure_gpt5_deployment_override_allows_top_p(config: AzureOpenAIGPT5Config, sampling_capability_override: None):
    """A deployment enabling none-effort must have top_p honoured, not stripped.

    top_p is only supported when reasoning_effort='none' is available.  The base
    model (azure/gpt-5) does not support it, so the parameter survives only when the
    supported-parameter list is resolved with the deployment override applied.
    """
    assert not litellm.model_cost["azure/gpt-5"].get("supports_none_reasoning_effort")

    params = config.map_openai_params(
        non_default_params={"top_p": 0.5},
        optional_params={},
        model="azure/gpt-5",
        drop_params=True,
        api_version="2025-01-01-preview",
        deployment_model="gpt-5-sampling-dz",
    )
    assert params["top_p"] == 0.5


def test_azure_gpt5_supported_params_reflect_deployment_override(
    config: AzureOpenAIGPT5Config, sampling_capability_override: None
):
    """get_supported_openai_params must reflect the deployment override.

    Guards the pre-mapping validation path, which filters parameters before the
    request-mapping gates run.
    """
    assert "top_p" not in config.get_supported_openai_params(model="azure/gpt-5")
    assert "top_p" in config.get_supported_openai_params(model="azure/gpt-5", deployment_model="gpt-5-sampling-dz")


def test_get_optional_params_honours_deployment_override_for_top_p(
    sampling_capability_override: None,
):
    """End-to-end: top_p survives for an overriding deployment through get_optional_params."""
    params = litellm.utils.get_optional_params(
        model="gpt-5-sampling-dz",
        custom_llm_provider="azure",
        base_model="azure/gpt-5",
        top_p=0.5,
        drop_params=True,
    )
    assert params["top_p"] == 0.5


def test_azure_gpt5_logprobs_still_gated_by_model_version(
    config: AzureOpenAIGPT5Config, sampling_capability_override: None
):
    """logprobs stays gated on gpt-5.2+ regardless of the capability override.

    Azure has only verified logprobs for gpt-5.2 and newer, and that rule keys off the
    model version rather than the reasoning-effort capability.  A registry gpt-5.1
    entry behaves identically, so the override must not widen logprobs support.
    """
    assert "logprobs" not in config.get_supported_openai_params(model="azure/gpt-5.1")
    assert "logprobs" not in config.get_supported_openai_params(
        model="azure/gpt-5", deployment_model="gpt-5-sampling-dz"
    )
    assert "logprobs" in config.get_supported_openai_params(model="azure/gpt-5.2")


@pytest.fixture()
def minimal_effort_overrides() -> Iterator[None]:
    """Register deployments overriding the minimal reasoning-effort capability both ways."""
    original_model_cost = litellm.model_cost.copy()
    litellm.register_model(
        model_cost={
            "azure/gpt-55-minimal-allowed-dz": {"supports_minimal_reasoning_effort": True},
            "azure/gpt-54-minimal-blocked-dz": {"supports_minimal_reasoning_effort": False},
        }
    )
    yield
    litellm.model_cost = original_model_cost


def test_azure_gpt5_deployment_override_re_enables_minimal_effort(
    config: AzureOpenAIGPT5Config, minimal_effort_overrides: None
):
    """A deployment may re-enable a minimal effort level its base model disables.

    azure/gpt-5.5-pro sets supports_minimal_reasoning_effort=false, so without the
    override reasoning_effort='minimal' is dropped.
    """
    assert litellm.model_cost["azure/gpt-5.5-pro"]["supports_minimal_reasoning_effort"] is False

    params = config.map_openai_params(
        non_default_params={"reasoning_effort": "minimal"},
        optional_params={},
        model="azure/gpt-5.5-pro",
        drop_params=True,
        api_version="2025-01-01-preview",
        deployment_model="gpt-55-minimal-allowed-dz",
    )
    assert params["reasoning_effort"] == "minimal"


def test_azure_gpt5_deployment_override_blocks_minimal_effort(
    config: AzureOpenAIGPT5Config, minimal_effort_overrides: None
):
    """A deployment may disable a minimal effort level its base model allows.

    azure/gpt-5.4 carries no explicit setting, so minimal passes through unless the
    deployment override marks it unsupported.
    """
    assert litellm.model_cost["azure/gpt-5.4"].get("supports_minimal_reasoning_effort") is None

    params = config.map_openai_params(
        non_default_params={"reasoning_effort": "minimal"},
        optional_params={},
        model="azure/gpt-5.4",
        drop_params=True,
        api_version="2025-01-01-preview",
        deployment_model="gpt-54-minimal-blocked-dz",
    )
    assert "reasoning_effort" not in params


def test_azure_gpt5_deployment_override_blocks_minimal_effort_strict(
    config: AzureOpenAIGPT5Config, minimal_effort_overrides: None
):
    """The blocking override must raise rather than forward when drop_params is off."""
    with pytest.raises(litellm.utils.UnsupportedParamsError):
        config.map_openai_params(
            non_default_params={"reasoning_effort": "minimal"},
            optional_params={},
            model="azure/gpt-5.4",
            drop_params=False,
            api_version="2025-01-01-preview",
            deployment_model="gpt-54-minimal-blocked-dz",
        )
