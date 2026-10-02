import pytest

import litellm
from litellm.litellm_core_utils.get_model_cost_map import get_model_cost_map
from litellm.llms.azure.chat.gpt_5_transformation import AzureOpenAIGPT5Config


@pytest.fixture()
def config() -> AzureOpenAIGPT5Config:
    return AzureOpenAIGPT5Config()


@pytest.fixture(autouse=True)
def use_local_model_cost_map(monkeypatch: pytest.MonkeyPatch):
    """Pin the bundled cost map: these gates read model-map capability keys, and the default
    import path fetches the published map, which lags a key added in this repo."""
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    monkeypatch.setattr(litellm, "model_cost", get_model_cost_map(url=litellm.model_cost_map_url))
    litellm.add_known_models(model_cost_map=litellm.model_cost)


def test_azure_gpt5_supports_reasoning_effort(config: AzureOpenAIGPT5Config):
    assert "reasoning_effort" in config.get_supported_openai_params(model="gpt-5")
    assert "reasoning_effort" in config.get_supported_openai_params(
        model="gpt5_series/my-deployment"
    )


def test_azure_gpt5_allows_tool_choice_for_deployment_names():
    supported_params = litellm.get_supported_openai_params(
        model="gpt-5-chat-2025-08-07", custom_llm_provider="azure"
    )
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
    assert "reasoning_effort" in config.get_supported_openai_params(
        model="gpt5_series/gpt-5-codex"
    )


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


class TestAzureResolvesTheDeclaredDefaultEffort:
    """Azure reaches the same models under names that are not cost-map keys. Every capability
    lookup therefore has to normalise the name identically, which is why the normalisation is
    one overridden resolver rather than a rewrite inside a single lookup.
    """

    @pytest.mark.parametrize(
        "model, temperature_survives",
        [
            ("azure/gpt-5.1", True),
            ("gpt5_series/gpt-5.1", True),
            ("gpt-5.1", True),
            ("azure/gpt-5.6-terra", False),
            ("gpt5_series/gpt-5.6-terra", False),
            ("azure/gpt-5.5", False),
        ],
    )
    def test_every_azure_name_shape_reads_the_same_entry(self, config, model, temperature_survives):
        mapped = config.map_openai_params(
            non_default_params={"temperature": 0},
            optional_params={},
            model=model,
            drop_params=True,
        )
        assert ("temperature" in mapped) is temperature_survives


def test_azure_gpt_6_astra_takes_the_reasoning_series_request_shape():
    params = litellm.get_optional_params(
        model="gpt-6-astra",
        custom_llm_provider="azure",
        max_tokens=100,
        reasoning_effort="xhigh",
    )
    assert params["max_completion_tokens"] == 100
    assert "max_tokens" not in params
    assert params["reasoning_effort"] == "xhigh"


@pytest.mark.parametrize("model", ["azure/gpt-6-sol", "azure/us/gpt-6-sol"])
def test_azure_reasoning_effort_none_unlocks_temperature_off_the_azure_row(
    config: AzureOpenAIGPT5Config, monkeypatch: pytest.MonkeyPatch, model: str
):
    """A deployment whose azure/ row takes none accepts a non-default temperature with it, read off
    that row even when OpenAI's twin row says the model refuses none."""
    monkeypatch.setitem(
        litellm.model_cost, "gpt-6-sol", {**litellm.model_cost["gpt-6-sol"], "supports_none_reasoning_effort": False}
    )
    params = config.map_openai_params(
        non_default_params={"temperature": 0.2, "reasoning_effort": "none"},
        optional_params={},
        model=model,
        drop_params=False,
        api_version="2025-04-01-preview",
    )
    assert params["temperature"] == 0.2
    assert params["reasoning_effort"] == "none"


def _azure_chat_row_forwards(row: dict, level: str) -> bool:
    """Azure chat polarity: max is never gated, xhigh and none need an explicit true, every other
    level only has to not be false."""
    if level == "max":
        return True
    flag = row.get(f"supports_{level}_reasoning_effort")
    return flag is True if level in ("xhigh", "none") else flag is not False


@pytest.mark.parametrize("model", ["azure/gpt-6-astra", "azure/us/gpt-6-astra"])
@pytest.mark.parametrize("level", ["none", "minimal", "low", "medium", "high", "xhigh", "max"])
def test_azure_gpt6_astra_forwards_exactly_the_effort_levels_its_map_row_allows(
    config: AzureOpenAIGPT5Config, model: str, level: str
):
    """The azure/ row is the whole contract for a Foundry deployment. A live Foundry gpt-6-astra
    (model gpt-6-astra-2026-09-03) answered reasoning_effort none with a 400 naming low, medium,
    high and xhigh on 2026-10-01, so its row turns none and minimal off. The row keeps max off too,
    since chat refuses it, but no gate reads that flag: the Responses route honors max, so chat
    forwards it for the bridge and a plain chat call gets the provider's own 400."""
    forwarded = _azure_chat_row_forwards(litellm.model_cost[model], level)
    dropped = config.map_openai_params(
        non_default_params={"reasoning_effort": level},
        optional_params={},
        model=model,
        drop_params=True,
        api_version="2025-04-01-preview",
    )
    assert ("reasoning_effort" in dropped) is forwarded
    if forwarded:
        kept = config.map_openai_params(
            non_default_params={"reasoning_effort": level},
            optional_params={},
            model=model,
            drop_params=False,
            api_version="2025-04-01-preview",
        )
        assert kept["reasoning_effort"] == level
    else:
        with pytest.raises(litellm.utils.UnsupportedParamsError):
            config.map_openai_params(
                non_default_params={"reasoning_effort": level},
                optional_params={},
                model=model,
                drop_params=False,
                api_version="2025-04-01-preview",
            )


def test_azure_gpt6_astra_row_turns_none_and_minimal_off():
    """Pinned on purpose so the row-derived test above cannot go vacuous: a Foundry gpt-6-astra
    deployment refused none on chat and on Responses on 2026-10-01 (400 unsupported_value naming
    low, medium, high, xhigh), and minimal the same way; max it refused on chat while honoring it
    on Responses, so the route-blind flag stays off. When Azure adds a level, flip the row and
    this assertion together."""
    row = litellm.model_cost["azure/gpt-6-astra"]
    assert row["supports_none_reasoning_effort"] is False
    assert row["supports_minimal_reasoning_effort"] is False
    assert row["supports_max_reasoning_effort"] is False


def test_azure_chat_gate_reads_the_foundry_row_when_azure_ai_prefix_survives_provider_remap(
    config: AzureOpenAIGPT5Config, monkeypatch: pytest.MonkeyPatch
):
    """A Foundry deployment on an OpenAI-v1 host is re-routed to the azure provider with its
    azure_ai/ prefix intact, and this config then gates it. The rows are made to disagree on low,
    a level neither real row flags, so the prefixed name has to read azure_ai/gpt-6-sol while the
    bare deployment name keeps reading azure/gpt-6-sol."""
    monkeypatch.setitem(
        litellm.model_cost,
        "azure_ai/gpt-6-sol",
        {**litellm.model_cost["azure_ai/gpt-6-sol"], "supports_low_reasoning_effort": False},
    )
    monkeypatch.setitem(
        litellm.model_cost,
        "azure/gpt-6-sol",
        {**litellm.model_cost["azure/gpt-6-sol"], "supports_low_reasoning_effort": True},
    )
    prefixed = config.map_openai_params(
        non_default_params={"reasoning_effort": "low"},
        optional_params={},
        model="azure_ai/gpt-6-sol",
        drop_params=True,
        api_version="2025-04-01-preview",
    )
    assert "reasoning_effort" not in prefixed
    bare = config.map_openai_params(
        non_default_params={"reasoning_effort": "low"},
        optional_params={},
        model="gpt-6-sol",
        drop_params=True,
        api_version="2025-04-01-preview",
    )
    assert bare["reasoning_effort"] == "low"


def test_azure_chat_gate_falls_back_to_the_azure_row_for_an_azure_ai_name_the_map_lacks(
    config: AzureOpenAIGPT5Config, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.delitem(litellm.model_cost, "azure_ai/gpt-6-sol")
    monkeypatch.setitem(
        litellm.model_cost,
        "azure/gpt-6-sol",
        {**litellm.model_cost["azure/gpt-6-sol"], "supports_low_reasoning_effort": False},
    )
    dropped = config.map_openai_params(
        non_default_params={"reasoning_effort": "low"},
        optional_params={},
        model="azure_ai/gpt-6-sol",
        drop_params=True,
        api_version="2025-04-01-preview",
    )
    assert "reasoning_effort" not in dropped


@pytest.mark.parametrize("key", ["azure/gpt-6.1-sol", "azure/gpt-6.1-sol-2026-09-29", "azure_ai/gpt-6.1-sol"])
def test_azure_gpt61_sol_rows_turn_none_off(key: str):
    """Pinned on purpose, as the astra row above: a Foundry gpt-6.1-sol deployment (model
    gpt-6.1-sol-2026-09-29) refused none on chat and on Responses on 2026-10-02 (400 unsupported_value
    naming low, medium, high, xhigh, and max on Responses), and minimal the same way, while these rows
    still said none was on. When Azure adds a level, flip the row and this assertion together."""
    row = litellm.model_cost[key]
    assert row["supports_none_reasoning_effort"] is False
    assert row["supports_minimal_reasoning_effort"] is False
