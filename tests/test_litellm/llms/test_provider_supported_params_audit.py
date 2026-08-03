"""
Regression tests for provider `get_supported_openai_params` / `map_openai_params`
gaps found by diffing LiteLLM against per-provider API references.

Each test names the provider doc that makes the expected behaviour authoritative.
Two failure modes are covered:

  * a param the provider accepts is missing from the supported list, so
    `drop_params=True` silently discards a setting the caller asked for
  * a param the provider rejects is present in the supported list, so it
    survives `drop_params=True` and the request 400s
"""

import os
import sys

sys.path.insert(0, os.path.abspath("../../.."))

import pytest

import litellm
from litellm.llms.cohere.chat.v2_transformation import THINKING_BY_REASONING_EFFORT, CohereV2ChatConfig
from litellm.llms.moonshot.chat.transformation import MoonshotChatConfig
from litellm.llms.xai.chat.transformation import XAIChatConfig
from litellm.llms.zai.chat.transformation import ZAIChatConfig
from litellm.utils import get_optional_params


def _optional_params(model: str, custom_llm_provider: str, **kwargs) -> dict:
    return get_optional_params(
        model=model,
        custom_llm_provider=custom_llm_provider,
        drop_params=True,
        **kwargs,
    )


class TestMistralPenalties:
    """Ref: https://docs.mistral.ai/api/ - chat/completions accepts
    frequency_penalty, presence_penalty and n."""

    @pytest.mark.parametrize("param", ["frequency_penalty", "presence_penalty", "n"])
    def test_param_is_supported(self, param):
        supported = litellm.get_supported_openai_params(model="mistral-large-latest", custom_llm_provider="mistral")
        assert param in supported

    def test_params_survive_drop_params(self):
        optional_params = _optional_params(
            "mistral-large-latest",
            "mistral",
            frequency_penalty=0.3,
            presence_penalty=0.2,
            n=2,
        )
        assert optional_params["frequency_penalty"] == 0.3
        assert optional_params["presence_penalty"] == 0.2
        assert optional_params["n"] == 2

    @pytest.mark.parametrize(
        "kwargs, expected",
        [
            ({"max_tokens": 10}, 10),
            ({"max_completion_tokens": 50}, 50),
            ({"max_tokens": 10, "max_completion_tokens": 50}, 50),
        ],
    )
    def test_max_completion_tokens_wins(self, kwargs, expected):
        assert _optional_params("mistral-large-latest", "mistral", **kwargs)["max_tokens"] == expected


class TestCohereV2:
    """Ref: https://docs.cohere.com/reference/chat - v2 accepts response_format,
    logprobs, k, tool_choice (REQUIRED/NONE) and thinking."""

    def test_response_format_and_logprobs_survive_drop_params(self):
        optional_params = _optional_params(
            "command-a-03-2025",
            "cohere_chat",
            response_format={"type": "json_object"},
            logprobs=True,
        )
        assert optional_params["response_format"] == {"type": "json_object"}
        assert optional_params["logprobs"] is True

    @pytest.mark.parametrize(
        "tool_choice, expected",
        [("required", "REQUIRED"), ("none", "NONE")],
    )
    def test_tool_choice_is_mapped_to_cohere_casing(self, tool_choice, expected):
        optional_params = _optional_params("command-a-03-2025", "cohere_chat", tool_choice=tool_choice)
        assert optional_params["tool_choice"] == expected

    def test_tool_choice_auto_is_omitted(self):
        """Cohere has no AUTO; omitting the field is the equivalent behaviour."""
        optional_params = _optional_params("command-a-03-2025", "cohere_chat", tool_choice="auto")
        assert "tool_choice" not in optional_params

    def test_top_k_is_renamed_to_k(self):
        """`top_k` bypasses map_openai_params, so the rename happens in transform."""
        data = CohereV2ChatConfig().transform_request(
            model="command-a-03-2025",
            messages=[{"role": "user", "content": "hi"}],
            optional_params={"top_k": 40},
            litellm_params={},
            headers={},
        )
        assert data["k"] == 40
        assert "top_k" not in data

    @pytest.mark.parametrize("param, expected", [("top_p", "p"), ("n", "num_generations"), ("stop", "stop_sequences")])
    def test_renamed_params(self, param, expected):
        optional_params = _optional_params("command-a-03-2025", "cohere_chat", **{param: 1})
        assert optional_params[expected] == 1

    def test_reasoning_effort_maps_to_thinking(self):
        optional_params = _optional_params("command-a-reasoning-08-2025", "cohere_chat", reasoning_effort="high")
        assert optional_params["thinking"]["type"] == "enabled"
        assert optional_params["thinking"]["token_budget"] > 0

    def test_reasoning_effort_none_disables_thinking(self):
        optional_params = _optional_params("command-a-reasoning-08-2025", "cohere_chat", reasoning_effort="none")
        assert optional_params["thinking"] == {"type": "disabled"}

    def test_unmapped_reasoning_effort_is_ignored(self):
        """Cohere has no budget for 'minimal'-below efforts we don't recognise."""
        assert THINKING_BY_REASONING_EFFORT.get("nonsense") is None

    def test_non_reasoning_model_does_not_advertise_thinking(self):
        supported = CohereV2ChatConfig().get_supported_openai_params("command-a-03-2025")
        assert "thinking" not in supported
        assert "reasoning_effort" not in supported


class TestZAI:
    """Ref: https://docs.z.ai/api-reference/llm/chat-completion - GLM accepts
    response_format; reasoning_effort is native to GLM-5.2+, earlier reasoning
    models only expose the thinking object."""

    def test_response_format_survives_drop_params(self):
        optional_params = _optional_params("glm-4.6", "zai", response_format={"type": "json_object"})
        assert optional_params["response_format"] == {"type": "json_object"}

    def test_max_completion_tokens_maps_to_max_tokens(self):
        optional_params = _optional_params("glm-4.6", "zai", max_completion_tokens=512)
        assert optional_params["max_tokens"] == 512
        assert "max_completion_tokens" not in optional_params

    def test_reasoning_effort_translated_to_thinking_before_glm_5_2(self):
        optional_params = _optional_params("glm-4.6", "zai", reasoning_effort="high")
        assert optional_params["thinking"] == {"type": "enabled"}
        assert "reasoning_effort" not in optional_params

    def test_reasoning_effort_none_disables_thinking(self):
        optional_params = _optional_params("glm-4.6", "zai", reasoning_effort="none")
        assert optional_params["thinking"] == {"type": "disabled"}

    def test_reasoning_effort_is_native_on_glm_5_2(self):
        optional_params = _optional_params("glm-5.2", "zai", reasoning_effort="high")
        assert optional_params["reasoning_effort"] == "high"
        assert "thinking" not in optional_params

    @pytest.mark.parametrize(
        "model, expected",
        [
            ("glm-4.5-air", False),
            ("glm-4.6", False),
            ("glm-5", False),
            ("glm-5.2", True),
            ("zai/glm-5.2", True),
            ("glm-5.3", True),
            ("not-a-glm-model", False),
        ],
    )
    def test_reasoning_effort_version_gate(self, model, expected):
        assert ZAIChatConfig._supports_reasoning_effort_param(model) is expected

    @pytest.mark.parametrize(
        "model, expected",
        [("glm-5.2", True), ("glm-4.5-air", True), ("glm-4-flash", False), ("not-a-glm-model", False)],
    )
    def test_reasoning_model_detection(self, model, expected):
        """Read from the id, so ids missing from the cost map still work."""
        assert ZAIChatConfig._is_reasoning_model(model) is expected


class TestMoonshot:
    """Ref: https://platform.kimi.ai/docs/api/chat - kimi-k2.5+ reject temperature,
    top_p, n, presence_penalty and frequency_penalty. kimi-k2.5/k2.6/k2.7 take a
    thinking object; kimi-k3 replaced it with reasoning_effort."""

    SAMPLING_PROBES = {
        "temperature": 0.5,
        "top_p": 0.9,
        "n": 2,
        "presence_penalty": 0.1,
        "frequency_penalty": 0.1,
    }

    @pytest.mark.parametrize("param", list(SAMPLING_PROBES))
    @pytest.mark.parametrize("model", ["kimi-k2.5", "kimi-k2.6", "kimi-k2.7-code", "kimi-k3"])
    def test_sampling_params_dropped_for_reasoning_models(self, model, param):
        supported = MoonshotChatConfig().get_supported_openai_params(model)
        assert param not in supported

        optional_params = _optional_params(model, "moonshot", **{param: self.SAMPLING_PROBES[param]})
        assert param not in optional_params

    @pytest.mark.parametrize("param", list(SAMPLING_PROBES))
    def test_sampling_params_kept_for_moonshot_v1(self, param):
        supported = MoonshotChatConfig().get_supported_openai_params("moonshot-v1-8k")
        assert param in supported

    @pytest.mark.parametrize("model", ["kimi-k2.5", "kimi-k2.6", "kimi-k2.7-code"])
    def test_thinking_supported_before_k3(self, model):
        optional_params = _optional_params(model, "moonshot", thinking={"type": "enabled"})
        assert optional_params["thinking"] == {"type": "enabled"}

    def test_k3_uses_reasoning_effort_not_thinking(self):
        supported = MoonshotChatConfig().get_supported_openai_params("kimi-k3")
        assert "reasoning_effort" in supported
        assert "thinking" not in supported

        optional_params = _optional_params("kimi-k3", "moonshot", reasoning_effort="high")
        assert optional_params["reasoning_effort"] == "high"

    @pytest.mark.parametrize(
        "model, expected",
        [
            ("kimi-k2", False),
            ("kimi-k2.5", True),
            ("kimi-k2.6", True),
            ("kimi-k2.7-code-highspeed", True),
            ("kimi-k3", True),
            ("moonshot-v1-8k", False),
            # only the model map marks this one; the version reads as 2.0
            ("kimi-k2-thinking", True),
        ],
    )
    def test_reasoning_model_detection(self, model, expected):
        assert MoonshotChatConfig._is_reasoning_model(model) is expected


class TestPerplexity:
    """Ref: https://docs.perplexity.ai/api-reference/chat-completions-post -
    `stop` is an accepted body param."""

    def test_stop_survives_drop_params(self):
        optional_params = _optional_params("sonar-pro", "perplexity", stop=["END"])
        assert optional_params["stop"] == ["END"]


class TestXAIPenalties:
    """Ref: https://docs.x.ai/docs/api-reference - presence_penalty and
    frequency_penalty are "Not supported by grok-3 and reasoning models"."""

    @pytest.mark.parametrize("model", ["grok-3", "grok-3-mini", "grok-4", "grok-4.3", "grok-code-fast-1"])
    @pytest.mark.parametrize("param", ["presence_penalty", "frequency_penalty"])
    def test_penalties_rejected(self, model, param):
        supported = XAIChatConfig().get_supported_openai_params(model=model)
        assert param not in supported

        optional_params = _optional_params(model, "xai", **{param: 0.2})
        assert param not in optional_params

    @pytest.mark.parametrize("param", ["presence_penalty", "frequency_penalty"])
    def test_penalties_kept_on_older_models(self, param):
        supported = XAIChatConfig().get_supported_openai_params(model="grok-beta")
        assert param in supported
