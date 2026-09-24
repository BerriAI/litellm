"""Boundary coverage for reasoning effort normalization on the ``/v1/messages`` adapter.

``test_reasoning_effort_fields.py`` pins ``normalize_reasoning_effort_value`` itself. These tests
sit one layer out, on the kwargs the handler actually hands to ``litellm.acompletion``, so the
regression they guard is the one a caller sees: a tier the proxy advertises has to be the tier that
leaves the adapter, in the shape the target expects.
"""

from typing import Final

import pytest

from litellm.llms.anthropic.experimental_pass_through.adapters.handler import (
    LiteLLMMessagesToCompletionTransformationHandler,
)

MESSAGES = [{"role": "user", "content": "hello"}]


def _reasoning_effort_sent(model: str, provider: str, reasoning_effort: object) -> object:
    completion_kwargs, _ = LiteLLMMessagesToCompletionTransformationHandler._prepare_completion_kwargs(
        max_tokens=1024,
        messages=MESSAGES,
        model=model,
        metadata=None,
        stop_sequences=None,
        stream=False,
        system=None,
        temperature=None,
        thinking=None,
        tool_choice=None,
        tools=None,
        top_k=None,
        top_p=None,
        output_format=None,
        extra_kwargs={"custom_llm_provider": provider, "reasoning_effort": reasoning_effort},
    )
    return completion_kwargs.get("reasoning_effort")


def _reasoning_effort_sent_for_thinking(
    model: str,
    provider: str | None,
    thinking: dict[str, object],
    *,
    tools: list[dict[str, object]] | None = None,
    api_base: str | None = None,
) -> object:
    extra_kwargs: Final = {
        key: value for key, value in (("custom_llm_provider", provider), ("api_base", api_base)) if value is not None
    }
    completion_kwargs, _ = LiteLLMMessagesToCompletionTransformationHandler._prepare_completion_kwargs(
        max_tokens=1024,
        messages=MESSAGES,
        model=model,
        metadata=None,
        stop_sequences=None,
        stream=False,
        system=None,
        temperature=None,
        thinking=thinking,
        tool_choice=None,
        tools=tools,
        top_k=None,
        top_p=None,
        output_format=None,
        extra_kwargs=extra_kwargs,
    )
    return completion_kwargs.get("reasoning_effort")


SUMMARIZED_THINKING = {"type": "enabled", "budget_tokens": 4096, "summary": "auto"}
PLAIN_THINKING = {"type": "enabled", "budget_tokens": 4096}
MULTIPLY_TOOL = {
    "name": "multiply",
    "description": "Multiply two integers",
    "input_schema": {"type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}}},
}


class TestTheSummaryWrappingOnlyRidesTheResponsesBridge:
    """Only the Responses API takes ``reasoning_effort`` as a dict. Databricks answered the wrapped
    ``{"effort", "summary"}`` with ``field 'reasoning_effort' expects input with json type 'string'
    but got 'object'``, so a target that stays on chat completions has to get the plain tier and a
    target the bridge picks up has to keep the summary it can honor."""

    @pytest.mark.parametrize(
        "model, provider",
        [
            ("databricks/databricks-qwen35-122b-a10b", "databricks"),
            ("databricks-qwen35-122b-a10b", "databricks"),
            ("fireworks_ai/kimi-k3", "fireworks_ai"),
        ],
    )
    def test_a_chat_target_gets_the_plain_tier(self, local_model_cost_map: None, model: str, provider: str) -> None:
        assert _reasoning_effort_sent_for_thinking(model, provider, SUMMARIZED_THINKING) == "high"

    def test_auto_summary_stays_a_plain_tier_on_a_chat_target(
        self, local_model_cost_map: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("LITELLM_REASONING_AUTO_SUMMARY", "true")

        sent = _reasoning_effort_sent_for_thinking("databricks/databricks-qwen35-122b-a10b", "databricks", PLAIN_THINKING)

        assert sent == "high"

    @pytest.mark.parametrize(
        "model, provider",
        [
            ("azure/responses/gpt-5-mini", "azure"),
            ("gpt-5-mini", "openai"),
            ("databricks/databricks-gpt-5-5", "databricks"),
        ],
    )
    def test_a_bridged_target_keeps_the_summary(self, local_model_cost_map: None, model: str, provider: str) -> None:
        sent = _reasoning_effort_sent_for_thinking(model, provider, SUMMARIZED_THINKING)

        assert sent == {"effort": "high", "summary": "auto"}

    def test_auto_summary_still_reaches_a_bridged_target(
        self, local_model_cost_map: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("LITELLM_REASONING_AUTO_SUMMARY", "true")

        sent = _reasoning_effort_sent_for_thinking("azure/responses/gpt-5-mini", "azure", PLAIN_THINKING)

        assert sent == {"effort": "high", "summary": "detailed"}

    @pytest.mark.parametrize(
        "api_base, expected",
        [
            ("https://foo.services.ai.azure.com/openai/v1", "high"),
            ("https://foo.eastus.models.ai.azure.com", {"effort": "high", "summary": "auto"}),
        ],
    )
    def test_a_foundry_deployment_is_judged_by_its_api_base(
        self, local_model_cost_map: None, api_base: str, expected: object
    ) -> None:
        """``completion()`` keeps a gpt-5.5 deployment with function tools on Foundry's chat route when
        its ``api_base`` is a Foundry OpenAI host, and bridges it to Responses when the base makes it an
        Azure OpenAI deployment. The adapter has to read the same ``api_base`` to land on the same call."""
        sent = _reasoning_effort_sent_for_thinking(
            "azure_ai/gpt-5.5", "azure_ai", SUMMARIZED_THINKING, tools=[MULTIPLY_TOOL], api_base=api_base
        )

        assert sent == expected

    def test_a_provider_resolved_from_the_api_base_gets_the_plain_tier(self, local_model_cost_map: None) -> None:
        sent = _reasoning_effort_sent_for_thinking(
            "kimi-k3", None, SUMMARIZED_THINKING, api_base="https://api.together.xyz/v1"
        )

        assert sent == "high"

    def test_a_chained_gateway_keeps_the_dict_for_its_own_bridge(self, local_model_cost_map: None) -> None:
        sent = _reasoning_effort_sent_for_thinking("litellm_proxy/gpt-5.4", "litellm_proxy", SUMMARIZED_THINKING)

        assert sent == {"effort": "high", "summary": "auto"}


class TestTheNormalizedTierIsTheTierSent:
    """The bug in the caller's terms: a proxy advertising kimi-k3 ``max`` accepted the request and
    then put ``high`` on the wire. Every spelling of the entry has to survive the adapter, including
    the provider-prefixed model name the handler is actually called with."""

    @pytest.mark.parametrize(
        "model, provider",
        [
            ("kimi-k3", "moonshot"),
            ("kimi-k3", "fireworks_ai"),
            ("fireworks_ai/kimi-k3", "fireworks_ai"),
            ("kimi-k3-us", "fireworks_ai"),
            ("FW-Kimi-K3", "azure_ai"),
        ],
    )
    def test_a_declared_tier_reaches_the_outgoing_request(self, local_model_cost_map, model, provider):
        assert _reasoning_effort_sent(model, provider, "max") == "max"

    @pytest.mark.parametrize("effort, expected", [("xhigh", "high"), ("minimal", "low")])
    def test_a_tier_the_entry_does_not_declare_still_degrades(self, local_model_cost_map, effort, expected):
        assert _reasoning_effort_sent("kimi-k3", "fireworks_ai", effort) == expected

    def test_the_fallback_is_a_tier_the_deployment_accepts(self, local_model_cost_map):
        """gpt-5.5-pro refuses ``low``, the floor the ``minimal`` chain used to stop on, so stopping
        there would have sent a level the model map says the model rejects."""
        assert _reasoning_effort_sent("gpt-5.5-pro", "azure", "minimal") == "medium"

    @pytest.mark.parametrize(
        "model, provider, expected",
        [("kimi-k3", "fireworks_ai", "max"), ("gpt-5-mini", "azure", "high")],
    )
    def test_the_dict_form_normalizes_effort_and_keeps_its_siblings(
        self, local_model_cost_map, model, provider, expected
    ):
        sent = _reasoning_effort_sent(model, provider, {"effort": "max", "summary": "detailed"})

        assert sent == {"effort": expected, "summary": "detailed"}

    @pytest.mark.parametrize(
        "model, provider, effort, expected",
        [("claude-opus-4-7", "anthropic", "max", "max"), ("gpt-5-mini", "azure", "max", "high")],
    )
    def test_an_entry_on_the_per_level_flags_is_unchanged(
        self, local_model_cost_map, model, provider, effort, expected
    ):
        assert _reasoning_effort_sent(model, provider, effort) == expected

    @pytest.mark.parametrize(
        "model, provider",
        [
            ("gpt-6-astra", "azure_ai"),
            ("azure_ai/gpt-6-astra", "azure_ai"),
            ("gpt-6-astra", "azure"),
            ("us/gpt-6-astra", "azure"),
        ],
    )
    def test_an_azure_hosted_astra_deployment_drops_to_the_tier_it_accepts(
        self, local_model_cost_map, model, provider
    ):
        """The deployment answers ``max`` with a 400 naming ``none`` through ``xhigh``, so the rows
        say so and the adapter sends the tier below instead of the rejected one."""
        assert _reasoning_effort_sent(model, provider, "max") == "xhigh"

    def test_the_openai_hosted_twin_still_sends_max(self, local_model_cost_map):
        assert _reasoning_effort_sent("gpt-6-astra", "openai", "max") == "max"
