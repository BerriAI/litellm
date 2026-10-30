"""Boundary coverage for reasoning effort normalization on the ``/v1/messages`` adapter.

``test_reasoning_effort_fields.py`` pins ``normalize_reasoning_effort_value`` itself. These tests
sit one layer out, on the kwargs the handler actually hands to ``litellm.acompletion``, so the
regression they guard is the one a caller sees: a tier the proxy advertises has to be the tier that
leaves the adapter, in the shape the target expects.
"""

import functools
import json
from collections.abc import Mapping
from typing import Final

import httpx
import pytest
import respx

import litellm
from litellm.llms.anthropic.messages.adapters.handler import (
    LiteLLMMessagesToCompletionTransformationHandler,
)
from litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import VertexGeminiConfig
from litellm.router import Router
from litellm.utils import reasoning_effort_with_summary_alias

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


def _completion_kwargs_for_thinking(
    model: str,
    provider: str | None,
    thinking: dict[str, object],
    *,
    tools: list[dict[str, object]] | None = None,
    api_base: str | None = None,
    model_info: dict[str, str] | None = None,
) -> Mapping[str, object]:
    extra_kwargs: Final = {
        key: value
        for key, value in (("custom_llm_provider", provider), ("api_base", api_base), ("model_info", model_info))
        if value is not None
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
    return completion_kwargs


def _reasoning_effort_sent_for_thinking(
    model: str,
    provider: str | None,
    thinking: dict[str, object],
    *,
    tools: list[dict[str, object]] | None = None,
    api_base: str | None = None,
    model_info: dict[str, str] | None = None,
) -> object:
    return _completion_kwargs_for_thinking(
        model, provider, thinking, tools=tools, api_base=api_base, model_info=model_info
    ).get("reasoning_effort")


SUMMARIZED_THINKING = {"type": "enabled", "budget_tokens": 4096, "summary": "auto"}
PLAIN_THINKING = {"type": "enabled", "budget_tokens": 4096}
MULTIPLY_TOOL = {
    "name": "multiply",
    "description": "Multiply two integers",
    "input_schema": {"type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}}},
}


class TestAThinkingSummaryAskReachesGemini:
    """Gemini maps a summary ask to ``includeThoughts``, so ``gemini_include_thoughts_default`` only
    decides for a ``/v1/messages`` request that asks for nothing, the same as on chat completions."""

    @pytest.mark.parametrize(
        "model, provider",
        [("gemini/gemini-3.1-pro-preview", "gemini"), ("vertex_ai/gemini-2.5-flash", "vertex_ai")],
    )
    @pytest.mark.parametrize(
        "thinking, sends_thoughts",
        [
            (SUMMARIZED_THINKING, True),
            ({**PLAIN_THINKING, "display": "summarized"}, True),
            ({**PLAIN_THINKING, "display": "updates"}, True),
            ({**PLAIN_THINKING, "display": "omitted"}, False),
            (PLAIN_THINKING, False),
            ({"type": "disabled", "display": "summarized"}, False),
        ],
    )
    def test_the_request_ask_beats_the_gateway_default(
        self,
        local_model_cost_map: None,
        monkeypatch: pytest.MonkeyPatch,
        model: str,
        provider: str,
        thinking: dict[str, object],
        sends_thoughts: bool,
    ) -> None:
        monkeypatch.setattr(litellm, "gemini_include_thoughts_default", False)

        sent: Final = _completion_kwargs_for_thinking(model, provider, thinking)
        gemini: Final = VertexGeminiConfig()
        reasoning_effort: Final = reasoning_effort_with_summary_alias(
            sent.get("reasoning_effort"), sent, accepts_summary=gemini.accepts_reasoning_summary_in_reasoning_effort
        )
        mapped: Final = gemini.map_openai_params(
            {"reasoning_effort": reasoning_effort}, {}, model.split("/", 1)[1], True
        )

        assert mapped["thinkingConfig"]["includeThoughts"] is sends_thoughts
        assert not isinstance(sent.get("reasoning_effort"), dict)

    def test_auto_summary_leaves_the_gateway_default_in_charge(
        self, local_model_cost_map: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(litellm, "gemini_include_thoughts_default", False)
        monkeypatch.setenv("LITELLM_REASONING_AUTO_SUMMARY", "true")

        sent: Final = _reasoning_effort_sent_for_thinking("gemini/gemini-3.1-pro-preview", "gemini", PLAIN_THINKING)

        assert sent == "high"

    def test_a_display_ask_leaves_other_chat_targets_on_the_plain_tier(self, local_model_cost_map: None) -> None:
        sent: Final = _reasoning_effort_sent_for_thinking(
            "databricks/databricks-qwen35-122b-a10b", "databricks", {**PLAIN_THINKING, "display": "summarized"}
        )

        assert sent == "high"


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

        sent: Final = _reasoning_effort_sent_for_thinking(
            "databricks/databricks-qwen35-122b-a10b", "databricks", PLAIN_THINKING
        )

        assert sent == "high"

    def test_a_chat_pinned_deployment_gets_the_plain_tier_on_a_responses_row(
        self, local_model_cost_map: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A ``mode: chat`` pin sends that deployment to chat completions even though its shared row
        says ``responses``, so the adapter has to read the pin too or chat gets the Responses-only dict."""
        monkeypatch.setenv("LITELLM_REASONING_AUTO_SUMMARY", "true")
        model: Final = "bedrock_mantle/openai.gpt-5.4"

        unpinned = _reasoning_effort_sent_for_thinking(model, "bedrock_mantle", PLAIN_THINKING)
        pinned = _reasoning_effort_sent_for_thinking(
            model, "bedrock_mantle", PLAIN_THINKING, model_info={"id": "pinned-id", "mode": "chat"}
        )

        assert isinstance(unpinned, dict)
        assert pinned == "high"

    @pytest.mark.parametrize(
        "model, provider",
        [
            ("azure/responses/gpt-5-mini", "azure"),
            ("gpt-5-mini", "openai"),
            ("databricks/databricks-gpt-5-5", "databricks"),
        ],
    )
    def test_a_bridged_target_keeps_the_summary(self, local_model_cost_map: None, model: str, provider: str) -> None:
        sent: Final = _reasoning_effort_sent_for_thinking(model, provider, SUMMARIZED_THINKING)

        assert sent == {"effort": "high", "summary": "auto"}

    def test_auto_summary_still_reaches_a_bridged_target(
        self, local_model_cost_map: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("LITELLM_REASONING_AUTO_SUMMARY", "true")

        sent: Final = _reasoning_effort_sent_for_thinking("azure/responses/gpt-5-mini", "azure", PLAIN_THINKING)

        assert sent == {"effort": "high", "summary": "detailed"}

    @pytest.mark.parametrize(
        "model, provider, summary",
        [
            ("bedrock_mantle/openai.gpt-5.4", "bedrock_mantle", "auto"),
            ("azure/responses/gpt-5-mini", "azure", "detailed"),
        ],
    )
    def test_auto_summary_asks_each_bridged_target_for_a_level_it_accepts(
        self, local_model_cost_map: None, monkeypatch: pytest.MonkeyPatch, model: str, provider: str, summary: str
    ) -> None:
        """Mantle's OpenAI path accepts only ``auto``, so the opt-in summary must not send it ``detailed``."""
        monkeypatch.setenv("LITELLM_REASONING_AUTO_SUMMARY", "true")

        sent = _reasoning_effort_sent_for_thinking(model, provider, PLAIN_THINKING)

        assert sent == {"effort": "high", "summary": summary}

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
        sent: Final = _reasoning_effort_sent_for_thinking(
            "azure_ai/gpt-5.5", "azure_ai", SUMMARIZED_THINKING, tools=[MULTIPLY_TOOL], api_base=api_base
        )

        assert sent == expected

    def test_a_provider_resolved_from_the_api_base_gets_the_plain_tier(self, local_model_cost_map: None) -> None:
        sent: Final = _reasoning_effort_sent_for_thinking(
            "kimi-k3", None, SUMMARIZED_THINKING, api_base="https://api.together.xyz/v1"
        )

        assert sent == "high"

    def test_a_chained_gateway_keeps_the_dict_for_its_own_bridge(self, local_model_cost_map: None) -> None:
        sent: Final = _reasoning_effort_sent_for_thinking("litellm_proxy/gpt-5.4", "litellm_proxy", SUMMARIZED_THINKING)

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
        sent: Final = _reasoning_effort_sent(model, provider, {"effort": "max", "summary": "detailed"})

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
            ("gpt-5.5", "azure_ai"),
            ("azure_ai/gpt-5.5", "azure_ai"),
            ("gpt-5.6", "azure"),
            ("us/gpt-5.6", "azure"),
        ],
    )
    def test_an_azure_hosted_deployment_without_the_flag_drops_to_the_tier_it_accepts(
        self, local_model_cost_map, model, provider
    ):
        """Chat Completions answers ``max`` on gpt-5.6 with a 400 naming ``none`` through ``xhigh``
        (OpenAI, 2026-10-01) and none of these Azure rows declares the level, so the adapter sends
        the tier below instead of the rejected one."""
        assert _reasoning_effort_sent(model, provider, "max") == "xhigh"

    @pytest.mark.parametrize(
        "model, provider",
        [
            ("gpt-6-astra", "azure_ai"),
            ("azure_ai/gpt-6-astra", "azure_ai"),
            ("gpt-6-astra", "azure"),
            ("us/gpt-6-astra", "azure"),
            ("eu/gpt-6-astra", "azure"),
        ],
    )
    def test_an_azure_hosted_astra_deployment_drops_to_the_tier_its_chat_route_accepts(
        self, local_model_cost_map: None, model: str, provider: str
    ) -> None:
        """A Foundry gpt-6-astra answers a Chat Completions ``max`` with a 400 naming low through
        xhigh while honoring the same level on the Responses route (2026-10-01). The row's flag is
        route-blind, so it keeps max off and the adapter, which cannot know which route the call
        takes, sends the tier below instead of the level chat rejects."""
        assert _reasoning_effort_sent(model, provider, "max") == "xhigh"

    def test_the_openai_hosted_twin_still_sends_max(self, local_model_cost_map):
        assert _reasoning_effort_sent("gpt-6-astra", "openai", "max") == "max"


_CHAT_COMPLETION: Final = {
    "id": "chatcmpl-1",
    "object": "chat.completion",
    "created": 1,
    "model": "my-kimi",
    "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
}
_RESPONSE: Final = {
    "id": "resp_1",
    "object": "response",
    "created_at": 1,
    "status": "completed",
    "model": "gpt-5.5-pro",
    "output": [
        {
            "type": "message",
            "id": "msg_1",
            "status": "completed",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "ok", "annotations": []}],
        }
    ],
    "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
}
_KIMI_LEVELS: Final = {"reasoning_effort_levels": ["low", "high", "max", "none"]}
_SELF_HOSTED: Final = {
    "model": "hosted_vllm/my-kimi",
    "api_base": "https://vllm.example.test/v1",
    "api_key": "fake-key",
}
_VLLM_CHAT_URL: Final = r"https://vllm\.example\.test/v1/chat/completions"
_MALFORMED_BUDGET_TARGETS: Final = [
    pytest.param(_SELF_HOSTED, _KIMI_LEVELS, _VLLM_CHAT_URL, _CHAT_COMPLETION, id="declaring-hosted_vllm"),
    pytest.param(_SELF_HOSTED, {}, _VLLM_CHAT_URL, _CHAT_COMPLETION, id="hosted_vllm"),
    pytest.param(
        {"model": "fireworks_ai/accounts/fireworks/models/kimi-k3", "api_key": "fake-key"},
        {},
        r"https://api\.fireworks\.ai/inference/v1/chat/completions",
        _CHAT_COMPLETION,
        id="fireworks_ai",
    ),
    pytest.param(
        {"model": "openai/gpt-5.5-pro", "api_key": "fake-key"},
        {},
        r"https://api\.openai\.com/v1/responses",
        _RESPONSE,
        id="openai-responses-bridge",
    ),
]


async def _body_sent_upstream(
    respx_mock: respx.MockRouter,
    litellm_params: Mapping[str, object],
    model_info: Mapping[str, object],
    upstream_url: str,
    upstream_body: Mapping[str, object],
    **messages_params: object,
) -> Mapping[str, object]:
    route: Final = respx_mock.post(url__regex=upstream_url).mock(return_value=httpx.Response(200, json=upstream_body))
    router: Final = Router(
        model_list=[{"model_name": "group", "litellm_params": {**litellm_params}, "model_info": {**model_info}}]
    )
    await router.aanthropic_messages(model="group", messages=MESSAGES, max_tokens=4096, **messages_params)
    return json.loads(route.calls.last.request.read())


class TestADeclaredDeploymentRoundsTheMessagesTierOntoItsLevels:
    @pytest.fixture(autouse=True)
    def _httpx_transport(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "messages_params, expected",
        [
            ({"thinking": {"type": "enabled", "budget_tokens": 2048}}, "high"),
            ({"thinking": {"type": "adaptive"}, "output_config": {"effort": "xhigh"}}, "max"),
        ],
    )
    async def test_a_self_hosted_deployment_receives_its_declared_level(
        self, respx_mock: respx.MockRouter, messages_params: Mapping[str, object], expected: str
    ) -> None:
        sent: Final = await _body_sent_upstream(
            respx_mock,
            {"model": "hosted_vllm/my-kimi", "api_base": "https://vllm.example.test/v1", "api_key": "fake-key"},
            _KIMI_LEVELS,
            r"https://vllm\.example\.test/v1/chat/completions",
            _CHAT_COMPLETION,
            **messages_params,
        )

        assert sent["reasoning_effort"] == expected

    @pytest.mark.asyncio
    @pytest.mark.parametrize("model_info", [_KIMI_LEVELS, {}])
    async def test_a_claude_named_self_hosted_deployment_maps_a_budget_like_any_other_model_name(
        self, respx_mock: respx.MockRouter, model_info: Mapping[str, object]
    ) -> None:
        thinking: Final = {"type": "enabled", "budget_tokens": 9000}

        claude_named: Final = await _body_sent_upstream(
            respx_mock,
            {**_SELF_HOSTED, "model": "hosted_vllm/claude-opus-5-5"},
            model_info,
            _VLLM_CHAT_URL,
            _CHAT_COMPLETION,
            thinking=thinking,
        )
        kimi_named: Final = await _body_sent_upstream(
            respx_mock, _SELF_HOSTED, model_info, _VLLM_CHAT_URL, _CHAT_COMPLETION, thinking=thinking
        )

        assert claude_named["reasoning_effort"] == kimi_named["reasoning_effort"]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("budget", ["2048", True])
    @pytest.mark.parametrize("litellm_params, model_info, upstream_url, upstream_body", _MALFORMED_BUDGET_TARGETS)
    async def test_a_non_numeric_budget_is_rejected_with_a_400(
        self,
        local_model_cost_map: None,
        respx_mock: respx.MockRouter,
        litellm_params: Mapping[str, object],
        model_info: Mapping[str, object],
        upstream_url: str,
        upstream_body: Mapping[str, object],
        budget: object,
    ) -> None:
        with pytest.raises(litellm.BadRequestError, match=r"thinking\.budget_tokens must be a number") as rejected:
            await _body_sent_upstream(
                respx_mock,
                litellm_params,
                model_info,
                upstream_url,
                upstream_body,
                thinking={"type": "enabled", "budget_tokens": budget},
            )

        assert rejected.value.status_code == 400

    @pytest.mark.asyncio
    @pytest.mark.parametrize("budget", ["2048", True])
    @pytest.mark.parametrize("litellm_params, model_info, upstream_url, upstream_body", _MALFORMED_BUDGET_TARGETS)
    async def test_drop_params_drops_a_non_numeric_budget(
        self,
        local_model_cost_map: None,
        respx_mock: respx.MockRouter,
        litellm_params: Mapping[str, object],
        model_info: Mapping[str, object],
        upstream_url: str,
        upstream_body: Mapping[str, object],
        budget: object,
    ) -> None:
        sent: Final = await _body_sent_upstream(
            respx_mock,
            {**litellm_params, "drop_params": True},
            model_info,
            upstream_url,
            upstream_body,
            thinking={"type": "enabled", "budget_tokens": budget},
        )

        assert not {"reasoning_effort", "reasoning", "thinking"} & sent.keys()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "model, api_base, upstream_url, upstream_body, effort_path",
        [
            ("openai/gpt-5.5-pro", None, r"https://api\.openai\.com/v1/responses", _RESPONSE, ("reasoning", "effort")),
            (
                "hosted_vllm/my-kimi",
                "https://vllm.example.test/v1",
                r"https://vllm\.example\.test/v1/chat/completions",
                _CHAT_COMPLETION,
                ("reasoning_effort",),
            ),
        ],
    )
    async def test_the_deployment_declaration_beats_the_flag_based_chain(
        self,
        local_model_cost_map: None,
        respx_mock: respx.MockRouter,
        model: str,
        api_base: str | None,
        upstream_url: str,
        upstream_body: Mapping[str, object],
        effort_path: tuple[str, ...],
    ) -> None:
        """The map says gpt-5.5-pro refuses minimal and knows nothing of my-kimi, so the chain alone
        would send medium and low."""
        route: Final = respx_mock.post(url__regex=upstream_url).mock(
            return_value=httpx.Response(200, json=upstream_body)
        )

        await litellm.anthropic.messages.acreate(
            model=model,
            api_base=api_base,
            api_key="fake-key",
            messages=MESSAGES,
            max_tokens=4096,
            thinking={"type": "adaptive"},
            output_config={"effort": "minimal"},
            model_info={"reasoning_effort_levels": ["minimal", "medium", "high"]},
        )

        sent: Final = json.loads(route.calls.last.request.read())
        assert functools.reduce(lambda body, key: body[key], effort_path, sent) == "minimal"
