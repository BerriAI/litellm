from collections.abc import Mapping, Sequence
from typing import Final, cast  # noqa: TID251  # moved unchanged from litellm/main.py
from urllib.parse import urlsplit

import litellm
from litellm._logging import verbose_logger
from litellm.litellm_core_utils.get_llm_provider_logic import (
    get_llm_provider,
    resolve_model_and_provider_for_metadata,
)
from litellm.llms.azure_ai.chat.transformation import AzureAIGPT5Config
from litellm.llms.azure_ai.common_utils import (
    azure_ai_supports_native_responses,
    foundry_chat_rejects_function_tools_while_reasoning,
)
from litellm.llms.openai.chat.gpt_5_transformation import OpenAIGPT5Config
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import OpenAIWebSearchOptions
from litellm.types.utils import LlmProviders
from litellm.utils import ProviderConfigManager, get_model_info_helper

_OPENAI_DEFAULT_API_BASE: Final = "https://api.openai.com/v1"
_OPENAI_API_HOST: Final = "api.openai.com"


def _is_openai_backed_api_base(api_base: str) -> bool:
    hostname: Final = urlsplit(api_base).hostname
    return hostname is not None and (hostname == _OPENAI_API_HOST or hostname.endswith(f".{_OPENAI_API_HOST}"))


def _resolve_openai_api_base(api_base: str | None) -> str:
    """Effective OpenAI base a chat request will hit: arg > global > env > default. The bridge gate
    and the ``_complete_custom_openai`` chat handler MUST resolve this identically, or a custom base
    set via ``litellm.api_base`` or ``OPENAI_BASE_URL``/``OPENAI_API_BASE`` is invisible to the gate,
    which then misreads it as the default OpenAI endpoint and bridges a request the backend can't serve."""
    return (
        api_base
        or litellm.api_base
        or get_secret_str("OPENAI_BASE_URL")
        or get_secret_str("OPENAI_API_BASE")
        or _OPENAI_DEFAULT_API_BASE
    )


def chat_target_accepts_reasoning_summary(model: str, custom_llm_provider: str | None) -> bool:
    try:
        local_model, provider, _, _ = get_llm_provider(model=model, custom_llm_provider=custom_llm_provider)
        chat_config: Final = ProviderConfigManager.get_provider_chat_config(
            model=local_model, provider=LlmProviders(provider)
        )
    except Exception as e:  # noqa: BLE001  # a capability probe must never fail the request it probes for
        verbose_logger.debug("chat target reasoning summary check failed: %s", e)
        return False
    return chat_config is not None and chat_config.accepts_reasoning_summary_in_reasoning_effort


def _chat_gate_refuses_reasoning_effort_none(model: str, custom_llm_provider: str) -> bool:
    match custom_llm_provider:
        case "azure":
            return litellm.AzureOpenAIGPT5Config.refuses_reasoning_effort_level(model, "none")
        case "azure_ai":
            return AzureAIGPT5Config.refuses_reasoning_effort_level(model, "none")
        case _:
            return OpenAIGPT5Config.refuses_reasoning_effort_level(model, "none")


def responses_api_bridge_check(
    model: str,
    custom_llm_provider: str,
    web_search_options: OpenAIWebSearchOptions | None = None,
    tools: Sequence[Mapping[str, object]] | None = None,
    reasoning_effort: str | Mapping[str, object] | None = None,
    reasoning_summary: object | None = None,
    api_base: str | None = None,
) -> tuple[dict[str, object], str]:
    model_info: dict[str, object] = {}

    # Global flag: route ALL OpenAI chat completions through Responses API.
    # Returns early with minimal model_info; callers only inspect the "mode" key.
    if litellm.route_all_chat_openai_to_responses and custom_llm_provider == "openai":
        model = model.replace("responses/", "")
        model_info["mode"] = "responses"
        return model_info, model

    try:
        model_info = cast(
            dict,
            get_model_info_helper(model=model, custom_llm_provider=custom_llm_provider),
        )
        if model_info.get("mode") is None and model.startswith("responses/"):
            model = model.replace("responses/", "")
            mode = "responses"
            model_info["mode"] = mode

    except Exception as e:
        verbose_logger.debug("Error getting model info: %s", e)

        if model.startswith("responses/"):  # handle azure models - `azure/responses/<deployment-name>`
            model = model.replace("responses/", "")
            mode = "responses"
            model_info["mode"] = mode

    if web_search_options is not None and custom_llm_provider == "xai":
        model_info["mode"] = "responses"
        model = model.replace("responses/", "")

    # Function tools with explicit active reasoning bridge on GPT-5.4+. With effort unset,
    # OpenAI/Azure and Responses-capable Mantle endpoints bridge only GPT-5.6+ whose map
    # default is not "none". Custom OpenAI-compatible endpoints keep unset effort on chat.
    # Foundry uses its separate measured boundaries. Custom-only tools remain chat-native;
    # reasoning summaries and tool_search retain their independent Responses requirements.
    has_function_tool: Final = any(
        (
            tool.get("type") == "function" and (isinstance(tool.get("function"), dict) or "name" in tool)
            if isinstance(tool, dict)
            else getattr(tool, "type", None) == "function"
        )
        for tool in (tools or ())
    )
    carries_tool_search: Final = any(
        (tool.get("type") if isinstance(tool, dict) else getattr(tool, "type", None)) == "tool_search"
        for tool in (tools or ())
    )
    requested_effort: Final = reasoning_effort.get("effort") if isinstance(reasoning_effort, dict) else reasoning_effort
    requested_summary: Final = reasoning_effort.get("summary") if isinstance(reasoning_effort, dict) else None
    reasoning_active: Final = (
        requested_effort != "none"
        or requested_summary is not None
        or _chat_gate_refuses_reasoning_effort_none(model, custom_llm_provider)
    )
    # The reasoning+tools constraint is enforced by the real OpenAI backend behind any api.openai.com
    # host (the default URL or a PrivateLink hostname such as <region>.privatelink.api.openai.com) and
    # by Azure OpenAI through the azure provider. Resolve the effective OpenAI base arg>global>env>default
    # exactly as the chat handler does, so a custom base set via litellm.api_base or
    # OPENAI_BASE_URL/OPENAI_API_BASE isn't misread as the default and bridged to a /responses route it
    # lacks. A whitespace-only base collapses to the default too.
    resolved_api_base: Final = _resolve_openai_api_base(api_base).strip()
    on_foundry_openai_endpoint: Final = custom_llm_provider == "azure_ai" and azure_ai_supports_native_responses(
        model, api_base
    )
    on_constraint_enforcing_endpoint: Final = (
        custom_llm_provider == "azure" or resolved_api_base == "" or _is_openai_backed_api_base(resolved_api_base)
    )
    chat_rejects_function_tools: Final = (
        has_function_tool
        and reasoning_active
        and (
            foundry_chat_rejects_function_tools_while_reasoning(model, reasoning_effort)
            if on_foundry_openai_endpoint
            else (
                OpenAIGPT5Config.is_model_gpt_5_4_plus_model(model)
                and (
                    reasoning_effort is not None
                    or (
                        on_constraint_enforcing_endpoint
                        and OpenAIGPT5Config.is_model_gpt_5_6_plus_model(model)
                        and model_info.get("default_reasoning_effort") != "none"
                    )
                )
            )
        )
    )
    if (
        (custom_llm_provider in ("openai", "azure") or on_foundry_openai_endpoint)
        and model_info.get("mode") != "responses"
        and OpenAIGPT5Config.is_model_gpt_5_model(model)
        and not OpenAIGPT5Config.is_model_gpt_5_search_model(model)
        and (
            (reasoning_effort is not None and reasoning_summary is not None)
            or chat_rejects_function_tools
            or (carries_tool_search and on_constraint_enforcing_endpoint)
        )
    ):
        model_info["mode"] = "responses"
        model = model.replace("responses/", "")

    return model_info, model


def chat_completion_ends_in_responses_api(
    model: str | None,
    custom_llm_provider: str | None,
    *,
    tools: Sequence[Mapping[str, object]] | None = None,
    reasoning_effort: str | Mapping[str, object] | None = None,
    reasoning_summary: object | None = None,
    api_base: str | None = None,
) -> bool:
    """Whether chat completion for this model is served by a Responses API.

    Encrypted reasoning can be replayed only there. A normal chat provider cannot
    verify the block, so the messages adapter strips it. ``chatgpt``, a ``responses/``
    model, and a request whose tools or reasoning effort the bridge sends to Responses keep it.
    """
    if model is None and custom_llm_provider is None:
        return False
    try:
        resolved_model, provider = resolve_model_and_provider_for_metadata(model or "", custom_llm_provider)
    except Exception as error:  # noqa: BLE001  # unknown model names must not fail message translation
        verbose_logger.debug("responses replay check failed for %s: %s", model, error)
        return custom_llm_provider == LlmProviders.CHATGPT.value or (
            isinstance(model, str) and (model == "chatgpt" or model.startswith("chatgpt/"))
        )
    model_info, _resolved = responses_api_bridge_check(
        model=resolved_model,
        custom_llm_provider=provider,
        tools=tools,
        reasoning_effort=reasoning_effort,
        reasoning_summary=reasoning_summary,
        api_base=api_base,
    )
    return model_info.get("mode") == "responses"
