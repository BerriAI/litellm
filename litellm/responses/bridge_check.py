from collections.abc import Mapping, Sequence
from typing import Final, cast  # noqa: TID251  # moved unchanged from litellm/main.py
from urllib.parse import urlsplit

import litellm
from litellm._logging import verbose_logger
from litellm.llms.azure_ai.common_utils import (
    azure_ai_supports_native_responses,
    foundry_chat_rejects_function_tools_while_reasoning,
)
from litellm.llms.openai.chat.gpt_5_transformation import OpenAIGPT5Config
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import OpenAIWebSearchOptions
from litellm.utils import get_model_info_helper

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


def responses_api_bridge_check(
    model: str,
    custom_llm_provider: str,
    web_search_options: OpenAIWebSearchOptions | None = None,
    tools: Sequence[Mapping[str, object]] | None = None,
    reasoning_effort: str | Mapping[str, object] | None = None,
    reasoning_summary: object | None = None,
    api_base: str | None = None,
) -> tuple[dict, str]:
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

    # OpenAI/Azure GPT-5 chat-completions that need Responses-only fields (e.g.
    # ``reasoningSummary`` in ``extra_body``) must be bridged; Chat Completions rejects
    # those keys.
    #
    # - gpt-5.4+: FUNCTION tools with reasoning active must be bridged. OpenAI enables
    #   reasoning by default for these models (unset reasoning_effort means medium
    #   server-side), and Chat Completions rejects function tools whenever reasoning is
    #   on ("Function tools with reasoning_effort are not supported ... use
    #   /v1/responses or set reasoning_effort to 'none'"), so only an explicit
    #   ``"none"`` keeps the request chat-servable. Custom (grammar) tools are served
    #   natively by Chat Completions with reasoning on, so custom-only requests stay on
    #   chat and keep their native custom tool_call response shape.
    # - The UNSET-effort arm only fires against endpoints known to enforce that
    #   constraint (any api.openai.com host, or Azure OpenAI where api_base is
    #   always set): chat-only OpenAI-compatible backends registered under the openai
    #   provider with a custom api_base and gpt-5.4+ model names serve tools without
    #   reasoning fine and have no /responses route, so they keep pre-existing
    #   behavior (bridge only on an explicit reasoning_effort).
    # - Azure AI Foundry's OpenAI v1 hosts (azure_ai provider) enforce it later in the series:
    #   an explicit effort with function tools is rejected from gpt-5.6 on, and the unset
    #   effort only from gpt-6 on (gpt-5.6 serves tools with reasoning silently off), so the
    #   azure_ai gate keys on those measured boundaries instead of gpt-5.4+.
    # - Older GPT-5 names (e.g. ``gpt-5``, ``gpt-5.1``): bridge only when a reasoning
    #   summary alias is present with ``reasoning_effort`` (tools alone stay on chat).
    has_function_tool: Final = any(
        (
            tool.get("type") == "function" and (isinstance(tool.get("function"), dict) or "name" in tool)
            if isinstance(tool, dict)
            else getattr(tool, "type", None) == "function"
        )
        for tool in (tools or ())
    )
    if isinstance(reasoning_effort, dict):
        reasoning_active = reasoning_effort.get("effort") != "none" or reasoning_effort.get("summary") is not None
    else:
        reasoning_active = reasoning_effort != "none"
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
                and (reasoning_effort is not None or on_constraint_enforcing_endpoint)
            )
        )
    )
    if (
        (custom_llm_provider in ("openai", "azure") or on_foundry_openai_endpoint)
        and model_info.get("mode") != "responses"
        and OpenAIGPT5Config.is_model_gpt_5_model(model)
        and not OpenAIGPT5Config.is_model_gpt_5_search_model(model)
        and ((reasoning_effort is not None and reasoning_summary is not None) or chat_rejects_function_tools)
    ):
        model_info["mode"] = "responses"
        model = model.replace("responses/", "")

    return model_info, model
