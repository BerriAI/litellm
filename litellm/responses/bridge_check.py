from collections.abc import Mapping, Sequence
from typing import Final, cast  # noqa: TID251  # moved unchanged from litellm/main.py
from urllib.parse import urlsplit

import litellm
from litellm._logging import verbose_logger
from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider
from litellm.llms.azure_ai.chat.transformation import AzureAIGPT5Config
from litellm.llms.azure_ai.common_utils import (
    azure_ai_supports_native_responses,
    foundry_chat_rejects_function_tools_while_reasoning,
)
from litellm.llms.openai.chat.gpt_5_transformation import OpenAIGPT5Config
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import OpenAIWebSearchOptions
from litellm.types.utils import LlmProviders
from litellm.utils import ProviderConfigManager, get_model_info_helper, model_cost_registration_row

_OPENAI_DEFAULT_API_BASE: Final = "https://api.openai.com/v1"
RESPONSES_ONLY_PROVIDERS: Final = frozenset({"chatgpt", "github_copilot"})
ROUTABLE_MODES: Final = frozenset({"chat", "responses"})


def deployment_routing_mode(deployment_model_info: Mapping[str, object] | None, custom_llm_provider: str) -> str | None:
    """The mode a router deployment's own ``model_info`` pins its chat requests to, or ``None`` to defer
    to the shared cost-map row. Lets two deployments of one backend model route differently without
    either rewriting the shared row the SDK and the other deployments read."""
    if deployment_model_info is None or custom_llm_provider in RESPONSES_ONLY_PROVIDERS:
        return None
    mode: Final = deployment_model_info.get("mode")
    return mode if isinstance(mode, str) and mode in ROUTABLE_MODES else None


def keep_registered_mode(backend_key: str, shared_model_info: Mapping[str, object]) -> Mapping[str, object]:
    """A deployment's ``chat`` or ``responses`` pin routes only that deployment, so it never replaces the
    routing mode registered for its shared backend row, which the SDK and every sibling deployment route by,
    and no other mode replaces a registered ``responses`` mode either.

    A ``responses`` pin still marks a row that has no routing mode, since that is how a model missing from
    the cost map opts into the Responses API. A mode the row inherits from the built-in entry its registration
    falls back to counts as registered, since the first registration writes it whichever deployment that is. A ``chatgpt`` or ``github_copilot`` row takes any mode until it
    is ``responses``, since those providers read it by name to pick their endpoint. Elsewhere a ``chat`` pin
    never marks a row, so the outcome does not depend on which deployment registers first. A ``mode`` that
    is not a string names no mode, so it never reaches the row."""
    deployment_mode: Final = shared_model_info.get("mode")
    without_mode: Final = {key: value for key, value in shared_model_info.items() if key != "mode"}
    if not isinstance(deployment_mode, str):
        return without_mode
    registered_mode: Final = model_cost_registration_row(
        backend_key, shared_model_info.get("litellm_provider", "")
    ).get("mode")
    if registered_mode != "responses" and (
        deployment_mode not in ROUTABLE_MODES or backend_key.split("/", 1)[0] in RESPONSES_ONLY_PROVIDERS
    ):
        return shared_model_info
    if registered_mode is None and deployment_mode == "chat":
        return without_mode
    if registered_mode is None:
        return shared_model_info
    return {**shared_model_info, "mode": registered_mode}


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


def _mantle_responses_series_name(model: str) -> str | None:
    """The bare OpenAI series name of a Bedrock Mantle model that also serves the Responses API, or ``None``."""
    from litellm.llms.bedrock_mantle.common_utils import mantle_supports_responses, split_mantle_region_prefix

    if not mantle_supports_responses(model, litellm.model_cost):
        return None
    return split_mantle_region_prefix(model)[1].removeprefix("openai.")


def str_keyed_mapping(value: object) -> Mapping[str, object] | None:
    """A request field such as a deployment's ``model_info`` when it is a mapping, read without narrowing the
    caller's variable (``litellm.completion`` reuses ``model_info`` for pricing registration)."""
    if not isinstance(value, Mapping):
        return None
    return cast(  # cast-ok: isinstance only narrows to Mapping[Unknown, Unknown]; request JSON keys are str
        Mapping[str, object], value
    )


def bridge_reasoning_effort(
    reasoning_effort: str | Mapping[str, object] | None, extra_body: object
) -> str | Mapping[str, object] | None:
    nested_body: Final = str_keyed_mapping(extra_body)
    if reasoning_effort is not None or nested_body is None:
        return reasoning_effort
    nested_effort: Final = nested_body.get("reasoning_effort")
    return nested_effort if isinstance(nested_effort, str) else str_keyed_mapping(nested_effort)


def backend_serves_chat_completions(shared_model_info: Mapping[str, object]) -> bool:
    """Whether the shared cost-map row lets a deployment pin ``mode: chat``. Responses-only backends
    (``gpt-5.1-codex``, ``o3-pro``, ``gpt-5.4-pro``) list only ``/v1/responses`` and must stay bridged, and
    a ``responses`` row with no ``supported_endpoints`` (regional Azure codex and pro, Databricks codex) is
    read the same way."""
    supported_endpoints: Final = shared_model_info.get("supported_endpoints")
    if not isinstance(supported_endpoints, Sequence) or isinstance(supported_endpoints, str):
        return shared_model_info.get("mode") != "responses"
    return "/v1/chat/completions" in supported_endpoints


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
    deployment_model_info: Mapping[str, object] | None = None,
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
            dict[str, object],
            get_model_info_helper(model=model, custom_llm_provider=custom_llm_provider),
        )
    except Exception as e:
        verbose_logger.debug("Error getting model info: %s", e)

    explicit_responses_prefix: Final = model.startswith("responses/")
    pinned_mode: Final = (
        None if explicit_responses_prefix else deployment_routing_mode(deployment_model_info, custom_llm_provider)
    )
    if explicit_responses_prefix or pinned_mode == "responses":
        model = model.replace("responses/", "")
        model_info["mode"] = "responses"
    elif pinned_mode == "chat" and backend_serves_chat_completions(model_info):
        model_info["mode"] = "chat"

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
    #   ``"none"`` the chat gate forwards keeps the request chat-servable: a ``none``
    #   the model's map row refuses is dropped (or refused) before the call, which
    #   leaves the provider's default effort in force. Custom (grammar) tools are served
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
    # - Bedrock Mantle serves the same API and enforces the gpt-5.4+ rule even with the effort
    #   unset (measured on gpt-5.6-terra), so a Mantle row that also serves Responses follows
    #   these rules, which keeps a deployment pinned to ``mode: chat`` from a 400 on them.
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
    mantle_series_name: Final = (
        _mantle_responses_series_name(model) if custom_llm_provider == "bedrock_mantle" else None
    )
    on_mantle_responses_endpoint: Final = mantle_series_name is not None
    summary_requested: Final = reasoning_summary is not None or (
        on_mantle_responses_endpoint
        and isinstance(reasoning_effort, Mapping)
        and reasoning_effort.get("summary") is not None
    )
    gpt_series_name: Final = mantle_series_name or model
    on_constraint_enforcing_endpoint: Final = (
        custom_llm_provider == "azure"
        or on_mantle_responses_endpoint
        or resolved_api_base == ""
        or _is_openai_backed_api_base(resolved_api_base)
    )
    chat_rejects_function_tools: Final = (
        has_function_tool
        and reasoning_active
        and (
            foundry_chat_rejects_function_tools_while_reasoning(model, reasoning_effort)
            if on_foundry_openai_endpoint
            else (
                OpenAIGPT5Config.is_model_gpt_5_4_plus_model(gpt_series_name)
                and (reasoning_effort is not None or on_constraint_enforcing_endpoint)
            )
        )
    )
    if (
        (custom_llm_provider in ("openai", "azure") or on_foundry_openai_endpoint or on_mantle_responses_endpoint)
        and model_info.get("mode") != "responses"
        and OpenAIGPT5Config.is_model_gpt_5_model(gpt_series_name)
        and not OpenAIGPT5Config.is_model_gpt_5_search_model(gpt_series_name)
        and (
            (reasoning_effort is not None and summary_requested)
            or chat_rejects_function_tools
            or (carries_tool_search and on_constraint_enforcing_endpoint)
            or (web_search_options is not None and on_mantle_responses_endpoint)
        )
    ):
        model_info["mode"] = "responses"
        model = model.replace("responses/", "")

    return model_info, model
