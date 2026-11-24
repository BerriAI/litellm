from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Final, cast  # noqa: TID251  # moved unchanged from litellm/main.py
from urllib.parse import urlsplit

import litellm
from litellm._logging import verbose_logger
from litellm.litellm_core_utils.core_helpers import as_str_mapping, drops_unsupported_params
from litellm.litellm_core_utils.get_llm_provider_logic import (
    get_llm_provider,
    resolve_model_and_provider_for_metadata,
)
from litellm.litellm_core_utils.get_supported_openai_params import get_supported_openai_params
from litellm.litellm_core_utils.prompt_templates.common_utils import is_whitespace_user_message
from litellm.litellm_core_utils.prompt_templates.mid_conversation_system import message_field
from litellm.litellm_core_utils.reasoning_effort_capability import deployment_reasoning_effort_levels
from litellm.llms.azure_ai.chat.transformation import AzureAIGPT5Config
from litellm.llms.azure_ai.common_utils import (
    azure_ai_supports_native_responses,
    foundry_chat_rejects_function_tools_while_reasoning,
    is_foundry_project_agent,
)
from litellm.llms.openai.chat.gpt_reasoning_transformation import OpenAIGPT5Config
from litellm.llms.vertex_ai.common_utils import VertexAIModelRoute, get_vertex_ai_model_route
from litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import VertexGeminiConfig
from litellm.llms.vertex_ai.responses.transformation import is_vertex_muse_model
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import ChatCompletionUserMessage, OpenAIWebSearchOptions
from litellm.types.utils import LlmProviders
from litellm.utils import ProviderConfigManager, get_model_info_helper, model_cost_registration_row

_OPENAI_DEFAULT_API_BASE: Final = "https://api.openai.com/v1"
_REASONING_OBJECT_KEYS: Final = ("reasoning_effort", "reasoning")
_SDK_MERGED_REASONING_KEYS: Final = ("reasoning",)
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


def _extra_body_reasoning_effort(
    nested_body: Mapping[str, object], *, reaches_openai: bool
) -> str | Mapping[str, object] | None:
    nested_effort: Final = nested_body.get("reasoning_effort")
    if isinstance(nested_effort, str):
        return nested_effort
    nested_object: Final = str_keyed_mapping(nested_effort)
    if nested_object is not None or not reaches_openai:
        return nested_object
    return str_keyed_mapping(nested_body.get("reasoning"))


def bridge_reasoning_effort(
    reasoning_effort: str | Mapping[str, object] | None,
    extra_body: object,
    *,
    custom_llm_provider: str,
    api_base: str | None,
) -> str | Mapping[str, object] | None:
    nested_body: Final = str_keyed_mapping(extra_body)
    if nested_body is None:
        return reasoning_effort
    reaches_openai: Final = _reaches_openai(custom_llm_provider, api_base)
    nested: Final = _extra_body_reasoning_effort(nested_body, reaches_openai=reaches_openai)
    return nested if nested is not None and (reasoning_effort is None or reaches_openai) else reasoning_effort


def _targets_openai_api(api_base: str | None) -> bool:
    resolved_api_base: Final = _resolve_openai_api_base(api_base).strip()
    return resolved_api_base == "" or _is_openai_backed_api_base(resolved_api_base)


def _reaches_openai(custom_llm_provider: str, api_base: str | None) -> bool:
    return custom_llm_provider == "openai" and _targets_openai_api(api_base)


def _chat_refuses_effort_level(model: str, level: str) -> bool:
    if "reasoning_effort" not in (get_supported_openai_params(model=model, custom_llm_provider="openai") or ()):
        return True
    return OpenAIGPT5Config.is_model_gpt_5_model(model) and OpenAIGPT5Config.refuses_reasoning_effort_level(
        model, level
    )


def _without_refused_effort(
    reasoning: Mapping[str, object],
    *,
    model: str,
    deployment_model_info: Mapping[str, object] | None,
    refuses_level: Callable[[str, str], bool],
) -> Mapping[str, object]:
    effort: Final = reasoning.get("effort")
    if (
        not isinstance(effort, str)
        or effort in (deployment_reasoning_effort_levels(deployment_model_info) or ())
        or not refuses_level(model.removeprefix("responses/"), effort)
    ):
        return reasoning
    return {key: value for key, value in reasoning.items() if key != "effort"}


@dataclass(frozen=True, slots=True)
class _OpenAIReasoningJudge:
    model: str
    base_model: str | None
    api_base: str | None
    deployment_model_info: Mapping[str, object] | None

    def accepted(self, requested: Mapping[str, object], additional_drop_params: object) -> Mapping[str, object] | None:
        from litellm.responses.utils import bridge_reasoning

        return str_keyed_mapping(
            bridge_reasoning(
                requested=requested,
                effort_kept=True,
                summary_alias=None,
                model=self.model,
                base_model=self.base_model,
                custom_llm_provider="openai",
                api_base=self.api_base,
                deployment_model_info=self.deployment_model_info,
                drop_params=True,
                additional_drop_params=additional_drop_params,
            )
        )

    def _with_effort_checked(self, nested: Mapping[str, object]) -> Mapping[str, object]:
        return _without_refused_effort(
            nested,
            model=self.model,
            deployment_model_info=self.deployment_model_info,
            refuses_level=OpenAIGPT5Config.row_disables_reasoning_effort_level,
        )

    def cleaned(self, body: Mapping[str, object], keys: Sequence[str] = _REASONING_OBJECT_KEYS) -> Mapping[str, object]:
        judged: Final = {
            key: self.accepted(self._with_effort_checked(nested), additional_drop_params=None)
            for key in keys
            if (nested := str_keyed_mapping(body.get(key))) is not None
        }
        return {
            **{key: value for key, value in body.items() if key not in judged},
            **{key: value for key, value in judged.items() if value is not None},
        }


def drop_refused_extra_body_reasoning(
    params: Mapping[str, object],
    *,
    model: str,
    base_model: str | None,
    custom_llm_provider: str,
    api_base: str | None,
    deployment_model_info: Mapping[str, object] | None,
    drop_params: bool,
) -> Mapping[str, object]:
    if not drop_params or not _reaches_openai(custom_llm_provider, api_base):
        return params
    judge: Final = _OpenAIReasoningJudge(model, base_model, api_base, deployment_model_info)
    sdk_merged: Final = judge.cleaned(params, keys=_SDK_MERGED_REASONING_KEYS)
    extra_body: Final = str_keyed_mapping(params.get("extra_body"))
    return sdk_merged if extra_body is None else {**sdk_merged, "extra_body": judge.cleaned(extra_body)}


def drop_refused_forwarded_reasoning(forwarded_kwargs: Mapping[str, object], *, model: str) -> Mapping[str, object]:
    requested_api_base: Final = forwarded_kwargs.get("api_base")
    api_base: Final = requested_api_base if isinstance(requested_api_base, str) else None
    if (
        forwarded_kwargs.get("custom_llm_provider") != "openai"
        or not _targets_openai_api(api_base)
        or not drops_unsupported_params(forwarded_kwargs.get("drop_params"))
    ):
        return forwarded_kwargs
    base_model: Final = forwarded_kwargs.get("base_model")
    judge: Final = _OpenAIReasoningJudge(
        model=model.removeprefix("openai/"),
        base_model=base_model if isinstance(base_model, str) else None,
        api_base=api_base,
        deployment_model_info=str_keyed_mapping(forwarded_kwargs.get("model_info")),
    )
    requested: Final = str_keyed_mapping(forwarded_kwargs.get("reasoning_effort"))
    additional_drop_params: Final = forwarded_kwargs.get("additional_drop_params")
    extra_body: Final = str_keyed_mapping(forwarded_kwargs.get("extra_body"))
    return {
        **judge.cleaned(forwarded_kwargs, keys=_SDK_MERGED_REASONING_KEYS),
        **({} if requested is None else {"reasoning_effort": judge.accepted(requested, additional_drop_params)}),
        **({} if extra_body is None else {"extra_body": _with_responses_reasoning(judge.cleaned(extra_body))}),
    }


def _with_responses_reasoning(extra_body: Mapping[str, object]) -> Mapping[str, object]:
    effort_object: Final = str_keyed_mapping(extra_body.get("reasoning_effort"))
    if effort_object is None:
        return extra_body
    kept: Final = {key: value for key, value in extra_body.items() if key != "reasoning_effort"}
    return kept if str_keyed_mapping(kept.get("reasoning")) is not None else {**kept, "reasoning": effort_object}


def _with_chat_extra_body_effort(
    optional_params: dict[str, object],  # mutable-ok: the provider handlers write to this dict and logging shares it
    *,
    model: str,
    deployment_model_info: Mapping[str, object] | None,
) -> dict[str, object]:  # mutable-ok: the provider handlers write to this dict and logging shares it
    extra_body: Final = str_keyed_mapping(optional_params.get("extra_body"))
    objects: Final = (
        ()
        if extra_body is None
        else tuple(
            nested for key in _REASONING_OBJECT_KEYS if (nested := str_keyed_mapping(extra_body.get(key))) is not None
        )
    )
    if extra_body is None or not objects:
        return optional_params
    kept: Final = {
        key: value
        for key, value in extra_body.items()
        if key not in _REASONING_OBJECT_KEYS or str_keyed_mapping(value) is None
    }
    effort: Final = _without_refused_effort(
        objects[0],
        model=model,
        deployment_model_info=deployment_model_info,
        refuses_level=_chat_refuses_effort_level,
    ).get("effort")
    return {
        **optional_params,
        "extra_body": (
            kept if "reasoning_effort" in kept or not isinstance(effort, str) else {**kept, "reasoning_effort": effort}
        ),
    }


def _collapse_loses_fields(sent: object) -> bool:
    sent_object: Final = str_keyed_mapping(sent)
    return sent_object is not None and any(key != "effort" and value is not None for key, value in sent_object.items())


def settle_chat_reasoning_effort(
    optional_params: dict[str, object],  # mutable-ok: the provider handlers write to this dict and logging shares it
    *,
    requested: object,
    model: str,
    base_model: str | None,
    custom_llm_provider: str,
    api_base: str | None,
    deployment_model_info: Mapping[str, object] | None,
    drop_params: bool,
    additional_drop_params: object,
) -> dict[str, object]:  # mutable-ok: the provider handlers write to this dict and logging shares it
    chat_params: Final = (
        _with_chat_extra_body_effort(optional_params, model=model, deployment_model_info=deployment_model_info)
        if drop_params and _reaches_openai(custom_llm_provider, api_base)
        else optional_params
    )
    reasoning_object: Final = str_keyed_mapping(requested)
    if (
        reasoning_object is None
        or "reasoning_effort" not in chat_params
        or not _reaches_openai(custom_llm_provider, api_base)
    ):
        return chat_params

    from litellm.responses.utils import (
        RefusedReasoningFields,
        chat_reasoning_effort,
        raise_refused_reasoning_fields,
    )

    effort: Final = chat_reasoning_effort(
        requested=reasoning_object,
        model=model,
        base_model=base_model,
        custom_llm_provider=custom_llm_provider,
        api_base=api_base,
        deployment_model_info=deployment_model_info,
        drop_params=drop_params,
        additional_drop_params=additional_drop_params,
    )
    if isinstance(effort, RefusedReasoningFields):
        raise_refused_reasoning_fields(effort, model, custom_llm_provider)
    if effort is not None and (drop_params or not _collapse_loses_fields(chat_params["reasoning_effort"])):
        return {**chat_params, "reasoning_effort": effort}
    if not drop_params:
        return chat_params
    return {key: value for key, value in chat_params.items() if key != "reasoning_effort"}


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


def serves_only_responses(model: str, custom_llm_provider: str) -> bool:
    return custom_llm_provider == LlmProviders.VERTEX_AI.value and is_vertex_muse_model(model)


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
    if custom_llm_provider == LlmProviders.CHATGPT or explicit_responses_prefix or pinned_mode == "responses":
        model = model.replace("responses/", "")
        model_info["mode"] = "responses"
    elif pinned_mode == "chat" and backend_serves_chat_completions(model_info):
        model_info["mode"] = "chat"

    if custom_llm_provider == LlmProviders.AZURE_AI.value and is_foundry_project_agent(model):
        model_info["mode"] = "responses"

    if serves_only_responses(model, custom_llm_provider):
        model_info["mode"] = "responses"

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
        custom_llm_provider == "azure" or on_mantle_responses_endpoint or _targets_openai_api(api_base)
    )
    chat_rejects_function_tools: Final = (
        has_function_tool
        and reasoning_active
        and (
            foundry_chat_rejects_function_tools_while_reasoning(model, reasoning_effort)
            if on_foundry_openai_endpoint
            else (
                OpenAIGPT5Config.is_model_gpt_5_4_plus_model(gpt_series_name)
                and (
                    reasoning_effort is not None
                    or (
                        on_constraint_enforcing_endpoint
                        and OpenAIGPT5Config.is_model_gpt_5_6_plus_model(gpt_series_name)
                        and model_info.get("default_reasoning_effort") != "none"
                    )
                )
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


def chat_completion_ends_in_responses_api(
    model: str | None,
    custom_llm_provider: str | None,
    deployment_model_info: Mapping[str, object] | None = None,
    *,
    tools: Sequence[Mapping[str, object]] | None = None,
    reasoning_effort: str | Mapping[str, object] | None = None,
    reasoning_summary: object | None = None,
    api_base: str | None = None,
) -> bool:
    """Whether chat completion for this model is served by a Responses API.

    Encrypted reasoning can be replayed only there. A normal chat provider cannot
    verify the block, so the messages adapter strips it. ``chatgpt``, a ``responses/``
    model, a deployment pinned to ``mode: responses``, and a request whose tools or
    reasoning effort the bridge sends to Responses keep it.
    """
    if not model:
        return False
    try:
        resolved_model, provider = resolve_model_and_provider_for_metadata(model, custom_llm_provider)
    except Exception as error:  # noqa: BLE001  # unknown model names must not fail message translation
        verbose_logger.debug("responses replay check failed for %s: %s", model, error)
        return False
    model_info, _resolved = responses_api_bridge_check(
        model=resolved_model,
        custom_llm_provider=provider,
        tools=tools,
        reasoning_effort=reasoning_effort,
        reasoning_summary=reasoning_summary,
        api_base=api_base,
        deployment_model_info=deployment_model_info,
    )
    return model_info.get("mode") == "responses"


_USER_TURN_ROLES: Final = frozenset({"user", "tool", "function"})


def _with_deployment_base_model(litellm_params: Mapping[str, object]) -> Mapping[str, object]:
    deployment_model_info: Final = as_str_mapping(litellm_params.get("model_info"))
    if litellm_params.get("base_model") or deployment_model_info is None:
        return litellm_params
    return {**litellm_params, "base_model": deployment_model_info.get("base_model")}


def _uses_gemini_converter(model: str, provider: LlmProviders, litellm_params: Mapping[str, object]) -> bool:
    if provider == LlmProviders.VERTEX_AI:
        return (
            get_vertex_ai_model_route(model, _with_deployment_base_model(litellm_params)) == VertexAIModelRoute.GEMINI
        )
    return isinstance(
        ProviderConfigManager.get_provider_chat_config(model=model, provider=provider), VertexGeminiConfig
    )


def target_rejects_model_turn_end(
    model: str, custom_llm_provider: str | None, litellm_params: Mapping[str, object]
) -> bool:
    try:
        bare_model, provider, _, _ = get_llm_provider(model=model, custom_llm_provider=custom_llm_provider)
        return _uses_gemini_converter(bare_model, LlmProviders(provider), litellm_params)
    except (litellm.BadRequestError, ValueError):
        return False


def needs_empty_user_turn_back(
    caller_last_message: Mapping[str, object] | None,
    sent_messages: Sequence[object],
    model: str,
    custom_llm_provider: str | None,
    litellm_params: Mapping[str, object],
) -> bool:
    if not is_whitespace_user_message(caller_last_message):
        return False
    if sent_messages and message_field(sent_messages[-1], "role") in _USER_TURN_ROLES:
        return False
    return target_rejects_model_turn_end(model, custom_llm_provider, litellm_params)


def empty_user_turn_for_model_turn_end(
    caller_last_message: Mapping[str, object] | None,
    sent_messages: Sequence[object],
    model: str,
    custom_llm_provider: str | None,
    litellm_params: Mapping[str, object],
) -> tuple[ChatCompletionUserMessage, ...]:
    if not needs_empty_user_turn_back(caller_last_message, sent_messages, model, custom_llm_provider, litellm_params):
        return ()
    return (ChatCompletionUserMessage(role="user", content=""),)
