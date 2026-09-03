"""
This file contains common utils for anthropic calls.
"""

import copy
import re
from collections.abc import Mapping, MutableMapping, Sequence
from datetime import datetime, timezone
from types import MappingProxyType
from typing import Any, ClassVar, Final, Literal
from urllib.parse import quote

import httpx
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

import litellm
from litellm.constants import (
    DEFAULT_MODEL_CREATED_AT_TIME,
    DEFAULT_REASONING_EFFORT_HIGH_THINKING_BUDGET,
    DEFAULT_REASONING_EFFORT_MEDIUM_THINKING_BUDGET,
    DEFAULT_REASONING_EFFORT_XHIGH_THINKING_BUDGET,
)
from litellm.litellm_core_utils.prompt_templates.common_utils import (
    get_file_ids_from_messages,
)
from litellm.litellm_core_utils.prompt_templates.factory import (
    THOUGHT_SIGNATURE_SEPARATOR,
)
from litellm.llms.anthropic.wif import (
    aget_anthropic_wif_token,
    anthropic_base_without_chat_suffix,
    get_anthropic_wif_token,
)
from litellm.llms.base_llm.base_utils import BaseLLMModelInfo, BaseTokenCounter
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.proxy._types import SpecialHeaders
from litellm.types.llms.anthropic import (
    ANTHROPIC_HOSTED_TOOLS,
    ANTHROPIC_OAUTH_BETA_HEADER,
    ANTHROPIC_OAUTH_TOKEN_PREFIX,
    AllAnthropicToolsValues,
    AnthropicMcpServerTool,
    AnthropicMessagesToolChoice,
)
from litellm.types.llms.openai import AllMessageValues
from litellm.types.proxy.model_listing import ModelInfoResponse

DROP_FORCED_TOOL_CHOICE_WARNING: Final = (
    "Downgrading forced tool_choice to 'auto' for model=%s (drop_params=True): this model rejects tool_choice type "
    "'any'/'tool' with a 400 because thinking is always on and a forced call would skip it."
)
DROP_DISABLED_THINKING_WARNING: Final = (
    "Dropping `thinking={'type': 'disabled'}` for model=%s: thinking is always on for this model and cannot be "
    "disabled (the alternative is a provider 400). The model will still think adaptively, its response can contain "
    "thinking blocks, and those thinking tokens are billed as output tokens."
)

# Anthropic error `type` (both the JSON error body and SSE `event: error`
# payloads use this field) mapped to the HTTP status code it corresponds to.
ANTHROPIC_ERROR_STATUS_CODE_MAP: Final = MappingProxyType(
    {
        "invalid_request_error": 400,
        "authentication_error": 401,
        "permission_error": 403,
        "not_found_error": 404,
        "rate_limit_error": 429,
        "api_error": 500,
        "overloaded_error": 503,
        "timeout_error": 504,
    }
)

_BEDROCK_VERSION_SUFFIX_RE: Final = re.compile(r"-v\d+(?::\d+)?$")
_INFERENCE_PROFILE_MINOR_RE: Final = re.compile(r":\d+$")
_DATED_RELEASE_SUFFIX_RE: Final = re.compile(r"-\d{8}$")
_DOTTED_VERSION_RE: Final = re.compile(r"(\d)\.(\d)")


def _strip_bedrock_id_suffixes(model: str) -> str:
    """Reduce a full Bedrock model id to its base cost-map key by rewriting a
    dotted family version then peeling a trailing ``-vN:rev`` and ``-YYYYMMDD``
    in that order, so the real ``-<date>-v1:0`` shape (e.g.
    ``us.anthropic.claude-sonnet-4-6-20251101-v1:0``) resolves rather than only
    the date or version in isolation."""
    return _DATED_RELEASE_SUFFIX_RE.sub(
        "",
        _BEDROCK_VERSION_SUFFIX_RE.sub("", _DOTTED_VERSION_RE.sub(r"\1-\2", model)),
    )


_SERVER_OWNED_AUTH_HEADERS: Final = SpecialHeaders.litellm_credential_header_names()
_WIF_ELIGIBILITY_ATTR: Final = "_workload_identity_eligible"


def without_caller_credential_headers(headers: Mapping[str, str]) -> Mapping[str, str]:
    """``headers`` minus every header that authenticates the caller to litellm.

    The deployment's own credential is applied on top of the result, so a caller-supplied
    credential must not survive into the upstream request: without this a minted federation
    Bearer travels beside the caller's own ``x-api-key``, and Anthropic sees two credentials.
    """
    return MappingProxyType(
        {name: value for name, value in headers.items() if name.lower() not in _SERVER_OWNED_AUTH_HEADERS}
    )


def config_allows_workload_identity(config: object) -> bool:
    """A federation token is an Anthropic-org credential and its exchange POSTs the workload's OIDC
    assertion to the deployment's own host, so eligibility is declared per class and read from that
    class's own ``__dict__``: a subclass written for another provider inherits nothing."""
    return type(config).__dict__.get(_WIF_ELIGIBILITY_ATTR, False) is True


def is_anthropic_oauth_key(value: str | None) -> bool:
    """Check if a value contains an Anthropic OAuth token (sk-ant-oat*)."""
    if value is None:
        return False
    # Handle both raw token and "Bearer <token>" format
    value = value.removeprefix("Bearer ")
    return value.startswith(ANTHROPIC_OAUTH_TOKEN_PREFIX)


def merge_anthropic_beta_headers(existing: str | Sequence[str] | None, new_beta: str | Sequence[str] | None) -> str:
    """Merge anthropic-beta header values, deduplicated and sorted.

    Either side may arrive as a list rather than a comma-separated string: the Skills surface
    accepted a list-valued header before it shared this helper, and callers still send one.
    """
    values: Final = (
        entry
        for side in (existing, new_beta)
        if side
        for entry in ((side,) if isinstance(side, str) else side)
        if isinstance(entry, str)
    )
    betas: Final = frozenset(b.strip() for value in values for b in value.split(",") if b.strip())
    return ",".join(sorted(betas))


def optionally_handle_anthropic_oauth(headers: dict, api_key: str | None) -> tuple[dict, str | None]:
    """
    Handle Anthropic OAuth token detection and header setup.

    If an OAuth token is detected in the Authorization header (any casing),
    extracts it and sets the required OAuth headers.

    Args:
        headers: Request headers dict
        api_key: Current API key (may be None)

    Returns:
        Tuple of (updated headers, api_key)
    """
    # Check Authorization header (passthrough / forwarded requests)
    auth_header: Final = next((value for name, value in headers.items() if name.lower() == "authorization"), "")
    if auth_header.startswith(f"Bearer {ANTHROPIC_OAUTH_TOKEN_PREFIX}"):
        api_key = auth_header.removeprefix("Bearer ")
        for name in tuple(
            header_name for header_name in headers if header_name.lower() in ("x-api-key", "authorization")
        ):
            headers.pop(name)
        headers["authorization"] = auth_header
        headers["anthropic-beta"] = merge_anthropic_beta_headers(
            headers.get("anthropic-beta"), ANTHROPIC_OAUTH_BETA_HEADER
        )
        headers["anthropic-dangerous-direct-browser-access"] = "true"
        return headers, api_key
    # Check api_key directly (standard chat/completion flow)
    if api_key and api_key.startswith(ANTHROPIC_OAUTH_TOKEN_PREFIX):
        for name in tuple(header_name for header_name in headers if header_name.lower() == "x-api-key"):
            headers.pop(name)
        headers["authorization"] = f"Bearer {api_key}"
        headers["anthropic-beta"] = merge_anthropic_beta_headers(
            headers.get("anthropic-beta"), ANTHROPIC_OAUTH_BETA_HEADER
        )
        headers["anthropic-dangerous-direct-browser-access"] = "true"
    return headers, api_key


class AnthropicError(BaseLLMException):
    def __init__(
        self,
        status_code: int,
        message,
        headers: httpx.Headers | None = None,
    ):
        super().__init__(status_code=status_code, message=message, headers=headers)


_MODEL_LIST_PAGE_CAP: Final = 20


def _litellm_params_str(litellm_params: Mapping[str, object] | None, key: str) -> str | None:
    value: Final = litellm_params.get(key) if litellm_params is not None else None
    return value if isinstance(value, str) else None


class _AnthropicModelListEntry(BaseModel):
    id: str


class _AnthropicModelsPage(BaseModel):
    data: Sequence[_AnthropicModelListEntry] = Field(default_factory=tuple)
    has_more: bool = False
    last_id: str | None = None


def _sanitized_anthropic_error(response: httpx.Response, detail: str | None = None) -> str:
    """A provider error detail built only from structured fields, never ``response.text``
    verbatim: the raw body is untrusted content the caller of ``/v1/models`` did not ask for
    and should not have echoed back to it wholesale."""
    if detail is not None:
        return f"HTTP {response.status_code}: {detail}"
    try:
        body: Final = response.json()
    except ValueError:
        return f"HTTP {response.status_code}"
    error: Final = body.get("error") if isinstance(body, dict) else None
    message: Final = error.get("message") if isinstance(error, dict) else None
    return f"HTTP {response.status_code}: {message}" if isinstance(message, str) else f"HTTP {response.status_code}"


def _fetch_anthropic_models_page(
    api_base: str, headers: Mapping[str, str], after_id: str | None
) -> _AnthropicModelsPage:
    # after_id rides the URL because the client mutates the params mapping it is handed,
    # which a read-only one cannot support
    query: Final = f"?after_id={quote(after_id)}" if after_id else ""
    response: Final = litellm.module_level_client.get(
        url=f"{api_base}/v1/models{query}",
        headers=headers,
        follow_redirects=False,
    )
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError:
        raise Exception(f"Failed to fetch models from Anthropic. {_sanitized_anthropic_error(response)}") from None
    try:
        return _AnthropicModelsPage.model_validate(response.json())
    except ValueError as e:
        raise Exception(
            f"Failed to fetch models from Anthropic. {_sanitized_anthropic_error(response, detail=str(e))}"
        ) from None


def _fetch_anthropic_model_ids(
    api_base: str, headers: Mapping[str, str], after_id: str | None, pages_left: int
) -> tuple[str, ...]:
    collected: tuple[str, ...] = ()  # rebind-ok: accumulates one page of ids per iteration
    cursor: str | None = after_id  # rebind-ok: advances to each page's last_id
    for _ in range(max(pages_left, 0)):
        page = _fetch_anthropic_models_page(api_base, headers, cursor)  # rebind-ok: one page per iteration
        collected += tuple(entry.id for entry in page.data)
        if not page.has_more or page.last_id is None:
            return collected
        cursor = page.last_id
    raise Exception(f"Anthropic /v1/models did not terminate within {_MODEL_LIST_PAGE_CAP} pages.")


class AnthropicModelInfo(BaseLLMModelInfo):
    _workload_identity_eligible: ClassVar[bool] = True

    def is_cache_control_set(self, messages: list[AllMessageValues]) -> bool:
        """
        Return if {"cache_control": ..} in message content block

        Used to check if anthropic prompt caching headers need to be set.
        """
        for message in messages:
            if message.get("cache_control", None) is not None:
                return True
            _message_content = message.get("content")
            if _message_content is not None and isinstance(_message_content, list):
                for content in _message_content:
                    if "cache_control" in content:
                        return True

        return False

    def is_file_id_used(self, messages: list[AllMessageValues]) -> bool:
        """
        Return if {"source": {"type": "file", "file_id": ..}} in message content block
        """
        file_ids: Final = get_file_ids_from_messages(messages)
        return len(file_ids) > 0

    def is_mcp_server_used(self, mcp_servers: list[AnthropicMcpServerTool] | None) -> bool:
        if mcp_servers is None:
            return False
        if mcp_servers:
            return True
        return False

    def is_computer_tool_used(self, tools: list[AllAnthropicToolsValues] | None) -> str | None:
        """Returns the computer tool version if used, e.g. 'computer_20250124' or None"""
        if tools is None:
            return None
        for tool in tools:
            if "type" in tool and tool["type"].startswith("computer_"):
                return tool["type"]
        return None

    def is_web_search_tool_used(self, tools: list[AllAnthropicToolsValues] | None) -> bool:
        """Returns True if web_search tool is used"""
        if tools is None:
            return False
        for tool in tools:
            if "type" in tool and tool["type"].startswith(ANTHROPIC_HOSTED_TOOLS.WEB_SEARCH.value):
                return True
        return False

    def is_pdf_used(self, messages: list[AllMessageValues]) -> bool:
        """
        Set to true if media passed into messages.

        """
        for message in messages:
            if "content" in message and message["content"] is not None and isinstance(message["content"], list):
                for content in message["content"]:
                    if "type" in content and content["type"] != "text":
                        return True
        return False

    def is_tool_search_used(self, tools: list | None) -> bool:
        """
        Check if tool search tools are present in the tools list.
        """
        if not tools:
            return False

        for tool in tools:
            tool_type = tool.get("type", "")
            if tool_type in [
                "tool_search_tool_regex_20251119",
                "tool_search_tool_bm25_20251119",
            ]:
                return True
        return False

    def is_programmatic_tool_calling_used(self, tools: list | None) -> bool:
        """
        Check if programmatic tool calling is being used (tools with allowed_callers field).

        Returns True if any tool has allowed_callers containing 'code_execution_20250825'.
        """
        if not tools:
            return False

        for tool in tools:
            # Check top-level allowed_callers
            allowed_callers = tool.get("allowed_callers", None)
            if allowed_callers and isinstance(allowed_callers, list):
                if "code_execution_20250825" in allowed_callers:
                    return True

            # Check function.allowed_callers for OpenAI format tools
            function = tool.get("function", {})
            if isinstance(function, dict):
                function_allowed_callers = function.get("allowed_callers", None)
                if function_allowed_callers and isinstance(function_allowed_callers, list):
                    if "code_execution_20250825" in function_allowed_callers:
                        return True

        return False

    def is_input_examples_used(self, tools: list | None) -> bool:
        """
        Check if input_examples is being used in any tools.

        Returns True if any tool has input_examples field.
        """
        if not tools:
            return False

        for tool in tools:
            # Check top-level input_examples
            input_examples = tool.get("input_examples", None)
            if input_examples and isinstance(input_examples, list) and len(input_examples) > 0:
                return True

            # Check function.input_examples for OpenAI format tools
            function = tool.get("function", {})
            if isinstance(function, dict):
                function_input_examples = function.get("input_examples", None)
                if (
                    function_input_examples
                    and isinstance(function_input_examples, list)
                    and len(function_input_examples) > 0
                ):
                    return True

        return False

    @staticmethod
    def _supports_sampling_params(model: str) -> bool:
        """Claude 4.7+ (Opus 4.7/4.8, Fable 5) removed sampling params: the API
        rejects ``top_p``, ``top_k``, and any ``temperature`` other than 1 with
        a 400 ("`temperature` is deprecated for this model").

        Driven by the ``supports_sampling_params`` flag in the model map; the
        name check remains only as a fallback for provider-routed ids whose
        map entries predate the flag."""
        flag: Final = AnthropicModelInfo._get_model_capability(model, "supports_sampling_params")
        if flag is not None:
            return flag
        model_lower: Final = model.lower()
        return not any(
            v in model_lower
            for v in (
                "fable",
                "opus-4-7",
                "opus_4_7",
                "opus-4.7",
                "opus_4.7",
                "opus-4-8",
                "opus_4_8",
                "opus-4.8",
                "opus_4.8",
            )
        )

    @staticmethod
    def _apply_sampling_param(
        optional_params: dict,
        model: str,
        param: str,
        value: Any,
        drop_params: bool,
        output_key: str,
    ) -> None:
        """Forward ``temperature``/``top_p``/``top_k`` to
        ``optional_params[output_key]`` unless the model removed sampling
        params, in which case drop the param (with drop_params) or raise a
        clean client-side 400."""
        if AnthropicModelInfo._supports_sampling_params(model) or (param == "temperature" and value == 1):
            optional_params[output_key] = value
        elif not (litellm.drop_params or drop_params):
            supported_hint: Final = "Only temperature=1 is supported. " if param == "temperature" else ""
            raise litellm.utils.UnsupportedParamsError(
                message=(
                    f"{model} does not support {param}={value}. {supported_hint}"
                    "To drop unsupported params, set `litellm.drop_params = True`."
                ),
                status_code=400,
            )

    @staticmethod
    def forced_tool_use_unsupported(model: str) -> bool:
        return AnthropicModelInfo._get_model_capability(model, "supports_forced_tool_use") is False

    @staticmethod
    def forced_tool_use_downgraded(model: str, drop_params: bool) -> bool:
        """True when the model map flags the model with
        ``supports_forced_tool_use: false`` (Fable 5.1 / Mythos 5.1 400 on
        ``any``/``tool``) and ``drop_params`` asks for the ``auto`` downgrade;
        raises a clean client-side 400 for such models without ``drop_params``."""
        if not AnthropicModelInfo.forced_tool_use_unsupported(model):
            return False
        if not (litellm.drop_params or drop_params):
            raise litellm.utils.UnsupportedParamsError(
                message=(
                    f"{model} does not support forced tool use (tool_choice='required' or a named tool). "
                    "Use tool_choice='auto' and tell the model in the prompt when to call the tool, or set "
                    "`litellm.drop_params = True` to downgrade to 'auto' automatically."
                ),
                status_code=400,
            )
        litellm.verbose_logger.warning(DROP_FORCED_TOOL_CHOICE_WARNING, model)
        return True

    @staticmethod
    def _apply_forced_tool_choice(
        model: str,
        tool_choice: AnthropicMessagesToolChoice,
        drop_params: bool,
    ) -> AnthropicMessagesToolChoice:
        if tool_choice["type"] not in ("any", "tool"):
            return tool_choice
        if not AnthropicModelInfo.forced_tool_use_downgraded(model, drop_params):
            return tool_choice
        disable_parallel: Final = tool_choice.get("disable_parallel_tool_use")
        if disable_parallel is None:
            return AnthropicMessagesToolChoice(type="auto")
        return AnthropicMessagesToolChoice(type="auto", disable_parallel_tool_use=disable_parallel)

    @staticmethod
    def _strip_version_suffix(model: str) -> str:
        at: Final = model.rfind("@")
        if at > 0:
            return model[:at]
        return model

    @staticmethod
    def _model_map_lookup_candidates(model: str) -> list[str]:
        """Model-map keys to try for ``model``: the id itself, the same id with a
        bedrock/vertex routing prefix removed, the Bedrock base model, and each of
        those normalized by stripping a Bedrock version suffix (``-v1:0`` fully or
        just the ``:0`` inference-profile minor), stripping a dated-release suffix
        (``-20260205``), or rewriting a dotted family version to hyphens
        (``4.6`` -> ``4-6``). Lets any reasonable alias (e.g.
        ``bedrock/invoke/global.anthropic.claude-opus-4-7-v1:0``,
        ``claude-sonnet-4-6-20260219`` or ``claude-sonnet-4.6``) resolve to its base
        cost-map entry so the capability flag on that entry stays authoritative."""
        prefixes: Final = (
            "bedrock/converse/",
            "bedrock/invoke/",
            "bedrock/",
            "vertex_ai/",
        )
        deprefixed: Final = tuple(model[len(p) :] for p in prefixes if model.startswith(p))
        try:
            from litellm.llms.bedrock.common_utils import BedrockModelInfo

            base = BedrockModelInfo.get_base_model(model)
        except Exception:
            base = None
        bedrock_base: Final = (base, f"bedrock/{base}") if base else ()
        primary: Final = (model, *deprefixed, *bedrock_base)
        normalized: Final = tuple(
            stripped
            for cand in primary
            for stripped in (
                _BEDROCK_VERSION_SUFFIX_RE.sub("", cand),
                _INFERENCE_PROFILE_MINOR_RE.sub("", cand),
                _DATED_RELEASE_SUFFIX_RE.sub("", cand),
                _DOTTED_VERSION_RE.sub(r"\1-\2", cand),
                _strip_bedrock_id_suffixes(cand),
                AnthropicModelInfo._strip_version_suffix(cand),
            )
        )
        return list(dict.fromkeys((*primary, *normalized)))

    @staticmethod
    def _get_model_capability(model: str, key: str) -> bool | None:
        """Read boolean capability ``key`` from the model map, or None when
        no entry declares it."""
        from litellm.utils import _get_bundled_model_cost_map

        try:
            candidates: Final = AnthropicModelInfo._model_map_lookup_candidates(model)
            for model_cost in (litellm.model_cost, _get_bundled_model_cost_map()):
                for cand in candidates:
                    value = model_cost.get(cand, {}).get(key)
                    if isinstance(value, bool):
                        return value
        except Exception:
            pass
        return None

    @staticmethod
    def _get_exact_model_capability(model: str, key: str) -> bool | None:
        """Read boolean capability ``key`` from the exact model-map entry only.

        Unlike ``_get_model_capability``, does not walk stripped provider aliases.
        Use when a feature is tied to a specific host (e.g. Anthropic API fast mode).
        """
        value: Final = litellm.model_cost.get(model, {}).get(key)
        return value if isinstance(value, bool) else None

    @staticmethod
    def _get_provider_resolved_capability(model: str, key: str, custom_llm_provider: str) -> bool | None:
        """Resolve boolean capability ``key`` for ``model`` under the caller's provider.

        Returns the flag when the provider-aware lookup resolves ``model`` to an
        entry (or fallback rule) that sets it explicitly, and ``None`` when the
        model does not resolve under that provider or the resolved entry has no
        opinion on ``key``.
        """
        from litellm.utils import _get_model_info_helper

        try:
            resolved_model, resolved_provider, _, _ = litellm.get_llm_provider(
                model=model, custom_llm_provider=custom_llm_provider
            )
            value: Final = _get_model_info_helper(model=resolved_model, custom_llm_provider=resolved_provider).get(key)
        except Exception:  # noqa: BLE001  # _get_model_info_helper raises bare Exception for unmapped models
            return None
        return value if isinstance(value, bool) else None

    @staticmethod
    def _supports_model_capability(model: str, key: str, custom_llm_provider: str) -> bool:
        """Check a boolean capability ``key`` in the model map under the caller's provider.

        The provider-aware lookup is authoritative when it resolves an explicit flag,
        so ``key: false`` on the provider-namespaced entry wins over every fallback.
        Otherwise ``_supports_factory``'s provider-level fallbacks and the raw
        model-map walk remain as backstops for alias forms the lookup misses.
        """
        from litellm.utils import _supports_factory

        resolved: Final = AnthropicModelInfo._get_provider_resolved_capability(model, key, custom_llm_provider)
        if resolved is not None:
            return resolved
        try:
            if _supports_factory(
                model=model,
                custom_llm_provider=custom_llm_provider,
                key=key,
            ):
                return True
        except Exception:
            pass
        return AnthropicModelInfo._get_model_capability(model, key) is True

    @staticmethod
    def _is_adaptive_thinking_model(model: str, custom_llm_provider: str) -> bool:
        """Whether ``model`` uses adaptive thinking (``output_config.effort``).

        The model cost map is authoritative: an explicit ``supports_adaptive_thinking``
        entry resolved under ``custom_llm_provider``, or a ``fallback_generalizations``
        rule for unknown Claude models. The version gate (>= 4.6, including
        provider-prefixed Bedrock/Vertex ids that map to no exact entry) lives entirely
        in that declarative rule, not here.
        """
        return AnthropicModelInfo._supports_model_capability(model, "supports_adaptive_thinking", custom_llm_provider)

    @staticmethod
    def _is_always_on_thinking_model(model: str, custom_llm_provider: str) -> bool:
        """Whether ``model`` always thinks and rejects ``thinking.type=disabled``
        (Fable 5 / Mythos 5 generation). The model cost map is authoritative: an
        explicit ``thinking_always_on`` entry resolved under ``custom_llm_provider``,
        or a ``fallback_generalizations`` rule for unmapped ids of those families.
        """
        return AnthropicModelInfo._supports_model_capability(model, "thinking_always_on", custom_llm_provider)

    @staticmethod
    def _supports_legacy_thinking(model: str, custom_llm_provider: str) -> bool:
        """Whether ``model`` is an adaptive-thinking model that still accepts legacy
        ``thinking.type=enabled`` with ``budget_tokens`` (the Claude 4.6 family).
        The model cost map is authoritative: an explicit ``supports_legacy_thinking``
        entry resolved under ``custom_llm_provider``, or a ``fallback_generalizations``
        rule for unmapped 4.6 ids. Absent flag means the model rejects the legacy shape.
        """
        return AnthropicModelInfo._supports_model_capability(model, "supports_legacy_thinking", custom_llm_provider)

    @staticmethod
    def maybe_drop_disabled_thinking(
        model: str,
        optional_params: MutableMapping[str, object],  # mutable-ok: in-place out-param, as in _maybe_drop_speed_param
        custom_llm_provider: str,
    ) -> None:
        """Omit ``thinking={'type': 'disabled'}`` for always-on-thinking models
        (Fable 5 / Mythos 5), which 400 on it; omission is the API-documented
        remedy and yields the model's default adaptive thinking."""
        thinking: Final = optional_params.get("thinking")
        if not isinstance(thinking, dict) or thinking.get("type") != "disabled":
            return
        if not AnthropicModelInfo._is_always_on_thinking_model(model, custom_llm_provider):
            return
        litellm.verbose_logger.warning(
            DROP_DISABLED_THINKING_WARNING,
            model,
        )
        optional_params.pop("thinking", None)

    @staticmethod
    def translate_legacy_thinking_for_adaptive_model(
        model: str,
        optional_params: MutableMapping[str, object],  # mutable-ok: in-place out-param like the sibling helpers
        custom_llm_provider: str,
    ) -> None:
        """Translate legacy ``thinking.type=enabled`` to adaptive for the
        adaptive-thinking models that reject it (4.7+ and the 5 families).
        Models flagged ``supports_legacy_thinking`` (the 4.6 family) accept the
        legacy shape natively, so it is forwarded verbatim and the caller's
        ``budget_tokens`` cap keeps applying. Caller-provided
        ``output_config.effort`` is never overridden.
        """
        if not AnthropicModelInfo._is_adaptive_thinking_model(model, custom_llm_provider):
            return
        if AnthropicModelInfo._supports_legacy_thinking(model, custom_llm_provider):
            return
        thinking: Final = optional_params.get("thinking")
        if not isinstance(thinking, dict) or thinking.get("type") != "enabled":
            return

        effort: Final = AnthropicModelInfo._legacy_budget_to_effort(
            model=model,
            budget_tokens=int(thinking.get("budget_tokens") or 0),
            custom_llm_provider=custom_llm_provider,
        )
        existing_output_config: Final = optional_params.get("output_config")
        optional_params["thinking"] = {"type": "adaptive"}
        optional_params["output_config"] = {
            "effort": effort,
            **(existing_output_config if isinstance(existing_output_config, dict) else MappingProxyType({})),
        }

    @staticmethod
    def _legacy_budget_to_effort(model: str, budget_tokens: int, custom_llm_provider: str) -> str:
        if budget_tokens >= DEFAULT_REASONING_EFFORT_XHIGH_THINKING_BUDGET and (
            AnthropicModelInfo._supports_model_capability(model, "supports_xhigh_reasoning_effort", custom_llm_provider)
        ):
            return "xhigh"
        if budget_tokens >= DEFAULT_REASONING_EFFORT_HIGH_THINKING_BUDGET:
            return "high"
        if budget_tokens >= DEFAULT_REASONING_EFFORT_MEDIUM_THINKING_BUDGET:
            return "medium"
        return "low"

    def is_effort_used(
        self,
        optional_params: dict | None,
        model: str | None = None,
        *,
        custom_llm_provider: str,
    ) -> bool:
        """
        Check if effort parameter is being used and requires a beta header.

        Returns True if effort-related parameters are present and
        the model requires the effort beta header. Claude 4.6+ models
        use output_config as a stable API feature — no beta header needed.
        """
        if not optional_params:
            return False

        # Claude 4.6+ models use output_config as a stable API feature — no beta header needed
        if model and self._is_adaptive_thinking_model(model, custom_llm_provider):
            return False

        # Check if reasoning_effort is provided for Claude Opus 4.5
        if model and ("opus-4-5" in model.lower() or "opus_4_5" in model.lower()):
            reasoning_effort: Final = optional_params.get("reasoning_effort")
            if reasoning_effort and isinstance(reasoning_effort, str):
                return True

        # Check if output_config is directly provided (for non-4.6 models)
        output_config: Final = optional_params.get("output_config")
        if output_config and isinstance(output_config, dict):
            effort: Final = output_config.get("effort")
            if effort and isinstance(effort, str):
                return True

        return False

    def is_code_execution_tool_used(self, tools: list | None) -> bool:
        """
        Check if code execution tool is being used.

        Returns True if any tool has type "code_execution_20250825".
        """
        if not tools:
            return False

        for tool in tools:
            tool_type = tool.get("type", "")
            if tool_type == "code_execution_20250825":
                return True
        return False

    def is_container_with_skills_used(self, optional_params: dict | None) -> bool:
        """
        Check if container with skills is being used.

        Returns True if optional_params contains container with skills.
        """
        if not optional_params:
            return False

        container: Final = optional_params.get("container")
        if container and isinstance(container, dict):
            skills: Final = container.get("skills")
            if skills and isinstance(skills, list) and len(skills) > 0:
                return True
        return False

    def _get_user_anthropic_beta_headers(self, anthropic_beta_header: str | None) -> list[str] | None:
        if anthropic_beta_header is None:
            return None
        return anthropic_beta_header.split(",")

    def get_computer_tool_beta_header(self, computer_tool_version: str) -> str:
        """
        Get the appropriate beta header for a given computer tool version.

        Args:
            computer_tool_version: The computer tool version (e.g., 'computer_20250124', 'computer_20241022')

        Returns:
            The corresponding beta header string
        """
        computer_tool_beta_mapping: Final = {
            "computer_20250124": "computer-use-2025-01-24",
            "computer_20241022": "computer-use-2024-10-22",
        }
        return computer_tool_beta_mapping.get(
            computer_tool_version,
            "computer-use-2024-10-22",  # Default fallback
        )

    def get_anthropic_beta_list(
        self,
        model: str,
        optional_params: dict | None = None,
        computer_tool_used: str | None = None,
        prompt_caching_set: bool = False,
        file_id_used: bool = False,
        mcp_server_used: bool = False,
        *,
        custom_llm_provider: str,
    ) -> list[str]:
        """
        Get list of common beta headers based on the features that are active.

        Returns:
            List of beta header strings
        """
        from litellm.types.llms.anthropic import ANTHROPIC_EFFORT_BETA_HEADER

        betas: Final = []

        # Detect features
        effort_used: Final = self.is_effort_used(optional_params, model, custom_llm_provider=custom_llm_provider)

        if effort_used:
            betas.append(ANTHROPIC_EFFORT_BETA_HEADER)  # effort-2025-11-24

        if computer_tool_used:
            beta_header: Final = self.get_computer_tool_beta_header(computer_tool_used)
            betas.append(beta_header)

        # Anthropic no longer requires the prompt-caching beta header
        # Prompt caching now works automatically when cache_control is used in messages
        # Reference: https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching

        if file_id_used:
            betas.append("files-api-2025-04-14")
            betas.append("code-execution-2025-05-22")

        if mcp_server_used:
            betas.append("mcp-client-2025-04-04")

        return list(set(betas))

    @staticmethod
    def _make_api_key_auth_header(
        api_key: str, api_base: str | None, use_bearer_for_custom_base: bool = False
    ) -> Mapping[str, str]:
        if use_bearer_for_custom_base and (
            api_base and "api.anthropic.com" not in api_base and not api_key.startswith("sk-ant-")
        ):
            value: Final = api_key if api_key.startswith("Bearer ") else f"Bearer {api_key}"
            return {"authorization": value}
        return {"x-api-key": api_key}

    def _credential_headers(
        self,
        *,
        api_key: str | None,
        auth_token: str | None,
        api_base: str | None,
        use_bearer_for_custom_base: bool,
        wif_minted: bool,
        betas: set[str],  # mutable-ok: the caller's beta accumulator, appended to by the oauth tier
    ) -> Mapping[str, str]:
        """The credential tier walk: a consumer OAuth token, then ANTHROPIC_AUTH_TOKEN, then an api key.

        A server-minted federation token takes the same Bearer shape as a consumer OAuth token but is
        not browser-forwarded, so it does not get the direct-browser-access header.
        """
        if api_key and api_key.startswith(ANTHROPIC_OAUTH_TOKEN_PREFIX):
            betas.add(ANTHROPIC_OAUTH_BETA_HEADER)
            oauth_headers: Final = {"authorization": f"Bearer {api_key}"}
            if wif_minted:
                return oauth_headers
            return {**oauth_headers, "anthropic-dangerous-direct-browser-access": "true"}
        if auth_token and not api_key:
            return {"authorization": f"Bearer {auth_token}"}
        if api_key:
            return self._make_api_key_auth_header(api_key, api_base, use_bearer_for_custom_base)
        return {}

    def get_anthropic_headers(
        self,
        api_key: str | None = None,
        auth_token: str | None = None,
        anthropic_version: str | None = None,
        computer_tool_used: str | None = None,
        prompt_caching_set: bool = False,
        pdf_used: bool = False,
        file_id_used: bool = False,
        mcp_server_used: bool = False,
        web_search_tool_used: bool = False,
        tool_search_used: bool = False,
        programmatic_tool_calling_used: bool = False,
        input_examples_used: bool = False,
        effort_used: bool = False,
        is_vertex_request: bool = False,
        user_anthropic_beta_headers: list[str] | None = None,
        code_execution_tool_used: bool = False,
        container_with_skills_used: bool = False,
        api_base: str | None = None,
        use_bearer_for_custom_base: bool = False,
        wif_minted: bool = False,
    ) -> dict:
        betas: Final = set()
        # Anthropic no longer requires the prompt-caching beta header
        # Prompt caching now works automatically when cache_control is used in messages
        # Reference: https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching
        if computer_tool_used:
            beta_header: Final = self.get_computer_tool_beta_header(computer_tool_used)
            betas.add(beta_header)
        # if pdf_used:
        #     betas.add("pdfs-2024-09-25")
        if file_id_used:
            betas.add("files-api-2025-04-14")
            betas.add("code-execution-2025-05-22")
        if mcp_server_used:
            betas.add("mcp-client-2025-04-04")
        # Tool search, programmatic tool calling, and input_examples all use the same beta header
        if tool_search_used or programmatic_tool_calling_used or input_examples_used:
            from litellm.types.llms.anthropic import ANTHROPIC_TOOL_SEARCH_BETA_HEADER

            betas.add(ANTHROPIC_TOOL_SEARCH_BETA_HEADER)

        # Effort parameter uses a separate beta header
        if effort_used:
            from litellm.types.llms.anthropic import ANTHROPIC_EFFORT_BETA_HEADER

            betas.add(ANTHROPIC_EFFORT_BETA_HEADER)

        # Code execution tool uses a separate beta header
        if code_execution_tool_used:
            betas.add("code-execution-2025-08-25")

        # Container with skills uses a separate beta header
        if container_with_skills_used:
            betas.add("skills-2025-10-02")

        headers: Final = {
            "anthropic-version": anthropic_version or "2023-06-01",
            "accept": "application/json",
            "content-type": "application/json",
        }
        headers.update(
            self._credential_headers(
                api_key=api_key,
                auth_token=auth_token,
                api_base=api_base,
                use_bearer_for_custom_base=use_bearer_for_custom_base,
                wif_minted=wif_minted,
                betas=betas,
            )
        )

        if user_anthropic_beta_headers is not None:
            betas.update(user_anthropic_beta_headers)

        # Don't send any beta headers to Vertex, except web search which is required
        if is_vertex_request is True:
            # Vertex AI requires web search beta header for web search to work
            if web_search_tool_used:
                from litellm.types.llms.anthropic import ANTHROPIC_BETA_HEADER_VALUES

                headers["anthropic-beta"] = ANTHROPIC_BETA_HEADER_VALUES.WEB_SEARCH_2025_03_05.value
        elif len(betas) > 0:
            headers["anthropic-beta"] = ",".join(betas)

        return headers

    def validate_environment(
        self,
        headers: dict,
        model: str,
        messages: list[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> dict:
        params_mapping: Final = litellm_params if isinstance(litellm_params, dict) else None
        if api_base is None and params_mapping is not None:
            api_base = params_mapping.get("api_base")
        use_bearer_for_custom_base: Final[bool] = bool(
            params_mapping is not None and params_mapping.get("use_bearer_for_custom_base", False)
        )
        # Check for Anthropic OAuth token in headers
        headers, api_key = optionally_handle_anthropic_oauth(headers=headers, api_key=api_key)
        api_key = AnthropicModelInfo.get_api_key(api_key)
        # Resolve auth_token from ANTHROPIC_AUTH_TOKEN if api_key is not set
        auth_token: str | None = None
        if api_key is None:
            auth_token = AnthropicModelInfo.get_auth_token()
        wif_token: Final = (
            get_anthropic_wif_token(params_mapping, api_base, model)
            if api_key is None and auth_token is None and config_allows_workload_identity(self)
            else None
        )
        wif_minted: Final = wif_token is not None
        resolved_api_key: Final = wif_token if wif_token is not None else api_key
        if resolved_api_key is None and auth_token is None:
            raise litellm.AuthenticationError(
                message=(
                    "Missing Anthropic API Key - A call is being made to anthropic but no key is set either in the "
                    "environment variables or via params. Please set `ANTHROPIC_API_KEY` or `ANTHROPIC_AUTH_TOKEN` "
                    "in your environment vars, or configure workload identity federation via "
                    "`ANTHROPIC_FEDERATION_RULE_ID`, `ANTHROPIC_ORGANIZATION_ID`, "
                    "`ANTHROPIC_SERVICE_ACCOUNT_ID` and "
                    "`ANTHROPIC_IDENTITY_TOKEN_FILE` (or `ANTHROPIC_IDENTITY_TOKEN`)"
                ),
                llm_provider="anthropic",
                model=model,
            )

        tools: Final = optional_params.get("tools")
        prompt_caching_set: Final = self.is_cache_control_set(messages=messages)
        computer_tool_used: Final = self.is_computer_tool_used(tools=tools)
        mcp_server_used: Final = self.is_mcp_server_used(mcp_servers=optional_params.get("mcp_servers"))
        pdf_used: Final = self.is_pdf_used(messages=messages)
        file_id_used: Final = self.is_file_id_used(messages=messages)
        web_search_tool_used: Final = self.is_web_search_tool_used(tools=tools)
        tool_search_used: Final = self.is_tool_search_used(tools=tools)
        programmatic_tool_calling_used: Final = self.is_programmatic_tool_calling_used(tools=tools)
        input_examples_used: Final = self.is_input_examples_used(tools=tools)
        effort_used = self.is_effort_used(optional_params=optional_params, model=model, custom_llm_provider="anthropic")
        code_execution_tool_used: Final = self.is_code_execution_tool_used(tools=tools)
        container_with_skills_used: Final = self.is_container_with_skills_used(optional_params=optional_params)
        user_anthropic_beta_headers: Final = self._get_user_anthropic_beta_headers(
            anthropic_beta_header=headers.get("anthropic-beta")
        )
        anthropic_headers: Final = self.get_anthropic_headers(
            computer_tool_used=computer_tool_used,
            prompt_caching_set=prompt_caching_set,
            pdf_used=pdf_used,
            api_key=resolved_api_key,
            auth_token=auth_token,
            file_id_used=file_id_used,
            web_search_tool_used=web_search_tool_used,
            is_vertex_request=optional_params.get("is_vertex_request", False),
            user_anthropic_beta_headers=user_anthropic_beta_headers,
            mcp_server_used=mcp_server_used,
            tool_search_used=tool_search_used,
            programmatic_tool_calling_used=programmatic_tool_calling_used,
            input_examples_used=input_examples_used,
            effort_used=effort_used,
            code_execution_tool_used=code_execution_tool_used,
            container_with_skills_used=container_with_skills_used,
            api_base=api_base,
            use_bearer_for_custom_base=use_bearer_for_custom_base,
            wif_minted=wif_minted,
        )

        caller_headers: Final = without_caller_credential_headers(headers) if wif_minted else headers

        return {**caller_headers, **anthropic_headers}

    @staticmethod
    def get_api_base(api_base: str | None = None) -> str | None:
        from litellm.secret_managers.main import get_secret_str

        return (
            api_base
            or get_secret_str("ANTHROPIC_API_BASE")
            or get_secret_str("ANTHROPIC_BASE_URL")
            or "https://api.anthropic.com"
        )

    @staticmethod
    def get_api_key(api_key: str | None = None) -> str | None:
        from litellm.secret_managers.main import get_secret_str

        return api_key or get_secret_str("ANTHROPIC_API_KEY")

    @staticmethod
    def get_auth_token(auth_token: str | None = None) -> str | None:
        """Get auth token from ANTHROPIC_AUTH_TOKEN env var.

        Unlike api_key (which uses X-Api-Key header), auth_token uses
        Authorization: Bearer header, matching the official Anthropic SDK behavior.
        """
        from litellm.secret_managers.main import get_secret_str

        return auth_token or get_secret_str("ANTHROPIC_AUTH_TOKEN")

    @staticmethod
    def get_auth_header(
        api_key: str | None = None,
        api_base: str | None = None,
        use_bearer_for_custom_base: bool = False,
        litellm_params: Mapping[str, object] | None = None,
        allow_workload_identity: bool = False,
    ) -> Mapping[str, str] | None:
        """Resolve Anthropic credentials and return the appropriate auth header dict.

        Checks ANTHROPIC_API_KEY first (-> x-api-key or Bearer depending on
        use_bearer_for_custom_base), then ANTHROPIC_AUTH_TOKEN (-> Authorization: Bearer),
        then workload identity federation (-> Authorization: Bearer with a minted
        sk-ant-oat01 token, honoring anthropic_* litellm_params when provided). Every
        Bearer built from an sk-ant-oat token carries the mandatory oauth anthropic-beta.
        Returns None if no credential source is available.
        """
        static_header: Final = AnthropicModelInfo._static_auth_header(api_key, api_base, use_bearer_for_custom_base)
        if static_header is not None:
            return static_header
        if not allow_workload_identity:
            return None
        wif_token: Final = get_anthropic_wif_token(litellm_params, api_base, "")
        if wif_token is not None:
            return AnthropicModelInfo._oauth_bearer_header(wif_token)
        return None

    @staticmethod
    async def aget_auth_header(
        api_key: str | None = None,
        api_base: str | None = None,
        use_bearer_for_custom_base: bool = False,
        litellm_params: Mapping[str, object] | None = None,
        allow_workload_identity: bool = False,
    ) -> Mapping[str, str] | None:
        """Async counterpart of get_auth_header: the WIF tier can block on a token
        exchange POST, so async callers await it off the event loop."""
        static_header: Final = AnthropicModelInfo._static_auth_header(api_key, api_base, use_bearer_for_custom_base)
        if static_header is not None:
            return static_header
        if not allow_workload_identity:
            return None
        wif_token: Final = await aget_anthropic_wif_token(litellm_params, api_base, "")
        if wif_token is not None:
            return AnthropicModelInfo._oauth_bearer_header(wif_token)
        return None

    @staticmethod
    def _static_auth_header(
        api_key: str | None,
        api_base: str | None,
        use_bearer_for_custom_base: bool,
    ) -> Mapping[str, str] | None:
        resolved_key: Final = AnthropicModelInfo.get_api_key(api_key)
        if resolved_key is not None:
            if is_anthropic_oauth_key(resolved_key):
                return AnthropicModelInfo._oauth_bearer_header(resolved_key)
            return AnthropicModelInfo._make_api_key_auth_header(resolved_key, api_base, use_bearer_for_custom_base)
        auth_token: Final = AnthropicModelInfo.get_auth_token()
        if auth_token is not None:
            return {"authorization": f"Bearer {auth_token}"}
        return None

    @staticmethod
    def _oauth_bearer_header(token: str) -> Mapping[str, str]:
        return {"authorization": f"Bearer {token}", "anthropic-beta": ANTHROPIC_OAUTH_BETA_HEADER}

    @staticmethod
    def get_base_model(model: str | None = None) -> str | None:
        return model.replace("anthropic/", "") if model else None

    def get_models(self, api_key: str | None = None, api_base: str | None = None) -> list[str]:
        return self._list_models(api_key=api_key, api_base=api_base, litellm_params=None)

    def discover_models(
        self, litellm_params: Mapping[str, object] | None = None
    ) -> list[str]:  # mutable-ok: matches get_models' list[str] contract shared by every provider override
        """Live discovery for a configured deployment: unlike ``get_models``, this threads the
        full ``litellm_params`` into ``get_auth_header`` so a workload-identity-federation source
        configured on the deployment (rather than the environment) is honored, gated the same way
        every other Anthropic auth surface is via ``config_allows_workload_identity``."""
        return self._list_models(
            api_key=_litellm_params_str(litellm_params, "api_key"),
            api_base=_litellm_params_str(litellm_params, "api_base"),
            litellm_params=litellm_params,
        )

    def _list_models(
        self,
        *,
        api_key: str | None,
        api_base: str | None,
        litellm_params: Mapping[str, object] | None,
    ) -> list[str]:  # mutable-ok: matches get_models' list[str] contract shared by every provider override
        resolved_api_base: Final = AnthropicModelInfo.get_api_base(api_base)
        auth_header: Final = AnthropicModelInfo.get_auth_header(
            api_key,
            resolved_api_base,
            litellm_params=litellm_params,
            allow_workload_identity=config_allows_workload_identity(self),
        )
        if resolved_api_base is None or auth_header is None:
            raise ValueError(
                "ANTHROPIC_API_BASE/ANTHROPIC_BASE_URL or ANTHROPIC_API_KEY/ANTHROPIC_AUTH_TOKEN (or workload "
                "identity federation via ANTHROPIC_FEDERATION_RULE_ID/ANTHROPIC_ORGANIZATION_ID/"
                "ANTHROPIC_IDENTITY_TOKEN_FILE) is not set. Please set the environment variable, to query "
                "Anthropic's `/models` endpoint."
            )
        headers: Final = MappingProxyType({"anthropic-version": "2023-06-01", **auth_header})
        # /v1/models is appended below, so a base the operator already wrote as .../v1 or
        # .../v1/messages would otherwise be asked for /v1/v1/models.
        model_ids: Final = _fetch_anthropic_model_ids(
            anthropic_base_without_chat_suffix(resolved_api_base),
            headers,
            after_id=None,
            pages_left=_MODEL_LIST_PAGE_CAP,
        )
        return [  # mutable-ok: matches get_models' list[str] contract shared by every provider override
            "anthropic/" + model_id for model_id in model_ids
        ]

    def get_token_counter(self) -> BaseTokenCounter | None:
        """
        Factory method to create an Anthropic token counter.

        Returns:
            AnthropicTokenCounter instance for this provider.
        """
        from litellm.llms.anthropic.count_tokens.token_counter import (
            AnthropicTokenCounter,
        )

        return AnthropicTokenCounter()


def strip_advisor_blocks_from_messages(messages: list[Any], replace_with_text: bool = False) -> list[Any]:
    """
    Remove (or replace) server_tool_use (name='advisor') and advisor_tool_result blocks
    from assistant message content.

    Prevents Anthropic 400 invalid_request_error: if advisor_tool_result blocks
    exist in history but the advisor tool is not in the tools array, the API rejects
    the request. This happens when the user has removed the advisor tool for cost
    control or on a follow-up turn.

    Args:
        messages: Conversation history to process (mutated in-place).
        replace_with_text: When True, replace the advisor exchange with an
            <advisor_feedback> text block so the executor retains the semantic
            context of what the advisor said.  When False (default), strip silently.
    """
    for message in messages:
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue

        # Collect advisor server_tool_use ids and their advice text (for replace mode).
        advisor_id_to_text: dict = {}
        for block in content:
            if isinstance(block, dict) and block.get("type") == "server_tool_use" and block.get("name") == "advisor":
                bid = block.get("id")
                if bid:
                    advisor_id_to_text[bid] = None  # text filled in below

        if not advisor_id_to_text:
            continue

        # If replacing, collect the advisor response text from advisor_tool_result blocks.
        if replace_with_text:
            for block in content:
                if (
                    isinstance(block, dict)
                    and block.get("type") == "advisor_tool_result"
                    and block.get("tool_use_id") in advisor_id_to_text
                ):
                    raw = block.get("content") or ""
                    text = (
                        raw
                        if isinstance(raw, str)
                        else next(
                            (b.get("text", "") for b in raw if isinstance(b, dict) and b.get("type") == "text"),
                            "",
                        )
                    )
                    advisor_id_to_text[block["tool_use_id"]] = text

        new_content = []
        for block in content:
            if not isinstance(block, dict):
                new_content.append(block)
                continue
            is_advisor_use = (
                block.get("type") == "server_tool_use"
                and block.get("name") == "advisor"
                and block.get("id") in advisor_id_to_text
            )
            is_advisor_result = (
                block.get("type") == "advisor_tool_result" and block.get("tool_use_id") in advisor_id_to_text
            )
            if is_advisor_use:
                if replace_with_text:
                    advice = advisor_id_to_text.get(block.get("id")) or ""
                    if advice:
                        new_content.append(
                            {
                                "type": "text",
                                "text": f"<advisor_feedback>\n{advice}\n</advisor_feedback>",
                            }
                        )
                # else: drop silently
            elif is_advisor_result:
                pass  # always drop — replaced above (or stripped)
            else:
                new_content.append(block)

        message["content"] = new_content
    return messages


def is_anthropic_invalid_thinking_block_error(error_text: str) -> bool:
    """
    Detect Anthropic 400 errors caused by invalid thinking blocks in replayed
    history: a missing or invalid signature, or a block with empty thinking text.

    Known error formats:
    {"message":"messages.2.content.0.thinking.signature.str: Input should be a valid string"}
    messages.N.content.M.thinking.signature.str: Input should be a valid string
    messages.N.content.M: Invalid `signature` in `thinking` block
    messages.N.content.M.thinking: each thinking block must contain thinking
    """
    if not error_text:
        return False
    lower: Final = error_text.lower()
    if "thinking" not in lower:
        return False
    if "signature" in lower and ("invalid" in lower or "valid string" in lower):
        return True
    return "must contain thinking" in lower


def strip_thinking_blocks_from_anthropic_messages(messages: list[Any]) -> list[Any]:
    """
    Return a new message list with thinking / redacted_thinking content blocks removed
    from each message. Used to recover from invalid thinking signatures on retry.

    Messages whose content is a list and becomes empty after stripping are omitted,
    since Anthropic rejects empty content arrays.
    """
    out: Final[list[Any]] = []
    for m in messages:
        if not isinstance(m, dict):
            out.append(m)
            continue
        mm = copy.deepcopy(m)
        content = mm.get("content")
        if isinstance(content, list):
            filtered = [
                b for b in content if not (isinstance(b, dict) and b.get("type") in ("thinking", "redacted_thinking"))
            ]
            if not filtered:
                continue
            mm["content"] = filtered
        out.append(mm)
    return out


def strip_thinking_blocks_from_anthropic_messages_request_dict(
    data: dict[str, Any],
) -> None:
    """
    Mutate an Anthropic Messages-style request dict: strip thinking blocks from
    ``messages`` and remove the top-level ``thinking`` extended-thinking param.
    """
    msgs: Final = data.get("messages")
    if isinstance(msgs, list):
        data["messages"] = strip_thinking_blocks_from_anthropic_messages(msgs)
    data.pop("thinking", None)


def strip_empty_content_blocks_from_anthropic_messages(
    messages: list[Any],
) -> list[Any]:
    """
    Return a new message list with empty or whitespace-only ``{"type": "text"}``
    and ``{"type": "thinking"}`` content blocks removed.

    Anthropic's API rejects requests containing such blocks with
    ``"messages: text content blocks must be non-empty"`` and
    ``"messages.N.content.M.thinking: each thinking block must contain
    thinking"`` respectively.  Assistant messages routinely arrive with
    ``{"type": "text", "text": ""}`` alongside ``tool_use`` blocks (see
    anthropics/anthropic-sdk-python#461), and a turn served by a
    non-Anthropic reasoning model through the /v1/messages bridge can carry
    ``{"type": "thinking", "thinking": ""}`` when the model produced no
    reasoning text (e.g. it went straight to parallel tool calls).
    Multi-turn tool-use clients (e.g. Claude Code) loop these prior responses
    back as conversation history, which then causes the next request to 400
    on the unified ``/v1/messages`` path.  ``/v1/chat/completions`` already
    handles this in ``anthropic_messages_pt``; this helper provides the
    equivalent guarantee for the native Anthropic Messages path.
    ``redacted_thinking`` blocks are never touched: they carry opaque
    ``data`` instead of thinking text.

    Messages whose content is a list and becomes empty after stripping are
    omitted, matching :func:`strip_thinking_blocks_from_anthropic_messages`.
    The caller's list and its content blocks are never mutated; modified
    messages are returned as shallow copies with a fresh content list.
    """
    out: Final[list[Any]] = []
    for m in messages:
        if not isinstance(m, dict) or not isinstance(m.get("content"), list):
            out.append(m)
            continue
        content = m["content"]
        filtered = [b for b in content if not _is_empty_text_block(b) and not is_empty_thinking_block(b)]
        if len(filtered) == len(content):
            out.append(m)
        elif filtered:
            out.append({**m, "content": filtered})
    return out


def _is_empty_text_block(block: object) -> bool:
    if not isinstance(block, dict) or block.get("type") != "text":
        return False
    text: Final = block.get("text")
    return not isinstance(text, str) or not text.strip()


def is_empty_thinking_block(block: object) -> bool:
    """
    True for a ``{"type": "thinking"}`` content block whose thinking text is
    missing, not a string, or empty/whitespace-only after ``.strip()``.
    Anthropic rejects such blocks with ``"each thinking block must contain
    thinking"`` (whitespace-only included, verified live), regardless of any
    signature they carry.  ``redacted_thinking`` blocks are a different type
    and always return False.
    """
    if not isinstance(block, dict) or block.get("type") != "thinking":
        return False
    thinking: Final = block.get("thinking")
    return not isinstance(thinking, str) or not thinking.strip()


def is_empty_unsigned_thinking_block(block: object) -> bool:
    """
    True for an empty ``{"type": "thinking"}`` block carrying no signature.

    The emit-side predicate: response paths drop a thinking block only when it
    holds nothing the client could need.  A signature-only block is a real
    provider response (Bedrock Converse under adaptive thinking emits a
    reasoning block with empty text and only a signature) and the client needs
    the signature to replay reasoning across tool-use turns, so it must be
    emitted.  Request paths keep using :func:`is_empty_thinking_block`:
    Anthropic rejects empty thinking blocks in request history regardless of
    signature, and the inbound strip self-heals a replayed signature-only
    block.
    """
    if not isinstance(block, dict) or not is_empty_thinking_block(block):
        return False
    return not block.get("signature")


def normalize_anthropic_tool_use_id(raw_id: str) -> str:
    """
    Normalize a tool_use / tool_result id for Anthropic's ``^[a-zA-Z0-9_-]+$``
    pattern.

    Strips Gemini thought-signature suffixes (``__thought__``) first, then
    replaces any remaining invalid characters with underscores.
    """
    base_id = raw_id.split(THOUGHT_SIGNATURE_SEPARATOR, 1)[0] if THOUGHT_SIGNATURE_SEPARATOR in raw_id else raw_id
    sanitized: Final = re.sub(r"[^a-zA-Z0-9_-]", "_", base_id)
    return sanitized or "tool_use_id"


def _sanitize_tool_use_id_content_block(block: object) -> object:
    if not isinstance(block, dict):
        return block
    block_type: Final = block.get("type")
    if block_type in ("tool_use", "server_tool_use"):
        raw_id = block.get("id")
        if isinstance(raw_id, str):
            normalized = normalize_anthropic_tool_use_id(raw_id)
            if normalized != raw_id:
                return {**block, "id": normalized}
    elif block_type == "tool_result":
        raw_id = block.get("tool_use_id")
        if isinstance(raw_id, str):
            normalized = normalize_anthropic_tool_use_id(raw_id)
            if normalized != raw_id:
                return {**block, "tool_use_id": normalized}
    return block


def sanitize_tool_use_ids_in_anthropic_messages(messages: list[Any]) -> list[Any]:
    """
    Return a new message list with ``tool_use`` / ``server_tool_use`` ``id`` and
    ``tool_result`` ``tool_use_id`` values rewritten to satisfy Anthropic's
    ``^[a-zA-Z0-9_-]+$`` requirement.

    Cross-provider clients (e.g. Claude Code routed through kimi) may replay
    conversation history containing ids like ``functions.Bash:0`` with ``.``
    and ``:`` — valid on the upstream provider but rejected by Anthropic when
    the session is switched to a native Anthropic deployment.
    """
    out: Final[list[Any]] = []
    for m in messages:
        if not isinstance(m, dict) or not isinstance(m.get("content"), list):
            out.append(m)
            continue
        content = m["content"]
        new_content = [_sanitize_tool_use_id_content_block(b) for b in content]
        if new_content == content:
            out.append(m)
        else:
            out.append({**m, "content": new_content})
    return out


class _ReplayedSearchQuery(BaseModel):
    model_config = ConfigDict(extra="allow")

    query: str = ""


class _ReplayedWebSearchResult(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: Literal["web_search_result"]
    url: str = ""
    title: str = ""
    snippet: str = ""
    encrypted_content: str = ""


class _ReplayedWebSearchToolResult(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: Literal["web_search_tool_result"]
    tool_use_id: str
    content: tuple[_ReplayedWebSearchResult, ...]


class _ReplayedServerToolUse(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: Literal["server_tool_use"]
    id: str
    input: _ReplayedSearchQuery = _ReplayedSearchQuery()


class _TextBlock(BaseModel):
    type: Literal["text"] = "text"
    text: str


_WEB_SEARCH_TOOL_RESULT_ADAPTER: Final = TypeAdapter(_ReplayedWebSearchToolResult)
_SERVER_TOOL_USE_ADAPTER: Final = TypeAdapter(_ReplayedServerToolUse)


def _flattenable_web_search_tool_result(block: object) -> _ReplayedWebSearchToolResult | None:
    """
    The parsed block when it is a ``web_search_tool_result`` carrying no
    ``encrypted_content``, else None for anything Anthropic itself issued.

    An empty ``content`` list is flattenable too. It is what the interceptor emits
    when a search legitimately returns nothing and when a search raises, and it
    carries neither evidence to preserve nor an ``encrypted_content`` to respect,
    so leaving it in place only buys the 400 this whole function exists to avoid.
    """
    try:
        parsed: Final = _WEB_SEARCH_TOOL_RESULT_ADAPTER.validate_python(block)
    except ValidationError:
        return None
    if any(result.encrypted_content for result in parsed.content):
        return None
    return parsed


def _replayed_server_tool_use(block: object) -> _ReplayedServerToolUse | None:
    try:
        return _SERVER_TOOL_USE_ADAPTER.validate_python(block)
    except ValidationError:
        return None


def _render_web_search_results(query: str, results: tuple[_ReplayedWebSearchResult, ...]) -> str:
    header: Final = f"Web search results for '{query}':" if query else "Web search results:"
    if not results:
        return f"{header}\n\nNo results were returned."
    body: Final = "\n\n".join(
        "\n".join(
            line
            for line in (
                f"Title: {result.title}" if result.title else "",
                f"URL: {result.url}" if result.url else "",
                f"Snippet: {result.snippet}" if result.snippet else "",
            )
            if line
        )
        for result in results
    )
    return f"{header}\n\n{body}" if body else header


def _rewrite_replayed_web_search_block(
    block: object,
    flattenable: Mapping[str, _ReplayedWebSearchToolResult],
    queries: Mapping[str, str],
) -> object | None:
    parsed_result: Final = _flattenable_web_search_tool_result(block)
    if parsed_result is not None:
        return _TextBlock(
            text=_render_web_search_results(queries.get(parsed_result.tool_use_id, ""), parsed_result.content)
        ).model_dump()
    parsed_use: Final = _replayed_server_tool_use(block)
    if parsed_use is not None and parsed_use.id in flattenable:
        return None
    return block


def _flatten_web_search_results_in_message(message: object) -> object:
    if not isinstance(message, Mapping) or not isinstance(message.get("content"), Sequence):
        return message
    content: Final = message["content"]
    if isinstance(content, str):
        return message
    flattenable: Final = MappingProxyType(
        {
            parsed.tool_use_id: parsed
            for parsed in (_flattenable_web_search_tool_result(block) for block in content)
            if parsed is not None
        }
    )
    if not flattenable:
        return message
    queries: Final = MappingProxyType(
        {
            parsed.id: parsed.input.query
            for parsed in (_replayed_server_tool_use(block) for block in content)
            if parsed is not None
        }
    )
    rewritten: Final = tuple(_rewrite_replayed_web_search_block(block, flattenable, queries) for block in content)
    return {**message, "content": [b for b in rewritten if b is not None]}  # mutable-ok: JSON wire format


def flatten_unencrypted_web_search_results_in_anthropic_messages(  # mutable-ok: as sibling sanitizers
    messages: list[Any],
) -> list[Any]:
    """
    Return a new message list with replayed ``web_search_tool_result`` blocks that
    carry no ``encrypted_content`` rewritten into plain ``text`` blocks holding the
    same title / url / snippet evidence.

    ``encrypted_content`` is an opaque blob only Anthropic's own search backend can
    mint, so blocks synthesized by LiteLLM (websearch interception against a search
    provider) are rejected with ``Invalid encrypted_content in search_result block``
    when a native client loops them back as history. Flattening them keeps the
    evidence in the conversation instead of 400ing the follow-up turn, and leaves
    genuine Anthropic-issued blocks untouched.
    """
    return [_flatten_web_search_results_in_message(m) for m in messages]  # mutable-ok: JSON wire format


def process_anthropic_headers(headers: httpx.Headers | dict) -> dict:
    openai_headers: Final = {}
    if "anthropic-ratelimit-requests-limit" in headers:
        openai_headers["x-ratelimit-limit-requests"] = headers["anthropic-ratelimit-requests-limit"]
    if "anthropic-ratelimit-requests-remaining" in headers:
        openai_headers["x-ratelimit-remaining-requests"] = headers["anthropic-ratelimit-requests-remaining"]
    if "anthropic-ratelimit-tokens-limit" in headers:
        openai_headers["x-ratelimit-limit-tokens"] = headers["anthropic-ratelimit-tokens-limit"]
    if "anthropic-ratelimit-tokens-remaining" in headers:
        openai_headers["x-ratelimit-remaining-tokens"] = headers["anthropic-ratelimit-tokens-remaining"]

    llm_response_headers: Final = {"{}-{}".format("llm_provider", k): v for k, v in headers.items()}

    additional_headers: Final = {**llm_response_headers, **openai_headers}
    return additional_headers


def _anthropic_model_entry(
    model: ModelInfoResponse, created_at: str, display_names: Mapping[str, str]
) -> Mapping[str, object]:
    return {  # mutable-ok: JSON response body, serialized by the route and never mutated
        "type": "model",
        "id": model["id"],
        "display_name": display_names.get(model["id"], model["id"]),
        "created_at": created_at,
        "max_input_tokens": model.get("max_input_tokens"),
        "max_tokens": model.get("max_output_tokens"),
    }


def create_anthropic_model_list_response(
    models: Sequence[ModelInfoResponse],
    display_names: Mapping[str, str] = MappingProxyType({}),
) -> Mapping[str, object]:
    """Build the Anthropic-native /v1/models envelope.

    Clients that send an anthropic-version header parse the Anthropic Models API
    shape (type/display_name/created_at plus has_more/first_id/last_id) and filter
    the list themselves, so every model is returned here. The token limits carry
    over from the OpenAI-shaped listing, named as the Messages API names them, and
    are always present because the vendor shape declares them nullable, not optional.
    display_names maps a listed model id to a configured human-readable name; ids
    without an entry fall back to the id itself, matching the vendor behavior
    """
    created_at: Final = (
        datetime.fromtimestamp(DEFAULT_MODEL_CREATED_AT_TIME, tz=timezone.utc).isoformat().replace("+00:00", "Z")
    )
    data: Final = [  # mutable-ok: JSON response body, serialized by the route and never mutated
        _anthropic_model_entry(model, created_at, display_names) for model in models
    ]
    return {  # mutable-ok: JSON response body, serialized by the route and never mutated
        "data": data,
        "has_more": False,
        "first_id": models[0]["id"] if models else None,
        "last_id": models[-1]["id"] if models else None,
    }
