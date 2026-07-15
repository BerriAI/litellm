"""
This file contains common utils for anthropic calls.
"""

import copy
import re
from typing import Any, Dict, List, Mapping, Optional, Union

import httpx

import litellm
from litellm.litellm_core_utils.prompt_templates.common_utils import (
    get_file_ids_from_messages,
)
from litellm.litellm_core_utils.prompt_templates.factory import (
    THOUGHT_SIGNATURE_SEPARATOR,
)
from litellm.llms.base_llm.base_utils import BaseLLMModelInfo, BaseTokenCounter
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.types.llms.anthropic import (
    ANTHROPIC_BETA_HEADER_VALUES,
    ANTHROPIC_HOSTED_TOOLS,
    ANTHROPIC_OAUTH_BETA_HEADER,
    ANTHROPIC_OAUTH_TOKEN_PREFIX,
    AllAnthropicToolsValues,
    AnthropicMcpServerTool,
)
from litellm.types.llms.openai import AllMessageValues

_BEDROCK_VERSION_SUFFIX_RE = re.compile(r"-v\d+(?::\d+)?$")
_INFERENCE_PROFILE_MINOR_RE = re.compile(r":\d+$")
_DATED_RELEASE_SUFFIX_RE = re.compile(r"-\d{8}$")
_DOTTED_VERSION_RE = re.compile(r"(\d)\.(\d)")
ANTHROPIC_SERVER_SIDE_FALLBACKS_PARAM = "anthropic_server_fallbacks"


def is_anthropic_server_side_fallback_request(
    request_data: Mapping[str, object],
    headers: Mapping[str, str],
) -> bool:
    beta_header = next(
        (value for key, value in headers.items() if key.lower() == "anthropic-beta"),
        None,
    )
    if beta_header is None or not isinstance(request_data.get("fallbacks"), list):
        return False
    return ANTHROPIC_BETA_HEADER_VALUES.SERVER_SIDE_FALLBACK_2026_06_01.value in {
        value.strip() for value in beta_header.split(",")
    }


def normalize_anthropic_server_side_fallbacks(
    request_data: Mapping[str, object],
    headers: Mapping[str, str],
) -> dict[str, object]:
    sanitized_request_data = {
        key: value for key, value in request_data.items() if key != ANTHROPIC_SERVER_SIDE_FALLBACKS_PARAM
    }
    if not is_anthropic_server_side_fallback_request(
        request_data=request_data,
        headers=headers,
    ):
        return sanitized_request_data
    return {
        **{key: value for key, value in sanitized_request_data.items() if key != "fallbacks"},
        ANTHROPIC_SERVER_SIDE_FALLBACKS_PARAM: request_data["fallbacks"],
    }


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


def is_anthropic_oauth_key(value: Optional[str]) -> bool:
    """Check if a value contains an Anthropic OAuth token (sk-ant-oat*)."""
    if value is None:
        return False
    # Handle both raw token and "Bearer <token>" format
    if value.startswith("Bearer "):
        value = value[7:]
    return value.startswith(ANTHROPIC_OAUTH_TOKEN_PREFIX)


def _merge_beta_headers(existing: Optional[str], new_beta: str) -> str:
    """Merge a new beta value into an existing comma-separated anthropic-beta header."""
    if not existing:
        return new_beta
    betas = {b.strip() for b in existing.split(",") if b.strip()}
    betas.add(new_beta)
    return ",".join(sorted(betas))


def optionally_handle_anthropic_oauth(headers: dict, api_key: Optional[str]) -> tuple[dict, Optional[str]]:
    """
    Handle Anthropic OAuth token detection and header setup.

    If an OAuth token is detected in the Authorization header, extracts it
    and sets the required OAuth headers.

    Args:
        headers: Request headers dict
        api_key: Current API key (may be None)

    Returns:
        Tuple of (updated headers, api_key)
    """
    # Check Authorization header (passthrough / forwarded requests)
    auth_header = headers.get("authorization", "")
    if auth_header and auth_header.startswith(f"Bearer {ANTHROPIC_OAUTH_TOKEN_PREFIX}"):
        api_key = auth_header.replace("Bearer ", "")
        headers.pop("x-api-key", None)
        headers["anthropic-beta"] = _merge_beta_headers(headers.get("anthropic-beta"), ANTHROPIC_OAUTH_BETA_HEADER)
        headers["anthropic-dangerous-direct-browser-access"] = "true"
        return headers, api_key
    # Check api_key directly (standard chat/completion flow)
    if api_key and api_key.startswith(ANTHROPIC_OAUTH_TOKEN_PREFIX):
        headers.pop("x-api-key", None)
        headers["authorization"] = f"Bearer {api_key}"
        headers["anthropic-beta"] = _merge_beta_headers(headers.get("anthropic-beta"), ANTHROPIC_OAUTH_BETA_HEADER)
        headers["anthropic-dangerous-direct-browser-access"] = "true"
    return headers, api_key


class AnthropicError(BaseLLMException):
    def __init__(
        self,
        status_code: int,
        message,
        headers: Optional[httpx.Headers] = None,
    ):
        super().__init__(status_code=status_code, message=message, headers=headers)


class AnthropicModelInfo(BaseLLMModelInfo):
    def is_cache_control_set(self, messages: List[AllMessageValues]) -> bool:
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

    def is_file_id_used(self, messages: List[AllMessageValues]) -> bool:
        """
        Return if {"source": {"type": "file", "file_id": ..}} in message content block
        """
        file_ids = get_file_ids_from_messages(messages)
        return len(file_ids) > 0

    def is_mcp_server_used(self, mcp_servers: Optional[List[AnthropicMcpServerTool]]) -> bool:
        if mcp_servers is None:
            return False
        if mcp_servers:
            return True
        return False

    def is_computer_tool_used(self, tools: Optional[List[AllAnthropicToolsValues]]) -> Optional[str]:
        """Returns the computer tool version if used, e.g. 'computer_20250124' or None"""
        if tools is None:
            return None
        for tool in tools:
            if "type" in tool and tool["type"].startswith("computer_"):
                return tool["type"]
        return None

    def is_web_search_tool_used(self, tools: Optional[List[AllAnthropicToolsValues]]) -> bool:
        """Returns True if web_search tool is used"""
        if tools is None:
            return False
        for tool in tools:
            if "type" in tool and tool["type"].startswith(ANTHROPIC_HOSTED_TOOLS.WEB_SEARCH.value):
                return True
        return False

    def is_pdf_used(self, messages: List[AllMessageValues]) -> bool:
        """
        Set to true if media passed into messages.

        """
        for message in messages:
            if "content" in message and message["content"] is not None and isinstance(message["content"], list):
                for content in message["content"]:
                    if "type" in content and content["type"] != "text":
                        return True
        return False

    def is_tool_search_used(self, tools: Optional[List]) -> bool:
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

    def is_programmatic_tool_calling_used(self, tools: Optional[List]) -> bool:
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

    def is_input_examples_used(self, tools: Optional[List]) -> bool:
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
        flag = AnthropicModelInfo._get_model_capability(model, "supports_sampling_params")
        if flag is not None:
            return flag
        model_lower = model.lower()
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
            supported_hint = "Only temperature=1 is supported. " if param == "temperature" else ""
            raise litellm.utils.UnsupportedParamsError(
                message=(
                    f"{model} does not support {param}={value}. {supported_hint}"
                    "To drop unsupported params, set `litellm.drop_params = True`."
                ),
                status_code=400,
            )

    @staticmethod
    def _strip_version_suffix(model: str) -> str:
        at = model.rfind("@")
        if at > 0:
            return model[:at]
        return model

    @staticmethod
    def _model_map_lookup_candidates(model: str) -> List[str]:
        """Model-map keys to try for ``model``: the id itself, the same id with a
        bedrock/vertex routing prefix removed, the Bedrock base model, and each of
        those normalized by stripping a Bedrock version suffix (``-v1:0`` fully or
        just the ``:0`` inference-profile minor), stripping a dated-release suffix
        (``-20260205``), or rewriting a dotted family version to hyphens
        (``4.6`` -> ``4-6``). Lets any reasonable alias (e.g.
        ``bedrock/invoke/global.anthropic.claude-opus-4-7-v1:0``,
        ``claude-sonnet-4-6-20260219`` or ``claude-sonnet-4.6``) resolve to its base
        cost-map entry so the capability flag on that entry stays authoritative."""
        prefixes = (
            "bedrock/converse/",
            "bedrock/invoke/",
            "bedrock/",
            "vertex_ai/",
        )
        deprefixed = tuple(model[len(p) :] for p in prefixes if model.startswith(p))
        try:
            from litellm.llms.bedrock.common_utils import BedrockModelInfo

            base = BedrockModelInfo.get_base_model(model)
        except Exception:
            base = None
        bedrock_base = (base, f"bedrock/{base}") if base else ()
        primary = (model, *deprefixed, *bedrock_base)
        normalized = tuple(
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
    def _get_model_capability(model: str, key: str) -> Optional[bool]:
        """Read boolean capability ``key`` from the model map, or None when
        no entry declares it."""
        from litellm.utils import _get_bundled_model_cost_map

        try:
            candidates = AnthropicModelInfo._model_map_lookup_candidates(model)
            for model_cost in (litellm.model_cost, _get_bundled_model_cost_map()):
                for cand in candidates:
                    value = model_cost.get(cand, {}).get(key)
                    if isinstance(value, bool):
                        return value
        except Exception:
            pass
        return None

    @staticmethod
    def _get_exact_model_capability(model: str, key: str) -> Optional[bool]:
        """Read boolean capability ``key`` from the exact model-map entry only.

        Unlike ``_get_model_capability``, does not walk stripped provider aliases.
        Use when a feature is tied to a specific host (e.g. Anthropic API fast mode).
        """
        value = litellm.model_cost.get(model, {}).get(key)
        return value if isinstance(value, bool) else None

    @staticmethod
    def _get_provider_resolved_capability(model: str, key: str, custom_llm_provider: str) -> Optional[bool]:
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
            value = _get_model_info_helper(model=resolved_model, custom_llm_provider=resolved_provider).get(key)
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

        resolved = AnthropicModelInfo._get_provider_resolved_capability(model, key, custom_llm_provider)
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

    def is_effort_used(
        self,
        optional_params: Optional[dict],
        model: Optional[str] = None,
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
            reasoning_effort = optional_params.get("reasoning_effort")
            if reasoning_effort and isinstance(reasoning_effort, str):
                return True

        # Check if output_config is directly provided (for non-4.6 models)
        output_config = optional_params.get("output_config")
        if output_config and isinstance(output_config, dict):
            effort = output_config.get("effort")
            if effort and isinstance(effort, str):
                return True

        return False

    def is_code_execution_tool_used(self, tools: Optional[List]) -> bool:
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

    def is_container_with_skills_used(self, optional_params: Optional[dict]) -> bool:
        """
        Check if container with skills is being used.

        Returns True if optional_params contains container with skills.
        """
        if not optional_params:
            return False

        container = optional_params.get("container")
        if container and isinstance(container, dict):
            skills = container.get("skills")
            if skills and isinstance(skills, list) and len(skills) > 0:
                return True
        return False

    def _get_user_anthropic_beta_headers(self, anthropic_beta_header: Optional[str]) -> Optional[List[str]]:
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
        computer_tool_beta_mapping = {
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
        optional_params: Optional[dict] = None,
        computer_tool_used: Optional[str] = None,
        prompt_caching_set: bool = False,
        file_id_used: bool = False,
        mcp_server_used: bool = False,
        *,
        custom_llm_provider: str,
    ) -> List[str]:
        """
        Get list of common beta headers based on the features that are active.

        Returns:
            List of beta header strings
        """
        from litellm.types.llms.anthropic import ANTHROPIC_EFFORT_BETA_HEADER

        betas = []

        # Detect features
        effort_used = self.is_effort_used(optional_params, model, custom_llm_provider=custom_llm_provider)

        if effort_used:
            betas.append(ANTHROPIC_EFFORT_BETA_HEADER)  # effort-2025-11-24

        if computer_tool_used:
            beta_header = self.get_computer_tool_beta_header(computer_tool_used)
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
    def _make_api_key_auth_header(api_key: str, api_base: str | None, use_bearer_for_custom_base: bool = False) -> dict:
        if use_bearer_for_custom_base and (
            api_base and "api.anthropic.com" not in api_base and not api_key.startswith("sk-ant-")
        ):
            value = api_key if api_key.startswith("Bearer ") else f"Bearer {api_key}"
            return {"authorization": value}
        return {"x-api-key": api_key}

    def get_anthropic_headers(
        self,
        api_key: Optional[str] = None,
        auth_token: Optional[str] = None,
        anthropic_version: Optional[str] = None,
        computer_tool_used: Optional[str] = None,
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
        user_anthropic_beta_headers: Optional[List[str]] = None,
        code_execution_tool_used: bool = False,
        container_with_skills_used: bool = False,
        api_base: str | None = None,
        use_bearer_for_custom_base: bool = False,
    ) -> dict:
        betas = set()
        # Anthropic no longer requires the prompt-caching beta header
        # Prompt caching now works automatically when cache_control is used in messages
        # Reference: https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching
        if computer_tool_used:
            beta_header = self.get_computer_tool_beta_header(computer_tool_used)
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

        _is_oauth = api_key and api_key.startswith(ANTHROPIC_OAUTH_TOKEN_PREFIX)
        headers = {
            "anthropic-version": anthropic_version or "2023-06-01",
            "accept": "application/json",
            "content-type": "application/json",
        }
        if _is_oauth:
            headers["authorization"] = f"Bearer {api_key}"
            headers["anthropic-dangerous-direct-browser-access"] = "true"
            betas.add(ANTHROPIC_OAUTH_BETA_HEADER)
        elif auth_token and not api_key:
            headers["authorization"] = f"Bearer {auth_token}"
        elif api_key:
            headers.update(self._make_api_key_auth_header(api_key, api_base, use_bearer_for_custom_base))

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
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> Dict:
        if api_base is None and isinstance(litellm_params, dict):
            api_base = litellm_params.get("api_base")
        use_bearer_for_custom_base: bool = bool(
            isinstance(litellm_params, dict) and litellm_params.get("use_bearer_for_custom_base", False)
        )
        # Check for Anthropic OAuth token in headers
        headers, api_key = optionally_handle_anthropic_oauth(headers=headers, api_key=api_key)
        api_key = AnthropicModelInfo.get_api_key(api_key)
        # Resolve auth_token from ANTHROPIC_AUTH_TOKEN if api_key is not set
        auth_token: Optional[str] = None
        if api_key is None:
            auth_token = AnthropicModelInfo.get_auth_token()
        if api_key is None and auth_token is None:
            raise litellm.AuthenticationError(
                message="Missing Anthropic API Key - A call is being made to anthropic but no key is set either in the environment variables or via params. Please set `ANTHROPIC_API_KEY` or `ANTHROPIC_AUTH_TOKEN` in your environment vars",
                llm_provider="anthropic",
                model=model,
            )

        tools = optional_params.get("tools")
        prompt_caching_set = self.is_cache_control_set(messages=messages)
        computer_tool_used = self.is_computer_tool_used(tools=tools)
        mcp_server_used = self.is_mcp_server_used(mcp_servers=optional_params.get("mcp_servers"))
        pdf_used = self.is_pdf_used(messages=messages)
        file_id_used = self.is_file_id_used(messages=messages)
        web_search_tool_used = self.is_web_search_tool_used(tools=tools)
        tool_search_used = self.is_tool_search_used(tools=tools)
        programmatic_tool_calling_used = self.is_programmatic_tool_calling_used(tools=tools)
        input_examples_used = self.is_input_examples_used(tools=tools)
        effort_used = self.is_effort_used(optional_params=optional_params, model=model, custom_llm_provider="anthropic")
        code_execution_tool_used = self.is_code_execution_tool_used(tools=tools)
        container_with_skills_used = self.is_container_with_skills_used(optional_params=optional_params)
        user_anthropic_beta_headers = self._get_user_anthropic_beta_headers(
            anthropic_beta_header=headers.get("anthropic-beta")
        )
        anthropic_headers = self.get_anthropic_headers(
            computer_tool_used=computer_tool_used,
            prompt_caching_set=prompt_caching_set,
            pdf_used=pdf_used,
            api_key=api_key,
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
        )

        headers = {**headers, **anthropic_headers}

        return headers

    @staticmethod
    def get_api_base(api_base: Optional[str] = None) -> Optional[str]:
        from litellm.secret_managers.main import get_secret_str

        return (
            api_base
            or get_secret_str("ANTHROPIC_API_BASE")
            or get_secret_str("ANTHROPIC_BASE_URL")
            or "https://api.anthropic.com"
        )

    @staticmethod
    def get_api_key(api_key: Optional[str] = None) -> Optional[str]:
        from litellm.secret_managers.main import get_secret_str

        return api_key or get_secret_str("ANTHROPIC_API_KEY")

    @staticmethod
    def get_auth_token(auth_token: Optional[str] = None) -> Optional[str]:
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
    ) -> dict | None:
        """Resolve Anthropic credentials and return the appropriate auth header dict.

        Checks ANTHROPIC_API_KEY first (-> x-api-key or Bearer depending on
        use_bearer_for_custom_base), then ANTHROPIC_AUTH_TOKEN (-> Authorization: Bearer).
        Returns None if neither is available.
        """
        resolved_key = AnthropicModelInfo.get_api_key(api_key)
        if resolved_key is not None:
            if is_anthropic_oauth_key(resolved_key):
                return {"authorization": f"Bearer {resolved_key}"}
            return AnthropicModelInfo._make_api_key_auth_header(resolved_key, api_base, use_bearer_for_custom_base)
        auth_token = AnthropicModelInfo.get_auth_token()
        if auth_token is not None:
            return {"authorization": f"Bearer {auth_token}"}
        return None

    @staticmethod
    def get_base_model(model: Optional[str] = None) -> Optional[str]:
        return model.replace("anthropic/", "") if model else None

    def get_models(self, api_key: Optional[str] = None, api_base: Optional[str] = None) -> List[str]:
        api_base = AnthropicModelInfo.get_api_base(api_base)
        auth_header = AnthropicModelInfo.get_auth_header(api_key, api_base)
        if api_base is None or auth_header is None:
            raise ValueError(
                "ANTHROPIC_API_BASE/ANTHROPIC_BASE_URL or ANTHROPIC_API_KEY/ANTHROPIC_AUTH_TOKEN is not set. Please set the environment variable, to query Anthropic's `/models` endpoint."
            )
        headers = {"anthropic-version": "2023-06-01"}
        headers.update(auth_header)
        response = litellm.module_level_client.get(
            url=f"{api_base}/v1/models",
            headers=headers,
        )

        try:
            response.raise_for_status()
        except httpx.HTTPStatusError:
            raise Exception(
                f"Failed to fetch models from Anthropic. Status code: {response.status_code}, Response: {response.text}"
            )

        models = response.json()["data"]

        litellm_model_names = []
        for model in models:
            stripped_model_name = model["id"]
            litellm_model_name = "anthropic/" + stripped_model_name
            litellm_model_names.append(litellm_model_name)
        return litellm_model_names

    def get_token_counter(self) -> Optional[BaseTokenCounter]:
        """
        Factory method to create an Anthropic token counter.

        Returns:
            AnthropicTokenCounter instance for this provider.
        """
        from litellm.llms.anthropic.count_tokens.token_counter import (
            AnthropicTokenCounter,
        )

        return AnthropicTokenCounter()


def strip_advisor_blocks_from_messages(messages: List[Any], replace_with_text: bool = False) -> List[Any]:
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


def is_anthropic_invalid_thinking_signature_error(error_text: str) -> bool:
    """
    Detect Anthropic 400 when encrypted thinking signatures in history do not match
    the current deployment (e.g. user rotated API key or switched model endpoint).

    Example API message:
    messages.N.content.M: Invalid `signature` in `thinking` block
    """
    if not error_text:
        return False
    lower = error_text.lower()
    return "invalid" in lower and "signature" in lower and "thinking" in lower and "block" in lower


def strip_thinking_blocks_from_anthropic_messages(messages: List[Any]) -> List[Any]:
    """
    Return a new message list with thinking / redacted_thinking content blocks removed
    from each message. Used to recover from invalid thinking signatures on retry.

    Messages whose content is a list and becomes empty after stripping are omitted,
    since Anthropic rejects empty content arrays.
    """
    out: List[Any] = []
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
    data: Dict[str, Any],
) -> None:
    """
    Mutate an Anthropic Messages-style request dict: strip thinking blocks from
    ``messages`` and remove the top-level ``thinking`` extended-thinking param.
    """
    msgs = data.get("messages")
    if isinstance(msgs, list):
        data["messages"] = strip_thinking_blocks_from_anthropic_messages(msgs)
    data.pop("thinking", None)


def strip_empty_text_blocks_from_anthropic_messages(
    messages: List[Any],
) -> List[Any]:
    """
    Return a new message list with empty or whitespace-only ``{"type": "text"}``
    content blocks removed.

    Anthropic's API rejects requests containing such blocks with
    ``"messages: text content blocks must be non-empty"``, but assistant
    messages from Anthropic routinely arrive with ``{"type": "text", "text": ""}``
    alongside ``tool_use`` blocks (see anthropics/anthropic-sdk-python#461).
    Multi-turn tool-use clients (e.g. Claude Code) loop these prior responses
    back as conversation history, which then causes the next request to 400
    on the unified ``/v1/messages`` path.  ``/v1/chat/completions`` already
    handles this in ``anthropic_messages_pt``; this helper provides the
    equivalent guarantee for the native Anthropic Messages path.

    Messages whose content is a list and becomes empty after stripping are
    omitted, matching :func:`strip_thinking_blocks_from_anthropic_messages`.
    The caller's list and its content blocks are never mutated; modified
    messages are returned as shallow copies with a fresh content list.
    """
    out: List[Any] = []
    for m in messages:
        if not isinstance(m, dict) or not isinstance(m.get("content"), list):
            out.append(m)
            continue
        content = m["content"]
        filtered = [b for b in content if not _is_empty_text_block(b)]
        if len(filtered) == len(content):
            out.append(m)
        elif filtered:
            out.append({**m, "content": filtered})
    return out


def _is_empty_text_block(block: Any) -> bool:
    if not isinstance(block, dict) or block.get("type") != "text":
        return False
    text = block.get("text")
    return not isinstance(text, str) or not text.strip()


def normalize_anthropic_tool_use_id(raw_id: str) -> str:
    """
    Normalize a tool_use / tool_result id for Anthropic's ``^[a-zA-Z0-9_-]+$``
    pattern.

    Strips Gemini thought-signature suffixes (``__thought__``) first, then
    replaces any remaining invalid characters with underscores.
    """
    base_id = raw_id.split(THOUGHT_SIGNATURE_SEPARATOR, 1)[0] if THOUGHT_SIGNATURE_SEPARATOR in raw_id else raw_id
    sanitized = re.sub(r"[^a-zA-Z0-9_-]", "_", base_id)
    return sanitized or "tool_use_id"


def _sanitize_tool_use_id_content_block(block: Any) -> Any:
    if not isinstance(block, dict):
        return block
    block_type = block.get("type")
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
    out: list[Any] = []
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


def process_anthropic_headers(headers: Union[httpx.Headers, dict]) -> dict:
    openai_headers = {}
    if "anthropic-ratelimit-requests-limit" in headers:
        openai_headers["x-ratelimit-limit-requests"] = headers["anthropic-ratelimit-requests-limit"]
    if "anthropic-ratelimit-requests-remaining" in headers:
        openai_headers["x-ratelimit-remaining-requests"] = headers["anthropic-ratelimit-requests-remaining"]
    if "anthropic-ratelimit-tokens-limit" in headers:
        openai_headers["x-ratelimit-limit-tokens"] = headers["anthropic-ratelimit-tokens-limit"]
    if "anthropic-ratelimit-tokens-remaining" in headers:
        openai_headers["x-ratelimit-remaining-tokens"] = headers["anthropic-ratelimit-tokens-remaining"]

    llm_response_headers = {"{}-{}".format("llm_provider", k): v for k, v in headers.items()}

    additional_headers = {**llm_response_headers, **openai_headers}
    return additional_headers
