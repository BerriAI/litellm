"""
Amazon Bedrock Mantle - Responses API backend.

Mantle serves Responses on two upstream paths: gpt frontier models (gpt-5.5 /
gpt-5.4) on `/openai/v1/responses`, and everything else that supports Responses
(e.g. gpt-oss) on the standard `/v1/responses`. The gate picks the path per
model and injects it via `use_openai_path`. Payloads and SSE follow the OpenAI
Responses spec, so this config inherits OpenAIResponsesAPIConfig and overrides
only the endpoint URL and authentication.

Auth: Bearer token (BEDROCK_MANTLE_API_KEY or the standard
AWS_BEARER_TOKEN_BEDROCK, or litellm_params.api_key) when present; otherwise
AWS SigV4 (service name "bedrock") using the standard credential chain (IAM
role / access key / profile / web identity), signed via the shared
BaseAWSLLM._sign_request after the request body is finalized.
"""

import json
from collections.abc import Mapping
from typing import Any, Final

from typing_extensions import ReadOnly, TypedDict

import litellm
from litellm._logging import verbose_logger
from litellm.llms.bedrock.base_aws_llm import BaseAWSLLM
from litellm.llms.bedrock_mantle.common_utils import (
    MANTLE_HOST_RE,
    BedrockMantleAuthMixin,
)
from litellm.llms.openai.responses.transformation import OpenAIResponsesAPIConfig
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import (
    ResponseInputParam,
    ResponsesAPIOptionalRequestParams,
)
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import LlmProviders

# Checked longest/most-specific first so a full endpoint URL collapses to host
# in one pass and the appended path never doubles.
_BASE_SUFFIXES_TO_STRIP: Final = (
    "/openai/v1/responses",
    "/v1/responses",
    "/responses",
    "/openai/v1",
    "/v1",
)

# Per Bedrock Mantle Responses API validation errors.
_BEDROCK_MANTLE_SUPPORTED_RESPONSE_TOOL_TYPES = frozenset({"function", "mcp", "custom", "namespace", "tool_search"})

_BEDROCK_MANTLE_SUPPORTED_SERVICE_TIERS: Final = frozenset({"auto", "default"})

_CODEX_ADDITIONAL_TOOLS_INPUT_ITEM_TYPE: Final = "additional_tools"

_CODEX_AGENT_MESSAGE_INPUT_ITEM_TYPE: Final = "agent_message"
_CODEX_CONTEXT_COMPACTION_INPUT_ITEM_TYPE: Final = "context_compaction"
_CODEX_LOCAL_SHELL_CALL_INPUT_ITEM_TYPE: Final = "local_shell_call"


class _RewrittenOutputTextBlock(TypedDict):
    type: ReadOnly[str]
    text: ReadOnly[str]


class _RewrittenAssistantMessageItem(TypedDict):
    type: ReadOnly[str]
    role: ReadOnly[str]
    content: ReadOnly[tuple[_RewrittenOutputTextBlock, ...]]


class _RewrittenCompactionItem(TypedDict):
    type: ReadOnly[str]
    encrypted_content: ReadOnly[str]


class _RewrittenFunctionCallItem(TypedDict):
    type: ReadOnly[str]
    call_id: ReadOnly[str]
    name: ReadOnly[str]
    arguments: ReadOnly[str]


class BedrockMantleResponsesAPIConfig(BedrockMantleAuthMixin, OpenAIResponsesAPIConfig):
    def __init__(
        self,
        aws_signer: BaseAWSLLM | None = None,
        use_openai_path: bool = True,
    ):
        super().__init__()
        self._aws_signer = aws_signer or BaseAWSLLM()
        self.use_openai_path = use_openai_path

    @property
    def custom_llm_provider(self) -> LlmProviders:
        return LlmProviders.BEDROCK_MANTLE

    def get_complete_url(
        self,
        api_base: str | None,
        litellm_params: dict,
    ) -> str:
        region: Final = self._resolve_region({**litellm_params, "api_base": api_base})
        base = api_base or get_secret_str("BEDROCK_MANTLE_API_BASE") or f"https://bedrock-mantle.{region}.api.aws"
        base = base.rstrip("/")
        for suffix in _BASE_SUFFIXES_TO_STRIP:
            if base.endswith(suffix):
                base = base[: -len(suffix)]
                break
        # For the standard Mantle host (including the default-region base that
        # responses/main.py auto-injects into litellm_params.api_base), pin to the
        # single resolved region so aws_region_name wins; preserve custom proxy hosts.
        if MANTLE_HOST_RE.match(base):
            base = f"https://bedrock-mantle.{region}.api.aws"
        path: Final = "/openai/v1/responses" if self.use_openai_path else "/v1/responses"
        return f"{base}{path}"

    def validate_environment(self, headers: dict, model: str, litellm_params: GenericLiteLLMParams | None) -> dict:
        litellm_params = litellm_params or GenericLiteLLMParams()
        bearer: Final = self._resolve_bearer_token(litellm_params.api_key)
        if bearer:
            headers["Authorization"] = f"Bearer {bearer}"
        if litellm_params.aws_bedrock_project_id:
            headers["OpenAI-Project"] = litellm_params.aws_bedrock_project_id
        return headers

    def supports_native_file_search(self) -> bool:
        return False

    def supports_native_websocket(self) -> bool:
        return False

    @staticmethod
    def _filter_unsupported_tools(tools: list[Any]) -> list[Any]:
        """Keep only tool types Mantle's Responses API accepts."""
        kept: Final[list[Any]] = []
        dropped_types: Final[list[str]] = []
        for tool in tools:
            if not isinstance(tool, dict):
                kept.append(tool)
                continue
            tool_type = tool.get("type")
            if tool_type in _BEDROCK_MANTLE_SUPPORTED_RESPONSE_TOOL_TYPES:
                kept.append(tool)
            else:
                dropped_types.append(str(tool_type))

        if dropped_types:
            verbose_logger.warning(
                "Bedrock Mantle Responses API: dropping unsupported tool type(s) %s (supported: %s).",
                sorted(set(dropped_types)),
                sorted(_BEDROCK_MANTLE_SUPPORTED_RESPONSE_TOOL_TYPES),
            )

        return kept

    @staticmethod
    def _handle_unsupported_service_tier(params: dict, drop_params: bool) -> dict:
        service_tier: Final = params.get("service_tier")
        if service_tier is None or service_tier in _BEDROCK_MANTLE_SUPPORTED_SERVICE_TIERS:
            return params
        if not drop_params:
            raise litellm.utils.UnsupportedParamsError(
                status_code=400,
                message=(
                    f"bedrock_mantle does not support service_tier={service_tier!r}; the Bedrock Mantle "
                    "Responses API only accepts 'auto' or 'default'. Set `drop_params: true` (litellm_settings "
                    "or this deployment's litellm_params) to have LiteLLM drop it, or remove service_tier from "
                    "the client (Codex CLI sends it when a speed tier is set in ~/.codex/config.toml)."
                ),
            )
        verbose_logger.warning(
            "Bedrock Mantle Responses API: dropping unsupported service_tier %r (supported: %s).",
            service_tier,
            sorted(_BEDROCK_MANTLE_SUPPORTED_SERVICE_TIERS),
        )
        return {key: value for key, value in params.items() if key != "service_tier"}

    def transform_responses_api_request(
        self,
        model: str,
        input: "str | ResponseInputParam",
        response_api_optional_request_params: dict,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> dict:
        remaining_input, hoisted_tools = self._hoist_codex_additional_tools(input)
        normalized_input: Final = self._normalize_codex_input_items(remaining_input)
        request_params: Final = (
            {
                **response_api_optional_request_params,
                "tools": [
                    *(response_api_optional_request_params.get("tools") or []),
                    *hoisted_tools,
                ],
            }
            if hoisted_tools
            else response_api_optional_request_params
        )
        return super().transform_responses_api_request(
            model=model,
            input=normalized_input,
            response_api_optional_request_params=request_params,
            litellm_params=litellm_params,
            headers=headers,
        )

    @staticmethod
    def _is_codex_additional_tools_item(item: Any) -> bool:
        return isinstance(item, dict) and item.get("type") == _CODEX_ADDITIONAL_TOOLS_INPUT_ITEM_TYPE

    @staticmethod
    def _tools_of_additional_tools_item(item: "dict[str, Any]") -> "list[Any]":
        tools: Final = item.get("tools")
        return tools if isinstance(tools, list) else []

    @classmethod
    def _hoist_codex_additional_tools(
        cls,
        input: "str | ResponseInputParam",
    ) -> "tuple[str | ResponseInputParam, list[Any]]":
        """Codex's "responses lite" wire mode ships tool definitions inside
        `input` as {"type": "additional_tools", "role": "developer",
        "tools": [...]} items. api.openai.com accepts that item type; Mantle
        rejects the whole request with 400 "Invalid 'input': value did not
        match any expected variant" but accepts the same tools at the top
        level, so move them there and strip the items from `input`.
        """
        if not isinstance(input, list):
            return input, []
        additional_tools_items: Final = [item for item in input if cls._is_codex_additional_tools_item(item)]
        if not additional_tools_items:
            return input, []
        remaining_input: Final = [item for item in input if not cls._is_codex_additional_tools_item(item)]
        hoisted_tools = [tool for item in additional_tools_items for tool in cls._tools_of_additional_tools_item(item)]
        verbose_logger.debug(
            "Bedrock Mantle Responses API: hoisting %d tool(s) out of %d 'additional_tools' input item(s) "
            "into the top-level tools param (Mantle rejects that input item type).",
            len(hoisted_tools),
            len(additional_tools_items),
        )
        return remaining_input, cls._filter_unsupported_tools(hoisted_tools)

    @staticmethod
    def _agent_message_text(item: "Mapping[str, object]") -> str:
        content: Final = item.get("content")
        if not isinstance(content, list):
            return ""
        return "".join(
            str(block.get("text") or block.get("encrypted_content") or "")
            for block in content
            if isinstance(block, dict)
        )

    @classmethod
    def _normalize_agent_message_item(cls, item: "Mapping[str, object]") -> "_RewrittenAssistantMessageItem | None":
        text: Final = cls._agent_message_text(item)
        if not text:
            return None
        rewritten: Final[_RewrittenAssistantMessageItem] = {
            "type": "message",
            "role": "assistant",
            "content": ({"type": "output_text", "text": text},),
        }
        return rewritten

    @staticmethod
    def _normalize_context_compaction_item(item: "Mapping[str, object]") -> "_RewrittenCompactionItem | None":
        encrypted_content: Final = item.get("encrypted_content")
        if not isinstance(encrypted_content, str) or not encrypted_content:
            return None
        rewritten: Final[_RewrittenCompactionItem] = {"type": "compaction", "encrypted_content": encrypted_content}
        return rewritten

    @staticmethod
    def _normalize_local_shell_call_item(item: "Mapping[str, object]") -> "_RewrittenFunctionCallItem | None":
        call_id: Final = item.get("call_id")
        if not isinstance(call_id, str) or not call_id:
            return None
        action: Final = item.get("action")
        rewritten: Final[_RewrittenFunctionCallItem] = {
            "type": "function_call",
            "call_id": call_id,
            "name": "local_shell",
            "arguments": json.dumps(action) if isinstance(action, dict) else "{}",
        }
        return rewritten

    @classmethod
    def _normalize_codex_input_item(cls, item: object) -> "tuple[object, str | None]":
        """Returns (normalized item or None to drop it, original type when rewritten)."""
        if not isinstance(item, dict):
            return item, None
        item_type: Final = item.get("type")
        if item_type == _CODEX_AGENT_MESSAGE_INPUT_ITEM_TYPE:
            return cls._normalize_agent_message_item(item), item_type
        if item_type == _CODEX_CONTEXT_COMPACTION_INPUT_ITEM_TYPE:
            return cls._normalize_context_compaction_item(item), item_type
        if item_type == _CODEX_LOCAL_SHELL_CALL_INPUT_ITEM_TYPE:
            return cls._normalize_local_shell_call_item(item), item_type
        return item, None

    @classmethod
    def _normalize_codex_input_items(
        cls,
        input: "str | ResponseInputParam",
    ) -> "str | ResponseInputParam":
        """Rewrite Codex history item types Mantle rejects with 400 "Invalid
        'input': value did not match any expected variant" into supported
        equivalents. `agent_message` (Codex multi-agent traffic; its
        encrypted_content slot carries the plaintext payload when the model
        never issued encrypted args) becomes an assistant message,
        `context_compaction` becomes the `compaction` spelling Mantle accepts,
        and `local_shell_call` becomes the function_call its recorded
        function_call_output already pairs with.
        """
        if not isinstance(input, list):
            return input
        normalized: Final = tuple(cls._normalize_codex_input_item(item) for item in input)
        rewritten_types: Final = sorted(frozenset(item_type for _, item_type in normalized if item_type is not None))
        if rewritten_types:
            verbose_logger.warning(
                "Bedrock Mantle Responses API: rewrote Codex input item type(s) %s that Mantle rejects.",
                rewritten_types,
            )
        kept: Final = [item for item, _ in normalized if item is not None]  # mutable-ok: ResponseInputParam is a list
        return kept  # pyright: ignore[reportReturnType]  # Codex passthrough items sit outside the OpenAI input union

    def map_openai_params(
        self,
        response_api_optional_params: ResponsesAPIOptionalRequestParams,
        model: str,
        drop_params: bool,
    ) -> dict:
        params: Final = self._handle_unsupported_service_tier(
            super().map_openai_params(
                response_api_optional_params=response_api_optional_params,
                model=model,
                drop_params=drop_params,
            ),
            drop_params=drop_params,
        )

        tools: Final = params.get("tools")
        if not tools:
            return params

        tools_list: Final = tools if isinstance(tools, list) else [tools]
        filtered: Final = self._filter_unsupported_tools(tools_list)
        if filtered:
            params["tools"] = filtered
        else:
            params.pop("tools", None)

        return params
