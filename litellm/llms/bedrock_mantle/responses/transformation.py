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

from collections.abc import Callable, Mapping, Sequence
from functools import partial
from types import MappingProxyType
from typing import Final, cast  # noqa: TID251  # map_openai_params returns the filtered params as a bare dict

import httpx
from pydantic import TypeAdapter

import litellm
from litellm._logging import verbose_logger
from litellm.litellm_core_utils import model_cost_table
from litellm.litellm_core_utils.core_helpers import drops_unsupported_params
from litellm.litellm_core_utils.prompt_templates.json_schema_utils import with_sanitized_schema_at
from litellm.litellm_core_utils.prompt_templates.prompt_cache_breakpoint_utils import (
    chat_completion_prompt_cache_breakpoint_for_wire,
)
from litellm.litellm_core_utils.url_utils import join_api_base
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.base_llm.responses.codex_compat import drop_unsupported_tools, normalize_codex_input_items
from litellm.llms.base_llm.responses.transformation import ReasoningSummaryLevel
from litellm.llms.bedrock.base_aws_llm import BaseAWSLLM
from litellm.llms.bedrock.common_utils import BedrockError
from litellm.llms.bedrock_mantle.common_utils import (
    MANTLE_HOST_RE,
    BedrockMantleAuthMixin,
    drop_schema_keywords_gemma_4_rejects,
    mantle_supports_all_json_schema_keywords,
    split_mantle_region_prefix,
    with_values_gemma_4_accepts,
)
from litellm.llms.openai.responses.transformation import OpenAIResponsesAPIConfig
from litellm.responses.additional_tools import HoistedAdditionalTools, hoist_additional_tools
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
_BEDROCK_MANTLE_SUPPORTED_RESPONSE_TOOL_TYPES: Final = frozenset(
    {"function", "mcp", "custom", "namespace", "tool_search", "web_search"}
)

_REQUEST_BODY: Final = TypeAdapter(dict[str, object])
_JSON_ARRAY: Final = TypeAdapter(list[object])
_JSON_OBJECT: Final = TypeAdapter(Mapping[str, object])
_BEDROCK_MANTLE_SUPPORTED_SERVICE_TIERS: Final = frozenset({"auto", "default"})
_BEDROCK_MANTLE_OPENAI_PATH_SUPPORTED_REASONING_SUMMARIES: Final = frozenset({"auto"})


def _prompt_cache_breakpoint_as_sent_or_none(marker: object) -> object:
    return marker if chat_completion_prompt_cache_breakpoint_for_wire(marker, drop_params=True) == marker else None


class BedrockMantleResponsesAPIConfig(BedrockMantleAuthMixin, OpenAIResponsesAPIConfig):
    def __init__(
        self,
        aws_signer: BaseAWSLLM | None = None,
        use_openai_path: bool = True,
        encrypted_agent_messages: bool = False,
    ):
        super().__init__()
        self._aws_signer = aws_signer or BaseAWSLLM()
        self.use_openai_path = use_openai_path
        self.encrypted_agent_messages = encrypted_agent_messages

    @property
    def custom_llm_provider(self) -> LlmProviders:
        return LlmProviders.BEDROCK_MANTLE

    def auto_reasoning_summary(self) -> ReasoningSummaryLevel:
        return "auto" if self.use_openai_path else super().auto_reasoning_summary()

    def supports_encrypted_agent_messages(self) -> bool:
        return self.use_openai_path and self.encrypted_agent_messages

    def get_error_class(
        self, error_message: str, status_code: int, headers: dict[str, object] | httpx.Headers
    ) -> BaseLLMException:
        return BedrockError(status_code=status_code, message=error_message, headers=headers)

    def get_complete_url(
        self,
        api_base: str | None,
        litellm_params: Mapping[str, object],
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
        return join_api_base(base, path)

    def validate_environment(
        self, headers: Mapping[str, object], model: str, litellm_params: GenericLiteLLMParams | None
    ) -> dict[str, object]:
        params: Final = litellm_params or GenericLiteLLMParams()
        return {**headers, **self._pinned_headers(params.api_key, params.aws_bedrock_project_id)}

    def sign_request(
        self,
        headers: Mapping[str, object],
        optional_params: Mapping[str, object],
        request_data: Mapping[str, object],
        api_base: str,
        api_key: str | None = None,
        model: str | None = None,
        stream: bool | None = None,
        fake_stream: bool | None = None,
    ) -> tuple[dict[str, object], bytes | None]:  # mutable-ok: the shared HTTP handler sends the returned headers dict
        pinned: Final = self._pinned_headers(api_key, optional_params.get("aws_bedrock_project_id"))
        pinned_names: Final = frozenset(name.lower() for name in pinned)
        return super().sign_request(
            headers={**{name: value for name, value in headers.items() if name.lower() not in pinned_names}, **pinned},
            optional_params=dict(optional_params),
            request_data=dict(request_data),
            api_base=api_base,
            api_key=api_key,
            model=model,
            stream=stream,
            fake_stream=fake_stream,
        )

    def _pinned_headers(self, api_key: str | None, project_id: object) -> Mapping[str, str]:
        bearer: Final = self._resolve_bearer_token(api_key)
        return MappingProxyType(
            {
                **({"Authorization": f"Bearer {bearer}"} if bearer else {}),
                **({"OpenAI-Project": project_id} if isinstance(project_id, str) and project_id else {}),
            }
        )

    def _prompt_cache_breakpoint_to_wire(
        self, litellm_params: GenericLiteLLMParams, reads_breakpoints: bool, known_support: bool | None
    ) -> Callable[[object], object]:
        if reads_breakpoints or known_support is True:
            return partial(chat_completion_prompt_cache_breakpoint_for_wire, drop_params=True)
        return _prompt_cache_breakpoint_as_sent_or_none

    def supports_native_file_search(self) -> bool:
        return False

    def supports_native_websocket(self) -> bool:
        return False

    @staticmethod
    def _filter_unsupported_tools(tools: "Sequence[object]") -> "list[object]":
        """Keep only tool types Mantle's Responses API accepts."""
        kept, dropped_types = drop_unsupported_tools(tools, _BEDROCK_MANTLE_SUPPORTED_RESPONSE_TOOL_TYPES)
        if dropped_types:
            verbose_logger.warning(
                "Bedrock Mantle Responses API: dropping unsupported tool type(s) %s (supported: %s).",
                list(dropped_types),
                sorted(_BEDROCK_MANTLE_SUPPORTED_RESPONSE_TOOL_TYPES),
            )
        return list(kept)

    @staticmethod
    def _handle_unsupported_service_tier(params: dict[str, object], drop_params: bool) -> dict[str, object]:
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

    def _handle_unsupported_reasoning_summary(
        self, params: dict[str, object], model: str, drop_params: bool
    ) -> dict[str, object]:
        reasoning: Final = params.get("reasoning")
        if not self.use_openai_path or not isinstance(reasoning, dict):
            return params
        summary: Final = reasoning.get("summary")
        if summary is None or (
            isinstance(summary, str) and summary in _BEDROCK_MANTLE_OPENAI_PATH_SUPPORTED_REASONING_SUMMARIES
        ):
            return params
        if not drop_params:
            raise litellm.utils.UnsupportedParamsError(
                status_code=400,
                message=(
                    f"bedrock_mantle does not support reasoning.summary={summary!r} for {model!r}; the Bedrock Mantle "
                    "OpenAI Responses path only accepts 'auto'. Set `drop_params: true` (litellm_settings or this "
                    'deployment\'s litellm_params) to have LiteLLM drop it, or set `model_reasoning_summary = "auto"` '
                    "in the client (Codex CLI: ~/.codex/config.toml)."
                ),
            )
        verbose_logger.warning(
            "Bedrock Mantle Responses API: dropping unsupported reasoning.summary %r (supported: %s).",
            summary,
            sorted(_BEDROCK_MANTLE_OPENAI_PATH_SUPPORTED_REASONING_SUMMARIES),
        )
        stripped: Final = {key: value for key, value in reasoning.items() if key != "summary"}
        return (
            {**params, "reasoning": stripped}
            if stripped
            else {key: value for key, value in params.items() if key != "reasoning"}
        )

    def transform_responses_api_request(
        self,
        model: str,
        input: "str | ResponseInputParam",
        response_api_optional_request_params: dict[str, object],
        litellm_params: GenericLiteLLMParams,
        headers: dict[str, str],
    ) -> dict[str, object]:
        params: Final = cast(  # cast-ok: the base signature leaves the params dict untyped
            "ResponsesAPIOptionalRequestParams", response_api_optional_request_params
        )
        hoisted: Final = hoist_additional_tools(
            input, params.get("tools"), drop_params=drops_unsupported_params(litellm_params.drop_params)
        )
        normalized_input, rewritten_types = normalize_codex_input_items(
            hoisted.input, keep_encrypted_agent_messages=self.supports_encrypted_agent_messages()
        )
        if rewritten_types:
            verbose_logger.warning(
                "Bedrock Mantle Responses API: rewrote Codex input item type(s) %s that Mantle rejects.",
                list(rewritten_types),
            )
        request_params: Final = (
            self._params_with_hoisted_tools(params, hoisted)
            if hoisted.hoisted
            else response_api_optional_request_params
        )
        request: Final = _REQUEST_BODY.validate_python(
            super().transform_responses_api_request(
                model=model,
                input=normalized_input,
                response_api_optional_request_params=request_params,
                litellm_params=litellm_params,
                headers=headers,
            )
        )
        return dict(self._with_schemas_model_accepts(request, model, litellm_params))

    def transform_extra_body(
        self,
        extra_body: Mapping[str, object],
        request: Mapping[str, object],
        model: str,
        litellm_params: GenericLiteLLMParams,
    ) -> Mapping[str, object]:
        return self._with_schemas_model_accepts(
            super().transform_extra_body(extra_body, request, model, litellm_params), model, litellm_params
        )

    @classmethod
    def _with_schemas_model_accepts(
        cls, params: Mapping[str, object], model: str, litellm_params: GenericLiteLLMParams
    ) -> Mapping[str, object]:
        region: Final = cls._resolve_region(
            {"aws_region_name": litellm_params.aws_region_name, "api_base": litellm_params.api_base}
        )
        if mantle_supports_all_json_schema_keywords(model, region, litellm_params, model_cost_table.published()):
            return params
        return with_values_gemma_4_accepts(params, model, cls._value_gemma_4_accepts)

    @classmethod
    def _value_gemma_4_accepts(cls, key: str, value: object) -> object:
        if key == "tools" and isinstance(value, list):
            return cls._sanitized_tools(_JSON_ARRAY.validate_python(value), drop_schema_keywords_gemma_4_rejects)
        if key == "text" and isinstance(value, dict):
            return with_sanitized_schema_at(
                _JSON_OBJECT.validate_python(value), ("format", "schema"), drop_schema_keywords_gemma_4_rejects
            )
        return value

    @classmethod
    def _params_with_hoisted_tools(
        cls, params: Mapping[str, object], hoisted: HoistedAdditionalTools
    ) -> dict[str, object]:
        supported_tools: Final = cls._filter_unsupported_tools(list(hoisted.tools))
        if supported_tools:
            return {**params, "tools": supported_tools}
        return {key: value for key, value in params.items() if key != "tools"}

    @staticmethod
    def _model_map_lookup_name(model: str) -> str:
        return model.split("/")[-1].removeprefix("openai.")

    def _model_map_row_keys(self, model: str) -> tuple[str, ...]:
        region, bare = split_mantle_region_prefix(model)
        regional: Final = (f"bedrock_mantle/{region}/{bare}",) if region else ()
        return (*regional, f"bedrock_mantle/{bare}")

    def map_openai_params(
        self,
        response_api_optional_params: ResponsesAPIOptionalRequestParams,
        model: str,
        drop_params: bool,
    ) -> dict[str, object]:
        params: Final = self._handle_unsupported_reasoning_summary(
            self._handle_unsupported_service_tier(
                super().map_openai_params(
                    response_api_optional_params=response_api_optional_params,
                    model=model,
                    drop_params=drop_params,
                ),
                drop_params=drop_params,
            ),
            model=model,
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
