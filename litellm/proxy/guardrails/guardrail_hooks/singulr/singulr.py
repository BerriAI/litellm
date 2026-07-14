"""
Singulr guardrail integration for LiteLLM.
Calls the Singulr Guard API to scan messages.
"""

import json
import os
from collections import defaultdict
from typing import Any, TypeAlias, Union, cast
from urllib.parse import urlparse

import httpx
import pydantic
from openai.types.chat import ChatCompletionMessageToolCall
from pydantic import BaseModel

from litellm._logging import verbose_proxy_logger
from litellm.exceptions import GuardrailRaisedException
from litellm.integrations.custom_guardrail import (
    CustomGuardrail,
    log_guardrail_information,
)
from litellm.litellm_core_utils.litellm_logging import (
    Logging as LiteLLMLoggingObj,
)
from litellm.litellm_core_utils.prompt_templates.common_utils import (
    convert_content_list_to_str,
)
from litellm.llms.custom_httpx.http_handler import (
    get_async_httpx_client,
    httpxSpecialProvider,
)
from litellm.types.guardrails import GuardrailEventHooks
from litellm.types.llms.openai import (
    ChatCompletionToolCallChunk,
    ChatCompletionToolParam,
    ChatCompletionToolParamFunctionChunk,
)
from litellm.types.proxy.guardrails.guardrail_hooks.base import (
    GuardrailConfigModel,
)
from litellm.types.proxy.guardrails.guardrail_hooks.singulr import (
    SingulrGuardrailPayload,
    SingulrGuardrailRequest,
    SingulrGuardrailResponse,
)
from litellm.types.utils import GenericGuardrailAPIInputs

_DEFAULT_API_BASE = "http://localhost:8003"
_GUARD_ENDPOINT = "/api/v1/ai-gateway/litellm"
_DEFAULT_TIMEOUT = 30.0

_SingulrToolCall: TypeAlias = Union[ChatCompletionToolCallChunk, ChatCompletionMessageToolCall]


class _LegacyFunctionDefinition(BaseModel):
    """Validates a request_data["functions"][i] entry (the deprecated
    top-level functions[] field) before it is re-shaped into a
    ChatCompletionToolParam, instead of casting an untyped dict."""

    name: str
    description: str | None = None
    parameters: dict | None = None


class SingulrGuardrail(CustomGuardrail):
    def __init__(
        self,
        singulr_api_key: str | None = None,
        singulr_api_base: str | None = None,
        singulr_application_id: str | None = None,
        singulr_guardrail_id: str | None = None,
        block_on_error: bool | None = None,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> None:
        self.singulr_api_key = singulr_api_key or os.environ.get("SINGULR_API_KEY")
        self.singulr_api_base = (singulr_api_base or os.environ.get("SINGULR_API_BASE") or _DEFAULT_API_BASE).rstrip(
            "/"
        )
        parsed = urlparse(self.singulr_api_base)
        if parsed.scheme == "http" and parsed.hostname not in (
            "localhost",
            "127.0.0.1",
        ):
            raise ValueError(
                f"Singulr: api_base {self.singulr_api_base} uses plain HTTP for a "
                "non-local endpoint. Guardrail payloads contain the API token, full "
                "conversation content, and the guardrail decision, so this endpoint "
                "must use HTTPS."
            )

        self.singulr_application_id = singulr_application_id or os.environ.get("SINGULR_ENFORCEMENT_ENTITY_ID")
        self.singulr_guardrail_id = singulr_guardrail_id or os.environ.get("SINGULR_GUARDRAIL_ID")

        if block_on_error is None:
            env = os.environ.get("SINGULR_BLOCK_ON_ERROR", "true")
            self.block_on_error = env.lower() in ("true", "1", "yes")
        else:
            self.block_on_error = block_on_error

        self.timeout = _DEFAULT_TIMEOUT if timeout is None else timeout

        self.async_handler = get_async_httpx_client(
            llm_provider=httpxSpecialProvider.GuardrailCallback,
        )

        if "supported_event_hooks" not in kwargs:
            kwargs["supported_event_hooks"] = [
                GuardrailEventHooks.pre_call,
                GuardrailEventHooks.post_call,
            ]

        super().__init__(**kwargs)

    @staticmethod
    def get_config_model() -> type["GuardrailConfigModel"] | None:
        from litellm.types.proxy.guardrails.guardrail_hooks.singulr import (
            SingulrGuardrailConfigModel,
        )

        return SingulrGuardrailConfigModel

    @staticmethod
    def _extract_texts_by_role(
        inputs: GenericGuardrailAPIInputs,
    ) -> dict[str, list[str]]:
        structured_messages = inputs.get("structured_messages")
        if not structured_messages:
            return {}

        result: dict[str, list[str]] = {}

        for message in structured_messages:
            role = message.get("role")
            if not role:
                continue

            text = convert_content_list_to_str(message=message)
            if not text:
                continue

            result.setdefault(role, []).append(text)

        return result

    @staticmethod
    def _legacy_functions_as_tools(
        request_data: dict[str, Any],
    ) -> list[ChatCompletionToolParam]:
        functions = request_data.get("functions")
        if not isinstance(functions, list):
            return []

        tools: list[ChatCompletionToolParam] = []
        for fn in functions:
            try:
                validated = _LegacyFunctionDefinition.model_validate(fn)
            except pydantic.ValidationError:
                continue

            function: ChatCompletionToolParamFunctionChunk = {"name": validated.name}
            if validated.description is not None:
                function["description"] = validated.description
            if validated.parameters is not None:
                function["parameters"] = validated.parameters

            tools.append({"type": "function", "function": function})
        return tools

    @staticmethod
    def _response_format_schema_prompt(request_data: dict[str, Any]) -> str | None:
        response_format = request_data.get("response_format")
        if not isinstance(response_format, dict):
            return None

        json_schema = response_format.get("json_schema")
        if not isinstance(json_schema, dict):
            return None

        return json.dumps(json_schema)

    @staticmethod
    def reconstruct_tool_calls(
        tool_call_chunks: list[ChatCompletionToolCallChunk],
    ) -> list[ChatCompletionMessageToolCall]:

        aggregated: dict[int, dict[str, Any]] = defaultdict(
            lambda: {
                "id": None,
                "type": "function",
                "function": {
                    "name": "",
                    "arguments": "",
                },
            }
        )

        for chunk in tool_call_chunks:
            index = chunk.get("index", 0)
            current = aggregated[index]

            if chunk.get("id"):
                current["id"] = chunk["id"]

            if chunk.get("type"):
                current["type"] = chunk["type"]

            function = chunk.get("function") or {}

            name = function.get("name")
            if name:
                current["function"]["name"] = name

            arguments = function.get("arguments")
            if arguments:
                current["function"]["arguments"] += arguments

        return [ChatCompletionMessageToolCall(**tool_call) for _, tool_call in sorted(aggregated.items())]

    def _build_payload(
        self,
        request_data: dict[str, Any],
        inputs: GenericGuardrailAPIInputs,
        input_type: str,
    ) -> dict[str, Any]:
        messages = self._extract_texts_by_role(inputs=inputs)
        model_response = inputs.get("texts") or []

        # Guardrail playground sends only texts
        if input_type == "request" and not messages:
            messages["user"] = model_response

        schema_prompt = self._response_format_schema_prompt(request_data) if input_type == "request" else None
        if schema_prompt:
            messages.setdefault("system", []).append(schema_prompt)

        tools: list[ChatCompletionToolParam] = list(inputs.get("tools") or [])
        if input_type == "request":
            tools.extend(self._legacy_functions_as_tools(request_data))

        raw_tool_calls = inputs.get("tool_calls") or []

        tool_calls: list[_SingulrToolCall] = []
        if raw_tool_calls and isinstance(raw_tool_calls[0], ChatCompletionMessageToolCall):
            tool_calls = cast(list[_SingulrToolCall], raw_tool_calls)  # noqa: LIT006  # Type casting for tool call validation
        elif raw_tool_calls:
            tool_calls = cast(  # noqa: LIT006  # Complex nested cast for tool call reconstruction
                list[_SingulrToolCall],
                self.reconstruct_tool_calls(cast(list[ChatCompletionToolCallChunk], raw_tool_calls)),
            )

        singulr_request = SingulrGuardrailRequest(
            model=inputs.get("model") or request_data.get("model"),
            prompts=messages if input_type == "request" else None,
            completions=model_response if input_type == "response" else None,
            tools=tools,
            tool_calls=tool_calls,
        )

        payload = SingulrGuardrailPayload(
            request=singulr_request,
            input_type=input_type,
        )

        return payload.model_dump(exclude_none=True)

    def _build_headers(self) -> dict[str, str]:
        return dict(
            (header, value)
            for header, value in (
                ("Content-Type", "application/json"),
                ("X-Singulr-Gateway-Token", self.singulr_api_key),
                (
                    "X-Singulr-Enforcement-Entity-Id",
                    self.singulr_application_id or "",
                ),
                ("X-Singulr-Guardrail-Id", self.singulr_guardrail_id or ""),
            )
            if value
        )

    async def _call_api(self, payload: dict[str, Any]) -> SingulrGuardrailResponse | None:
        endpoint = f"{self.singulr_api_base}{_GUARD_ENDPOINT}"
        verbose_proxy_logger.debug("Singulr: %s", endpoint)

        try:
            response = await self.async_handler.post(
                url=endpoint,
                headers=self._build_headers(),
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
            result = SingulrGuardrailResponse.model_validate(response.json())
            verbose_proxy_logger.debug("Singulr: result=%s", result)
            return result

        except httpx.HTTPStatusError as exc:
            verbose_proxy_logger.error(
                "Singulr API returned HTTP %s: %s",
                exc.response.status_code,
                str(exc),
            )
            if self.block_on_error:
                raise GuardrailRaisedException(
                    guardrail_name=self.guardrail_name,
                    message=(f"Singulr API returned HTTP {exc.response.status_code}: {exc.response.text}"),
                ) from exc
            return None

        except httpx.TransportError as exc:
            verbose_proxy_logger.error("Singulr API unreachable: %s", str(exc))
            if self.block_on_error:
                raise GuardrailRaisedException(
                    guardrail_name=self.guardrail_name,
                    message=f"Singulr API unreachable (block_on_error=True): {exc}",
                ) from exc
            return None

        except (ValueError, pydantic.ValidationError) as exc:
            verbose_proxy_logger.error("Singulr API returned an invalid response: %s", str(exc))
            if self.block_on_error:
                raise GuardrailRaisedException(
                    guardrail_name=self.guardrail_name,
                    message=f"Singulr API returned an invalid response: {exc}",
                ) from exc
            return None

    @log_guardrail_information
    async def apply_guardrail(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict,
        input_type: str,
        logging_obj: "LiteLLMLoggingObj | None" = None,
    ) -> GenericGuardrailAPIInputs:
        payload = self._build_payload(request_data, inputs, input_type)
        # verbose_proxy_logger.debug("Singulr: payload=%s", payload)
        if not payload:
            return inputs

        result = await self._call_api(payload)
        if result is None:
            return inputs

        verbose_proxy_logger.debug(
            "Singulr: should_block=%s blocking_due_to=%s",
            result.should_block,
            result.blocking_due_to,
        )

        if result.should_block:
            raise GuardrailRaisedException(
                guardrail_name=self.guardrail_name,
                message=f"Blocked by Singulr: {result.blocking_due_to or 'unknown'}",
            )

        return inputs
