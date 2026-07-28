# +-------------------------------------------------------------+
#
#           Use Akamai Firewall for AI Guardrails for your LLM calls
#                   https://www.akamai.com/products/firewall-for-ai
#
# +-------------------------------------------------------------+
import json
import os
import uuid
from itertools import chain
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncGenerator,
    Iterator,
    TypedDict,
    cast,
)

from fastapi import HTTPException

from litellm import DualCache
from litellm._logging import verbose_proxy_logger
from litellm.integrations.custom_guardrail import (
    CustomGuardrail,
    log_guardrail_information,
)
from litellm.llms.custom_httpx.http_handler import (
    get_async_httpx_client,
    httpxSpecialProvider,
)
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.guardrails._content_utils import iter_message_text
from litellm.types.guardrails import GuardrailEventHooks
from litellm.types.llms.openai import ResponsesAPIResponse
from litellm.types.utils import (
    CallTypesLiteral,
    EmbeddingResponse,
    ImageResponse,
    ModelResponse,
    ModelResponseStream,
)

if TYPE_CHECKING:
    from litellm.types.llms.anthropic import AnthropicMessagesRequest
    from litellm.types.proxy.guardrails.guardrail_hooks.base import GuardrailConfigModel


DEFAULT_API_BASE = "https://aisec.akamai.com"
BLOCKING_ACTIONS = frozenset({"deny", "block"})
ANTHROPIC_MESSAGES_CALL_TYPES = frozenset({"anthropic_messages", "aanthropic_messages"})


def _item_get(item: Any, key: str) -> Any:
    return item.get(key) if isinstance(item, dict) else getattr(item, key, None)


def _iter_function_fragments(function: Any) -> Iterator[str]:
    name = _item_get(function, "name")
    if isinstance(name, str) and name:
        yield name
    for key in ("arguments", "input"):
        value = _item_get(function, key)
        if isinstance(value, str) and value:
            yield value


def _iter_request_tool_call_text(data: dict) -> Iterator[str]:
    """Yield tool-call and legacy function_call names + arguments from a request body.

    ``iter_message_text`` only inspects message *content*, so tool-call
    arguments carried in prior assistant turns (chat ``tool_calls`` /
    ``function_call``) or in Responses-API ``input`` ``function_call`` items
    would otherwise reach the model without being sent to Akamai.
    """
    messages = data.get("messages")
    if isinstance(messages, list):
        for message in messages:
            if not isinstance(message, dict):
                continue
            for tool_call in message.get("tool_calls") or []:
                yield from _iter_function_fragments(_item_get(tool_call, "function"))
            yield from _iter_function_fragments(message.get("function_call"))

    input_value = data.get("input")
    if isinstance(input_value, list):
        for item in input_value:
            if _item_get(item, "type") == "function_call":
                yield from _iter_function_fragments(item)


def _iter_request_prompt_text(data: dict) -> Iterator[str]:
    """Yield the legacy Completions ``prompt`` and Responses-API ``instructions``.

    ``iter_message_text`` only walks ``messages`` and ``input``; the
    ``/completions`` ``prompt`` (string or list of strings) and the
    Responses-API top-level ``instructions`` are forwarded to the model but
    live in neither field, so without this they would reach the model
    uninspected.
    """
    for key in ("prompt", "instructions"):
        value = data.get(key)
        if isinstance(value, str):
            if value:
                yield value
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, str) and item:
                    yield item


def _iter_request_tool_definition_text(data: dict) -> Iterator[str]:
    """Yield names, descriptions and parameter schemas of request ``tools``.

    A tool *definition* (Chat-Completions ``tools[].function`` or the flattened
    Responses-API ``tools[]`` shape) is handed to the model as usable
    instructions, so an injected description or JSON-schema field reaches the
    model even though ``_iter_request_tool_call_text`` only inspects tool
    *calls*.
    """
    tools = data.get("tools")
    if not isinstance(tools, list):
        return
    for tool in tools:
        function = _item_get(tool, "function")
        definition = function if function is not None else tool
        name = _item_get(definition, "name")
        if isinstance(name, str) and name:
            yield name
        description = _item_get(definition, "description")
        if isinstance(description, str) and description:
            yield description
        parameters = _item_get(definition, "parameters")
        if isinstance(parameters, dict) and parameters:
            yield json.dumps(parameters, sort_keys=True)


def _translate_anthropic_to_openai_request(data: dict) -> dict:
    """Translate an Anthropic ``/v1/messages`` request into Chat-Completions shape.

    Hook-based guardrails receive the provider-native body, so the top-level
    ``system`` prompt, ``tool_use`` / ``tool_result`` content blocks and tool
    ``input_schema`` never match the OpenAI-shaped iterators. Reusing the shared
    Anthropic adapter lifts ``system`` into a system message, ``tool_use`` /
    ``tool_result`` into ``tool_calls`` / tool messages and ``input_schema`` into
    ``tools[].function.parameters`` so the standard extraction inspects them all.
    On a translation failure the raw body is returned so text content is still
    inspected rather than the whole request being dropped.
    """
    from litellm.llms.anthropic.experimental_pass_through.adapters.transformation import (
        LiteLLMAnthropicMessagesAdapter,
    )

    try:
        body = cast("AnthropicMessagesRequest", data.copy())  # cast-ok: dict passed to adapter TypedDict param
        openai_request, _ = LiteLLMAnthropicMessagesAdapter().translate_anthropic_to_openai(
            anthropic_message_request=body
        )
    except Exception as exc:
        verbose_proxy_logger.warning(
            "Akamai Firewall for AI: could not translate Anthropic /v1/messages request for inspection; "
            "falling back to raw extraction: %s",
            exc,
        )
        return data
    return dict(openai_request)


def _iter_responses_api_output_text(response: ResponsesAPIResponse) -> Iterator[str]:
    """Yield text and function-call arguments from a Responses API result.

    ``/v1/responses`` returns a ``ResponsesAPIResponse`` whose generated text
    lives in ``output[].content[].text`` and whose tool-call payloads live in
    ``output[].arguments`` / ``output[].input``; none of it is reachable via
    the Chat-Completions ``choices`` shape.
    """
    for item in response.output or []:
        content = _item_get(item, "content")
        if isinstance(content, list):
            for part in content:
                text = _item_get(part, "text")
                if isinstance(text, str) and text:
                    yield text
        yield from _iter_function_fragments(item)


class AkamaiRuleTriggered(TypedDict, total=False):
    action: str
    category: str
    details: dict[str, Any]
    message: str
    riskScore: int
    ruleId: str
    selector: str
    tags: list[str]
    version: str


class AkamaiDetectResponse(TypedDict, total=False):
    clientRequestId: str
    overallRiskScore: int
    rulesTriggered: list[AkamaiRuleTriggered]
    userApplicationId: str


class AkamaiFirewallForAIMissingSecrets(Exception):
    pass


class AkamaiFirewallForAIGuardrail(CustomGuardrail):
    @classmethod
    def get_supported_event_hooks(cls) -> list[GuardrailEventHooks]:
        return [
            GuardrailEventHooks.pre_call,
            GuardrailEventHooks.during_call,
            GuardrailEventHooks.post_call,
        ]

    def __init__(
        self,
        api_key: str | None = None,
        api_base: str | None = None,
        fai_configuration_id: str | None = None,
        user_application_id: str | None = None,
        **kwargs,
    ):
        kwargs.setdefault("supported_event_hooks", list(self.get_supported_event_hooks()))
        self.async_handler = get_async_httpx_client(llm_provider=httpxSpecialProvider.GuardrailCallback)

        self.api_key = api_key or os.environ.get("AKAMAI_FIREWALL_API_KEY")
        self.fai_configuration_id = fai_configuration_id or os.environ.get("AKAMAI_FIREWALL_CONFIGURATION_ID")
        self.user_application_id = user_application_id or os.environ.get("AKAMAI_FIREWALL_USER_APPLICATION_ID")

        missing = [
            name
            for name, value in (
                ("AKAMAI_FIREWALL_API_KEY", self.api_key),
                ("AKAMAI_FIREWALL_CONFIGURATION_ID", self.fai_configuration_id),
                ("AKAMAI_FIREWALL_USER_APPLICATION_ID", self.user_application_id),
            )
            if not value
        ]
        if missing:
            raise AkamaiFirewallForAIMissingSecrets(
                "Couldn't configure the Akamai Firewall for AI guardrail. Missing "
                + ", ".join(missing)
                + ". Set them in the environment or pass api_key, fai_configuration_id and "
                "user_application_id to the guardrail in the config file."
            )

        self.api_base = (api_base or os.environ.get("AKAMAI_FIREWALL_API_BASE") or DEFAULT_API_BASE).rstrip("/")
        super().__init__(**kwargs)

    @property
    def detect_url(self) -> str:
        return f"{self.api_base}/fai/v1/fai-configurations/{self.fai_configuration_id}/detect"

    @staticmethod
    def _input_text(data: dict, call_type: str) -> str:
        request = _translate_anthropic_to_openai_request(data) if call_type in ANTHROPIC_MESSAGES_CALL_TYPES else data
        fragments = chain(
            iter_message_text(request),
            _iter_request_tool_call_text(request),
            _iter_request_tool_definition_text(request),
            _iter_request_prompt_text(request),
        )
        return "\n".join(fragment for fragment in fragments if fragment)

    @staticmethod
    def _output_text(response: ModelResponse | Any) -> str:
        from litellm.litellm_core_utils.prompt_templates.common_utils import (
            get_content_from_model_response,
        )

        if isinstance(response, ModelResponse):
            return get_content_from_model_response(response)
        if isinstance(response, ResponsesAPIResponse):
            return "\n".join(_iter_responses_api_output_text(response))
        return ""

    async def _detect(
        self,
        client_request_id: str,
        llm_input: str | None = None,
        llm_output: str | None = None,
    ) -> None:
        payload: dict[str, str] = {
            "clientRequestId": client_request_id,
            "userApplicationId": self.user_application_id or "",
        }
        if llm_input:
            payload["llmInput"] = llm_input
        if llm_output:
            payload["llmOutput"] = llm_output

        if "llmInput" not in payload and "llmOutput" not in payload:
            return

        response = await self.async_handler.post(
            self.detect_url,
            headers={
                "Fai-Api-Key": self.api_key or "",
                "accept": "application/json",
                "content-type": "application/json",
            },
            json=payload,
        )
        response.raise_for_status()
        self._handle_detection(response.json())

    def _handle_detection(self, result: AkamaiDetectResponse) -> None:
        rules_triggered = result.get("rulesTriggered") or []
        blocking_rules = [rule for rule in rules_triggered if str(rule.get("action", "")).lower() in BLOCKING_ACTIONS]
        if not blocking_rules:
            if rules_triggered:
                verbose_proxy_logger.info(
                    "Akamai Firewall for AI: non-blocking rules triggered: %s",
                    [rule.get("ruleId") for rule in rules_triggered],
                )
            return

        verbose_proxy_logger.warning(
            "Akamai Firewall for AI: blocked request. overallRiskScore=%s rules=%s",
            result.get("overallRiskScore"),
            [rule.get("ruleId") for rule in blocking_rules],
        )
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Blocked by Akamai Firewall for AI",
                "overallRiskScore": result.get("overallRiskScore"),
                "rulesTriggered": [
                    {
                        "ruleId": rule.get("ruleId"),
                        "category": rule.get("category"),
                        "message": rule.get("message"),
                        "riskScore": rule.get("riskScore"),
                        "selector": rule.get("selector"),
                    }
                    for rule in blocking_rules
                ],
            },
        )

    @staticmethod
    def _client_request_id(data: dict) -> str:
        return str(data.get("litellm_call_id") or uuid.uuid4())

    @log_guardrail_information
    async def async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: DualCache,
        data: dict,
        call_type: CallTypesLiteral,
    ) -> Exception | str | dict | None:
        if self.should_run_guardrail(data=data, event_type=GuardrailEventHooks.pre_call) is not True:
            return data
        await self._detect(
            client_request_id=self._client_request_id(data),
            llm_input=self._input_text(data, call_type),
        )
        return data

    @log_guardrail_information
    async def async_moderation_hook(
        self,
        data: dict,
        user_api_key_dict: UserAPIKeyAuth,
        call_type: CallTypesLiteral,
    ) -> Exception | str | dict | None:
        if self.should_run_guardrail(data=data, event_type=GuardrailEventHooks.during_call) is not True:
            return data
        await self._detect(
            client_request_id=self._client_request_id(data),
            llm_input=self._input_text(data, call_type),
        )
        return data

    @log_guardrail_information
    async def async_post_call_success_hook(
        self,
        data: dict,
        user_api_key_dict: UserAPIKeyAuth,
        response: Any | ModelResponse | EmbeddingResponse | ImageResponse,
    ) -> Any:
        if self.should_run_guardrail(data=data, event_type=GuardrailEventHooks.post_call) is not True:
            return response
        await self._detect(client_request_id=self._client_request_id(data), llm_output=self._output_text(response))
        return response

    @classmethod
    def _streaming_output_text(cls, chunks: list) -> str:
        """Extract inspectable output text from a fully buffered stream.

        Chat streams (``ModelResponse`` / ``ModelResponseStream`` chunks) are
        assembled with ``stream_chunk_builder``. Responses-API streams instead
        emit events, the terminal one of which carries the complete
        ``ResponsesAPIResponse``; reuse ``_output_text`` on it so streamed
        Responses output and tool calls are inspected as well.
        """
        if isinstance(chunks[0], (ModelResponse, ModelResponseStream)):
            from litellm.main import stream_chunk_builder

            assembled = stream_chunk_builder(chunks=chunks)
            return cls._output_text(assembled) if isinstance(assembled, ModelResponse) else ""

        for chunk in reversed(chunks):
            candidate = _item_get(chunk, "response")
            if isinstance(candidate, ResponsesAPIResponse):
                return cls._output_text(candidate)
        return ""

    async def async_post_call_streaming_iterator_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        response: Any,
        request_data: dict,
    ) -> AsyncGenerator[Any, None]:
        if self.should_run_guardrail(data=request_data, event_type=GuardrailEventHooks.post_call) is not True:
            async for chunk in response:
                yield chunk
            return

        chunks = [chunk async for chunk in response]
        if not chunks:
            return

        try:
            await self._detect(
                client_request_id=self._client_request_id(request_data),
                llm_output=self._streaming_output_text(chunks),
            )
        except HTTPException as exc:
            error_obj = dict(exc.detail) if isinstance(exc.detail, dict) else {"message": str(exc.detail)}
            error_obj["code"] = exc.status_code
            yield f"data: {json.dumps({'error': error_obj})}\n\n"
            return
        except Exception as exc:
            verbose_proxy_logger.exception("Akamai Firewall for AI: streaming output scan failed: %s", exc)
            error_obj = {
                "message": "Akamai Firewall for AI scan failed; response withheld",
                "type": "guardrail_scan_error",
                "code": 500,
                "guardrail": self.guardrail_name,
            }
            yield f"data: {json.dumps({'error': error_obj})}\n\n"
            return

        for chunk in chunks:
            yield chunk

    @staticmethod
    def get_config_model() -> type["GuardrailConfigModel"] | None:
        from litellm.types.proxy.guardrails.guardrail_hooks.akamai_firewall_for_ai import (
            AkamaiFirewallForAIGuardrailConfigModel,
        )

        return AkamaiFirewallForAIGuardrailConfigModel
