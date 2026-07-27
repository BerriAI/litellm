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
    from litellm.types.proxy.guardrails.guardrail_hooks.base import GuardrailConfigModel


DEFAULT_API_BASE = "https://aisec.akamai.com"
BLOCKING_ACTIONS = frozenset({"deny", "block"})


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
    def _input_text(data: dict) -> str:
        fragments = chain(iter_message_text(data), _iter_request_tool_call_text(data))
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
        await self._detect(client_request_id=self._client_request_id(data), llm_input=self._input_text(data))
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
        await self._detect(client_request_id=self._client_request_id(data), llm_input=self._input_text(data))
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
