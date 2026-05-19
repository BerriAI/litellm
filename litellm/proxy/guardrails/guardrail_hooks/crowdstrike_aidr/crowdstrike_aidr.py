from collections.abc import Mapping, Sequence
import json
import os
from typing import TYPE_CHECKING, Annotated, Literal, Optional, Type, Union, cast
from pydantic import BaseModel, ConfigDict, Field
from typing_extensions import Any, override

from fastapi import HTTPException

from litellm._logging import verbose_proxy_logger
from litellm.integrations.custom_guardrail import (
    CustomGuardrail,
    log_guardrail_information,
)
from litellm.llms.custom_httpx.http_handler import (
    get_async_httpx_client,
    httpxSpecialProvider,
)
from litellm.proxy.common_utils.callback_utils import (
    add_guardrail_to_applied_guardrails_header,
)
from litellm.types.llms.openai import OpenAIChatCompletionToolParam
from litellm.types.utils import GenericGuardrailAPIInputs

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
    from litellm.types.proxy.guardrails.guardrail_hooks.base import GuardrailConfigModel


class CrowdStrikeAIDRGuardrailMissingSecrets(Exception):
    """Custom exception for missing CrowdStrike AIDR secrets."""

    pass


class _TextContentPart(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["text"] = "text"
    text: str


class _ImageUrl(BaseModel):
    url: str


class _ImageUrlContentPart(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["image_url"] = "image_url"
    image_url: _ImageUrl


_ContentPart = Annotated[
    Union[_TextContentPart, _ImageUrlContentPart], Field(discriminator="type")
]


class _Message(BaseModel):
    role: str
    content: Optional[Union[str, list[_ContentPart]]] = None


class _GuardInput(BaseModel):
    messages: list[_Message]
    tools: Optional[Sequence[OpenAIChatCompletionToolParam]] = None


def _normalize_content(raw: object) -> str | list[_ContentPart] | None:
    if raw is None:
        return None
    if isinstance(raw, str):
        return raw
    if not isinstance(raw, list):
        return json.dumps(raw)
    parts: list[_ContentPart] = []
    for block in raw:
        if not isinstance(block, dict):
            parts.append(_TextContentPart(text=json.dumps(block)))
            continue

        t = block.get("type")
        if t == "text" and isinstance(block.get("text"), str):
            parts.append(_TextContentPart(text=cast(str, block["text"])))
        elif t == "image_url":
            iu = block.get("image_url")
            url = iu if isinstance(iu, str) else str((iu or {}).get("url", ""))
            parts.append(_ImageUrlContentPart(image_url=_ImageUrl(url=url)))

        # Any other types are not recognized by the CrowdStrike AIDR API.

    return parts


def _extract_text_from_content(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            item.get("text", "")
            for item in content
            if isinstance(item, dict) and item.get("type") == "text"
        ]
        return "\n".join(parts)
    return ""


class CrowdStrikeAIDRHandler(CustomGuardrail):
    """
    CrowdStrike AIDR AI Guardrail handler to interact with the CrowdStrike AIDR
    AI Guard service.
    """

    def __init__(
        self,
        guardrail_name: str,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        **kwargs,
    ):
        """
        Initializes the CrowdStrikeAIDRHandler.

        Args:
            guardrail_name (str): The name of the guardrail instance.
            api_key (Optional[str]): The CrowdStrike AIDR API key. Reads from CS_AIDR_TOKEN env var if None.
            api_base (Optional[str]): The CrowdStrike AIDR API base URL. Reads from CS_AIDR_BASE_URL env var if None.
            **kwargs: Additional arguments passed to the CustomGuardrail base class.
        """
        self.async_handler = get_async_httpx_client(
            llm_provider=httpxSpecialProvider.GuardrailCallback
        )

        self.api_key = api_key or os.environ.get("CS_AIDR_TOKEN")
        if not self.api_key:
            raise CrowdStrikeAIDRGuardrailMissingSecrets(
                "CrowdStrike AIDR API Key not found. Set CS_AIDR_TOKEN environment variable or pass it in litellm_params."
            )

        self.api_base = api_base or os.environ.get("CS_AIDR_BASE_URL")
        if not self.api_base:
            raise CrowdStrikeAIDRGuardrailMissingSecrets(
                "CrowdStrike AIDR API base URL is required. Set CS_AIDR_BASE_URL environment variable or pass it in litellm_params."
            )

        # Pass relevant kwargs to the parent class
        super().__init__(guardrail_name=guardrail_name, **kwargs)
        verbose_proxy_logger.debug(
            f"Initialized CrowdStrike AIDR Guardrail: name={guardrail_name}, api_base={self.api_base}"
        )

    async def _call_crowdstrike_aidr_guard(
        self, payload: dict[str, Any], hook_name: str
    ) -> dict[str, Any]:
        """
        Makes the API call to the CrowdStrike AIDR AI Guard endpoint.
        The function itself will raise an error if a response should be blocked,
        but otherwise will return a list of redacted messages that the caller
        should act on.

        Args:
            payload (dict): The request payload.
            hook_name (str): Name of the hook calling this function (for logging).

        Raises:
            HTTPException: If the CrowdStrike AIDR API returns a 'blocked: true' response.
            Exception: For other API call failures.

        Returns:
            dict: The API response body
        """
        endpoint = f"{self.api_base}/v1/guard_chat_completions"

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        verbose_proxy_logger.debug(
            f"CrowdStrike AIDR Guardrail ({hook_name}): Calling endpoint {endpoint} with payload: {payload}"
        )

        response = await self.async_handler.post(
            url=endpoint, json=payload, headers=headers
        )
        response.raise_for_status()

        result: dict[str, Any] = response.json()

        if result.get("result", {}).get("blocked"):
            verbose_proxy_logger.warning(
                f"CrowdStrike AIDR Guardrail ({hook_name}): Request blocked. Response: {result}"
            )
            raise HTTPException(
                status_code=400,  # Bad Request, indicating violation
                detail={
                    "error": "Violated CrowdStrike AIDR guardrail policy",
                    "guardrail_name": self.guardrail_name,
                },
            )
        verbose_proxy_logger.debug(
            f"CrowdStrike AIDR Guardrail ({hook_name}): Request passed. Response: {result.get('result', {}).get('detectors')}"
        )

        return result

    def _build_guard_input_for_request(
        self, inputs: GenericGuardrailAPIInputs
    ) -> Optional[_GuardInput]:
        guard_input = _GuardInput(messages=[], tools=[])
        structured_messages = inputs.get("structured_messages")
        texts = inputs.get("texts", [])
        tools = inputs.get("tools")

        if structured_messages:
            for message in structured_messages:
                content = _normalize_content(message.get("content"))
                if content is None or len(content) == 0:
                    content = ""
                guard_input.messages.append(
                    _Message(role=message["role"], content=content)
                )
        elif texts:
            guard_input.messages = [
                _Message(role="user", content=text) for text in texts
            ]
        else:
            verbose_proxy_logger.warning(
                "CrowdStrike AIDR Guardrail: No messages or texts provided for input request"
            )
            return None

        if tools:
            guard_input.tools = tools

        return guard_input

    def _build_guard_input_for_response(
        self, inputs: GenericGuardrailAPIInputs, request_data: Mapping[str, Any]
    ) -> Optional[_GuardInput]:
        output_texts: list[str] = inputs.get("texts", [])
        if len(output_texts) == 0:
            verbose_proxy_logger.warning(
                "CrowdStrike AIDR Guardrail: No text in output response."
            )
            return None

        input_messages = request_data.get("messages", [])

        return _GuardInput(
            messages=[
                _Message(role=role, content=content)
                for (role, content) in (
                    (message["role"], _normalize_content(message.get("content")))
                    for message in input_messages
                )
                if content is not None and len(content) > 0
            ]
            + [_Message(role="assistant", content=text) for text in output_texts]
        )

    def _extract_transformed_texts(
        self,
        guard_output: Mapping[str, Any],
        num_assistant_messages: int,
    ) -> list[str]:
        transformed_messages = guard_output.get("messages", [])
        tail = (
            transformed_messages[-num_assistant_messages:]
            if num_assistant_messages > 0
            else []
        )
        return [
            (
                _extract_text_from_content(msg.get("content"))
                if isinstance(msg, dict)
                else ""
            )
            for msg in tail
        ]

    @log_guardrail_information
    @override
    async def apply_guardrail(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict,
        input_type: Literal["request", "response"],
        logging_obj: Optional["LiteLLMLoggingObj"] = None,
    ) -> GenericGuardrailAPIInputs:
        verbose_proxy_logger.debug(
            f"CrowdStrike AIDR Guardrail: Applying guardrail to {input_type}"
        )

        # Extract inputs
        texts = inputs.get("texts", [])
        structured_messages = inputs.get("structured_messages")
        tools = inputs.get("tools")
        tool_calls = inputs.get("tool_calls")

        # Build guard_input based on input_type
        if input_type == "request":
            guard_input = self._build_guard_input_for_request(inputs)
            if guard_input is None:
                return inputs
            event_type = "input"
            hook_name = "apply_guardrail (request)"
        else:
            guard_input = self._build_guard_input_for_response(inputs, request_data)
            if guard_input is None:
                return inputs
            event_type = "output"
            hook_name = "apply_guardrail (response)"

        ai_guard_payload = {
            "guard_input": guard_input.model_dump(mode="json"),
            "event_type": event_type,
        }

        ai_guard_response = await self._call_crowdstrike_aidr_guard(
            ai_guard_payload, hook_name
        )

        if "body" in request_data or "messages" in request_data:
            add_guardrail_to_applied_guardrails_header(
                request_data=request_data, guardrail_name=self.guardrail_name
            )

        result = ai_guard_response.get("result", {})
        if not result.get("transformed"):
            return inputs

        guard_output = result.get("guard_output", {})

        if input_type == "request":
            # For requests, all messages were in the guard_input. Extract texts
            # for every message in guard_output.
            all_messages = guard_output.get("messages", [])
            transformed_texts = [
                _extract_text_from_content(
                    msg.get("content") if isinstance(msg, dict) else ""
                )
                for msg in all_messages
            ]
        else:
            # For responses, guard_input contained history + assistant messages
            # appended at the end. Extract only the assistant tail.
            num_assistant = len(texts)
            transformed_texts = self._extract_transformed_texts(
                guard_output, num_assistant
            )

        result_inputs: GenericGuardrailAPIInputs = {"texts": transformed_texts}
        if tools:
            result_inputs["tools"] = tools
        if tool_calls:
            result_inputs["tool_calls"] = tool_calls
        if structured_messages:
            result_inputs["structured_messages"] = structured_messages

        return result_inputs

    @override
    @staticmethod
    def get_config_model() -> Optional[Type["GuardrailConfigModel"]]:
        from litellm.types.proxy.guardrails.guardrail_hooks.crowdstrike_aidr import (
            CrowdStrikeAIDRGuardrailConfigModel,
        )

        return CrowdStrikeAIDRGuardrailConfigModel
