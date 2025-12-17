import os
from typing import TYPE_CHECKING, Literal, Optional, Type
from typing_extensions import Any, override

from fastapi import HTTPException

from litellm._logging import verbose_proxy_logger
from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.llms.custom_httpx.http_handler import (
    get_async_httpx_client,
    httpxSpecialProvider,
)
from litellm.proxy.common_utils.callback_utils import (
    add_guardrail_to_applied_guardrails_header,
)
from litellm.types.utils import GenericGuardrailAPIInputs

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
    from litellm.types.proxy.guardrails.guardrail_hooks.base import GuardrailConfigModel


class CrowdStrikeAIDRGuardrailMissingSecrets(Exception):
    """Custom exception for missing CrowdStrike AIDR secrets."""

    pass


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
    ) -> Optional[dict[str, Any]]:
        guard_input: dict[str, Any] = {}
        structured_messages = inputs.get("structured_messages")
        texts = inputs.get("texts", [])
        tools = inputs.get("tools")

        if structured_messages:
            guard_input["messages"] = structured_messages
        elif texts:
            guard_input["messages"] = [
                {"role": "user", "content": text} for text in texts
            ]
        else:
            verbose_proxy_logger.warning(
                "CrowdStrike AIDR Guardrail: No messages or texts provided for input request"
            )
            return None

        if tools:
            guard_input["tools"] = tools

        return guard_input

    def _build_guard_input_for_response(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict,
        logging_obj: Optional["LiteLLMLoggingObj"],
    ) -> Optional[dict[str, Any]]:
        guard_input: dict[str, Any] = {}
        response = request_data.get("response")
        if not response:
            verbose_proxy_logger.warning(
                "CrowdStrike AIDR Guardrail: No response object in request_data for output response"
            )
            return None

        input_messages = None
        if "body" in request_data:
            input_messages = request_data["body"].get("messages")
        if not input_messages:
            input_messages = request_data.get("messages")
        if not input_messages and logging_obj:
            try:
                if hasattr(logging_obj, "model_call_details"):
                    model_call_details = logging_obj.model_call_details
                    if isinstance(model_call_details, dict):
                        input_messages = model_call_details.get("messages")
            except Exception:
                pass

        guard_input["messages"] = input_messages if input_messages else []

        if tools := inputs.get("tools"):
            guard_input["tools"] = tools
        elif tools := request_data.get("body", {}).get("tools"):
            guard_input["tools"] = tools

        return guard_input

    def _extract_transformed_texts_from_messages(
        self,
        guard_output: dict[str, Any],
        structured_messages: Optional[list],
        texts: list[str],
    ) -> list[str]:
        transformed_texts: list[str] = []
        transformed_messages = guard_output.get("messages", [])

        if structured_messages and len(transformed_messages) == len(
            structured_messages
        ):
            for msg in transformed_messages:
                if isinstance(msg, dict):
                    content = msg.get("content")
                    if isinstance(content, str):
                        transformed_texts.append(content)
                    elif isinstance(content, list):
                        text_found = False
                        for item in content:
                            if isinstance(item, dict) and item.get("type") == "text":
                                transformed_texts.append(item.get("text", ""))
                                text_found = True
                                break
                        if not text_found:
                            transformed_texts.append("")
        else:
            for msg in transformed_messages:
                if isinstance(msg, dict):
                    content = msg.get("content")
                    if isinstance(content, str):
                        transformed_texts.append(content)
                    elif isinstance(content, list):
                        for item in content:
                            if isinstance(item, dict) and item.get("type") == "text":
                                transformed_texts.append(item.get("text", ""))
                                break

        while len(transformed_texts) < len(texts):
            transformed_texts.append(texts[len(transformed_texts)])
        return transformed_texts[: len(texts)]

    def _extract_transformed_texts_from_choices(
        self, guard_output: dict[str, Any], texts: list[str]
    ) -> list[str]:
        transformed_texts: list[str] = []
        transformed_choices = guard_output.get("choices", [])

        for choice in transformed_choices:
            if isinstance(choice, dict):
                message = choice.get("message", {})
                content = message.get("content")
                if isinstance(content, str):
                    transformed_texts.append(content)
                elif isinstance(content, list):
                    text_found = False
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "text":
                            transformed_texts.append(item.get("text", ""))
                            text_found = True
                            break
                    if not text_found:
                        transformed_texts.append("")
                else:
                    transformed_texts.append("")
            else:
                transformed_texts.append("")

        while len(transformed_texts) < len(texts):
            transformed_texts.append(texts[len(transformed_texts)])
        return transformed_texts[: len(texts)]

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
            guard_input = self._build_guard_input_for_response(
                inputs, request_data, logging_obj
            )
            if guard_input is None:
                return inputs
            event_type = "output"
            hook_name = "apply_guardrail (response)"

        ai_guard_payload = {
            "guard_input": guard_input,
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
            # Not transformed, return original inputs.
            return inputs

        guard_output = result.get("guard_output", {})

        transformed_texts = (
            self._extract_transformed_texts_from_messages(
                guard_output, structured_messages, texts
            )
            if input_type == "request"
            else self._extract_transformed_texts_from_choices(guard_output, texts)
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
