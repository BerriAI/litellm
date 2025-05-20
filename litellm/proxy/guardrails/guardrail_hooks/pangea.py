# litellm/proxy/guardrails/guardrail_hooks/pangea.py
import os
from typing import Any, Optional, Protocol

from fastapi import HTTPException

from litellm._logging import verbose_proxy_logger
from litellm.caching.dual_cache import DualCache
from litellm.integrations.custom_guardrail import (
    CustomGuardrail,
    log_guardrail_information,
)

from litellm.llms.custom_httpx.http_handler import (
    get_async_httpx_client,
    httpxSpecialProvider,
)
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.common_utils.callback_utils import (
    add_guardrail_to_applied_guardrails_header,
)
from litellm.types.guardrails import GuardrailEventHooks
from litellm.types.utils import LLMResponseTypes, ModelResponse, TextCompletionResponse


class PangeaGuardrailMissingSecrets(Exception):
    """Custom exception for missing Pangea secrets."""

    pass


class _Transformer(Protocol):
    def get_messages(self) -> list[dict]:
        ...

    def update_original_body(self, prompt_messages: list[dict]) -> Any:
        ...


class _TextCompletionRequest:
    def __init__(self, body):
        self.body = body

    def get_messages(self) -> list[dict]:
        return [{"role": "user", "content": self.body["prompt"]}]

    # This mutates the original dict, but we'll still return it anyways
    def update_original_body(self, prompt_messages: list[dict]) -> Any:
        assert(len(prompt_messages) == 1)
        self.body["prompt"] = prompt_messages[0]["content"]
        return self.body


class _TextCompletionResponse:
    def __init__(self, body):
        self.body = body

    def get_messages(self) -> list[dict]:
        messages = []
        for choice in self.body["choices"]:
            messages.append({"role": "assistant", "content": choice["text"]})

        return messages

    def update_original_body(self, prompt_messages: list[dict]) -> Any:
        assert(len(prompt_messages) == len(self.body["choices"]))

        for choice, prompt_message in zip(self.body["choices"], prompt_messages):
            choice["text"] = prompt_message["content"]

        return self.body


class _ChatCompletionRequest:
    def __init__(self, body):
        self.body = body

    def get_messages(self) -> list[dict]:
        messages = []

        for message in self.body["messages"]:
            role = message["role"]
            content = message["content"]
            if isinstance(content, str):
                messages.append({"role": role, "content": content})
            if isinstance(content, list):
                for content_part in content:
                    if content_part["type"] == "text":
                        messages.append({"role": role, "content": content_part["text"]})

        return messages

    def update_original_body(self, prompt_messages: list[dict]) -> Any:
        count = 0

        for message in self.body["messages"]:
            content = message["content"]
            if isinstance(content, str):
                message["content"] = prompt_messages[count]["content"]
                count += 1
            if isinstance(content, list):
                for content_part in content:
                    if content_part["type"] == "text":
                        content_part["text"] = prompt_messages[count]["content"]
                        count += 1

        assert(len(prompt_messages) == count)
        return self.body


class _ChatCompletionResponse:
    def __init__(self, body):
        self.body = body

    def get_messages(self) -> list[dict]:
        messages = []

        for choice in self.body["choices"]:
            messages.append({"role": choice["message"]["role"], "content": choice["message"]["content"]})

        return messages

    def update_original_body(self, prompt_messages: list[dict]) -> Any:
        assert(len(prompt_messages) == len(self.body["choices"]))

        for choice, prompt_message in zip(self.body["choices"], prompt_messages):
            choice["message"]["content"] = prompt_message["content"]

        return self.body


def _get_transformer_for_request(body, call_type) -> Optional[_Transformer]:
    match call_type:
        case "text_completion" | "atext_completion":
            return _TextCompletionRequest(body)
        case "completion" | "acompletion":
            return _ChatCompletionRequest(body)

    return None


def _get_transformer_for_response(body) -> Optional[_Transformer]:
    match body:
        case TextCompletionResponse():
            return _TextCompletionResponse(body)
        case ModelResponse():
            return _ChatCompletionResponse(body)

    return None



class PangeaHandler(CustomGuardrail):
    """
    Pangea AI Guardrail handler to interact with the Pangea AI Guard service.

    This class implements the necessary hooks to call the Pangea AI Guard API
    for input and output scanning based on the configured recipe.
    """

    def __init__(
        self,
        guardrail_name: str,
        pangea_input_recipe: Optional[str] = None,
        pangea_output_recipe: Optional[str] = None,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        **kwargs,
    ):
        """
        Initializes the PangeaHandler.

        Args:
            guardrail_name (str): The name of the guardrail instance.
            pangea_recipe (str): The Pangea recipe key to use for scanning.
            api_key (Optional[str]): The Pangea API key. Reads from PANGEA_API_KEY env var if None.
            api_base (Optional[str]): The Pangea API base URL. Reads from PANGEA_API_BASE env var or uses default if None.
            **kwargs: Additional arguments passed to the CustomGuardrail base class.
        """
        self.async_handler = get_async_httpx_client(
            llm_provider=httpxSpecialProvider.GuardrailCallback
        )
        self.api_key = api_key or os.environ.get("PANGEA_API_KEY")
        if not self.api_key:
            raise PangeaGuardrailMissingSecrets(
                "Pangea API Key not found. Set PANGEA_API_KEY environment variable or pass it in litellm_params."
            )

        # Default Pangea base URL if not provided
        self.api_base = (
            api_base
            or os.environ.get("PANGEA_API_BASE")
            or "https://ai-guard.aws.us.pangea.cloud"
        )
        self.pangea_input_recipe = pangea_input_recipe
        self.pangea_output_recipe = pangea_output_recipe
        self.guardrail_endpoint = f"{self.api_base}/v1/text/guard"

        # Pass relevant kwargs to the parent class
        super().__init__(guardrail_name=guardrail_name, **kwargs)
        verbose_proxy_logger.info(
            f"Initialized Pangea Guardrail: name={guardrail_name}, recipe={pangea_input_recipe}, api_base={self.api_base}"
        )

    async def _call_pangea_guard(
        self, payload: dict, hook_name: str
    ) -> dict:
        """
        Makes the API call to the Pangea AI Guard endpoint.
        The function itself will raise an error in the case that a response
        should be blocked, but will return a list of redacted messages that the caller
        should act on.

        Args:
            payload (dict): The request payload.
            request_data (dict): Original request data (used for logging/headers).
            hook_name (str): Name of the hook calling this function (for logging).

        Raises:
            HTTPException: If the Pangea API returns a 'blocked: true' response.
            Exception: For other API call failures.

        Returns:
            list[dict]: The original response body
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        try:
            verbose_proxy_logger.debug(
                f"Pangea Guardrail ({hook_name}): Calling endpoint {self.guardrail_endpoint} with payload: {payload}"
            )
            response = await self.async_handler.post(
                url=self.guardrail_endpoint, json=payload, headers=headers
            )
            response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)

            result = response.json()
            verbose_proxy_logger.debug(
                f"Pangea Guardrail ({hook_name}): Received response: {result}"
            )

            # Check if the request was blocked
            if result.get("result", {}).get("blocked") is True:
                verbose_proxy_logger.warning(
                    f"Pangea Guardrail ({hook_name}): Request blocked. Response: {result}"
                )
                raise HTTPException(
                    status_code=400,  # Bad Request, indicating violation
                    detail={
                        "error": "Violated Pangea guardrail policy",
                        "guardrail_name": self.guardrail_name,
                        "pangea_response": result.get("result"),
                    },
                )
            else:
                verbose_proxy_logger.info(
                    f"Pangea Guardrail ({hook_name}): Request passed. Response: {result.get('result', {}).get('detectors')}"
                )

            return result

        except HTTPException as e:
            # Re-raise HTTPException if it's the one we raised for blocking
            raise e
        except Exception as e:
            verbose_proxy_logger.error(
                    f"Pangea Guardrail ({hook_name}): Error calling API: {e}. Response text: {getattr(e, 'response', None) and getattr(e.response, 'text', None)}"  # type: ignore
            )
            # Decide if you want to block by default on error, or allow through
            # Raising an exception here will block the request.
            # To allow through on error, you might just log and return.
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "Error communicating with Pangea Guardrail",
                    "guardrail_name": self.guardrail_name,
                    "exception": str(e),
                },
            ) from e

    @log_guardrail_information
    async def async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: DualCache,
        data: dict,
        call_type: str
    ):
        event_type = GuardrailEventHooks.pre_call
        if self.should_run_guardrail(data=data, event_type=event_type) is not True:
            verbose_proxy_logger.debug(
                f"Pangea Guardail (async_pre_call_hook): Guardrail is disabled {self.guardrail_name}."
            )
            return data


        transformer = _get_transformer_for_request(data, call_type)
        if not transformer:
            verbose_proxy_logger.warning(
                f"Pangea Guardrail (async_pre_call_hook): Skipping guardrail {self.guardrail_name}"
                f" because we cannot determine type of request: call_type '{call_type}'"
            )
            return

        messages = transformer.get_messages()
        if not messages:
            verbose_proxy_logger.warning(
                f"Pangea Guardrail (async_pre_call_hook): Skipping guardrail {self.guardrail_name}"
                " because messages is empty."
            )
            return

        ai_guard_payload = {
            "debug": False,  # Or make this configurable if needed
            "messages": messages,
        }
        if self.pangea_input_recipe:
            ai_guard_payload["recipe"] = self.pangea_input_recipe

        ai_guard_response = await self._call_pangea_guard(ai_guard_payload, "async_pre_call_hook")
        # Add guardrail name to header if passed
        add_guardrail_to_applied_guardrails_header(
            request_data=data, guardrail_name=self.guardrail_name
        )
        prompt_messages = ai_guard_response.get("result", {}).get("prompt_messages", [])

        try:
            return transformer.update_original_body(prompt_messages)
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "Failed to update original request body",
                    "guardrail_name": self.guardrail_name,
                    "exceptions": str(e),
                }
            ) from e

    @log_guardrail_information
    async def async_post_call_success_hook(
        self,
        data: dict,
        user_api_key_dict: UserAPIKeyAuth,
        # This union isn't actually correct -- it can get other response types depending on the API called
        response: LLMResponseTypes,
    ):
        """
        Guardrail hook run after a successful LLM call (scans output).

        Args:
            data (dict): The original request data.
            user_api_key_dict (UserAPIKeyAuth): User API key details.
            response (LLMResponseTypes): The response object from the LLM call.
        """
        event_type = GuardrailEventHooks.post_call
        if self.should_run_guardrail(data=data, event_type=event_type) is not True:
            verbose_proxy_logger.debug(
                f"Pangea Guardail (async_pre_call_hook): Guardrail is disabled {self.guardrail_name}."
            )
            return data

        transformer = _get_transformer_for_response(response)
        if not transformer:
            verbose_proxy_logger.warning(
                f"Pangea Guardrail (async_post_call_success_hook): Skipping guardrail {self.guardrail_name}"
                " because we cannot determine type of request"
            )
            return

        messages = transformer.get_messages()
        verbose_proxy_logger.warning(
            f"GOT MESSAGES: {messages}"
        )
        ai_guard_payload = {
            "debug": False,  # Or make this configurable if needed
            "messages": messages,
        }
        if self.pangea_output_recipe:
            ai_guard_payload["recipe"] = self.pangea_output_recipe

        ai_guard_response = await self._call_pangea_guard(ai_guard_payload, "post_call_success_hook")
        prompt_messages = ai_guard_response.get("result", {}).get("prompt_messages", [])

        try:
            return transformer.update_original_body(prompt_messages)
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "Failed to update original response body",
                    "guardrail_name": self.guardrail_name,
                    "exceptions": str(e),
                }
            ) from e
