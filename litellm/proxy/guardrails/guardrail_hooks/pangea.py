# litellm/proxy/guardrails/guardrail_hooks/pangea.py
import os
from typing import Any, List, Literal, Optional, Union

from fastapi import HTTPException

from litellm._logging import verbose_proxy_logger
from litellm.integrations.custom_guardrail import (
    CustomGuardrail,
    log_guardrail_information,
)
from litellm.litellm_core_utils.logging_utils import (
    convert_litellm_response_object_to_str,
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
from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import ModelResponse


class PangeaGuardrailMissingSecrets(Exception):
    """Custom exception for missing Pangea secrets."""

    pass


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

    def _prepare_payload(
        self,
        messages: Optional[List[AllMessageValues]] = None,
        text_input: Optional[str] = None,
        request_data: Optional[dict] = None,
        recipe: Optional[str] = None,
    ) -> dict:
        """
        Prepares the payload for the Pangea AI Guard API request.

        Args:
            messages (Optional[List[AllMessageValues]]): List of messages for structured input.
            text_input (Optional[str]): Plain text input/output.
            request_data (Optional[dict]): Original request data (used for overrides).

        Returns:
            dict: The payload dictionary for the API request.
        """
        payload: dict[str, Any] = {
            "debug": False,  # Or make this configurable if needed
        }

        if recipe:
            payload["recipe"] = recipe

        if messages:
            # Ensure messages are in the format Pangea expects (list of dicts with 'role' and 'content')
            payload["messages"] = [
                {"role": msg.get("role"), "content": msg.get("content")}
                for msg in messages
                if msg.get("role") and msg.get("content")
            ]
        elif text_input:
            payload["text"] = text_input
        else:
            raise ValueError("Either messages or text_input must be provided.")

        # Add overrides if present in request metadata
        if (
            request_data
            and isinstance(request_data.get("metadata"), dict)
            and isinstance(
                request_data["metadata"].get("pangea_overrides"), dict
            )
        ):
            payload["overrides"] = request_data["metadata"]["pangea_overrides"]
            verbose_proxy_logger.debug(
                f"Pangea Guardrail: Applying overrides: {payload['overrides']}"
            )

        return payload

    async def _call_pangea_guard(
        self, payload: dict, request_data: dict, hook_name: str
    ) -> None:
        """
        Makes the API call to the Pangea AI Guard endpoint.

        Args:
            payload (dict): The request payload.
            request_data (dict): Original request data (used for logging/headers).
            hook_name (str): Name of the hook calling this function (for logging).

        Raises:
            HTTPException: If the Pangea API returns a 'blocked: true' response.
            Exception: For other API call failures.
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
                # Add guardrail name to header if passed
                add_guardrail_to_applied_guardrails_header(
                    request_data=request_data, guardrail_name=self.guardrail_name
                )

        except HTTPException as e:
            # Re-raise HTTPException if it's the one we raised for blocking
            raise e
        except Exception as e:
            verbose_proxy_logger.error(
                f"Pangea Guardrail ({hook_name}): Error calling API: {e}. Response text: {getattr(e, 'response', None) and getattr(e.response, 'text', None)}"
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
    async def async_moderation_hook(
        self,
        data: dict,
        user_api_key_dict: UserAPIKeyAuth,
        call_type: Literal[
            "completion",
            "embeddings",
            "image_generation",
            "moderation",
            "audio_transcription",
            "responses",
        ],
    ):
        """
        Guardrail hook run during the LLM call (scans input).

        Args:
            data (dict): The request data containing messages or input text.
            user_api_key_dict (UserAPIKeyAuth): User API key details.
            call_type (Literal): The type of the call.
        """
        event_type: GuardrailEventHooks = GuardrailEventHooks.during_call
        if not self.should_run_guardrail(data=data, event_type=event_type):
            verbose_proxy_logger.debug(
                f"Pangea Guardrail (moderation_hook): Skipping guardrail {self.guardrail_name} based on should_run_guardrail."
            )
            return

        messages: Optional[List[AllMessageValues]] = data.get("messages")
        text_input: Optional[str] = data.get(
            "input"
        )  # Assuming 'input' for non-chat models

        if not messages and not text_input:
            verbose_proxy_logger.warning(
                f"Pangea Guardrail (moderation_hook): No 'messages' or 'input' found in data for guardrail {self.guardrail_name}. Skipping."
            )
            return

        try:
            payload = self._prepare_payload(
                messages=messages, text_input=text_input, request_data=data, recipe=self.pangea_input_recipe
            )
            await self._call_pangea_guard(
                payload=payload, request_data=data, hook_name="moderation_hook"
            )
        except ValueError as ve:
            verbose_proxy_logger.error(
                f"Pangea Guardrail (moderation_hook): Error preparing payload: {ve}"
            )
            # Decide how to handle payload errors (e.g., block or allow)
            raise HTTPException(
                status_code=400,
                detail={"error": str(ve), "guardrail_name": self.guardrail_name},
            )

    @log_guardrail_information
    async def async_post_call_success_hook(
        self,
        data: dict,
        user_api_key_dict: UserAPIKeyAuth,
        response: Union[Any, ModelResponse],
    ):
        """
        Guardrail hook run after a successful LLM call (scans output).

        Args:
            data (dict): The original request data.
            user_api_key_dict (UserAPIKeyAuth): User API key details.
            response (Union[Any, ModelResponse]): The response object from the LLM call.
        """
        event_type: GuardrailEventHooks = GuardrailEventHooks.post_call
        if not self.should_run_guardrail(data=data, event_type=event_type):
            verbose_proxy_logger.debug(
                f"Pangea Guardrail (post_call_success_hook): Skipping guardrail {self.guardrail_name} based on should_run_guardrail."
            )
            return

        response_str: Optional[str] = convert_litellm_response_object_to_str(
            response
        )

        if response_str is None or not response_str.strip():
            verbose_proxy_logger.warning(
                f"Pangea Guardrail (post_call_success_hook): No valid response content found for guardrail {self.guardrail_name}. Skipping output scan."
            )
            return

        try:
            # Scan only the output text in the post-call hook
            payload = self._prepare_payload(
                text_input=response_str, request_data=data, recipe=self.pangea_output_recipe
            )
            await self._call_pangea_guard(
                payload=payload,
                request_data=data,
                hook_name="post_call_success_hook",
            )
        except ValueError as ve:
            verbose_proxy_logger.error(
                f"Pangea Guardrail (post_call_success_hook): Error preparing payload: {ve}"
            )
            # Block if payload prep fails for output
            raise HTTPException(
                status_code=500,  # Internal error as response couldn't be processed
                detail={
                    "error": f"Error preparing Pangea payload for response: {ve}",
                    "guardrail_name": self.guardrail_name,
                },
            )
