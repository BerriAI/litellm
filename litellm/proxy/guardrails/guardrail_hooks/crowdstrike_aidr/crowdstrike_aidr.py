import os
from typing import TYPE_CHECKING, Any, Optional, Type

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
from litellm.types.utils import (
    Choices,
    LLMResponseTypes,
    ModelResponse,
    TextCompletionResponse,
)

if TYPE_CHECKING:
    from litellm.types.proxy.guardrails.guardrail_hooks.base import GuardrailConfigModel


class CrowdStrikeAIDRGuardrailMissingSecrets(Exception):
    """Custom exception for missing CrowdStrike AIDR secrets."""

    pass


class _TextCompletionRequest:
    def __init__(self, body):
        self.body = body

    def get_messages(self) -> list[dict]:
        return [{"role": "user", "content": self.body["prompt"]}]

    # This mutates the original dict, but we'll still return it anyways
    def update_original_body(self, output_messages: list[dict]) -> Any:
        assert len(output_messages) == 1
        self.body["prompt"] = output_messages[0]["content"]
        return self.body


class CrowdStrikeAIDRHandler(CustomGuardrail):
    """
    CrowdStrike AIDR AI Guardrail handler to interact with the CrowdStrike AIDR AI Guard service.

    This class implements the necessary hooks to call the CrowdStrike AIDR AI Guard API
    for input and output scanning.
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
        self, api: str, payload: dict, hook_name: str
    ) -> dict:
        """
        Makes the API call to the CrowdStrike AIDR AI Guard endpoint.
        The function itself will raise an error in the case that a response
        should be blocked, but will return a list of redacted messages that the
        caller should act on.

        Args:
            api (str): Which API to use (v1/guard_chat_completions)
            payload (dict): The request payload.
            hook_name (str): Name of the hook calling this function (for logging).

        Raises:
            HTTPException: If the CrowdStrike AIDR API returns a 'blocked: true' response.
            Exception: For other API call failures.

        Returns:
            dict: The API response body
        """
        endpoint = f"{self.api_base}/{api}"

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

        result = response.json()

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

    async def _async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: DualCache,
        data: dict,
        call_type: str,
    ):
        transformer = None
        messages: Any = None
        if call_type == "text_completion" or call_type == "atext_completion":
            transformer = _TextCompletionRequest(data)
            messages = transformer.get_messages()
        else:
            messages = data.get("messages")

        guard_input = {
            "messages": messages,  # type: ignore
        }
        if data.get("tools"):
            guard_input["tools"] = data.get("tools")

        ai_guard_payload = {
            "guard_input": guard_input,
            "event_type": "input",
        }

        ai_guard_response = await self._call_crowdstrike_aidr_guard(
            "v1/guard_chat_completions", ai_guard_payload, "async_pre_call_hook"
        )
        add_guardrail_to_applied_guardrails_header(
            request_data=data, guardrail_name=self.guardrail_name
        )

        if not ai_guard_response.get("result", {}).get("transformed"):
            return

        guard_output = ai_guard_response.get("result", {}).get("guard_output", {})
        if call_type == "text_completion" or call_type == "atext_completion":
            data = transformer.update_original_body(guard_output["messages"])  # type: ignore
        else:
            data["messages"] = guard_output["messages"]
        return data

    @log_guardrail_information
    async def async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: DualCache,
        data: dict,
        call_type: str,
    ):
        event_type = GuardrailEventHooks.pre_call
        if self.should_run_guardrail(data=data, event_type=event_type) is not True:
            verbose_proxy_logger.debug(
                f"CrowdStrike AIDR Guardrail (async_pre_call_hook): Guardrail is disabled {self.guardrail_name}."
            )
            return data

        try:
            return await self._async_pre_call_hook(
                user_api_key_dict, cache, data, call_type
            )
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "Error in CrowdStrike AIDR Guardrail",
                    "guardrail_name": self.guardrail_name,
                    "exceptions": str(e),
                },
            ) from e

    async def _async_post_call_success_hook(
        self,
        data: dict,
        user_api_key_dict: UserAPIKeyAuth,
        # This union isn't actually correct -- it can get other response types depending on the API called
        response: LLMResponseTypes,
    ):
        if isinstance(response, TextCompletionResponse):
            # Assume the earlier call type as well
            input_messages = _TextCompletionRequest(data).get_messages()
        if not isinstance(response, ModelResponse):
            return
        else:
            input_messages = data.get("messages")

        if choices := response.get("choices"):
            if isinstance(choices, list):
                serialized_choices = []
                for c in choices:
                    if isinstance(c, Choices):
                        try:
                            serialized_choices.append(c.model_dump())
                        except Exception:
                            serialized_choices.append(c.dict())
                    else:
                        serialized_choices.append(c)
                choices = serialized_choices

        guard_input = {
            "messages": input_messages,
            "choices": choices,
        }
        if data.get("tools"):
            guard_input["tools"] = data.get("tools")

        ai_guard_payload = {
            "guard_input": guard_input,
            "event_type": "output",
        }

        ai_guard_response = await self._call_crowdstrike_aidr_guard(
            "v1/guard_chat_completions",
            ai_guard_payload,
            "async_post_call_success_hook",
        )
        add_guardrail_to_applied_guardrails_header(
            request_data=data, guardrail_name=self.guardrail_name
        )

        if not ai_guard_response.get("result", {}).get("transformed"):
            return

        guard_output = ai_guard_response.get("result", {}).get("guard_output", {})
        response.choices = guard_output["choices"]
        return response

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
                f"CrowdStrike AIDR Guardrail (async_post_call_success_hook): Guardrail is disabled {self.guardrail_name}."
            )
            return data
        try:
            return await self._async_post_call_success_hook(
                data, user_api_key_dict, response
            )
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "Error in CrowdStrike AIDR Guardrail",
                    "guardrail_name": self.guardrail_name,
                    "exceptions": str(e),
                },
            ) from e

    @staticmethod
    def get_config_model() -> Optional[Type["GuardrailConfigModel"]]:
        from litellm.types.proxy.guardrails.guardrail_hooks.crowdstrike_aidr import (
            CrowdStrikeAIDRGuardrailConfigModel,
        )

        return CrowdStrikeAIDRGuardrailConfigModel
