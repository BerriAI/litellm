# +-------------------------------------------------------------+
#
#           Use DynamoAI Guardrails for your LLM calls
#           https://dynamo.ai
#
# +-------------------------------------------------------------+

import os
from datetime import datetime
from typing import Any, AsyncGenerator, Dict, List, Optional, Type, Union

import httpx

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.caching.caching import DualCache
from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.llms.custom_httpx.http_handler import (
    get_async_httpx_client,
    httpxSpecialProvider,
)
from litellm.proxy._types import UserAPIKeyAuth
from litellm.types.guardrails import GuardrailEventHooks
from litellm.types.proxy.guardrails.guardrail_hooks.base import GuardrailConfigModel
from litellm.types.proxy.guardrails.guardrail_hooks.dynamoai import (
    DynamoAIProcessedResult,
    DynamoAIRequest,
    DynamoAIResponse,
)
from litellm.types.utils import CallTypesLiteral, GuardrailStatus, ModelResponseStream

GUARDRAIL_NAME = "dynamoai"


class DynamoAIGuardrails(CustomGuardrail):
    """
    DynamoAI Guardrails integration for LiteLLM.

    Provides content moderation and policy enforcement using DynamoAI's guardrail API.
    """

    def __init__(
        self,
        guardrail_name: str = "litellm_test",
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        model_id: str = "",
        policy_ids: List[str] = [],
        **kwargs,
    ):
        self.async_handler = get_async_httpx_client(
            llm_provider=httpxSpecialProvider.GuardrailCallback
        )

        # Set API configuration
        self.api_key = api_key or os.getenv("DYNAMOAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "DynamoAI API key is required. Set DYNAMOAI_API_KEY environment variable or pass api_key parameter."
            )

        self.api_base = api_base or os.getenv(
            "DYNAMOAI_API_BASE", "https://api.dynamo.ai"
        )
        self.api_url = f"{self.api_base}/v1/moderation/analyze/"

        # Model ID for tracking/logging purposes
        self.model_id = model_id or os.getenv("DYNAMOAI_MODEL_ID", "")

        # Policy IDs - get from parameter, env var, or use empty list
        env_policy_ids = os.getenv("DYNAMOAI_POLICY_IDS", "")
        self.policy_ids = policy_ids or (
            env_policy_ids.split(",") if env_policy_ids else []
        )
        self.guardrail_name = guardrail_name
        self.guardrail_provider = "dynamoai"

        # store kwargs as optional_params
        self.optional_params = kwargs

        # Set supported event hooks
        if "supported_event_hooks" not in kwargs:
            kwargs["supported_event_hooks"] = [
                GuardrailEventHooks.pre_call,
                GuardrailEventHooks.post_call,
                GuardrailEventHooks.during_call,
            ]

        super().__init__(guardrail_name=guardrail_name, **kwargs)

        verbose_proxy_logger.debug(
            "DynamoAI Guardrail initialized with guardrail_name=%s, model_id=%s",
            self.guardrail_name,
            self.model_id,
        )

    async def _call_dynamoai_guardrails(
        self,
        messages: List[Dict[str, Any]],
        text_type: str = "input",
        request_data: Optional[dict] = None,
    ) -> DynamoAIResponse:
        """
        Call DynamoAI Guardrails API to analyze messages for policy violations.

        Args:
            messages: List of messages to analyze
            text_type: Type of text being analyzed ("input" or "output")
            request_data: Optional request data for logging purposes

        Returns:
            DynamoAIResponse: Response from the DynamoAI Guardrails API
        """
        start_time = datetime.now()

        payload: DynamoAIRequest = {
            "messages": messages,
        }

        # Add optional fields if provided
        if self.policy_ids:
            payload["policyIds"] = self.policy_ids
        if self.model_id:
            payload["modelId"] = self.model_id

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        verbose_proxy_logger.debug(
            "DynamoAI request to %s with payload=%s",
            self.api_url,
            payload,
        )

        try:
            response = await self.async_handler.post(
                url=self.api_url,
                json=dict(payload),
                headers=headers,
            )
            response.raise_for_status()
            response_json = response.json()

            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()

            # Add guardrail information to request trace
            if request_data:
                guardrail_status = self._determine_guardrail_status(response_json)
                self.add_standard_logging_guardrail_information_to_request_data(
                    guardrail_provider=self.guardrail_provider,
                    guardrail_json_response=response_json,
                    request_data=request_data,
                    guardrail_status=guardrail_status,
                    start_time=start_time.timestamp(),
                    end_time=end_time.timestamp(),
                    duration=duration,
                )

            return response_json

        except httpx.HTTPError as e:
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()

            verbose_proxy_logger.error("DynamoAI API request failed: %s", str(e))

            # Add guardrail information with failure status
            if request_data:
                self.add_standard_logging_guardrail_information_to_request_data(
                    guardrail_provider=self.guardrail_provider,
                    guardrail_json_response={"error": str(e)},
                    request_data=request_data,
                    guardrail_status="guardrail_failed_to_respond",
                    start_time=start_time.timestamp(),
                    end_time=end_time.timestamp(),
                    duration=duration,
                )

            raise

    def _process_dynamoai_guardrails_response(
        self, response: DynamoAIResponse
    ) -> DynamoAIProcessedResult:
        """
        Process the response from the DynamoAI Guardrails API

        Args:
            response: The response from the API with 'finalAction' and 'appliedPolicies' keys

        Returns:
            DynamoAIProcessedResult: Processed response with detected violations
        """
        final_action = response.get("finalAction", "NONE")
        applied_policies = response.get("appliedPolicies", [])

        violations_detected: List[str] = []
        violation_details: Dict[str, Any] = {}

        # For now, only handle BLOCK action
        if final_action == "BLOCK":
            for applied_policy in applied_policies:
                policy_info = applied_policy.get("policy", {})
                policy_outputs = applied_policy.get("outputs", {})

                # Get policy name and action
                policy_name = policy_info.get("name", "unknown")

                # Check for action in multiple places
                policy_action = (
                    applied_policy.get("action")
                    or (policy_outputs.get("action") if policy_outputs else None)
                    or "NONE"
                )

                # Only include policies with BLOCK action
                if policy_action == "BLOCK":
                    violations_detected.append(policy_name)
                    violation_details[policy_name] = {
                        "policyId": policy_info.get("id"),
                        "action": policy_action,
                        "method": policy_info.get("method"),
                        "description": policy_info.get("description"),
                        "message": (
                            policy_outputs.get("message") if policy_outputs else None
                        ),
                    }

        return {
            "violations_detected": violations_detected,
            "violation_details": violation_details,
        }

    def _determine_guardrail_status(
        self, response_json: DynamoAIResponse
    ) -> GuardrailStatus:
        """
        Determine the guardrail status based on DynamoAI API response.

        Returns:
            "success": Content allowed through with no violations (finalAction is NONE)
            "guardrail_intervened": Content blocked (finalAction is BLOCK)
            "guardrail_failed_to_respond": Technical error or API failure
        """
        try:
            if not isinstance(response_json, dict):
                return "guardrail_failed_to_respond"

            # Check for error in response
            if response_json.get("error"):
                return "guardrail_failed_to_respond"

            final_action = response_json.get("finalAction", "NONE")

            if final_action == "NONE":
                return "success"
            elif final_action == "BLOCK":
                return "guardrail_intervened"

            # For now, treat other actions as success (WARN, REDACT, SANITIZE not implemented yet)
            return "success"

        except Exception as e:
            verbose_proxy_logger.error(
                "Error determining DynamoAI guardrail status: %s", str(e)
            )
            return "guardrail_failed_to_respond"

    def _create_error_message(self, processed_result: DynamoAIProcessedResult) -> str:
        """
        Create a detailed error message from processed guardrail results.

        Args:
            processed_result: Processed response with detected violations

        Returns:
            Formatted error message string
        """
        violations_detected = processed_result["violations_detected"]
        violation_details = processed_result["violation_details"]

        error_message = (
            f"Guardrail failed: {len(violations_detected)} violation(s) detected\n\n"
        )

        for policy_name in violations_detected:
            error_message += f"- {policy_name.upper()}:\n"
            details = violation_details.get(policy_name, {})

            # Format violation details
            if details.get("action"):
                error_message += f"  Action: {details['action']}\n"
            if details.get("method"):
                error_message += f"  Method: {details['method']}\n"
            if details.get("description"):
                error_message += f"  Description: {details['description']}\n"
            if details.get("message"):
                error_message += f"  Message: {details['message']}\n"
            if details.get("policyId"):
                error_message += f"  Policy ID: {details['policyId']}\n"
            error_message += "\n"

        return error_message.strip()

    async def async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: DualCache,
        data: dict,
        call_type: CallTypesLiteral,
    ) -> Union[Exception, str, dict, None]:
        """
        Runs before the LLM API call
        Runs on only Input
        Use this if you want to MODIFY the input
        """
        verbose_proxy_logger.debug("Running DynamoAI pre-call hook")

        from litellm.proxy.common_utils.callback_utils import (
            add_guardrail_to_applied_guardrails_header,
        )

        event_type: GuardrailEventHooks = GuardrailEventHooks.pre_call
        if self.should_run_guardrail(data=data, event_type=event_type) is not True:
            return data

        _messages = data.get("messages")
        if _messages:
            result = await self._call_dynamoai_guardrails(
                messages=_messages,
                text_type="input",
                request_data=data,
            )

            verbose_proxy_logger.debug(
                "Guardrails async_pre_call_hook result=%s", result
            )

            # Process the guardrails response
            processed_result = self._process_dynamoai_guardrails_response(result)
            violations_detected = processed_result["violations_detected"]

            # If any violations are detected, raise an error
            if violations_detected:
                error_message = self._create_error_message(processed_result)
                raise ValueError(error_message)

        # Add guardrail to applied guardrails header
        add_guardrail_to_applied_guardrails_header(
            request_data=data, guardrail_name=self.guardrail_name
        )

        return data

    async def async_moderation_hook(
        self,
        data: dict,
        user_api_key_dict: UserAPIKeyAuth,
        call_type: CallTypesLiteral,
    ):
        """
        Runs in parallel to LLM API call
        Runs on only Input

        This can NOT modify the input, only used to reject or accept a call before going to LLM API
        """
        from litellm.proxy.common_utils.callback_utils import (
            add_guardrail_to_applied_guardrails_header,
        )

        event_type: GuardrailEventHooks = GuardrailEventHooks.during_call
        if self.should_run_guardrail(data=data, event_type=event_type) is not True:
            return

        _messages = data.get("messages")
        if _messages:
            result = await self._call_dynamoai_guardrails(
                messages=_messages,
                text_type="input",
                request_data=data,
            )

            verbose_proxy_logger.debug(
                "Guardrails async_moderation_hook result=%s", result
            )

            # Process the guardrails response
            processed_result = self._process_dynamoai_guardrails_response(result)
            violations_detected = processed_result["violations_detected"]

            # If any violations are detected, raise an error
            if violations_detected:
                error_message = self._create_error_message(processed_result)
                raise ValueError(error_message)

        # Add guardrail to applied guardrails header
        add_guardrail_to_applied_guardrails_header(
            request_data=data, guardrail_name=self.guardrail_name
        )

        return data

    async def async_post_call_success_hook(
        self,
        data: dict,
        user_api_key_dict: UserAPIKeyAuth,
        response,
    ):
        """
        Runs on response from LLM API call

        It can be used to reject a response

        Uses DynamoAI guardrails to check the response for policy violations
        """
        from litellm.proxy.common_utils.callback_utils import (
            add_guardrail_to_applied_guardrails_header,
        )
        from litellm.types.guardrails import GuardrailEventHooks

        if (
            self.should_run_guardrail(
                data=data, event_type=GuardrailEventHooks.post_call
            )
            is not True
        ):
            return

        verbose_proxy_logger.debug("async_post_call_success_hook response=%s", response)

        # Check if the ModelResponse has text content in its choices
        # to avoid sending empty content to DynamoAI (e.g., during tool calls)
        if isinstance(response, litellm.ModelResponse):
            has_text_content = False
            dynamoai_messages: List[Dict[str, Any]] = []

            for choice in response.choices:
                if isinstance(choice, litellm.Choices):
                    if choice.message.content and isinstance(
                        choice.message.content, str
                    ):
                        has_text_content = True
                        dynamoai_messages.append(
                            {
                                "role": choice.message.role or "assistant",
                                "content": choice.message.content,
                            }
                        )

            if not has_text_content:
                verbose_proxy_logger.warning(
                    "DynamoAI: not running guardrail. No output text in response"
                )
                return

            if dynamoai_messages:
                result = await self._call_dynamoai_guardrails(
                    messages=dynamoai_messages,
                    text_type="output",
                    request_data=data,
                )

                verbose_proxy_logger.debug(
                    "Guardrails async_post_call_success_hook result=%s", result
                )

                # Process the guardrails response
                processed_result = self._process_dynamoai_guardrails_response(result)
                violations_detected = processed_result["violations_detected"]

                # If any violations are detected, raise an error
                if violations_detected:
                    error_message = self._create_error_message(processed_result)
                    raise ValueError(error_message)

        # Add guardrail to applied guardrails header
        add_guardrail_to_applied_guardrails_header(
            request_data=data, guardrail_name=self.guardrail_name
        )

    async def async_post_call_streaming_iterator_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        response: Any,
        request_data: dict,
    ) -> AsyncGenerator[ModelResponseStream, None]:
        """
        Passes the entire stream to the guardrail

        This is useful for guardrails that need to see the entire response, such as PII masking.

        Triggered by mode: 'post_call'
        """
        async for item in response:
            yield item

    @staticmethod
    def get_config_model() -> Optional[Type[GuardrailConfigModel]]:
        from litellm.types.proxy.guardrails.guardrail_hooks.dynamoai import (
            DynamoAIGuardrailConfigModel,
        )

        return DynamoAIGuardrailConfigModel
