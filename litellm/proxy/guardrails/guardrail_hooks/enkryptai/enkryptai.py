# +-------------------------------------------------------------+
#
#           Use EnkryptAI Guardrails for your LLM calls
#           https://enkryptai.com
#
# +-------------------------------------------------------------+

import os
from datetime import datetime
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncGenerator,
    Dict,
    List,
    Literal,
    Optional,
    Union,
)

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
from litellm.types.proxy.guardrails.guardrail_hooks.enkryptai import (
    EnkryptAIProcessedResult,
    EnkryptAIResponse,
)
from litellm.types.utils import (
    CallTypesLiteral,
    GenericGuardrailAPIInputs,
    GuardrailStatus,
    ModelResponseStream,
)

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj

GUARDRAIL_NAME = "enkryptai"


class EnkryptAIGuardrails(CustomGuardrail):
    def __init__(
        self,
        guardrail_name: str = "litellm_test",
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        policy_name: Optional[str] = None,
        **kwargs,
    ):
        self.async_handler = get_async_httpx_client(
            llm_provider=httpxSpecialProvider.GuardrailCallback
        )

        # Set API configuration
        self.api_key = api_key or os.getenv("ENKRYPTAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "EnkryptAI API key is required. Set ENKRYPTAI_API_KEY environment variable or pass api_key parameter."
            )

        self.api_base = api_base or os.getenv(
            "ENKRYPTAI_API_BASE", "https://api.enkryptai.com"
        )
        self.api_url = f"{self.api_base}/guardrails/policy/detect"

        # Policy name can be passed as parameter or use guardrail_name
        self.policy_name = policy_name
        self.guardrail_name = guardrail_name
        self.guardrail_provider = "enkryptai"

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
            "EnkryptAI Guardrail initialized with guardrail_name: %s, policy_name: %s",
            self.guardrail_name,
            self.policy_name,
        )

    async def _call_enkryptai_guardrails(
        self,
        prompt: str,
        request_data: Optional[dict] = None,
    ) -> EnkryptAIResponse:
        """
        Call Enkrypt AI Guardrails API to detect potential issues in the given prompt.

        Args:
            prompt (str): The text to analyze for potential violations
            request_data (dict): Optional request data for logging purposes

        Returns:
            EnkryptAIResponse: Response from the Enkrypt AI Guardrails API
        """
        start_time = datetime.now()

        payload = {"text": prompt}

        headers = {"Content-Type": "application/json", "apikey": self.api_key}

        # Add policy header if policy_name is set
        if self.policy_name:
            headers["x-enkrypt-policy"] = self.policy_name

        verbose_proxy_logger.debug(
            "EnkryptAI request to %s with payload: %s",
            self.api_url,
            payload,
        )

        try:
            verbose_proxy_logger.debug(
                "EnkryptAI request to %s with payload: %s",
                self.api_url,
                payload,
            )
            response = await self.async_handler.post(
                url=self.api_url,
                json=payload,
                headers=headers,
            )
            response.raise_for_status()
            response_json = response.json()

            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()

            verbose_proxy_logger.debug(
                "EnkryptAI response from %s with payload: %s",
                self.api_url,
                response_json,
            )

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

            verbose_proxy_logger.error("EnkryptAI API request failed: %s", str(e))

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

    def _process_enkryptai_guardrails_response(
        self, response: EnkryptAIResponse
    ) -> EnkryptAIProcessedResult:
        """
        Process the response from the Enkrypt AI Guardrails API

        Args:
            response: The response from the API with 'summary' and 'details' keys

        Returns:
            EnkryptAIProcessedResult: Processed response with detected attacks and their details
        """
        summary = response.get("summary", {})
        details = response.get("details", {})

        detected_attacks: List[str] = []
        attack_details: Dict[str, Any] = {}

        for key, value in summary.items():
            # Check if attack is detected
            # For toxicity, it's a list (non-empty list means detected)
            # For others, it's 1 for detected, 0 for not detected
            if key == "toxicity":
                if isinstance(value, list) and len(value) > 0:
                    detected_attacks.append(key)
                    attack_details[key] = details.get(key, {})
            else:
                if value == 1:
                    detected_attacks.append(key)
                    attack_details[key] = details.get(key, {})

        return {"attacks_detected": detected_attacks, "attack_details": attack_details}

    def _determine_guardrail_status(
        self, response_json: EnkryptAIResponse
    ) -> GuardrailStatus:
        """
        Determine the guardrail status based on EnkryptAI API response.

        Returns:
            "success": Content allowed through with no violations
            "guardrail_intervened": Content blocked due to policy violations
            "guardrail_failed_to_respond": Technical error or API failure
        """
        try:
            if not isinstance(response_json, dict):
                return "guardrail_failed_to_respond"

            # Process the response to check for violations
            processed_result = self._process_enkryptai_guardrails_response(
                response_json
            )
            attacks_detected = processed_result["attacks_detected"]

            if attacks_detected:
                return "guardrail_intervened"

            return "success"

        except Exception as e:
            verbose_proxy_logger.error(
                "Error determining EnkryptAI guardrail status: %s", str(e)
            )
            return "guardrail_failed_to_respond"

    def _create_error_message(self, processed_result: EnkryptAIProcessedResult) -> str:
        """
        Create a detailed error message from processed guardrail results.

        Args:
            processed_result: Processed response with detected attacks and their details

        Returns:
            Formatted error message string
        """
        attacks_detected = processed_result["attacks_detected"]
        attack_details = processed_result["attack_details"]

        error_message = (
            f"Guardrail failed: {len(attacks_detected)} violation(s) detected\n\n"
        )

        for attack_type in attacks_detected:
            error_message += f"- {attack_type.upper()}:\n"
            details = attack_details.get(attack_type, {})

            # Format details based on attack type
            if attack_type == "policy_violation":
                error_message += f"  Policy: {details.get('violating_policy', 'N/A')}\n"
                error_message += f"  Explanation: {details.get('explanation', 'N/A')}\n"
            elif attack_type == "pii":
                error_message += f"  PII Detected: {details.get('pii', {})}\n"
            elif attack_type == "toxicity":
                toxic_types = [
                    k
                    for k, v in details.items()
                    if isinstance(v, (int, float)) and v > 0.5
                ]
                error_message += f"  Types: {', '.join(toxic_types)}\n"
            elif attack_type == "keyword_detected":
                error_message += f"  Keywords: {details.get('detected_keywords', [])}\n"
            elif attack_type == "bias":
                error_message += (
                    f"  Bias Detected: {details.get('bias_detected', False)}\n"
                )
            else:
                error_message += f"  Details: {details}\n"
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
        verbose_proxy_logger.debug("Running EnkryptAI pre-call hook")

        from litellm.proxy.common_utils.callback_utils import (
            add_guardrail_to_applied_guardrails_header,
        )

        event_type: GuardrailEventHooks = GuardrailEventHooks.pre_call
        if self.should_run_guardrail(data=data, event_type=event_type) is not True:
            return data

        _messages = data.get("messages")
        if _messages:
            for message in _messages:
                _content = message.get("content")
                if isinstance(_content, str):
                    result = await self._call_enkryptai_guardrails(
                        prompt=_content,
                        request_data=data,
                    )

                    verbose_proxy_logger.debug(
                        "Guardrails async_pre_call_hook result: %s", result
                    )

                    # Process the guardrails response
                    processed_result = self._process_enkryptai_guardrails_response(
                        result
                    )
                    attacks_detected = processed_result["attacks_detected"]

                    # If any attacks are detected, raise an error
                    if attacks_detected:
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
            for message in _messages:
                _content = message.get("content")
                if isinstance(_content, str):
                    result = await self._call_enkryptai_guardrails(
                        prompt=_content,
                        request_data=data,
                    )

                    verbose_proxy_logger.debug(
                        "Guardrails async_moderation_hook result: %s", result
                    )

                    # Process the guardrails response
                    processed_result = self._process_enkryptai_guardrails_response(
                        result
                    )
                    attacks_detected = processed_result["attacks_detected"]

                    # If any attacks are detected, raise an error
                    if attacks_detected:
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

        Uses Enkrypt AI guardrails to check the response for policy violations, PII, and injection attacks
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

        verbose_proxy_logger.debug(
            "async_post_call_success_hook response: %s", response
        )

        # Check if the ModelResponse has text content in its choices
        # to avoid sending empty content to EnkryptAI (e.g., during tool calls)
        if isinstance(response, litellm.ModelResponse):
            has_text_content = False
            for choice in response.choices:
                if isinstance(choice, litellm.Choices):
                    if choice.message.content and isinstance(
                        choice.message.content, str
                    ):
                        has_text_content = True
                        break

            if not has_text_content:
                verbose_proxy_logger.warning(
                    "EnkryptAI: not running guardrail. No output text in response"
                )
                return

            for choice in response.choices:
                if isinstance(choice, litellm.Choices):
                    verbose_proxy_logger.debug(
                        "async_post_call_success_hook choice: %s", choice
                    )
                    if choice.message.content and isinstance(
                        choice.message.content, str
                    ):
                        result = await self._call_enkryptai_guardrails(
                            prompt=choice.message.content,
                            request_data=data,
                        )

                        verbose_proxy_logger.debug(
                            "Guardrails async_post_call_success_hook result: %s", result
                        )

                        # Process the guardrails response
                        processed_result = self._process_enkryptai_guardrails_response(
                            result
                        )
                        attacks_detected = processed_result["attacks_detected"]

                        # If any attacks are detected, raise an error
                        if attacks_detected:
                            error_message = self._create_error_message(processed_result)
                            raise ValueError(error_message)

        # Add guardrail to applied guardrails header
        add_guardrail_to_applied_guardrails_header(
            request_data=data, guardrail_name=self.guardrail_name
        )

    async def apply_guardrail(
        self,
        inputs: "GenericGuardrailAPIInputs",
        request_data: dict,
        input_type: Literal["request", "response"],
        logging_obj: Optional["LiteLLMLoggingObj"] = None,
    ) -> "GenericGuardrailAPIInputs":
        """
        Apply EnkryptAI guardrail to a batch of texts.

        Args:
            inputs: Dictionary containing texts and optional images
            request_data: Request data dictionary containing metadata
            input_type: Whether this is a "request" or "response"
            logging_obj: Optional logging object

        Returns:
            GenericGuardrailAPIInputs - texts unchanged if passed, images unchanged

        Raises:
            ValueError: If any attacks are detected
        """
        texts = inputs.get("texts", [])

        # Check each text for attacks
        for text in texts:
            result = await self._call_enkryptai_guardrails(
                prompt=text,
                request_data=request_data,
            )
            # Process the guardrails response
            processed_result = self._process_enkryptai_guardrails_response(result)
            attacks_detected = processed_result["attacks_detected"]

            # If any attacks are detected, raise an error
            if attacks_detected:
                error_message = self._create_error_message(processed_result)
                raise ValueError(error_message)

        return inputs

    async def async_post_call_streaming_iterator_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        response: Any,
        request_data: dict,
    ) -> AsyncGenerator[ModelResponseStream, None]:
        """
        Passes the entire stream to the guardrail

        This is useful for guardrails that need to see the entire response, such as PII masking.

        See Aim guardrail implementation for an example - https://github.com/BerriAI/litellm/blob/d0e022cfacb8e9ebc5409bb652059b6fd97b45c0/litellm/proxy/guardrails/guardrail_hooks/aim.py#L168

        Triggered by mode: 'post_call'
        """
        async for item in response:
            yield item

    @staticmethod
    def get_config_model():
        from litellm.types.proxy.guardrails.guardrail_hooks.enkryptai import (
            EnkryptAIGuardrailConfigModel,
        )

        return EnkryptAIGuardrailConfigModel
