# +-------------------------------------------------------------+
#
#           Use Zscaler AI Guard for your LLM calls
#
# +-------------------------------------------------------------+
import os
from typing import Any, Literal, Optional, Union
from fastapi import HTTPException

from litellm.proxy._types import UserAPIKeyAuth
from litellm.integrations.custom_guardrail import (
    CustomGuardrail,
    log_guardrail_information,
)
from litellm.types.utils import (
    TextCompletionResponse,
    ModelResponse,
    TextChoices,
    Choices,
    StreamingChoices
)
from litellm import ModelResponse as LiteLLMModelResponse

from litellm._logging import verbose_proxy_logger
from litellm.llms.custom_httpx.http_handler import (
    get_async_httpx_client,
    httpxSpecialProvider,
)

GUARDRAIL_TIMEOUT = 5


class ZscalerAIGuard(CustomGuardrail):
    def __init__(
        self,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        policy_id: Optional[int] = None,
        send_user_api_key_alias: Optional[bool] = False,
        send_user_api_key_user_id: Optional[bool] = False,
        send_user_api_key_team_id: Optional[bool] = False,
        **kwargs,
    ):
        # store kwargs as optional_params
        self.optional_params = kwargs
        self.zscaler_ai_guard_url =  api_base or os.environ.get("ZSCALER_AI_GUARD_URL", "https://api.us1.zseclipse.net/v1/detection/execute-policy")
        self.policy_id = policy_id or int(
            os.environ.get("ZSCALER_AI_GUARD_POLICY_ID", -1)
        )
        self.api_key = api_key or os.environ["ZSCALER_AI_GUARD_API_KEY"]
        self.send_user_api_key_alias = send_user_api_key_alias or os.environ.get(
            "SEND_USER_API_KEY_ALIAS", False
        )
        self.send_user_api_key_user_id = send_user_api_key_user_id or os.environ.get(
            "SEND_USER_API_KEY_USER_ID", False
        )
        self.send_user_api_key_team_id = send_user_api_key_team_id or os.environ.get(
            "SEND_USER_API_KEY_TEAM_ID", False
        )

        verbose_proxy_logger.debug(
            f'''send_user_api_key_alias: {self.send_user_api_key_alias}, 
            send_user_api_key_user_id:{self.send_user_api_key_user_id}, 
            send_user_api_key_team_id:{self.send_user_api_key_team_id}'''
        )

        super().__init__(default_on=True)

        verbose_proxy_logger.debug("ZscalerAIGuard Initializing ...")

    def extract_blocking_info(self, response):
        """
        Extracts transaction ID and blocking detector details from a response.
        """
        transaction_id = response.get("transactionId", None)

        # Find which detectors are invoked and blocking
        blocking_detectors = []
        detector_responses = response.get("detectorResponses", {})
        for detector, details in detector_responses.items():
            if details.get("action") == "BLOCK":
                blocking_detectors.append(detector)

        # Return the extracted information
        return {
            "transactionId": transaction_id,
            "blockingDetectors": blocking_detectors,
        }

    def _create_user_facing_error(self, reason: str):
        """
        create an error dictionary that return to use
        """
        return {
            "error_type": "Zscaler AI Guard Service Operational Issue",
            "reason": reason,
        }
    
    def _prepare_headers(self, api_key, **kwargs):
        headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
        }
        extra_headers = headers.copy()
        if self.send_user_api_key_alias:
            user_api_key_alias = kwargs.get("user_api_key_alias", "N/A")
            extra_headers.update({"user-api-key-alias": user_api_key_alias})

        if self.send_user_api_key_team_id:
            user_api_key_team_id = kwargs.get("user_api_key_team_id", "N/A")
            extra_headers.update({"user-api-key-team-id": user_api_key_team_id})

        if self.send_user_api_key_user_id:
            user_api_key_user_id = kwargs.get("user-api-key-user-id", "N/A")
            extra_headers.update({"user-api-key-user-id": user_api_key_user_id})
        return extra_headers
    
    async def _send_request(self, url, headers, data):
        # Use LiteLLM's async HTTP client
        async_client = get_async_httpx_client(
            llm_provider=httpxSpecialProvider.LoggingCallback
        )

        response = await async_client.post(
            f"{url}",
            headers=headers,
            json=data,
            timeout=GUARDRAIL_TIMEOUT,
        )
        response.raise_for_status()
        return response



    def _handle_response(self, response, direction):
        # Raise exceptions on critical errors to stop the request
        if response.status_code == 429:  # Rate limit
            verbose_proxy_logger.error(
                "Zscaler AI Guard rate limit reached. Blocking request."
            )
            user_facing_error = self._create_user_facing_error(
                "Rate limit reached. status_code: 429"
            )
            # This exception will be caught by the proxy and returned to the user
            raise HTTPException(status_code=500, detail=user_facing_error)

        if response.status_code >= 500:  # Server error
            verbose_proxy_logger.error(
                f"Zscaler AI Guard service is unavailable (Status: {response.status_code}). Blocking request."
            )
            user_facing_error = self._create_user_facing_error(
                f"Service is unavailable (HTTP {response.status_code})"
            )
            raise HTTPException(status_code=500, detail=user_facing_error)

        if response.status_code == 200:
            json_response = response.json()
            statusCode_in_response = json_response.get("statusCode", None)
            if statusCode_in_response == 200:
                guardrail_result = json_response.get("action", None)
                verbose_proxy_logger.info(
                    f"Zscaler AI Guard response: {json_response}"
                )

                if guardrail_result == "BLOCK":
                    verbose_proxy_logger.info(
                        f"Violated Zscaler AI Guard guardrail policy. zscaler_ai_guard_response: {json_response}"
                    )
                    return {
                        "action": "BLOCK",
                        "zscaler_ai_guard_response": json_response,
                    }
                elif guardrail_result == "ALLOW" or guardrail_result == "DETECT":
                    verbose_proxy_logger.debug(
                        f"{direction} is allowed by Zscaler AI Guard. guardrail_result: {guardrail_result}"
                    )
                    return {
                        "action": "ALLOW",
                        "zscaler_ai_guard_response": json_response,
                        "direction": direction,
                    }
                else:
                    verbose_proxy_logger.error(
                        f"Action field in response is {guardrail_result}, expecting 'ALLOW', 'BLOCK' or 'DETECT'"
                    )
                    user_facing_error = self._create_user_facing_error(
                        f"Action field in response is {guardrail_result}, expecting 'ALLOW', 'BLOCK' or 'DETECT'"
                    )
                    raise HTTPException(status_code=500, detail=user_facing_error)
            else:
                errorMsg = json_response.get("errorMsg", None)
                verbose_proxy_logger.error(
                    f"statusCode in response: {statusCode_in_response}, errorMsg: {errorMsg}"
                )
                user_facing_error = self._create_user_facing_error(
                    f"statusCode in response: {statusCode_in_response}, errorMsg: {errorMsg}"
                )
                raise HTTPException(status_code=500, detail=user_facing_error)
        else:
            verbose_proxy_logger.error(
                f"Zscaler AI Guard status_code - {response.status_code}"
            )
            user_facing_error = self._create_user_facing_error(
                f"Response status code: {response.status_code}"
            )
            raise HTTPException(
                status_code=response.status_code, detail=user_facing_error
            )

    async def make_zscaler_ai_guard_api_call(
        self, zscaler_ai_guard_url, api_key, policy_id, direction, content, **kwargs
    ):
        """
        Makes an API call to the Zscaler AI Guard service and handles retries, errors, and response parsing.
        """

        extra_headers = self._prepare_headers(api_key, **kwargs)

        data = {
            "policyId": policy_id,
            "direction": direction,
            "content": content,
        }

        try:
            response =  await self._send_request(zscaler_ai_guard_url, extra_headers, data)
            return self._handle_response(response, direction)
        except Exception as e:
            verbose_proxy_logger.error(
                f"Hit exception when request Zscaler AI Guard: {e}. Blocking request."
            )
            user_facing_error = self._create_user_facing_error(
                f"Hit exception when request Zscaler AI Guard: {str(e)})"
            )
            # This exception will be caught by the proxy and returned to the user
            raise HTTPException(status_code=500, detail=user_facing_error)

    def _handle_guardrail_result(self, data, zscaler_ai_guard_result):
        if zscaler_ai_guard_result:
            action = zscaler_ai_guard_result.get("action")

            if action == "BLOCK":
                blocking_info = zscaler_ai_guard_result.get("zscaler_ai_guard_response")
                # Construct a formal error response with guardrail details
                error_response = {
                    "error_type": "Guardrail Policy Violation",
                    "message": "Prompt violated your Zscaler AI Guard Policy",
                    "blocking_info": self.extract_blocking_info(blocking_info),
                }
                verbose_proxy_logger.debug(f"{error_response}")
                raise HTTPException(
                    status_code=400,
                    detail=error_response,
                )
            elif action == "ALLOW":
                return data
            else:
                error_msg = self._create_user_facing_error(
                    f"Action field in guardrail response is {action}, expecting 'ALLOW', 'BLOCK' or 'DETECT'"
                )
                raise HTTPException(status_code=500, detail={"error": error_msg})
        else:
            err_msg = self._create_user_facing_error(
                reason="No response from Zscaler AI Guard."
            )
            raise HTTPException(status_code=500, detail={"error": err_msg})
        
    @log_guardrail_information
    async def async_moderation_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        data: dict,
        call_type: Literal[
            "completion",
            "text_completion",
            "embeddings",
            "image_generation",
            "moderation",
            "audio_transcription",
            "pass_through_endpoint",
            "rerank",
        ],
    ) -> Optional[Union[Exception, str, dict]]:
        """
        Moderation hook.
        Runs during the LLM API call to check if input meets guardrail criteria.
        Can allow or block the input based on policy violations.
        """
        verbose_proxy_logger.debug(
            f"inside async_moderation_hook... call_type: {call_type}"
        )

        custom_policy_id = data.get("metadata", {}).get("zguard_policy_id", self.policy_id)

        kwargs = {}
        if self.send_user_api_key_alias:
            user_api_key_alias = data.get("metadata", {}).get(
                "user_api_key_alias", "N/A"
            )
            if user_api_key_alias is not None:
                user_api_key_alias = str(user_api_key_alias).strip()
            kwargs["user_api_key_alias"] = user_api_key_alias

        if self.send_user_api_key_team_id:
            user_api_key_team_id = data.get("metadata", {}).get(
                "user_api_key_team_id", "N/A"
            )
            if user_api_key_team_id is not None:
                user_api_key_team_id = str(user_api_key_team_id).strip()
            kwargs["user_api_key_team_id"] = user_api_key_team_id

        if self.send_user_api_key_user_id:
            user_api_key_user_id = data.get("metadata", {}).get(
                "user_api_key_user_id", "N/A"
            )
            if user_api_key_user_id is not None:
                user_api_key_user_id = str(user_api_key_user_id).strip()
            kwargs["user_api_key_user_id"] = user_api_key_user_id

        try:
            # Extract content from different input formats
            prompt = ""
            messages = data.get("messages")
            if messages:
                for message in messages:
                    _content = message.get("content")
                    if isinstance(_content, str):
                        prompt += " " + _content
                    elif isinstance(_content, list):
                        for item in _content:
                            if isinstance(item, dict) and "text" in item:
                                prompt += " " + item["text"]
            elif data.get("prompt"):
                _content = data.get("prompt")
                if isinstance(_content, str):
                    prompt = _content
            elif data.get("inputs"):
                _content = data.get("inputs")
                if isinstance(_content, str):
                    prompt = _content
            else:
                verbose_proxy_logger.warning(
                    "no 'message' and 'prompt' and 'inputs' in input, didn't call Zscaler AI Guard"
                )
                return data
        except Exception as e:
            verbose_proxy_logger.error(
                f"Moderation hook hit exception before call Zscaler AI Guard: {e} "
            )
            return data

        if prompt == "":
            verbose_proxy_logger.error(
                "prompt is empty. Didn't call Zscaler AI Guardrail."
            )
            return data

        zscaler_ai_guard_result = await self.make_zscaler_ai_guard_api_call(
            self.zscaler_ai_guard_url,
            self.api_key,
            custom_policy_id,
            "IN",
            prompt,
            **kwargs,
        )

        return self._handle_guardrail_result(data, zscaler_ai_guard_result)
    
    def convert_litellm_response_object_to_str(
        self, response_obj: Union[Any, LiteLLMModelResponse]
    ) -> Optional[str]:
        """
        Converts LiteLLM response object to a string for further analysis.
        Parses specific response types to extract textual data.
        """
        response_str = ""
        try:
            if isinstance(response_obj, TextCompletionResponse):
                for choice in response_obj.choices:
                    if isinstance(choice, TextChoices):
                        if choice.text and isinstance(choice.text, str):
                            response_str += choice.text

            elif isinstance(response_obj, ModelResponse):
                for choice in response_obj.choices:  # type: ignore[assignment]
                    if isinstance(choice, Choices):
                        if choice.message.content and isinstance(
                            choice.message.content, str
                        ):
                            response_str += choice.message.content
                    elif isinstance(choice, StreamingChoices):
                        if choice.delta.content and isinstance(choice.delta.content, str):
                            response_str += choice.delta.content
            else:
                verbose_proxy_logger.error(
                    f"isinstance(response_obj : {type(response_obj)}, currently only handle TextCompletionResponse, ModelResponse"
                )
            return response_str if response_str else None
        except Exception as e:
            error_msg = f"Error converting response to string: {str(e)}"
            verbose_proxy_logger.error(f"{error_msg}")
            return None
            
    @log_guardrail_information
    async def async_post_call_success_hook(
        self, data: dict, user_api_key_dict: UserAPIKeyAuth, response
    ):
        """
        Post-call moderation hook.
        Runs after the LLM API call and checks if the output complies with guardrail policies.
        Can block or allow the output based on violations detected.
        """
        verbose_proxy_logger.debug("inside async_post_call_success_hook ...")
        custom_policy_id = data.get("metadata", {}).get(
            "zguard_policy_id", self.policy_id
        )
        kwargs = {}
        if self.send_user_api_key_alias:
            user_api_key_alias = data.get("metadata", {}).get(
                "user_api_key_alias", "N/A"
            )
            if user_api_key_alias is not None:
                user_api_key_alias = str(user_api_key_alias).strip()
            kwargs["user_api_key_alias"] = user_api_key_alias

        if self.send_user_api_key_team_id:
            user_api_key_team_id = data.get("metadata", {}).get(
                "user_api_key_team_id", "N/A"
            )
            if user_api_key_team_id is not None:
                user_api_key_team_id = str(user_api_key_team_id).strip()
            kwargs["user_api_key_team_id"] = user_api_key_team_id

        if self.send_user_api_key_user_id:
            user_api_key_user_id = data.get("metadata", {}).get(
                "user_api_key_user_id", "N/A"
            )
            if user_api_key_user_id is not None:
                user_api_key_user_id = str(user_api_key_user_id).strip()
            kwargs["user_api_key_user_id"] = user_api_key_user_id

        try:
            response_str = self.convert_litellm_response_object_to_str(response)
            verbose_proxy_logger.debug(f"response_str: {response_str}")
        except Exception as e:
            verbose_proxy_logger.error(
                f"Post call hook hit exception before Zscaler AI Guard API call. {e}"
            )
            return
        if response_str is not None:
            zscaler_ai_guard_result = await self.make_zscaler_ai_guard_api_call(
                self.zscaler_ai_guard_url,
                self.api_key,
                custom_policy_id,
                "OUT",
                response_str,
                **kwargs,
            )

            if zscaler_ai_guard_result:
                action = zscaler_ai_guard_result.get("action")

                if action == "BLOCK":

                    blocking_info = zscaler_ai_guard_result.get(
                        "zscaler_ai_guard_response"
                    )

                    # Construct a formal error response with guardrail details
                    error_response = {
                        "error_type": "Guardrail Policy Violation",
                        "message": "LLM response violated your Zscaler AI Guard Policy",
                        "blocking_info": self.extract_blocking_info(blocking_info),
                    }
                    # Raise the exception with detailed information
                    raise HTTPException(
                        status_code=400,
                        detail=error_response,
                    )
                elif action == "ALLOW":
                    return data
                else:
                    error_msg = self._create_user_facing_error(
                        f"Action field in response is {action}, expecting 'ALLOW', 'BLOCK' or 'DETECT'"
                    )
                    raise HTTPException(status_code=500, detail={"error": error_msg})
        else:
            verbose_proxy_logger.warning(
                "No response content to analyze, Didn't call Zscaler AI Guard"
            )
            return
