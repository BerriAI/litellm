# +-------------------------------------------------------------+
#
#           Use Zscaler AI Guard for your LLM calls
#
# +-------------------------------------------------------------+
import os
from typing import TYPE_CHECKING, Literal, Optional

from fastapi import HTTPException

from litellm._logging import verbose_proxy_logger
from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.llms.custom_httpx.http_handler import (
    get_async_httpx_client,
    httpxSpecialProvider,
)
from litellm.types.utils import GenericGuardrailAPIInputs

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj

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
        self.optional_params = kwargs
        self.zscaler_ai_guard_url = api_base or os.getenv(
            "ZSCALER_AI_GUARD_URL",
            "https://api.us1.zseclipse.net/v1/detection/execute-policy",
        )
        self.policy_id = policy_id or int(os.getenv("ZSCALER_AI_GUARD_POLICY_ID", -1))
        self.api_key = api_key or os.getenv("ZSCALER_AI_GUARD_API_KEY")
        self.send_user_api_key_alias = send_user_api_key_alias or os.getenv(
            "SEND_USER_API_KEY_ALIAS", "False"
        ).lower() in ("true", "1")
        self.send_user_api_key_user_id = send_user_api_key_user_id or os.getenv(
            "SEND_USER_API_KEY_USER_ID", "False"
        ).lower() in ("true", "1,")
        self.send_user_api_key_team_id = send_user_api_key_team_id or os.getenv(
            "SEND_USER_API_KEY_TEAM_ID", "False"
        ).lower() in ("true", "1")

        verbose_proxy_logger.debug(
            f"""send_user_api_key_alias: {self.send_user_api_key_alias}, 
            send_user_api_key_user_id:{self.send_user_api_key_user_id}, 
            send_user_api_key_team_id:{self.send_user_api_key_team_id}"""
        )

        super().__init__(default_on=True)

        verbose_proxy_logger.debug("ZscalerAIGuard Initializing ...")

    def _get_stripped_metadata_value(
        self, request_data: Optional[dict], key: str
    ) -> Optional[str]:
        if request_data is None:
            return "N/A"
        value = request_data.get("metadata", {}).get(key, "N/A")
        if value is not None:
            return str(value).strip()
        return "N/A"

    async def apply_guardrail(
        self,
        inputs: "GenericGuardrailAPIInputs",
        request_data: dict,
        input_type: Literal["request", "response"],
        logging_obj: Optional["LiteLLMLoggingObj"] = None,
    ) -> "GenericGuardrailAPIInputs":
        """
        Apply Zscaler AI Guard guardrail to batch of texts.

        Args:
            inputs: Dictionary containing texts and optional images
            request_data: Request data dictionary containing metadata
            input_type: Whether this is a "request" or "response"
            logging_obj: Optional logging object

        Returns:
            GenericGuardrailAPIInputs - texts unchanged if passed, images unchanged

        Raises:
            Exception: If content is blocked by Zscaler AI Guard
        """
        texts = inputs.get("texts", [])
        try:
            verbose_proxy_logger.debug(f"ZscalerAIGuard: Checking {len(texts)} text(s)")

            custom_policy_id = request_data.get("metadata", {}).get(
                "zguard_policy_id", self.policy_id
            )
            verbose_proxy_logger.debug(f"custom_policy_id: {custom_policy_id}")

            kwargs = {}
            if self.send_user_api_key_alias:
                kwargs["user_api_key_alias"] = self._get_stripped_metadata_value(
                    request_data, "user_api_key_alias"
                )
            if self.send_user_api_key_team_id:
                kwargs["user_api_key_team_id"] = self._get_stripped_metadata_value(
                    request_data, "user_api_key_team_id"
                )
            if self.send_user_api_key_user_id:
                kwargs["user_api_key_user_id"] = self._get_stripped_metadata_value(
                    request_data, "user_api_key_user_id"
                )
            verbose_proxy_logger.debug(f"inside apply_guardrail kwargs: {kwargs}")

            # Check each text (Zscaler processes one at a time)
            for text in texts:
                zscaler_ai_guard_result = await self.make_zscaler_ai_guard_api_call(
                    zscaler_ai_guard_url=self.zscaler_ai_guard_url,
                    api_key=self.api_key,
                    policy_id=self.policy_id,
                    direction="IN",
                    content=text,
                    **kwargs,
                )

                if (
                    zscaler_ai_guard_result
                    and zscaler_ai_guard_result.get("action") == "BLOCK"
                ):
                    blocking_info = zscaler_ai_guard_result.get(
                        "zscaler_ai_guard_response"
                    )
                    error_message = f"Content blocked by Zscaler AI Guard: {self.extract_blocking_info(blocking_info)}"
                    raise Exception(error_message)

        except Exception as e:
            verbose_proxy_logger.error(
                "ZscalerAIGuard: Failed to apply guardrail: %s", str(e)
            )
            raise e

        verbose_proxy_logger.debug("ZscalerAIGuard: Successfully applied guardrail.")
        return inputs

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

        return {
            "transactionId": transaction_id,
            "blockingDetectors": blocking_detectors,
        }

    def _create_user_facing_error(self, reason: str):
        """
        create an error dictionary that return to use
        """
        return {
            "error_type": "Zscaler AI Guard Error",
            "reason": reason,
        }

    def _prepare_headers(self, api_key, **kwargs):
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        extra_headers = headers.copy()
        if self.send_user_api_key_alias:
            verbose_proxy_logger.debug(f"kwargs: {kwargs}")
            user_api_key_alias = kwargs.get("user_api_key_alias", "N/A")
            verbose_proxy_logger.debug(
                f"kwargs user_api_key_alias: {user_api_key_alias}"
            )
            extra_headers.update({"user-api-key-alias": user_api_key_alias})

        if self.send_user_api_key_team_id:
            user_api_key_team_id = kwargs.get("user_api_key_team_id", "N/A")
            extra_headers.update({"user-api-key-team-id": user_api_key_team_id})

        if self.send_user_api_key_user_id:
            user_api_key_user_id = kwargs.get("user-api-key-user-id", "N/A")
            extra_headers.update({"user-api-key-user-id": user_api_key_user_id})

        verbose_proxy_logger.debug(f"extra_headers: {extra_headers}")
        return extra_headers

    async def _send_request(self, url, headers, data):
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
                verbose_proxy_logger.info(f"Zscaler AI Guard response: {json_response}")

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
            response = await self._send_request(
                zscaler_ai_guard_url, extra_headers, data
            )
            return self._handle_response(response, direction)
        except Exception as e:
            verbose_proxy_logger.error(f"{e}. Blocking request.")
            user_facing_error = self._create_user_facing_error(f"{str(e)})")
            # This exception will be caught by the proxy and returned to the user
            raise HTTPException(status_code=500, detail=user_facing_error)
