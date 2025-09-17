import json
import os
from typing import Any, Dict, List, Literal, Optional, Union

import litellm
from fastapi import HTTPException
from litellm._logging import verbose_proxy_logger
from litellm.caching.caching import DualCache
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
from litellm.proxy.guardrails.guardrail_helpers import (
    should_proceed_based_on_metadata,  # noqa: F401
)
from litellm.types.guardrails import GuardrailEventHooks

GUARDRAIL_NAME = "POINTGUARDAI"


class PointGuardAIGuardrail(CustomGuardrail):
    def __init__(
        self,
        api_base: str,
        api_key: str,
        api_email: str,
        org_id: str,
        policy_config_name: str,
        model_provider_name: Optional[str] = None,
        model_name: Optional[str] = None,
        guardrail_name: Optional[str] = None,
        event_hook: Optional[str] = None,
        default_on: Optional[bool] = False,
        **kwargs,
    ):
        self.async_handler = get_async_httpx_client(
            llm_provider=httpxSpecialProvider.GuardrailCallback
        )

        # Validate required parameters
        if not api_base:
            raise HTTPException(status_code=500, detail="Missing required parameter: api_base")
        if not api_key:
            raise HTTPException(status_code=500, detail="Missing required parameter: api_key")
        if not api_email:
            raise HTTPException(status_code=500, detail="Missing required parameter: api_email")
        if not org_id:
            raise HTTPException(status_code=500, detail="Missing required parameter: org_id")
        if not policy_config_name:
            raise HTTPException(status_code=500, detail="Missing required parameter: policy_config_name")

        self.pointguardai_api_base = api_base or os.getenv("POINTGUARDAI_API_URL_BASE")
        self.pointguardai_org_id = org_id or os.getenv("POINTGUARDAI_ORG_CODE", None)
        self.pointguardai_policy_config_name = policy_config_name or os.getenv(
            "POINTGUARDAI_CONFIG_NAME", None
        )
        self.pointguardai_api_key = api_key or os.getenv("POINTGUARDAI_API_KEY", None)
        self.pointguardai_api_email = api_email or os.getenv(
            "POINTGUARDAI_API_EMAIL", None
        )

        # Set default API base if not provided
        if not self.pointguardai_api_base:
            self.pointguardai_api_base = "https://api.appsoc.com"
            verbose_proxy_logger.debug(
                "PointGuardAI: Using default API base URL: %s",
                self.pointguardai_api_base,
            )

        if self.pointguardai_api_base and not self.pointguardai_api_base.endswith(
            "/policies/inspect"
        ):
            # If a base URL is provided, append the full path
            self.pointguardai_api_base = (
                self.pointguardai_api_base.rstrip("/")
                + "/aisec-rdc/api/v1/orgs/{{org}}/policies/inspect"
            )
            verbose_proxy_logger.debug(
                "PointGuardAI: Constructed full API URL: %s", self.pointguardai_api_base
            )

        # Configure headers with API key and email from kwargs or environment
        self.headers = {
            "X-appsoc-api-key": self.pointguardai_api_key,
            "X-appsoc-api-email": self.pointguardai_api_email,
            "Content-Type": "application/json",
        }

        # Fill in the API URL with the org ID
        if self.pointguardai_api_base and "{{org}}" in self.pointguardai_api_base:
            if self.pointguardai_org_id:
                self.pointguardai_api_base = self.pointguardai_api_base.replace(
                    "{{org}}", self.pointguardai_org_id
                )
            else:
                verbose_proxy_logger.warning(
                    "API URL contains {{org}} template but no org_id provided"
                )

        # Store new parameters
        self.model_provider_name = model_provider_name
        self.model_name = model_name
        
        # store kwargs as optional_params
        self.optional_params = kwargs

        # Set guardrail name
        self.guardrail_name = guardrail_name or GUARDRAIL_NAME
        self.event_hook = event_hook
        self.default_on = default_on

        # Debug logging for configuration
        verbose_proxy_logger.debug(
            "PointGuardAI: Configured with api_base: %s", self.pointguardai_api_base
        )
        verbose_proxy_logger.debug(
            "PointGuardAI: Configured with org_id: %s", self.pointguardai_org_id
        )
        verbose_proxy_logger.debug(
            "PointGuardAI: Configured with policy_config_name: %s",
            self.pointguardai_policy_config_name,
        )
        verbose_proxy_logger.debug(
            "PointGuardAI: Configured with api_email: %s", self.pointguardai_api_email
        )
        verbose_proxy_logger.debug(
            "PointGuardAI: Headers configured with API key: %s",
            "***" if self.pointguardai_api_key else "None",
        )

        super().__init__(**kwargs)

    def transform_messages(self, messages: List[dict]) -> List[dict]:
        """Transform messages to the format expected by PointGuard AI"""
        supported_openai_roles = ["system", "user", "assistant"]
        default_role = "user"  # for unsupported roles - e.g. tool
        new_messages = []
        for m in messages:
            if m.get("role", "") in supported_openai_roles:
                new_messages.append(m)
            else:
                new_messages.append(
                    {
                        "role": default_role,
                        **{key: value for key, value in m.items() if key != "role"},
                    }
                )
        return new_messages

    async def prepare_pointguard_ai_runtime_scanner_request(
        self, new_messages: List[dict], response_string: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Prepare the request data for PointGuard AI API"""
        try:
            # Validate required parameters
            if (
                not hasattr(self, "pointguardai_policy_config_name")
                or not self.pointguardai_policy_config_name
            ):
                verbose_proxy_logger.warning(
                    "PointGuardAI: Missing required policy configuration parameters"
                )
                return None

            data: dict[str, Any] = {
                "policyConfigName": self.pointguardai_policy_config_name,
                "input": [],
                "output": [],
            }

            # Add model_provider_name and model_name to the request data only if provided
            if hasattr(self, "model_provider_name") and self.model_provider_name:
                data["modelProviderName"] = self.model_provider_name
            if hasattr(self, "model_name") and self.model_name:
                data["modelName"] = self.model_name

            # Validate that we have either input messages or response string
            if not new_messages and not response_string:
                verbose_proxy_logger.warning(
                    "PointGuardAI: No input messages or response string provided"
                )
                return None

            if new_messages:
                data["input"] = new_messages
            if response_string:
                data["output"] = [{"role": "assistant", "content": response_string}]

            verbose_proxy_logger.debug("PointGuard AI request: %s", data)
            return data

        except Exception as e:
            verbose_proxy_logger.error(
                "Error preparing PointGuardAI request: %s", str(e)
            )
            return None

    async def make_pointguard_api_request(
        self,
        request_data: dict,
        new_messages: List[dict],
        response_string: Optional[str] = None,
    ):
        """Make the API request to PointGuard AI"""
        try:
            if not self.pointguardai_api_base:
                raise HTTPException(
                    status_code=500, detail="PointGuardAI API Base URL not configured"
                )

            pointguardai_data = (
                await self.prepare_pointguard_ai_runtime_scanner_request(
                    new_messages=new_messages, response_string=response_string
                )
            )

            if pointguardai_data is None:
                verbose_proxy_logger.warning(
                    "PointGuardAI: No data prepared for request"
                )
                return None

            pointguardai_data.update(
                self.get_guardrail_dynamic_request_body_params(
                    request_data=request_data
                )
            )

            _json_data = json.dumps(pointguardai_data)

            response = await self.async_handler.post(
                url=self.pointguardai_api_base,
                data=_json_data,
                headers=self.headers,
            )

            verbose_proxy_logger.debug(
                "PointGuard AI response status: %s", response.status_code
            )
            verbose_proxy_logger.debug("PointGuard AI response: %s", response.text)

            if response.status_code == 200:
                try:
                    response_data = response.json()
                except json.JSONDecodeError as e:
                    verbose_proxy_logger.error(
                        "Failed to parse PointGuardAI response JSON: %s", e
                    )
                    raise HTTPException(
                        status_code=500,
                        detail="Invalid JSON response from PointGuardAI",
                    )

                # Check if input or output sections are present
                input_section_present = False
                output_section_present = False
                if (
                    response_data.get("input") is not None
                    and response_data.get("input") != []
                    and response_data.get("input") != {}
                ):
                    input_section_present = True
                if (
                    response_data.get("output") is not None
                    and response_data.get("output") != []
                    and response_data.get("output") != {}
                ):
                    output_section_present = True

                # Check for blocking conditions
                input_blocked = (
                    response_data.get("input", {}).get("blocked", False)
                    if input_section_present
                    else False
                )
                output_blocked = (
                    response_data.get("output", {}).get("blocked", False)
                    if output_section_present
                    else False
                )

                if input_blocked or output_blocked:
                    verbose_proxy_logger.warning(
                        "PointGuardAI blocked the request: %s", response_data
                    )
                    # Get violations from the appropriate section
                    violations = []
                    if input_blocked and "input" in response_data:
                        if isinstance(response_data["input"], dict):
                            violations.extend(
                                response_data["input"].get("violations", [])
                            )
                        elif isinstance(response_data["input"], list):
                            # Handle case where violations are in content array
                            for item in response_data["input"]:
                                if isinstance(item, dict):
                                    violations.extend(item.get("violations", []))
                    if output_blocked and "output" in response_data:
                        if isinstance(response_data["output"], dict):
                            violations.extend(
                                response_data["output"].get("violations", [])
                            )
                        elif isinstance(response_data["output"], list):
                            # Handle case where violations are in content array
                            for item in response_data["output"]:
                                if isinstance(item, dict):
                                    violations.extend(item.get("violations", []))

                    raise HTTPException(
                        status_code=403,
                        detail={
                            "error": "Request blocked by PointGuardAI due to detected violations",
                            "violations": violations,
                            "pointguardai_response": response_data,
                        },
                    )

                # Check for modifications
                input_modified = (
                    response_data.get("input", {}).get("modified", False)
                    if input_section_present
                    else False
                )
                output_modified = (
                    response_data.get("output", {}).get("modified", False)
                    if output_section_present
                    else False
                )

                if input_modified or output_modified:
                    verbose_proxy_logger.info(
                        "PointGuardAI modified the request: %s", response_data
                    )
                    # Return the modified content
                    if input_modified and "input" in response_data:
                        return response_data["input"].get("content", [])
                    elif output_modified and "output" in response_data:
                        return response_data["output"].get("content", [])

                # No blocking or modification needed
                return None

            else:
                verbose_proxy_logger.error(
                    "PointGuardAI API request failed with status %s: %s",
                    response.status_code,
                    response.text,
                )
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"PointGuardAI API request failed: {response.text}",
                )

        except HTTPException:
            # Re-raise HTTP exceptions as-is
            raise
        except Exception as e:
            verbose_proxy_logger.error(
                "Unexpected error in PointGuardAI API request: %s",
                str(e),
                exc_info=True,
            )
            raise HTTPException(
                status_code=500,
                detail=f"Unexpected error in PointGuardAI integration: {str(e)}",
            )

    @log_guardrail_information
    async def async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: DualCache,
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
        Runs before the LLM API call
        Runs on only Input
        Use this if you want to MODIFY the input
        """
        from litellm.proxy.common_utils.callback_utils import (
            add_guardrail_to_applied_guardrails_header,
        )

        try:
            event_type: GuardrailEventHooks = GuardrailEventHooks.pre_call
            if self.should_run_guardrail(data=data, event_type=event_type) is not True:
                return data

            if call_type in [
                "embeddings",
                "audio_transcription",
                "image_generation",
                "rerank",
                "pass_through_endpoint",
            ]:
                verbose_proxy_logger.debug(
                    "PointGuardAI: Skipping unsupported call type: %s", call_type
                )
                return data

            new_messages: Optional[List[dict]] = None
            if "messages" in data and isinstance(data["messages"], list):
                new_messages = self.transform_messages(messages=data["messages"])

            if new_messages is not None:
                modified_content = await self.make_pointguard_api_request(
                    request_data=data,
                    new_messages=new_messages,
                )

                if modified_content is None:
                    verbose_proxy_logger.warning(
                        "PointGuardAI: No modifications made to the input messages. Returning original data."
                    )
                    return data

                add_guardrail_to_applied_guardrails_header(
                    request_data=data, guardrail_name=self.guardrail_name
                )
                if modified_content is not None and isinstance(modified_content, list):
                    if "messages" in data:
                        for i, message in enumerate(data["messages"]):
                            if "content" in message and isinstance(
                                message["content"], str
                            ):
                                # Update the content with the modified content
                                for mod in modified_content:
                                    if mod.get("originalContent") == message["content"]:
                                        # Handle null modifiedContent as content removal
                                        if mod.get("modifiedContent") is None:
                                            # Remove the message or set to empty
                                            data["messages"][i]["content"] = ""
                                        else:
                                            data["messages"][i]["content"] = mod.get(
                                                "modifiedContent", message["content"]
                                            )
                                        break
                    verbose_proxy_logger.info(
                        "PointGuardAI modified the input messages: %s", modified_content
                    )

                return data
            else:
                verbose_proxy_logger.warning(
                    "PointGuardAI: not running guardrail. No messages in data"
                )
                return data

        except HTTPException:
            # Re-raise HTTP exceptions (blocks/violations)
            raise
        except Exception as e:
            verbose_proxy_logger.error(
                "Error in PointGuardAI pre_call_hook: %s", str(e)
            )
            # Return original data on unexpected errors to avoid breaking the flow
            return data

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
        ],
    ):
        """
        Runs in parallel to LLM API call
        Runs on only Input

        This can NOT modify the input, only used to reject or accept a call before going to LLM API
        """
        from litellm.proxy.common_utils.callback_utils import (
            add_guardrail_to_applied_guardrails_header,
        )

        try:
            event_type: GuardrailEventHooks = GuardrailEventHooks.during_call
            if self.should_run_guardrail(data=data, event_type=event_type) is not True:
                return

            if call_type in [
                "embeddings",
                "audio_transcription",
                "image_generation",
                "rerank",
            ]:
                verbose_proxy_logger.debug(
                    "PointGuardAI: Skipping unsupported call type: %s", call_type
                )
                return data

            new_messages: Optional[List[dict]] = None
            if "messages" in data and isinstance(data["messages"], list):
                new_messages = self.transform_messages(messages=data["messages"])

            if new_messages is not None:
                await self.make_pointguard_api_request(
                    request_data=data,
                    new_messages=new_messages,
                )
                add_guardrail_to_applied_guardrails_header(
                    request_data=data, guardrail_name=self.guardrail_name
                )
            else:
                verbose_proxy_logger.warning(
                    "PointGuardAI: not running guardrail. No messages in data"
                )

        except HTTPException:
            # Re-raise HTTP exceptions (blocks/violations)
            raise
        except Exception as e:
            verbose_proxy_logger.error(
                "Error in PointGuardAI moderation_hook: %s", str(e)
            )
            # Don't raise on unexpected errors in moderation hook to avoid breaking the flow
            pass

    @log_guardrail_information
    async def async_post_call_success_hook(
        self,
        data: dict,
        user_api_key_dict: UserAPIKeyAuth,
        response: Union[litellm.ModelResponse, litellm.TextCompletionResponse],
    ) -> Optional[Union[Exception, str, dict]]:
        """
        Runs on response from LLM API call

        It can be used to reject a response or modify the response content
        """
        from litellm.proxy.common_utils.callback_utils import (
            add_guardrail_to_applied_guardrails_header,
        )

        try:
            """
            Use this for the post call moderation with Guardrails
            """
            event_type: GuardrailEventHooks = GuardrailEventHooks.post_call
            if self.should_run_guardrail(data=data, event_type=event_type) is not True:
                return response

            response_str: Optional[str] = convert_litellm_response_object_to_str(
                response
            )
            if response_str is not None:
                modified_content = await self.make_pointguard_api_request(
                    request_data=data,
                    response_string=response_str,
                    new_messages=data.get("messages", []),
                )

                add_guardrail_to_applied_guardrails_header(
                    request_data=data, guardrail_name=self.guardrail_name
                )

                if modified_content is not None and isinstance(modified_content, list):
                    # Import here to avoid circular imports
                    from litellm.utils import StreamingChoices

                    if isinstance(response, litellm.ModelResponse) and not isinstance(
                        response.choices[0], StreamingChoices
                    ):
                        # Handle non-streaming chat completions
                        if (
                            response.choices
                            and response.choices[0].message
                            and response.choices[0].message.content
                        ):
                            original_content = response.choices[0].message.content

                            # Find the matching modified content
                            for mod in modified_content:
                                if (
                                    isinstance(mod, dict)
                                    and mod.get("originalContent") == original_content
                                ):
                                    # Handle null modifiedContent as content removal
                                    if mod.get("modifiedContent") is None:
                                        response.choices[0].message.content = ""
                                    else:
                                        response.choices[0].message.content = mod.get(
                                            "modifiedContent", original_content
                                        )
                                    verbose_proxy_logger.info(
                                        "PointGuardAI modified the response content: %s",
                                        mod,
                                    )
                                    break

                            return response
                    else:
                        verbose_proxy_logger.debug(
                            "PointGuardAI: Unsupported response type for output modification: %s",
                            type(response),
                        )
                        return response
                else:
                    verbose_proxy_logger.debug(
                        "PointGuardAI: No modifications made to the response content"
                    )
                    return response
            else:
                verbose_proxy_logger.warning(
                    "PointGuardAI: No response string found for post-call validation"
                )
                return response

        except HTTPException:
            # Re-raise HTTP exceptions (blocks/violations)
            raise
        except Exception as e:
            verbose_proxy_logger.error(
                "Error in PointGuardAI post_call_success_hook: %s", str(e)
            )
            return response
