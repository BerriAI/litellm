import json
import os
from typing import Any, Dict, List, Literal, Optional, Union

import httpx
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
from litellm.types.guardrails import GuardrailEventHooks, PiiEntityType

GUARDRAIL_NAME = "POINTGUARDAI"


class PointGuardAIGuardrail(CustomGuardrail):
    def __init__(
        self,
        api_base: str,
        api_key: str,
        api_email: str,
        org_code: str,
        policy_config_name: str,
        model_provider_name: Optional[str] = None,
        model_name: Optional[str] = None,
        guardrail_name: Optional[str] = None,
        event_hook: Optional[Union[GuardrailEventHooks, List[GuardrailEventHooks], Any]] = None,
        default_on: bool = False,
        **kwargs,
    ):
        self.async_handler = get_async_httpx_client(
            llm_provider=httpxSpecialProvider.GuardrailCallback
        )

        # Validate required parameters
        if not api_base:
            raise HTTPException(status_code=401, detail="Missing required parameter: api_base")
        if not api_key:
            raise HTTPException(status_code=401, detail="Missing required parameter: api_key")
        if not api_email:
            raise HTTPException(status_code=401, detail="Missing required parameter: api_email")
        if not org_code:
            raise HTTPException(status_code=401, detail="Missing required parameter: org_code")
        if not policy_config_name:
            raise HTTPException(status_code=401, detail="Missing required parameter: policy_config_name")

        self.pointguardai_api_base = api_base or os.getenv("POINTGUARDAI_API_URL_BASE")
        self.pointguardai_org_code = org_code or os.getenv("POINTGUARDAI_ORG_CODE", "")
        self.pointguardai_policy_config_name = policy_config_name or os.getenv("POINTGUARDAI_CONFIG_NAME", "")
        self.pointguardai_api_key = api_key or os.getenv("POINTGUARDAI_API_KEY", "")
        self.pointguardai_api_email = api_email or os.getenv("POINTGUARDAI_API_EMAIL", "")

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
            if self.pointguardai_org_code:
                self.pointguardai_api_base = self.pointguardai_api_base.replace(
                    "{{org}}", self.pointguardai_org_code
                )
            else:
                verbose_proxy_logger.warning(
                    "API URL contains {{org}} template but no org_code provided"
                )

        # Store new parameters
        self.model_provider_name = model_provider_name
        self.model_name = model_name
        
        # store kwargs as optional_params
        self.optional_params = kwargs

        # Debug logging for configuration
        verbose_proxy_logger.debug(
            "PointGuardAI: Configured with api_base: %s", self.pointguardai_api_base
        )
        verbose_proxy_logger.debug(
            "PointGuardAI: Configured with org_code: %s", self.pointguardai_org_code
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

        super().__init__(
            guardrail_name=guardrail_name or GUARDRAIL_NAME,
            event_hook=event_hook,
            default_on=default_on,
            **kwargs
        )

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
                "configName": self.pointguardai_policy_config_name,
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

            # Only add input field if there are input messages
            if new_messages:
                data["input"] = new_messages
            
            # Only add output field if there's a response string
            if response_string:
                data["output"] = [{"role": "assistant", "content": response_string}]

            verbose_proxy_logger.debug("PointGuard AI request: %s", data)
            return data

        except Exception as e:
            verbose_proxy_logger.error(
                "Error preparing PointGuardAI request: %s", str(e)
            )
            return None

    def _check_sections_present(self, response_data: dict, new_messages: List[dict], response_string: Optional[str]) -> tuple[bool, bool]:
        """Check if input or output sections are present in response"""
        input_section_present = (
            bool(new_messages and len(new_messages) > 0 and
                 response_data.get("input") is not None and
                 response_data.get("input") != [] and
                 response_data.get("input") != {})
        )

        output_section_present = (
            bool(response_string and
                 response_data.get("output") is not None and
                 response_data.get("output") != [] and
                 response_data.get("output") != {})
        )

        return input_section_present, output_section_present

    def _extract_status_flags(self, response_data: dict, input_section_present: bool, output_section_present: bool) -> tuple[bool, bool, bool, bool]:
        """Extract blocking and modification flags from response"""
        input_blocked = response_data.get("input", {}).get("blocked", False) if input_section_present else False
        output_blocked = response_data.get("output", {}).get("blocked", False) if output_section_present else False
        input_modified = response_data.get("input", {}).get("modified", False) if input_section_present else False
        output_modified = response_data.get("output", {}).get("modified", False) if output_section_present else False
        
        return input_blocked, output_blocked, input_modified, output_modified

    def _extract_violations(self, response_data: dict, input_blocked: bool, output_blocked: bool) -> List[dict]:
        """Extract violations from blocked sections"""
        violations = []
        if input_blocked and "input" in response_data:
            input_content = response_data["input"].get("content", [])
            if isinstance(input_content, list):
                for content_item in input_content:
                    if isinstance(content_item, dict):
                        violations.extend(content_item.get("violations", []))
        if output_blocked and "output" in response_data:
            output_content = response_data["output"].get("content", [])
            if isinstance(output_content, list):
                for content_item in output_content:
                    if isinstance(content_item, dict):
                        violations.extend(content_item.get("violations", []))
        return violations

    def _create_violation_details(self, violations: List[dict]) -> List[dict]:
        """Create detailed violation information"""
        violation_details = []
        for violation in violations:
            if isinstance(violation, dict):
                categories = violation.get("categories", [])
                violation_details.append({
                    "severity": violation.get("severity", "UNKNOWN"),
                    "scanner": violation.get("scanner", "unknown"),
                    "inspector": violation.get("inspector", "unknown"),
                    "categories": categories,
                    "confidenceScore": violation.get("confidenceScore", 0.0),
                    "mode": violation.get("mode", "UNKNOWN")
                })
        return violation_details

    def _handle_blocked_request(self, violation_details: List[dict]) -> None:
        """Handle blocked request by raising HTTPException"""
        error_message = "Content blocked by PointGuardAI policy"
        
        verbose_proxy_logger.warning(
            "PointGuardAI blocking request with violations: %s", violation_details
        )
        
        pointguardai_response = {
            "action": "block",
            "revised_prompt": None,
            "revised_response": error_message,
            "explain_log": violation_details
        }
        
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Violated PointGuardAI policy",
                "pointguardai_response": pointguardai_response,
            }
        )

    def _handle_modifications(self, response_data: dict, input_modified: bool, output_modified: bool) -> Optional[List[dict]]:
        """Handle content modifications"""
        verbose_proxy_logger.info(
            "PointGuardAI modification detected - Input: %s, Output: %s", 
            input_modified, output_modified
        )
        
        if input_modified and "input" in response_data:
            input_data = response_data["input"]
            if isinstance(input_data, dict) and "content" in input_data:
                verbose_proxy_logger.info(
                    "PointGuardAI input modifications: %s", 
                    input_data.get("content", [])
                )
            return response_data["input"].get("content", [])
        elif output_modified and "output" in response_data:
            output_data = response_data["output"]
            if isinstance(output_data, dict) and "content" in output_data:
                verbose_proxy_logger.info(
                    "PointGuardAI output modifications: %s", 
                    output_data.get("content", [])
                )
            return response_data["output"].get("content", [])
        return None

    def _handle_http_status_error(self, e: httpx.HTTPStatusError) -> None:
        """Handle HTTP status errors"""
        status_code = e.response.status_code
        response_text = e.response.text if hasattr(e.response, 'text') else str(e)
        
        verbose_proxy_logger.error(
            "PointGuardAI API HTTP error %s: %s",
            status_code,
            response_text,
        )
        
        error_messages = {
            401: "PointGuardAI authentication failed: Invalid API credentials",
            400: "PointGuardAI bad request: Invalid configuration or parameters",
            403: "PointGuardAI access denied: Insufficient permissions",
            404: "PointGuardAI resource not found: Invalid endpoint or organization"
        }
        
        detail = error_messages.get(status_code, f"PointGuardAI API error ({status_code}): {response_text}")
        raise HTTPException(status_code=status_code, detail=detail)

    def _handle_network_errors(self, e: Union[httpx.ConnectError, httpx.TimeoutException, httpx.RequestError]) -> None:
        """Handle network-related errors"""
        if isinstance(e, httpx.TimeoutException):
            verbose_proxy_logger.error("PointGuardAI timeout error: %s", str(e))
            raise HTTPException(
                status_code=504,
                detail="PointGuardAI request timeout: API request took too long to complete",
            )
        else:
            verbose_proxy_logger.error("PointGuardAI connection error: %s", str(e))
            raise HTTPException(
                status_code=503,
                detail="PointGuardAI service unavailable: Cannot connect to API endpoint. Please check the API URL configuration.",
            )

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
            
            # Raise HTTPStatusError for 4xx and 5xx responses
            response.raise_for_status()

            # If we reach here, response.status_code is 2xx (success)
            if response.status_code == 200:
                try:
                    response_data = response.json()
                except json.JSONDecodeError as e:
                    verbose_proxy_logger.error("Failed to parse PointGuardAI response JSON: %s", e)
                    raise HTTPException(status_code=500, detail="Invalid JSON response from PointGuardAI")

                # Check sections and extract status flags
                input_section_present, output_section_present = self._check_sections_present(
                    response_data, new_messages, response_string
                )
                input_blocked, output_blocked, input_modified, output_modified = self._extract_status_flags(
                    response_data, input_section_present, output_section_present
                )

                verbose_proxy_logger.info(
                    "PointGuardAI API response analysis - Input: blocked=%s, modified=%s | Output: blocked=%s, modified=%s",
                    input_blocked, input_modified, output_blocked, output_modified
                )
                verbose_proxy_logger.debug("PointGuardAI full response data: %s", response_data)

                # Priority rule: If both blocked=true AND modified=true, BLOCK takes precedence
                if input_blocked or output_blocked:
                    verbose_proxy_logger.warning(
                        "PointGuardAI blocked the request - Input blocked: %s, Output blocked: %s", 
                        input_blocked, output_blocked
                    )
                    
                    violations = self._extract_violations(response_data, input_blocked, output_blocked)
                    violation_details = self._create_violation_details(violations)
                    self._handle_blocked_request(violation_details)

                # Check for modifications only if not blocked
                elif input_modified or output_modified:
                    return self._handle_modifications(response_data, input_modified, output_modified)

                # No blocking or modification needed
                verbose_proxy_logger.debug("PointGuardAI: No blocking or modifications required")
                return None

        except HTTPException:
            # Re-raise HTTP exceptions as-is
            raise
        except httpx.HTTPStatusError as e:
            self._handle_http_status_error(e)
        except (httpx.ConnectError, httpx.TimeoutException, httpx.RequestError) as e:
            self._handle_network_errors(e)
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
                # For pre_call hook, only send input messages (no response)
                modified_content = await self.make_pointguard_api_request(
                    request_data=data,
                    new_messages=new_messages,
                    response_string=None,  # Explicitly no response for pre_call
                )

                if modified_content is None:
                    verbose_proxy_logger.debug(
                        "PointGuardAI: No modifications made to the input messages. Returning original data."
                    )
                    return data

                add_guardrail_to_applied_guardrails_header(
                    request_data=data, guardrail_name=self.guardrail_name
                )
                if modified_content is not None and isinstance(modified_content, list):
                    verbose_proxy_logger.info(
                        "PointGuardAI applying %d modifications to input messages", 
                        len(modified_content)
                    )
                    
                    modifications_applied = 0
                    if "messages" in data:
                        for i, message in enumerate(data["messages"]):
                            if "content" in message and isinstance(
                                message["content"], str
                            ):
                                # Update the content with the modified content
                                for mod in modified_content:
                                    if mod.get("originalContent") == message["content"]:
                                        original_preview = message["content"][:100] + "..." if len(message["content"]) > 100 else message["content"]
                                        
                                        # Handle null modifiedContent as content removal
                                        if mod.get("modifiedContent") is None:
                                            # Remove the message or set to empty
                                            data["messages"][i]["content"] = ""
                                            verbose_proxy_logger.info(
                                                "PointGuardAI removed content from message %d: '%s' -> [REMOVED]", 
                                                i, original_preview
                                            )
                                        else:
                                            modified_preview = mod.get("modifiedContent", "")[:100] + "..." if len(mod.get("modifiedContent", "")) > 100 else mod.get("modifiedContent", "")
                                            data["messages"][i]["content"] = mod.get(
                                                "modifiedContent", message["content"]
                                            )
                                            verbose_proxy_logger.info(
                                                "PointGuardAI modified message %d: '%s' -> '%s'", 
                                                i, original_preview, modified_preview
                                            )
                                        modifications_applied += 1
                                        break
                    
                    if modifications_applied == 0:
                        verbose_proxy_logger.warning(
                            "PointGuardAI: Received modifications but no content matched for application: %s", 
                            modified_content
                        )
                    else:
                        verbose_proxy_logger.info(
                            "PointGuardAI successfully applied %d/%d modifications to input messages", 
                            modifications_applied, len(modified_content)
                        )

                return data
            else:
                verbose_proxy_logger.debug(
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
                # For during_call hook, only send input messages (no response)
                modified_content = await self.make_pointguard_api_request(
                    request_data=data,
                    new_messages=new_messages,
                    response_string=None,  # Explicitly no response for during_call
                )
                
                if modified_content is not None:
                    verbose_proxy_logger.info(
                        "PointGuardAI detected modifications during during_call hook: %s", 
                        modified_content
                    )
                    verbose_proxy_logger.warning(
                        "PointGuardAI: Content was modified but during_call hook cannot apply changes. Consider using pre_call mode instead."
                    )
                else:
                    verbose_proxy_logger.debug(
                        "PointGuardAI during_call hook: No modifications detected"
                    )
                
                add_guardrail_to_applied_guardrails_header(
                    request_data=data, guardrail_name=self.guardrail_name
                )
            else:
                verbose_proxy_logger.debug(
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

        return None

    @log_guardrail_information
    async def async_post_call_success_hook(
        self,
        data: dict,
        user_api_key_dict: UserAPIKeyAuth,
        response: Union[litellm.ModelResponse, litellm.TextCompletionResponse],
    ):
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
                # For post_call hook, send both input messages and output response
                new_messages = []
                if "messages" in data and isinstance(data["messages"], list):
                    new_messages = self.transform_messages(messages=data["messages"])
                
                modified_content = await self.make_pointguard_api_request(
                    request_data=data,
                    new_messages=new_messages,
                    response_string=response_str,
                )

                add_guardrail_to_applied_guardrails_header(
                    request_data=data, guardrail_name=self.guardrail_name
                )

                if modified_content is not None and isinstance(modified_content, list):
                    verbose_proxy_logger.info(
                        "PointGuardAI attempting to apply %d modifications to response content", 
                        len(modified_content)
                    )
                    
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
                            modifications_applied = False

                            # Find the matching modified content
                            for mod in modified_content:
                                if (
                                    isinstance(mod, dict)
                                    and mod.get("originalContent") == original_content
                                ):
                                    original_preview = original_content[:100] + "..." if len(original_content) > 100 else original_content
                                    
                                    # Handle null modifiedContent as content removal
                                    if mod.get("modifiedContent") is None:
                                        response.choices[0].message.content = ""
                                        verbose_proxy_logger.info(
                                            "PointGuardAI removed response content: '%s' -> [REMOVED]",
                                            original_preview
                                        )
                                    else:
                                        modified_preview = mod.get("modifiedContent", "")[:100] + "..." if len(mod.get("modifiedContent", "")) > 100 else mod.get("modifiedContent", "")
                                        response.choices[0].message.content = mod.get(
                                            "modifiedContent", original_content
                                        )
                                        verbose_proxy_logger.info(
                                            "PointGuardAI modified response content: '%s' -> '%s'",
                                            original_preview, modified_preview
                                        )
                                    modifications_applied = True
                                    break
                            
                            if not modifications_applied:
                                verbose_proxy_logger.warning(
                                    "PointGuardAI: Received response modifications but no content matched: %s", 
                                    modified_content
                                )

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
                verbose_proxy_logger.debug(
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

    async def apply_guardrail(
        self,
        text: str,
        language: Optional[str] = None,
        entities: Optional[List[PiiEntityType]] = None,
    ) -> str:
        """
        Apply PointGuard AI guardrail to the given text.
        
        Args:
            text: The text to analyze and potentially modify
            language: Optional language parameter (not used by PointGuard AI)
            entities: Optional entities parameter (not used by PointGuard AI)
            
        Returns:
            str: The original or modified text based on PointGuard AI's response
            
        Raises:
            HTTPException: If content is blocked by PointGuard AI policy
        """
        try:
            # Transform text into message format that PointGuard AI expects
            new_messages = [{"role": "user", "content": text}]
            
            # Make request to PointGuard AI API (input only, no response)
            modified_content = await self.make_pointguard_api_request(
                request_data={},  # Empty request data for standalone usage
                new_messages=new_messages,
                response_string=None,  # No response for input-only analysis
            )
            
            # If no modifications returned, return original text
            if modified_content is None:
                verbose_proxy_logger.debug(
                    "PointGuardAI apply_guardrail: No modifications made to input text"
                )
                return text
            
            # Apply modifications if present
            if isinstance(modified_content, list) and len(modified_content) > 0:
                verbose_proxy_logger.info(
                    "PointGuardAI apply_guardrail: Applying %d modifications to input text", 
                    len(modified_content)
                )
                
                # Find matching modification for the input text
                for mod in modified_content:
                    if isinstance(mod, dict) and mod.get("originalContent") == text:
                        # Handle null modifiedContent as content removal
                        if mod.get("modifiedContent") is None:
                            verbose_proxy_logger.info(
                                "PointGuardAI apply_guardrail: Content removed by policy"
                            )
                            return ""
                        else:
                            modified_text = mod.get("modifiedContent", text)
                            verbose_proxy_logger.info(
                                "PointGuardAI apply_guardrail: Content modified by policy"
                            )
                            return modified_text
                
                # If no exact match found, log warning and return original
                verbose_proxy_logger.warning(
                    "PointGuardAI apply_guardrail: Received modifications but no content matched: %s", 
                    modified_content
                )
                return text
            
            return text
            
        except HTTPException:
            # Re-raise HTTP exceptions (blocks/violations) as-is
            raise
        except Exception as e:
            verbose_proxy_logger.error(
                "Error in PointGuardAI apply_guardrail: %s", str(e)
            )
            # Return original text on unexpected errors to avoid breaking the flow
            return text
