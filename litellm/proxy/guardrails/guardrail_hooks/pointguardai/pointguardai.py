import json
import os
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Type, Union

import httpx
from fastapi import HTTPException
from litellm._logging import verbose_proxy_logger
from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.llms.custom_httpx.http_handler import (
    get_async_httpx_client,
    httpxSpecialProvider,
)
from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import GenericGuardrailAPIInputs

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import (
        Logging as LiteLLMLoggingObj,
    )
    from litellm.types.proxy.guardrails.guardrail_hooks.base import GuardrailConfigModel

GUARDRAIL_NAME = "POINTGUARDAI"


class PointGuardAIGuardrail(CustomGuardrail):
    def __init__(
        self,
        api_base: str,
        api_key: str,
        org_code: str,
        policy_config_name: str,
        correlation_key: Optional[str] = None,
        guardrail_name: Optional[str] = None,
        event_hook: Optional[Union[str, List[str]]] = None,
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
        if not org_code:
            raise HTTPException(status_code=401, detail="Missing required parameter: org_code")
        if not policy_config_name:
            raise HTTPException(status_code=401, detail="Missing required parameter: policy_config_name")

        self.pointguardai_api_base = api_base or os.getenv("POINTGUARDAI_API_URL_BASE")
        self.pointguardai_org_code = org_code or os.getenv("POINTGUARDAI_ORG_CODE", "")
        self.pointguardai_policy_config_name = policy_config_name or os.getenv("POINTGUARDAI_CONFIG_NAME", "")
        self.pointguardai_api_key = api_key or os.getenv("POINTGUARDAI_API_KEY", "")
        self.pointguardai_correlation_key = correlation_key  # Optional parameter for request tracking

        # Set default API base if not provided
        if not self.pointguardai_api_base:
            self.pointguardai_api_base = "https://api.appsoc.com"
            verbose_proxy_logger.debug(
                "PointGuardAI: Using default API base URL: %s",
                self.pointguardai_api_base,
            )

        # Construct v2 API endpoints
        base_url = self.pointguardai_api_base.rstrip("/")
        self.input_endpoint = f"{base_url}/aisec-rdc-v2/api/v1/orgs/{self.pointguardai_org_code}/inspect/input"
        self.output_endpoint = f"{base_url}/aisec-rdc-v2/api/v1/orgs/{self.pointguardai_org_code}/inspect/output"
        
        verbose_proxy_logger.debug(
            "PointGuardAI v2: Input endpoint: %s", self.input_endpoint
        )
        verbose_proxy_logger.debug(
            "PointGuardAI v2: Output endpoint: %s", self.output_endpoint
        )

        # Configure headers with API key only (email not required in v2)
        self.headers = {
            "X-appsoc-api-key": self.pointguardai_api_key,
            "Content-Type": "application/json",
        }
        
        # store kwargs as optional_params
        self.optional_params = kwargs

        # Debug logging for configuration
        verbose_proxy_logger.debug(
            "PointGuardAI v2: Configured with org_code: %s", self.pointguardai_org_code
        )
        verbose_proxy_logger.debug(
            "PointGuardAI v2: Configured with policy_config_name: %s",
            self.pointguardai_policy_config_name,
        )
        verbose_proxy_logger.debug(
            "PointGuardAI v2: Correlation key: %s",
            self.pointguardai_correlation_key or "(auto-generated)",
        )
        verbose_proxy_logger.debug(
            "PointGuardAI v2: API key configured: %s",
            "Yes" if self.pointguardai_api_key else "No",
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
        """Prepare the request data for PointGuard AI v2 API"""
        try:
            # Validate required parameters
            if (
                not hasattr(self, "pointguardai_policy_config_name")
                or not self.pointguardai_policy_config_name
            ):
                verbose_proxy_logger.warning(
                    "PointGuardAI v2: Missing required policy configuration parameters"
                )
                return None

            # v2 API uses policyName instead of configName
            data: dict[str, Any] = {
                "policyName": self.pointguardai_policy_config_name,
            }

            # Add optional correlationKey if provided
            if self.pointguardai_correlation_key:
                data["correlationKey"] = self.pointguardai_correlation_key

            # Validate that we have either input messages or response string
            if not new_messages and not response_string:
                verbose_proxy_logger.warning(
                    "PointGuardAI v2: No input messages or response string provided"
                )
                return None

            # Only add input field if there are input messages
            if new_messages:
                data["input"] = new_messages
            
            # Only add output field if there's a response string (for output endpoint)
            if response_string:
                data["output"] = [{"role": "assistant", "content": response_string}]

            verbose_proxy_logger.debug("PointGuardAI v2 request: %s", data)
            return data

        except Exception as e:
            verbose_proxy_logger.error(
                "Error preparing PointGuardAI v2 request: %s", str(e)
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
        """Extract violations from blocked sections in v2 format"""
        violations = []
        
        # Helper function to extract from content items
        def extract_from_content(content_items: List[dict]) -> List[dict]:
            all_violations = []
            for content_item in content_items:
                if not isinstance(content_item, dict):
                    continue
                    
                # Extract DLP violations
                dlp_violations = content_item.get("dlpViolations", [])
                for dlp in dlp_violations:
                    all_violations.append({
                        "type": "DLP",
                        "name": dlp.get("name", "Unknown"),
                        "dlp_data_type_id": dlp.get("dlpDataTypeId"),
                        "action": dlp.get("action", "UNKNOWN"),
                        "categories": dlp.get("categories", []),
                        "match_count": dlp.get("matchCount", 0)
                    })
                
                # Extract AI violations
                ai_violations = content_item.get("aiViolations", [])
                for ai in ai_violations:
                    all_violations.append({
                        "type": "AI_THREAT",
                        "name": ai.get("name", "Unknown"),
                        "ai_threat_category_id": ai.get("aiThreatCategoryId"),
                        "threat_type": ai.get("type", "UNKNOWN"),
                        "action": ai.get("action", "UNKNOWN")
                    })
            return all_violations
        
        # Extract from input if blocked
        if input_blocked and "input" in response_data:
            input_content = response_data["input"].get("content", [])
            if isinstance(input_content, list):
                violations.extend(extract_from_content(input_content))
        
        # Extract from output if blocked
        if output_blocked and "output" in response_data:
            output_content = response_data["output"].get("content", [])
            if isinstance(output_content, list):
                violations.extend(extract_from_content(output_content))
        
        return violations

    def _create_violation_details(self, violations: List[dict]) -> List[dict]:
        """Create detailed violation information for v2 format"""
        violation_details = []
        for violation in violations:
            if not isinstance(violation, dict):
                continue
                
            violation_type = violation.get("type", "UNKNOWN")
            
            if violation_type == "DLP":
                # DLP violation format
                categories = violation.get("categories", [])
                category_names = [cat.get("name", cat.get("code", "")) for cat in categories if isinstance(cat, dict)]
                
                violation_details.append({
                    "type": "DLP",
                    "name": violation.get("name", "Unknown DLP"),
                    "action": violation.get("action", "UNKNOWN"),
                    "categories": category_names,
                    "match_count": violation.get("match_count", 0),
                    "dlp_data_type_id": violation.get("dlp_data_type_id")
                })
            elif violation_type == "AI_THREAT":
                # AI threat violation format
                violation_details.append({
                    "type": "AI_THREAT",
                    "name": violation.get("name", "Unknown Threat"),
                    "threat_type": violation.get("threat_type", "UNKNOWN"),
                    "action": violation.get("action", "UNKNOWN"),
                    "ai_threat_category_id": violation.get("ai_threat_category_id")
                })
            else:
                # Generic violation
                violation_details.append(violation)
        
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
        """Handle content modifications in v2 format"""
        verbose_proxy_logger.info(
            "PointGuardAI v2 modification detected - Input: %s, Output: %s", 
            input_modified, output_modified
        )
        
        # Extract modified content from content items
        def extract_modified_content(content_items: List[dict]) -> List[dict]:
            modified_messages = []
            for item in content_items:
                if not isinstance(item, dict):
                    continue
                
                # Use modifiedContent if present, otherwise use originalContent
                content = item.get("modifiedContent") or item.get("originalContent", "")
                role = item.get("role", "user")
                
                modified_messages.append({
                    "role": role,
                    "content": content
                })
                
                # Log if content was actually modified
                if item.get("modifiedContent"):
                    verbose_proxy_logger.info(
                        "PointGuardAI v2: Content modified for role '%s'",
                        role
                    )
            
            return modified_messages
        
        # Handle input modifications
        if input_modified and "input" in response_data:
            input_data = response_data["input"]
            if isinstance(input_data, dict) and "content" in input_data:
                content_items = input_data.get("content", [])
                if isinstance(content_items, list):
                    verbose_proxy_logger.info(
                        "PointGuardAI v2 input modifications: %d items", 
                        len(content_items)
                    )
                    return extract_modified_content(content_items)
        
        # Handle output modifications
        elif output_modified and "output" in response_data:
            output_data = response_data["output"]
            if isinstance(output_data, dict) and "content" in output_data:
                content_items = output_data.get("content", [])
                if isinstance(content_items, list):
                    verbose_proxy_logger.info(
                        "PointGuardAI v2 output modifications: %d items", 
                        len(content_items)
                    )
                    return extract_modified_content(content_items)
        
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
        """Make the API request to PointGuardAI v2 API"""
        try:
            # Select appropriate endpoint based on whether we have output
            # pre_call mode: use input endpoint
            # post_call mode: use output endpoint
            if response_string:
                endpoint = self.output_endpoint
                verbose_proxy_logger.debug("PointGuardAI v2: Using output endpoint")
            else:
                endpoint = self.input_endpoint
                verbose_proxy_logger.debug("PointGuardAI v2: Using input endpoint")

            pointguardai_data = (
                await self.prepare_pointguard_ai_runtime_scanner_request(
                    new_messages=new_messages, response_string=response_string
                )
            )

            if pointguardai_data is None:
                verbose_proxy_logger.warning(
                    "PointGuardAI v2: No data prepared for request"
                )
                return None

            pointguardai_data.update(
                self.get_guardrail_dynamic_request_body_params(
                    request_data=request_data
                )
            )

            _json_data = json.dumps(pointguardai_data)

            verbose_proxy_logger.debug(
                "PointGuardAI v2: Sending request to %s", endpoint
            )

            response = await self.async_handler.post(
                url=endpoint,
                data=_json_data,
                headers=self.headers,
            )

            verbose_proxy_logger.debug(
                "PointGuardAI v2 response status: %s", response.status_code
            )
            verbose_proxy_logger.debug("PointGuardAI v2 response: %s", response.text)
            
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

    async def apply_guardrail(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict,
        input_type: Literal["request", "response"],
        logging_obj: Optional["LiteLLMLoggingObj"] = None,
    ) -> GenericGuardrailAPIInputs:
        """
        Apply PointGuardAI guardrail to the given inputs.

        This method is called by the unified guardrail system for both
        input (request) and output (response) validation.

        Args:
            inputs: Dictionary containing:
                - texts: List of texts to check
                - structured_messages: Structured messages from the request (pre-call only)
                - tool_calls: Tool calls if present
            request_data: The original request data
            input_type: "request" for pre-call, "response" for post-call
            logging_obj: Optional logging object

        Returns:
            GenericGuardrailAPIInputs - unchanged or modified if allowed through

        Raises:
            HTTPException: If content is blocked
        """
        try:
            # Extract messages from structured_messages or request_data
            messages: Optional[List[AllMessageValues]] = inputs.get("structured_messages")
            if not messages:
                messages = request_data.get("messages")

            # For response (post_call), get output text
            output: Optional[str] = None
            texts = inputs.get("texts", [])

            if input_type == "response":
                # Extract output text from response
                output = texts[-1] if texts else None

            # Transform messages if available
            new_messages = []
            if messages:
                new_messages = self.transform_messages(messages=messages)

            # Make request to PointGuard AI API
            modified_content = await self.make_pointguard_api_request(
                request_data=request_data,
                new_messages=new_messages,
                response_string=output,
            )

            # If no modifications, return inputs as-is
            if modified_content is None:
                verbose_proxy_logger.debug(
                    "PointGuardAI: No modifications made to content"
                )
                return inputs

            # Apply modifications based on input_type
            if isinstance(modified_content, list) and len(modified_content) > 0:
                if input_type == "request":
                    # Modify request messages
                    verbose_proxy_logger.info(
                        "PointGuardAI applying %d modifications to input messages",
                        len(modified_content)
                    )

                    if messages:
                        modifications_applied = 0
                        for i, message in enumerate(messages):
                            content = message.get("content")
                            if content and isinstance(content, str):
                                # Find matching modification
                                for mod in modified_content:
                                    if mod.get("originalContent") == content:
                                        # Handle null modifiedContent as content removal
                                        if mod.get("modifiedContent") is None:
                                            messages[i]["content"] = ""
                                            verbose_proxy_logger.info(
                                                "PointGuardAI removed content from message %d", i
                                            )
                                        else:
                                            messages[i]["content"] = mod.get(
                                                "modifiedContent", content
                                            )
                                            verbose_proxy_logger.info(
                                                "PointGuardAI modified message %d", i
                                            )
                                        modifications_applied += 1
                                        break

                        if modifications_applied > 0:
                            verbose_proxy_logger.info(
                                "PointGuardAI successfully applied %d/%d modifications",
                                modifications_applied, len(modified_content)
                            )

                        # Return modified inputs
                        return GenericGuardrailAPIInputs(
                            texts=texts,
                            structured_messages=messages
                        )

                elif input_type == "response":
                    # Modify response text
                    verbose_proxy_logger.info(
                        "PointGuardAI applying %d modifications to response content",
                        len(modified_content)
                    )

                    if output:
                        # Find matching modification for output
                        for mod in modified_content:
                            if mod.get("originalContent") == output:
                                # Handle null modifiedContent as content removal
                                if mod.get("modifiedContent") is None:
                                    modified_output = ""
                                    verbose_proxy_logger.info(
                                        "PointGuardAI removed response content"
                                    )
                                else:
                                    modified_output = mod.get("modifiedContent", output)
                                    verbose_proxy_logger.info(
                                        "PointGuardAI modified response content"
                                    )

                                # Update texts with modified content
                                new_texts = texts.copy()
                                new_texts[-1] = modified_output

                                return GenericGuardrailAPIInputs(
                                    texts=new_texts
                                )

            return inputs

        except HTTPException:
            # Re-raise HTTP exceptions (blocks/violations)
            raise
        except Exception as e:
            verbose_proxy_logger.error(
                "Error in PointGuardAI apply_guardrail: %s", str(e)
            )
            # Return original inputs on unexpected errors to avoid breaking the flow
            return inputs

    @staticmethod
    def get_config_model() -> Optional[Type["GuardrailConfigModel"]]:
        from litellm.types.proxy.guardrails.guardrail_hooks.pointguardai import (
            PointGuardAIGuardrailConfigModel,
        )
        return PointGuardAIGuardrailConfigModel
