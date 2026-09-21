import json
from typing import TYPE_CHECKING, Final, Literal, NoReturn

import httpx
from fastapi import HTTPException

from litellm._logging import verbose_proxy_logger
from litellm.exceptions import GuardrailRaisedException
from litellm.integrations.custom_guardrail import (
    CustomGuardrail,
    log_guardrail_information,
)
from litellm.llms.base_llm.guardrail_translation.utils import (
    effective_scan_only_tool_results_for_guardrail,
    effective_skip_system_message_for_guardrail,
    effective_skip_tool_message_for_guardrail,
    role_out_of_guardrail_scope,
)
from litellm.llms.custom_httpx.http_handler import (
    AsyncHTTPHandler,
    get_async_httpx_client,
    httpxSpecialProvider,
)
from litellm.types.guardrails import GuardrailEventHooks, Mode
from litellm.types.utils import GenericGuardrailAPIInputs, OpenAIObject

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import (
        Logging as LiteLLMLoggingObj,
    )
    from litellm.types.proxy.guardrails.guardrail_hooks.base import GuardrailConfigModel

GUARDRAIL_NAME: Final = "POINTGUARDAI"
DEFAULT_POINTGUARDAI_API_BASE: Final = "https://api.appsoc.com"


class _PointGuardAIUnavailableError(HTTPException):
    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(status_code=status_code, detail=detail)


class PointGuardAIGuardrail(CustomGuardrail):
    @classmethod
    def get_supported_event_hooks(cls) -> list[GuardrailEventHooks]:  # mutable-ok: LiteLLM hook API requires a list
        return [  # mutable-ok: LiteLLM hook API requires a list
            GuardrailEventHooks.pre_call,
            GuardrailEventHooks.post_call,
        ]

    def __init__(
        self,
        api_base: str | None = None,
        api_key: str | None = None,
        org_code: str | None = None,
        policy_config_name: str | None = None,
        unreachable_fallback: Literal["fail_closed", "fail_open"] = "fail_closed",
        guardrail_name: str | None = None,
        event_hook: GuardrailEventHooks  # mutable-ok: inherited LiteLLM hook contract
        | list[GuardrailEventHooks]  # mutable-ok: inherited LiteLLM hook contract
        | Mode
        | None = None,  # mutable-ok: inherited LiteLLM hook contract
        default_on: bool = False,
        async_handler: AsyncHTTPHandler | None = None,
        **kwargs: object,  # kwargs-ok: forwarded to the inherited LiteLLM guardrail constructor
    ) -> None:
        self.pointguardai_api_base = api_base or DEFAULT_POINTGUARDAI_API_BASE
        self.pointguardai_org_code = org_code or ""
        self.pointguardai_policy_config_name = policy_config_name or ""
        self.pointguardai_api_key = api_key or ""
        self.unreachable_fallback: Literal["fail_closed", "fail_open"] = unreachable_fallback
        self.streaming_transform_mode: Literal["block_only", "incremental_diff"] = "incremental_diff"
        self.streaming_end_of_stream_only: bool = True

        # Validate required parameters
        if not self.pointguardai_api_key:
            raise HTTPException(status_code=401, detail="Missing required parameter: api_key")
        if not self.pointguardai_org_code:
            raise HTTPException(status_code=401, detail="Missing required parameter: org_code")
        if not self.pointguardai_policy_config_name:
            raise HTTPException(status_code=401, detail="Missing required parameter: policy_config_name")

        supported_event_hooks: Final = self.get_supported_event_hooks()
        self._validate_event_hook(event_hook, supported_event_hooks)

        self.async_handler = (
            async_handler
            if async_handler is not None
            else get_async_httpx_client(llm_provider=httpxSpecialProvider.GuardrailCallback)
        )

        # Construct API endpoints
        base_url: Final = self.pointguardai_api_base.rstrip("/")
        self.input_endpoint = f"{base_url}/aisec-rdc-v2/api/v1/orgs/{self.pointguardai_org_code}/inspect/input"
        self.output_endpoint = f"{base_url}/aisec-rdc-v2/api/v1/orgs/{self.pointguardai_org_code}/inspect/output"

        self.headers = {  # mutable-ok: HTTP client headers contract requires a dictionary
            "X-appsoc-api-key": self.pointguardai_api_key,
            "Content-Type": "application/json",
        }

        # store kwargs as optional_params
        self.optional_params = kwargs

        verbose_proxy_logger.debug(
            "PointGuardAI configured: api_base_present=%s org_code_present=%s policy_present=%s api_key_present=%s",
            bool(self.pointguardai_api_base),
            bool(self.pointguardai_org_code),
            bool(self.pointguardai_policy_config_name),
            bool(self.pointguardai_api_key),
        )

        kwargs.setdefault("supported_event_hooks", supported_event_hooks)
        super().__init__(
            guardrail_name=guardrail_name or GUARDRAIL_NAME,
            event_hook=event_hook,
            default_on=default_on,
            **kwargs,  # pyright: ignore[reportArgumentType]  # inherited constructor accepts provider options dynamically
        )

    @staticmethod
    def _extract_text_content(content: object) -> str:
        """Convert supported message text content to PointGuard's string format."""
        if isinstance(content, str):
            return content
        if not isinstance(content, list):
            return ""

        text_parts: Final[list[str]] = []  # mutable-ok: local text accumulator joined before return
        for content_item in content:
            if isinstance(content_item, str):
                text_parts.append(content_item)
            elif isinstance(content_item, dict):
                text = content_item.get("text")
                if isinstance(text, str):
                    text_parts.append(text)
        return "\n".join(text_parts)

    def transform_messages(
        self,
        messages: list[dict],  # mutable-ok: LiteLLM and PointGuard exchange JSON message arrays
    ) -> list[dict]:  # mutable-ok: LiteLLM and PointGuard exchange JSON message arrays
        """Transform messages to PointGuard's text-only message format."""
        new_messages: Final[list[dict]] = []  # mutable-ok: outbound JSON message accumulator
        for m in messages:
            role = m.get("role")
            new_messages.append(
                {  # mutable-ok: outbound JSON message object
                    "role": role if isinstance(role, str) and role else "user",
                    "content": self._extract_text_content(m.get("content")),
                }
            )
        return new_messages

    @staticmethod
    def _replace_text_in_message_content(
        message: dict,  # mutable-ok: LiteLLM message payload is rewritten after redaction
        original: str,
        replacement: str,
    ) -> bool:  # mutable-ok: LiteLLM message payload is rewritten after redaction
        """Replace text in string or multimodal text blocks, leaving other blocks intact."""
        content: Final = message.get("content")
        if isinstance(content, str):
            if original not in content:
                return False
            redacted_content: Final = content.replace(original, replacement)
            message["content"] = redacted_content  # rebind-ok: apply PointGuard redaction
            return True

        if not isinstance(content, list):
            return False

        modified = False  # rebind-ok: tracks whether any multimodal text block was redacted
        for index, content_item in enumerate(content):
            if isinstance(content_item, str):
                if original in content_item:
                    content[index] = content_item.replace(original, replacement)
                    modified = True
                continue
            if not isinstance(content_item, dict):
                continue
            text = content_item.get("text")
            if isinstance(text, str) and original in text:
                content_item["text"] = text.replace(original, replacement)
                modified = True
        return modified

    @staticmethod
    def _serialize_tool_payload(payload: object) -> str | None:
        normalized_payload: Final[object] = (
            payload.model_dump(exclude_none=True) if isinstance(payload, OpenAIObject) else payload
        )
        if not isinstance(normalized_payload, dict):
            return None
        return json.dumps(normalized_payload, sort_keys=True, separators=(",", ":"))

    @staticmethod
    def _deserialize_tool_payload(content: str) -> dict | None:  # mutable-ok: LiteLLM tool payloads are JSON objects
        try:
            payload: Final = json.loads(content)
        except json.JSONDecodeError:
            return None
        return payload if isinstance(payload, dict) else None

    @staticmethod
    def _replace_text_in_all_entries(
        texts: list[str],  # mutable-ok: LiteLLM hook payload is rewritten after redaction
        original: str,
        replacement: str,
    ) -> bool:
        modified = False  # rebind-ok: tracks whether any duplicate text entry was redacted
        for index, text in enumerate(texts):
            if original not in text:
                continue
            texts[index] = text.replace(  # rebind-ok: applies provider redaction to the mutable hook payload
                original, replacement
            )
            modified = True
        return modified

    @classmethod
    def _resolve_tool_call_modification(
        cls,
        response_index: int,
        text_output_count: int,
        serialized_tool_calls: tuple[tuple[int, str], ...],
        original: str,
        replacement: str,
    ) -> tuple[int, dict] | None:  # mutable-ok: LiteLLM tool calls are JSON objects
        tool_call_position: Final = response_index - text_output_count
        if not 0 <= tool_call_position < len(serialized_tool_calls):
            return None
        tool_call_index, serialized_tool_call = serialized_tool_calls[tool_call_position]
        if serialized_tool_call != original:
            return None
        replacement_tool_call: Final = cls._deserialize_tool_payload(replacement)
        if replacement_tool_call is None:
            return None
        return tool_call_index, replacement_tool_call

    async def prepare_pointguard_ai_runtime_scanner_request(
        self,
        new_messages: list[dict],  # mutable-ok: outbound PointGuard JSON message array
        response_string: str | None = None,
        response_strings: list[str] | None = None,  # mutable-ok: LiteLLM output hook supplies a text list
    ) -> dict[str, object] | None:  # mutable-ok: HTTP client requires a JSON dictionary
        """Prepare the request data for PointGuardAI API"""
        try:
            # Validate required parameters
            if not hasattr(self, "pointguardai_policy_config_name") or not self.pointguardai_policy_config_name:
                raise HTTPException(
                    status_code=500,
                    detail="PointGuardAI policy configuration is unavailable",
                )

            data: Final[dict[str, object]] = {  # mutable-ok: outbound JSON body assembled by inspection mode
                "policyName": self.pointguardai_policy_config_name,
            }

            output_strings = response_strings  # rebind-ok: normalizes singular and plural output inputs
            if output_strings is None and response_string is not None:
                output_strings = [  # mutable-ok: PointGuard output schema requires an array  # rebind-ok: normalizes a singular output
                    response_string
                ]  # mutable-ok: PointGuard output schema requires an array  # rebind-ok: normalizes a singular output

            if not new_messages and not output_strings:
                verbose_proxy_logger.warning("PointGuardAI: No input messages or response string provided")
                return None

            # Output endpoint requires BOTH input and output fields
            # Input endpoint requires only input field
            if output_strings is not None:
                # Output endpoint - include both fields (input can be empty array)
                data["input"] = (
                    new_messages if new_messages else []  # mutable-ok: PointGuard input schema requires an array
                )  # mutable-ok: PointGuard input schema requires an array
                data["output"] = [  # mutable-ok: PointGuard output schema requires a JSON array
                    {  # mutable-ok: PointGuard output schema requires a JSON object
                        "role": "assistant",
                        "content": text,
                    }  # mutable-ok: PointGuard output schema requires a JSON object
                    for text in output_strings  # mutable-ok: PointGuard output schema requires a JSON array
                ]
            else:
                # Input endpoint - include only input field
                data["input"] = new_messages

            verbose_proxy_logger.debug(
                "PointGuardAI request prepared: input_messages=%d output_present=%s",
                len(new_messages),
                output_strings is not None,
            )
            return data

        except Exception as e:
            verbose_proxy_logger.error("Error preparing PointGuardAI request: %s", str(e))
            raise

    def _check_sections_present(
        self,
        response_data: dict,  # mutable-ok: decoded PointGuard JSON response
        new_messages: list[dict],  # mutable-ok: outbound PointGuard JSON message array
        response_string: str | None,
        response_strings: list[str] | None = None,  # mutable-ok: LiteLLM output hook supplies a text list
    ) -> tuple[bool, bool]:
        """Check if input or output sections are present in response"""
        input_section_present: Final = bool(new_messages and response_data.get("input"))

        output_section_present: Final = bool((response_strings or response_string) and response_data.get("output"))

        return input_section_present, output_section_present

    @staticmethod
    def _validate_response_data(response_data: object, output_present: bool) -> None:
        if not isinstance(response_data, dict):
            raise HTTPException(status_code=502, detail="Invalid response from PointGuardAI")

        policy_name: Final = response_data.get("policyName")
        if not isinstance(policy_name, str) or not policy_name:
            raise HTTPException(status_code=502, detail="Invalid PointGuardAI response: missing policyName")

        required_sections: Final = ("input", "output") if output_present else ("input",)
        for section_name in required_sections:
            section = response_data.get(
                section_name
            )  # rebind-ok: each loop iteration validates a different response section
            if (
                not isinstance(section, dict)
                or not isinstance(section.get("blocked"), bool)
                or not isinstance(section.get("modified"), bool)
                or not isinstance(section.get("content"), list)
            ):
                raise HTTPException(
                    status_code=502,
                    detail=f"Invalid PointGuardAI response: missing or malformed {section_name} inspection result",
                )

    def _extract_status_flags(
        self,
        response_data: dict,  # mutable-ok: decoded PointGuard JSON response
        input_section_present: bool,
        output_section_present: bool,
    ) -> tuple[bool, bool, bool, bool]:
        """Extract blocking and modification flags from response"""
        input_blocked: Final = (
            response_data.get("input", {}).get(  # mutable-ok: read-only fallback for absent JSON section
                "blocked", False
            )  # mutable-ok: read-only fallback for absent JSON section
            if input_section_present
            else False
        )
        output_blocked: Final = (
            response_data.get("output", {}).get(  # mutable-ok: read-only fallback for absent JSON section
                "blocked", False
            )  # mutable-ok: read-only fallback for absent JSON section
            if output_section_present
            else False
        )
        input_modified: Final = (
            response_data.get("input", {}).get(  # mutable-ok: read-only fallback for absent JSON section
                "modified", False
            )  # mutable-ok: read-only fallback for absent JSON section
            if input_section_present
            else False
        )
        output_modified: Final = (
            response_data.get("output", {}).get(  # mutable-ok: read-only fallback for absent JSON section
                "modified", False
            )  # mutable-ok: read-only fallback for absent JSON section
            if output_section_present
            else False
        )

        return input_blocked, output_blocked, input_modified, output_modified

    def _extract_violations(
        self,
        response_data: dict,  # mutable-ok: decoded provider JSON is normalized for LiteLLM error details
        input_blocked: bool,
        output_blocked: bool,
    ) -> list[dict]:  # mutable-ok: decoded provider JSON is normalized for LiteLLM error details
        """Extract violations from blocked sections in format"""
        violations: Final[list[dict]] = []  # mutable-ok: local violation accumulator returned to LiteLLM

        # Helper function to extract from content items
        def extract_from_content(
            content_items: list[dict],  # mutable-ok: decoded provider JSON is normalized into JSON error details
        ) -> list[dict]:  # mutable-ok: decoded provider JSON is normalized into JSON error details
            all_violations: Final[list[dict]] = []  # mutable-ok: local violation accumulator
            for content_item in content_items:
                if not isinstance(content_item, dict):  # pyright: ignore[reportUnnecessaryIsInstance]  # defensively validate untrusted provider JSON
                    continue

                # Extract DLP violations
                dlp_violations = content_item.get(  # rebind-ok: each provider content item has independent violations
                    "dlpViolations",
                    [],  # mutable-ok: read-only fallback for absent provider array
                )
                all_violations.extend(
                    {  # mutable-ok: normalized LiteLLM violation detail
                        "type": "DLP",
                        "name": dlp.get("name", "Unknown"),
                        "dlp_data_type_id": dlp.get("dlpDataTypeId"),
                        "action": dlp.get("action", "UNKNOWN"),
                        "categories": dlp.get(
                            "categories",
                            [],  # mutable-ok: read-only fallback for absent category array
                        ),  # mutable-ok: read-only fallback for absent category array
                        "match_count": dlp.get("matchCount", 0),
                    }
                    for dlp in dlp_violations
                )

                # Extract AI violations
                ai_violations = content_item.get(  # rebind-ok: each provider content item has independent violations
                    "aiViolations",
                    [],  # mutable-ok: read-only fallback for absent provider array
                )
                all_violations.extend(
                    {  # mutable-ok: normalized LiteLLM violation detail
                        "type": "AI_THREAT",
                        "name": ai.get("name", "Unknown"),
                        "ai_threat_category_id": ai.get("aiThreatCategoryId"),
                        "threat_type": ai.get("type", "UNKNOWN"),
                        "action": ai.get("action", "UNKNOWN"),
                    }
                    for ai in ai_violations
                )
            return all_violations

        # Extract from input if blocked
        if input_blocked and "input" in response_data:
            input_content: Final = response_data["input"].get(
                "content",
                [],  # mutable-ok: read-only fallback for absent provider content
            )
            if isinstance(input_content, list):
                violations.extend(extract_from_content(input_content))

        # Extract from output if blocked
        if output_blocked and "output" in response_data:
            output_content: Final = response_data["output"].get(
                "content",
                [],  # mutable-ok: read-only fallback for absent provider content
            )
            if isinstance(output_content, list):
                violations.extend(extract_from_content(output_content))

        return violations

    def _create_violation_details(
        self,
        violations: list[dict],  # mutable-ok: LiteLLM guardrail errors expose JSON detail arrays
    ) -> list[dict]:  # mutable-ok: LiteLLM guardrail errors expose JSON detail arrays
        """Create detailed violation information"""
        violation_details: Final[list[dict]] = []  # mutable-ok: local JSON detail accumulator
        for violation in violations:
            if not isinstance(violation, dict):  # pyright: ignore[reportUnnecessaryIsInstance]  # defensively validate normalized JSON before rendering details
                continue

            violation_type = violation.get("type", "UNKNOWN")

            if violation_type == "DLP":
                # DLP violation format
                categories = violation.get(  # rebind-ok: each DLP violation has independent categories
                    "categories",
                    [],  # mutable-ok: read-only fallback for absent category array
                )
                category_names = [  # mutable-ok: JSON detail array  # rebind-ok: one list per violation
                    cat.get("name", cat.get("code", "")) for cat in categories if isinstance(cat, dict)
                ]

                violation_details.append(
                    {  # mutable-ok: normalized LiteLLM violation detail
                        "type": "DLP",
                        "name": violation.get("name", "Unknown DLP"),
                        "action": violation.get("action", "UNKNOWN"),
                        "categories": category_names,
                        "match_count": violation.get("match_count", 0),
                        "dlp_data_type_id": violation.get("dlp_data_type_id"),
                    }
                )
            elif violation_type == "AI_THREAT":
                # AI threat violation format
                violation_details.append(
                    {  # mutable-ok: normalized LiteLLM violation detail
                        "type": "AI_THREAT",
                        "name": violation.get("name", "Unknown Threat"),
                        "threat_type": violation.get("threat_type", "UNKNOWN"),
                        "action": violation.get("action", "UNKNOWN"),
                        "ai_threat_category_id": violation.get("ai_threat_category_id"),
                    }
                )
            else:
                # Generic violation
                violation_details.append(violation)

        return violation_details

    def _handle_blocked_request(
        self,
        violation_details: list[dict],  # mutable-ok: LiteLLM guardrail errors expose JSON detail arrays
    ) -> None:  # mutable-ok: LiteLLM guardrail errors expose JSON detail arrays
        """Raise LiteLLM's standard exception for a PointGuard policy block."""
        error_message: Final = "Content blocked by PointGuardAI policy"

        verbose_proxy_logger.warning("PointGuardAI blocking request with violations: %s", violation_details)

        raise GuardrailRaisedException(
            guardrail_name=getattr(self, "guardrail_name", None) or GUARDRAIL_NAME,
            message=error_message,
            should_wrap_with_default_message=False,
            status_code=400,
            blocked_content=True,
        )

    def _handle_modifications(
        self,
        response_data: dict,  # mutable-ok: decoded PointGuard JSON response
        input_modified: bool,
        output_modified: bool,  # mutable-ok: decoded PointGuard JSON response
    ) -> list[dict] | None:  # mutable-ok: modifications are returned as LiteLLM JSON objects
        """Handle content modifications"""
        verbose_proxy_logger.info(
            "PointGuardAI modification detected - Input: %s, Output: %s",
            input_modified,
            output_modified,
        )

        # Extract modified content from content items
        # Returns items with originalContent and modifiedContent for comparison
        def extract_modified_content(
            content_items: list[dict],  # mutable-ok: decoded provider content is normalized into JSON modifications
        ) -> list[dict]:  # mutable-ok: decoded provider content is normalized into JSON modifications
            modified_messages: Final[list[dict]] = []  # mutable-ok: local modification accumulator
            for index, item in enumerate(content_items):
                if not isinstance(item, dict):  # pyright: ignore[reportUnnecessaryIsInstance]  # defensively validate untrusted provider JSON
                    continue

                original_content = item.get(
                    "originalContent"
                )  # rebind-ok: each provider item carries independent content
                modified_content = item.get(
                    "modifiedContent"
                )  # rebind-ok: each provider item carries independent content
                if not isinstance(original_content, str) or not original_content:
                    continue
                if not isinstance(modified_content, str):
                    continue

                # Return with both original and modified content for apply_guardrail to use
                modified_messages.append(
                    {  # mutable-ok: normalized PointGuard modification object
                        "role": item.get("role", "user"),
                        "originalContent": original_content,
                        "modifiedContent": modified_content,
                        "index": index,
                    }
                )

                # Log if content was actually modified
                if item.get("modifiedContent") is not None:
                    verbose_proxy_logger.info(
                        "PointGuardAI: Content modified for role '%s'",
                        item.get("role", "user"),
                    )

            return modified_messages

        # Post-call redactions take precedence when both sections are modified.
        if output_modified and "output" in response_data:
            output_data: Final = response_data["output"]
            if isinstance(output_data, dict) and "content" in output_data:
                content_items: Final = output_data.get(
                    "content",
                    [],  # mutable-ok: read-only fallback for absent provider content
                )
                if isinstance(content_items, list):
                    verbose_proxy_logger.info(
                        "PointGuardAI output modifications: %d items",
                        len(content_items),
                    )
                    modifications: Final = extract_modified_content(content_items)
                    if modifications:
                        return modifications
                    raise HTTPException(
                        status_code=502,
                        detail="Invalid PointGuardAI response: output marked modified without replacement content",
                    )

        if input_modified and "input" in response_data:
            input_data: Final = response_data["input"]
            if isinstance(input_data, dict) and "content" in input_data:
                input_content_items: Final = input_data.get(
                    "content",
                    [],  # mutable-ok: read-only fallback for absent provider content
                )
                if isinstance(input_content_items, list):
                    verbose_proxy_logger.info(
                        "PointGuardAI input modifications: %d items",
                        len(input_content_items),
                    )
                    input_modifications: Final = extract_modified_content(input_content_items)
                    if input_modifications:
                        return input_modifications
                    raise HTTPException(
                        status_code=502,
                        detail="Invalid PointGuardAI response: input marked modified without replacement content",
                    )

        return None

    def _handle_http_status_error(self, e: httpx.HTTPStatusError) -> NoReturn:
        """Handle HTTP status errors"""
        status_code: Final = e.response.status_code
        verbose_proxy_logger.error("PointGuardAI API HTTP error %s", status_code)

        error_messages: Final = {  # mutable-ok: fixed HTTP status lookup table
            401: "PointGuardAI authentication failed: Invalid API credentials",
            400: "PointGuardAI bad request: Invalid configuration or parameters",
            403: "PointGuardAI access denied: Insufficient permissions",
            404: "PointGuardAI resource not found: Invalid endpoint or organization",
        }

        detail: Final = error_messages.get(status_code, f"PointGuardAI API error ({status_code})")
        if 500 <= status_code < 600:
            raise _PointGuardAIUnavailableError(status_code=status_code, detail=detail)
        raise HTTPException(status_code=status_code, detail=detail)

    def _handle_network_errors(self, e: httpx.ConnectError | httpx.TimeoutException | httpx.RequestError) -> NoReturn:
        """Handle network-related errors"""
        if isinstance(e, httpx.TimeoutException):
            verbose_proxy_logger.error("PointGuardAI timeout error: %s", str(e))
            raise _PointGuardAIUnavailableError(
                status_code=504,
                detail="PointGuardAI request timeout: API request took too long to complete",
            )
        else:
            verbose_proxy_logger.error("PointGuardAI connection error: %s", str(e))
            raise _PointGuardAIUnavailableError(
                status_code=503,
                detail="PointGuardAI service unavailable: Cannot connect to API endpoint. Please check the API URL configuration.",
            )

    async def make_pointguard_api_request(
        self,
        request_data: dict,  # mutable-ok: inherited LiteLLM hook request payload
        new_messages: list[dict],  # mutable-ok: outbound PointGuard JSON message array
        response_string: str | None = None,
        response_strings: list[str] | None = None,  # mutable-ok: LiteLLM output hook supplies a text list
    ) -> list[dict] | None:  # mutable-ok: modifications are returned as LiteLLM JSON objects
        """Make the API request to PointGuardAI API"""
        try:
            # Select appropriate endpoint based on whether we have output
            # pre_call mode: use input endpoint
            # post_call mode: use output endpoint
            output_present: Final = response_strings is not None or response_string is not None
            endpoint: Final = self.output_endpoint if output_present else self.input_endpoint
            if output_present:
                verbose_proxy_logger.debug("PointGuardAI: Using output endpoint")
            else:
                verbose_proxy_logger.debug("PointGuardAI: Using input endpoint")

            pointguardai_data: Final = await self.prepare_pointguard_ai_runtime_scanner_request(
                new_messages=new_messages,
                response_string=response_string,
                response_strings=response_strings,
            )

            if pointguardai_data is None:
                verbose_proxy_logger.warning("PointGuardAI: No data prepared for request")
                return None

            _json_data: Final = json.dumps(pointguardai_data)

            verbose_proxy_logger.debug("PointGuardAI: Sending request to %s", endpoint)

            response: Final = await self.async_handler.post(
                url=endpoint,
                data=_json_data,
                headers=self.headers,
            )

            verbose_proxy_logger.debug("PointGuardAI response status: %s", response.status_code)
            # Raise HTTPStatusError for 4xx and 5xx responses
            response.raise_for_status()

            # If we reach here, response.status_code is 2xx (success)
            if response.status_code == 200:
                try:
                    response_data: Final = response.json()
                except json.JSONDecodeError as e:
                    verbose_proxy_logger.error("Failed to parse PointGuardAI response JSON: %s", e)
                    raise HTTPException(
                        status_code=502,
                        detail="Invalid JSON response from PointGuardAI",
                    )

                self._validate_response_data(response_data, output_present)

                # Check sections and extract status flags
                input_section_present, output_section_present = self._check_sections_present(
                    response_data,
                    new_messages,
                    response_string,
                    response_strings,
                )
                input_blocked, output_blocked, input_modified, output_modified = self._extract_status_flags(
                    response_data, input_section_present, output_section_present
                )

                verbose_proxy_logger.info(
                    "PointGuardAI API response analysis - Input: blocked=%s, modified=%s | Output: blocked=%s, modified=%s",
                    input_blocked,
                    input_modified,
                    output_blocked,
                    output_modified,
                )
                # Priority rule: If both blocked=true AND modified=true, BLOCK takes precedence
                if input_blocked or output_blocked:
                    verbose_proxy_logger.warning(
                        "PointGuardAI blocked the request - Input blocked: %s, Output blocked: %s",
                        input_blocked,
                        output_blocked,
                    )

                    violations: Final = self._extract_violations(response_data, input_blocked, output_blocked)
                    violation_details: Final = self._create_violation_details(violations)
                    self._handle_blocked_request(violation_details)

                # Check for modifications only if not blocked
                elif output_present and output_modified:
                    return self._handle_modifications(response_data, False, True)
                elif not output_present and input_modified:
                    return self._handle_modifications(response_data, True, False)

                # No blocking or modification needed
                verbose_proxy_logger.debug("PointGuardAI: No blocking or modifications required")
                return None

            raise HTTPException(
                status_code=502,
                detail=f"Invalid PointGuardAI success status: {response.status_code}",
            )

        except (HTTPException, GuardrailRaisedException):
            # Re-raise HTTP exceptions as-is
            raise
        except httpx.HTTPStatusError as e:
            self._handle_http_status_error(e)
        except (httpx.ConnectError, httpx.TimeoutException, httpx.RequestError) as e:
            self._handle_network_errors(e)
        except (KeyError, TypeError, ValueError) as e:
            verbose_proxy_logger.error(
                "Unexpected error in PointGuardAI API request: %s",
                str(e),
                exc_info=True,
            )
            raise HTTPException(
                status_code=500,
                detail=f"Unexpected error in PointGuardAI integration: {e!s}",
            )

    @log_guardrail_information
    async def apply_guardrail(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict,  # mutable-ok: inherited LiteLLM guardrail hook contract
        input_type: Literal["request", "response"],
        logging_obj: "LiteLLMLoggingObj | None" = None,
    ) -> GenericGuardrailAPIInputs:
        """
        Apply PointGuardAI guardrail to the given inputs using the unified guardrail system.

        Args:
            inputs: Dictionary containing:
                - texts: List of texts to check
                - structured_messages: Structured messages from the request (pre-call only)
            request_data: The original request data
            input_type: "request" for pre-call input validation, "response" for post-call output validation
            logging_obj: Optional logging object

        Returns:
            GenericGuardrailAPIInputs - modified if content changes are applied

        Raises:
            HTTPException: If content is blocked by PointGuardAI
        """
        texts: Final = inputs.get(
            "texts",
            [],  # mutable-ok: read-only fallback required by LiteLLM hook payload
        )
        structured_messages: Final = inputs.get(
            "structured_messages",
            [],  # mutable-ok: read-only fallback required by LiteLLM hook payload
        )

        verbose_proxy_logger.debug(
            "PointGuardAI: apply_guardrail called with input_type=%s, texts=%d, structured_messages=%d",
            input_type,
            len(texts),
            len(structured_messages),
        )

        try:
            if input_type == "request":
                return await self._apply_guardrail_on_request(
                    inputs=inputs,
                    texts=texts,
                    structured_messages=structured_messages,
                    request_data=request_data,
                )

            # Post-call: validate output
            return await self._apply_guardrail_on_response(
                inputs=inputs,
                texts=texts,
                request_data=request_data,
            )
        except _PointGuardAIUnavailableError as error:
            if self.unreachable_fallback == "fail_open":
                verbose_proxy_logger.critical(
                    "PointGuardAI unreachable (fail-open). Proceeding without guardrail. "
                    "status_code=%s guardrail_name=%s api_base=%s input_type=%s "
                    "litellm_call_id=%s litellm_trace_id=%s",
                    error.status_code,
                    getattr(self, "guardrail_name", None),
                    self.pointguardai_api_base,
                    input_type,
                    getattr(logging_obj, "litellm_call_id", None) if logging_obj else None,
                    getattr(logging_obj, "litellm_trace_id", None) if logging_obj else None,
                    exc_info=error,
                )
                passthrough_inputs: Final[
                    GenericGuardrailAPIInputs
                ] = {}  # mutable-ok: LiteLLM hook contract requires a mutable payload copy
                passthrough_inputs.update(inputs)
                return passthrough_inputs
            raise

    async def _apply_guardrail_on_request(
        self,
        inputs: GenericGuardrailAPIInputs,
        texts: list[str],  # mutable-ok: inherited LiteLLM guardrail hook payload
        structured_messages: list,  # mutable-ok: inherited LiteLLM guardrail hook payload
        request_data: dict,  # mutable-ok: inherited LiteLLM guardrail hook payload
    ) -> GenericGuardrailAPIInputs:
        """Handle request-side (pre-call) guardrail checks for input messages."""
        request_messages: Final = self._get_input_messages_from_request_data(request_data)
        messages: Final = (
            request_messages
            if request_messages is not None
            else (
                self.transform_messages(self._select_input_messages(structured_messages))
                if structured_messages
                else [  # mutable-ok: PointGuard input schema requires a JSON array
                    {"role": "user", "content": text}  # mutable-ok: PointGuard input schema requires a JSON object
                    for text in texts[-1:]  # mutable-ok: PointGuard input schema requires a JSON array
                ]
            )
        )

        scan_only_tool_results: Final = effective_scan_only_tool_results_for_guardrail(self)
        tools: Final = (
            []  # mutable-ok: tool definitions are out of scope when scanning only tool results
            if scan_only_tool_results
            else inputs.get(
                "tools",
                [],  # mutable-ok: read-only fallback required by LiteLLM hook payload
            )
        )
        serialized_tools: Final = tuple(
            (index, serialized)
            for index, tool in enumerate(tools)
            if (serialized := self._serialize_tool_payload(tool)) is not None
        )

        if not messages and not serialized_tools:
            return inputs

        new_messages: Final = self.transform_messages(
            messages=messages
        ) + [  # mutable-ok: PointGuard input schema requires an array
            {"role": "tool", "content": serialized}  # mutable-ok: PointGuard input schema requires a JSON object
            for _, serialized in serialized_tools
        ]

        # Make PointGuardAI API request (input only - no output)
        modified_content: Final = await self.make_pointguard_api_request(
            request_data=request_data,
            new_messages=new_messages,
            response_string=None,
        )

        # Apply modifications if present
        if modified_content:
            verbose_proxy_logger.info(
                "PointGuardAI: Applying %d modifications to input",
                len(modified_content),
            )

            modifications_applied = False  # rebind-ok: tracks whether any inspected input was redacted
            new_tools: Final = tools.copy()

            # Modify the structured_messages or texts with string replacement
            for mod_item in modified_content:
                if not isinstance(mod_item, dict):  # pyright: ignore[reportUnnecessaryIsInstance]  # defensively validate provider modifications
                    continue

                original = mod_item.get("originalContent")
                modified = mod_item.get("modifiedContent")

                if not isinstance(original, str) or not original:
                    continue
                if modified is not None and not isinstance(modified, str):
                    continue
                replacement = modified or ""

                for inspected_message in self._select_input_messages(structured_messages):
                    if self._replace_text_in_message_content(inspected_message, original, replacement):
                        modifications_applied = True
                        verbose_proxy_logger.info("PointGuardAI: Modified input message content")

                if self._replace_text_in_all_entries(texts, original, replacement):
                    modifications_applied = True

                matching_tool_index = next(  # rebind-ok: each modification searches for its matching tool
                    (index for index, serialized in serialized_tools if serialized == original),
                    None,
                )
                if matching_tool_index is not None:
                    replacement_tool = self._deserialize_tool_payload(replacement)
                    if replacement_tool is None:
                        continue
                    new_tools[matching_tool_index] = replacement_tool  # pyright: ignore[reportCallIssue, reportArgumentType]  # provider returns the inspected tool JSON shape
                    modifications_applied = True

            if modifications_applied:
                modified_inputs: Final[
                    GenericGuardrailAPIInputs
                ] = {}  # mutable-ok: LiteLLM hook contract requires a mutable payload copy
                modified_inputs.update(inputs)
                modified_inputs["texts"] = texts
                modified_inputs["structured_messages"] = structured_messages
                if new_tools != tools:
                    modified_inputs["tools"] = new_tools
                return modified_inputs

            raise HTTPException(
                status_code=502,
                detail="Invalid PointGuardAI response: input modification did not match inspected content",
            )

        return inputs

    def _select_input_messages(
        self,
        messages: list[dict],  # mutable-ok: inherited LiteLLM messages and PointGuard JSON use arrays of objects
    ) -> list[dict]:  # mutable-ok: inherited LiteLLM messages and PointGuard JSON use arrays of objects
        skip_system: Final = effective_skip_system_message_for_guardrail(self)
        skip_tool: Final = effective_skip_tool_message_for_guardrail(self)
        scan_only_tool_results: Final = effective_scan_only_tool_results_for_guardrail(self)
        scoped_messages: Final = [  # mutable-ok: selected LiteLLM messages remain a JSON-compatible list
            message
            for message in messages
            if not role_out_of_guardrail_scope(
                str(message.get("role") or "").lower(),
                skip_system_message=skip_system,
                skip_tool_message=skip_tool,
                scan_only_tool_results=scan_only_tool_results,
            )
        ]
        context_roles: Final = ("system", "developer")
        context_messages: Final = (
            []  # mutable-ok: LiteLLM message selection returns a list
            if skip_system
            else [  # mutable-ok: LiteLLM message selection returns a list
                message for message in scoped_messages if str(message.get("role") or "").lower() in context_roles
            ]
        )
        latest_assistant_index: Final = next(
            (
                index
                for index in range(len(scoped_messages) - 1, -1, -1)
                if str(scoped_messages[index].get("role") or "").lower() == "assistant"
            ),
            -1,
        )
        current_turn_messages: Final = [  # mutable-ok: selected LiteLLM messages remain a JSON-compatible list
            message
            for message in scoped_messages[latest_assistant_index + 1 :]
            if str(message.get("role") or "").lower() not in context_roles
        ]
        return context_messages + current_turn_messages

    def _get_input_messages_from_request_data(
        self,
        request_data: dict,  # mutable-ok: inherited LiteLLM request and PointGuard messages are JSON containers
    ) -> list[dict] | None:  # mutable-ok: inherited LiteLLM request and PointGuard messages are JSON containers
        """Read input messages from the request payload without copying them into metadata."""

        messages: Final = request_data.get("messages")
        if isinstance(messages, list):
            chat_messages: Final = [  # mutable-ok: LiteLLM messages remain a JSON-compatible list
                message for message in messages if isinstance(message, dict)
            ]
            return self.transform_messages(self._select_input_messages(chat_messages))

        request_input = request_data.get("input")  # rebind-ok: normalizes Responses API input into a message array
        if isinstance(request_input, str):
            return [  # mutable-ok: PointGuard input schema requires a JSON array
                {"role": "user", "content": request_input}  # mutable-ok: PointGuard input schema requires a JSON object
            ]
        if isinstance(request_input, dict):
            request_input = [  # mutable-ok: normalizes a single Responses API object into an array  # rebind-ok: replaces the singular form
                request_input
            ]  # mutable-ok: normalizes a single Responses API object into an array  # rebind-ok: replaces the singular form
        if isinstance(request_input, list):
            input_messages: Final[list[dict]] = []  # mutable-ok: local JSON message accumulator
            for item in request_input:
                if isinstance(item, str):
                    input_messages.append(
                        {"role": "user", "content": item}  # mutable-ok: PointGuard input schema requires a JSON object
                    )
                elif isinstance(item, dict) and "content" in item:
                    input_messages.append(item)
            if input_messages:
                return self.transform_messages(self._select_input_messages(input_messages))

        return None

    async def _apply_guardrail_on_response(
        self,
        inputs: GenericGuardrailAPIInputs,
        texts: list[str],  # mutable-ok: inherited LiteLLM guardrail hook payload
        request_data: dict,  # mutable-ok: inherited LiteLLM guardrail hook payload
    ) -> GenericGuardrailAPIInputs:
        """Handle response-side (post-call) guardrail checks for output."""
        output_indices: Final = tuple(index for index, text in enumerate(texts) if text)
        tool_calls: Final = inputs.get(
            "tool_calls",
            [],  # mutable-ok: read-only fallback required by LiteLLM hook payload
        )
        serialized_tool_calls: Final = tuple(
            (index, serialized)
            for index, tool_call in enumerate(tool_calls)
            if (serialized := self._serialize_tool_payload(tool_call)) is not None
        )
        if not output_indices and not serialized_tool_calls:
            return inputs
        output_texts: Final = [  # mutable-ok: PointGuard output schema requires a JSON array
            texts[index] for index in output_indices
        ] + [  # mutable-ok: PointGuard output schema requires a JSON array
            serialized for _, serialized in serialized_tool_calls
        ]

        # For /output endpoint, we need both input and output
        request_input_messages: Final = self._get_input_messages_from_request_data(request_data)
        input_messages: Final = (
            request_input_messages
            if request_input_messages is not None
            else []  # mutable-ok: Swagger permits an empty input array for output inspection
        )

        # Swagger requires input but permits an empty array.
        if request_input_messages is None:
            verbose_proxy_logger.warning(
                "PointGuardAI: Original input messages are unavailable for output validation; using an empty input array"
            )
        else:
            verbose_proxy_logger.info(
                "PointGuardAI: Using %d request input messages for output validation",
                len(input_messages),
            )

        # Make PointGuardAI API request with actual input and output
        modified_content: Final = await self.make_pointguard_api_request(
            request_data=request_data,
            new_messages=input_messages,
            response_strings=output_texts,
        )

        # Apply modifications to output if present
        if modified_content:
            verbose_proxy_logger.info(
                "PointGuardAI: Applying %d modifications to output",
                len(modified_content),
            )

            new_texts: Final = texts.copy()
            new_tool_calls: Final = tool_calls.copy()
            modifications_applied = False  # rebind-ok: tracks whether any current output was redacted

            for mod_item in modified_content:
                if not isinstance(mod_item, dict):  # pyright: ignore[reportUnnecessaryIsInstance]  # defensively validate provider modifications
                    continue

                original = mod_item.get("originalContent")
                modified = mod_item.get("modifiedContent")
                response_index = mod_item.get("index")

                if not isinstance(original, str) or not original:
                    continue
                if modified is not None and not isinstance(modified, str):
                    continue
                if not isinstance(response_index, int) or not 0 <= response_index < len(output_texts):
                    continue
                replacement = modified or ""
                if response_index < len(output_indices):
                    text_index = output_indices[  # rebind-ok: each modification maps to its inspected output position
                        response_index
                    ]
                    if original in new_texts[text_index]:
                        new_texts[text_index] = new_texts[text_index].replace(original, replacement)
                        modifications_applied = True
                        verbose_proxy_logger.info("PointGuardAI: Modified sensitive content in output")
                    continue

                tool_call_modification = self._resolve_tool_call_modification(
                    response_index=response_index,
                    text_output_count=len(output_indices),
                    serialized_tool_calls=serialized_tool_calls,
                    original=original,
                    replacement=replacement,
                )
                if tool_call_modification is None:
                    continue
                tool_call_index, replacement_tool_call = tool_call_modification
                new_tool_calls[tool_call_index] = replacement_tool_call  # pyright: ignore[reportCallIssue, reportArgumentType]  # provider returns the inspected tool-call JSON shape
                modifications_applied = True

            if modifications_applied:
                modified_inputs: Final[
                    GenericGuardrailAPIInputs
                ] = {}  # mutable-ok: LiteLLM hook contract requires a mutable payload copy
                modified_inputs.update(inputs)
                modified_inputs["texts"] = new_texts
                if new_tool_calls != tool_calls:
                    modified_inputs["tool_calls"] = new_tool_calls
                return modified_inputs

            raise HTTPException(
                status_code=502,
                detail="Invalid PointGuardAI response: output modification did not match inspected content",
            )

        return inputs

    @staticmethod
    def get_config_model() -> type["GuardrailConfigModel"]:
        from litellm.types.proxy.guardrails.guardrail_hooks.pointguardai import (
            PointGuardAIGuardrailConfigModel,
        )

        return PointGuardAIGuardrailConfigModel
