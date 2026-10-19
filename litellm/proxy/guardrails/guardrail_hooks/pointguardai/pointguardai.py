import json
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Final, Literal, NoReturn

import httpx
from fastapi import HTTPException
from typing_extensions import TypeIs  # noqa: TID251  # narrows JSON without copying, so redactions stay in place

from litellm._logging import verbose_proxy_logger
from litellm.exceptions import GuardrailRaisedException
from litellm.integrations.custom_guardrail import (
    CustomGuardrail,
    log_guardrail_information,  # pyright: ignore[reportUnknownVariableType]  # decorator is untyped upstream
)
from litellm.llms.base_llm.guardrail_translation.utils import (
    effective_scan_only_tool_results_for_guardrail,
    effective_skip_system_message_for_guardrail,
    effective_skip_tool_message_for_guardrail,
    role_out_of_guardrail_scope,
)
from litellm.llms.custom_httpx.http_handler import (
    AsyncHTTPHandler,
    get_async_httpx_client,  # pyright: ignore[reportUnknownVariableType]  # the factory's params are untyped upstream
    httpxSpecialProvider,
)
from litellm.types.guardrails import GuardrailEventHooks, Mode
from litellm.types.utils import GenericGuardrailAPIInputs, OpenAIObject

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import (
        Logging as LiteLLMLoggingObj,
    )
    from litellm.types.llms.openai import AllMessageValues
    from litellm.types.proxy.guardrails.guardrail_hooks.pointguardai import PointGuardAIGuardrailConfigModel

GUARDRAIL_NAME: Final = "POINTGUARDAI"
DEFAULT_POINTGUARDAI_API_BASE: Final = "https://api.appsoc.com"


def _is_json_object(
    value: object,
) -> TypeIs[dict[str, object]]:  # guard-ok: isinstance; JSON keys are str  # mutable-ok: messages are redacted in place
    return isinstance(value, dict)


def _is_json_array(
    value: object,
) -> TypeIs[list[object]]:  # guard-ok: trivial isinstance narrowing  # mutable-ok: content blocks are redacted in place
    return isinstance(value, list)


class _PointGuardAIUnavailableError(HTTPException):
    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(status_code=status_code, detail=detail)


class PointGuardAIGuardrail(CustomGuardrail):
    @classmethod
    def get_supported_event_hooks(cls) -> list[GuardrailEventHooks]:  # mutable-ok: LiteLLM hook API requires a list
        return [
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
        | list[GuardrailEventHooks]
        | Mode
        | None = None,
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

        self.headers = {
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
        super().__init__(  # pyright: ignore[reportUnknownMemberType]  # base init takes untyped **kwargs
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
        if not _is_json_array(content):
            return ""

        text_parts: Final[list[str]] = []  # mutable-ok: local text accumulator joined before return
        for content_item in content:
            if isinstance(content_item, str):
                text_parts.append(content_item)
            elif _is_json_object(content_item):
                text = content_item.get("text")
                if isinstance(text, str):
                    text_parts.append(text)
        return "\n".join(text_parts)

    def transform_messages(
        self,
        messages: Sequence[Mapping[str, object]],
    ) -> list[dict[str, object]]:  # mutable-ok: LiteLLM and PointGuard exchange JSON message arrays
        """Transform messages to PointGuard's text-only message format."""
        new_messages: Final[list[dict[str, object]]] = []  # mutable-ok: outbound JSON message accumulator
        for m in messages:
            role = m.get("role")
            new_messages.append(
                {
                    "role": role if isinstance(role, str) and role else "user",
                    "content": self._extract_text_content(m.get("content")),
                }
            )
        return new_messages

    @staticmethod
    def _replace_text_in_message_content(
        message: dict[str, object],  # mutable-ok: LiteLLM message payload is rewritten after redaction
        original: str,
        replacement: str,
    ) -> bool:
        """Replace text in string or multimodal text blocks, leaving other blocks intact."""
        content: Final = message.get("content")
        if isinstance(content, str):
            if original not in content:
                return False
            redacted_content: Final = content.replace(original, replacement)
            message["content"] = redacted_content  # rebind-ok: apply PointGuard redaction
            return True

        if not _is_json_array(content):
            return False

        modified = False  # rebind-ok: tracks whether any multimodal text block was redacted
        for index, content_item in enumerate(content):
            if isinstance(content_item, str):
                if original in content_item:
                    content[index] = content_item.replace(original, replacement)
                    modified = True
                continue
            if not _is_json_object(content_item):
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
    def _deserialize_tool_payload(
        content: str,
    ) -> dict[str, object] | None:  # mutable-ok: LiteLLM tool payloads are JSON objects
        try:
            payload: Final[object] = json.loads(content)
        except json.JSONDecodeError:
            return None
        return payload if _is_json_object(payload) else None

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
    ) -> tuple[int, dict[str, object]] | None:  # mutable-ok: LiteLLM tool calls are JSON objects
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
        new_messages: Sequence[Mapping[str, object]],
        response_string: str | None = None,
        response_strings: Sequence[str] | None = None,
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
                output_strings = [  # rebind-ok: normalizes a singular output
                    response_string
                ]

            if not new_messages and not output_strings:
                verbose_proxy_logger.warning("PointGuardAI: No input messages or response string provided")
                return None

            # Output endpoint requires BOTH input and output fields
            # Input endpoint requires only input field
            if output_strings is not None:
                # Output endpoint - include both fields (input can be empty array)
                data["input"] = new_messages if new_messages else []
                data["output"] = [
                    {
                        "role": "assistant",
                        "content": text,
                    }
                    for text in output_strings
                ]
            else:
                # Input endpoint - include only input field
                data["input"] = new_messages

            verbose_proxy_logger.debug(
                "PointGuardAI request prepared: input_messages=%d output_present=%s",
                len(new_messages),
                output_strings is not None,
            )
        except Exception as e:
            verbose_proxy_logger.error("Error preparing PointGuardAI request: %s", str(e))
            raise
        else:
            return data

    def _check_sections_present(
        self,
        response_data: Mapping[str, object],
        new_messages: Sequence[object],
        response_string: str | None,
        response_strings: Sequence[str] | None = None,
    ) -> tuple[bool, bool]:
        """Check if input or output sections are present in response"""
        input_section_present: Final = bool(new_messages and response_data.get("input"))

        output_section_present: Final = bool((response_strings or response_string) and response_data.get("output"))

        return input_section_present, output_section_present

    @staticmethod
    def _validate_response_data(response_data: object, output_present: bool) -> Mapping[str, object]:
        if not _is_json_object(response_data):
            raise HTTPException(status_code=502, detail="Invalid response from PointGuardAI")

        policy_name: Final = response_data.get("policyName")
        if not isinstance(policy_name, str) or not policy_name:
            raise HTTPException(status_code=502, detail="Invalid PointGuardAI response: missing policyName")

        required_sections: Final = ("input", "output") if output_present else ("input",)
        for section_name in required_sections:
            section = response_data.get(section_name)
            if (
                not _is_json_object(section)
                or not isinstance(section.get("blocked"), bool)
                or not isinstance(section.get("modified"), bool)
                or not isinstance(section.get("content"), list)
            ):
                raise HTTPException(
                    status_code=502,
                    detail=f"Invalid PointGuardAI response: missing or malformed {section_name} inspection result",
                )
        return response_data

    @staticmethod
    def _section_flag(response_data: Mapping[str, object], section_name: str, flag: str) -> bool:
        section: Final = response_data.get(section_name)
        return _is_json_object(section) and section.get(flag) is True

    def _extract_status_flags(
        self,
        response_data: Mapping[str, object],
        input_section_present: bool,
        output_section_present: bool,
    ) -> tuple[bool, bool, bool, bool]:
        """Extract blocking and modification flags from response"""
        input_blocked: Final = input_section_present and self._section_flag(response_data, "input", "blocked")
        output_blocked: Final = output_section_present and self._section_flag(response_data, "output", "blocked")
        input_modified: Final = input_section_present and self._section_flag(response_data, "input", "modified")
        output_modified: Final = output_section_present and self._section_flag(response_data, "output", "modified")

        return input_blocked, output_blocked, input_modified, output_modified

    @staticmethod
    def _section_content(response_data: Mapping[str, object], section_name: str) -> Sequence[object]:
        """The ``content`` array of a PointGuard response section; empty when absent or malformed."""
        section: Final = response_data.get(section_name)
        content: Final = section.get("content") if _is_json_object(section) else None
        return content if _is_json_array(content) else ()

    def _extract_violations(
        self,
        response_data: Mapping[str, object],
        input_blocked: bool,
        output_blocked: bool,
    ) -> list[dict[str, object]]:  # mutable-ok: decoded provider JSON is normalized for LiteLLM error details
        """Extract violations from blocked sections in format"""
        violations: Final[list[dict[str, object]]] = []  # mutable-ok: local violation accumulator returned to LiteLLM

        # Helper function to extract from content items
        def extract_from_content(
            content_items: Sequence[object],
        ) -> list[dict[str, object]]:  # mutable-ok: decoded provider JSON is normalized into JSON error details
            all_violations: Final[list[dict[str, object]]] = []  # mutable-ok: local violation accumulator
            for content_item in content_items:
                if not _is_json_object(content_item):
                    continue

                # Extract DLP violations
                dlp_violations = content_item.get("dlpViolations")
                all_violations.extend(
                    {
                        "type": "DLP",
                        "name": dlp.get("name", "Unknown"),
                        "dlp_data_type_id": dlp.get("dlpDataTypeId"),
                        "action": dlp.get("action", "UNKNOWN"),
                        "categories": dlp.get("categories", []),
                        "match_count": dlp.get("matchCount", 0),
                    }
                    for dlp in (dlp_violations if _is_json_array(dlp_violations) else ())
                    if _is_json_object(dlp)
                )

                # Extract AI violations
                ai_violations = content_item.get("aiViolations")
                all_violations.extend(
                    {
                        "type": "AI_THREAT",
                        "name": ai.get("name", "Unknown"),
                        "ai_threat_category_id": ai.get("aiThreatCategoryId"),
                        "threat_type": ai.get("type", "UNKNOWN"),
                        "action": ai.get("action", "UNKNOWN"),
                    }
                    for ai in (ai_violations if _is_json_array(ai_violations) else ())
                    if _is_json_object(ai)
                )
            return all_violations

        # Extract from input if blocked
        if input_blocked:
            violations.extend(extract_from_content(self._section_content(response_data, "input")))

        # Extract from output if blocked
        if output_blocked:
            violations.extend(extract_from_content(self._section_content(response_data, "output")))

        return violations

    def _create_violation_details(
        self,
        violations: Sequence[object],
    ) -> list[dict[str, object]]:  # mutable-ok: LiteLLM guardrail errors expose JSON detail arrays
        """Create detailed violation information"""
        violation_details: Final[list[dict[str, object]]] = []  # mutable-ok: local JSON detail accumulator
        for violation in violations:
            if not _is_json_object(violation):
                continue

            violation_type = violation.get("type", "UNKNOWN")

            if violation_type == "DLP":
                # DLP violation format
                categories = violation.get("categories")
                category_names = [
                    cat.get("name", cat.get("code", ""))
                    for cat in (categories if _is_json_array(categories) else ())
                    if _is_json_object(cat)
                ]

                violation_details.append(
                    {
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
                    {
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
        violation_details: Sequence[Mapping[str, object]],
    ) -> None:
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
        response_data: Mapping[str, object],
        input_modified: bool,
        output_modified: bool,
    ) -> list[dict[str, object]] | None:  # mutable-ok: modifications are returned as LiteLLM JSON objects
        """Handle content modifications"""
        verbose_proxy_logger.info(
            "PointGuardAI modification detected - Input: %s, Output: %s",
            input_modified,
            output_modified,
        )

        # Extract modified content from content items
        # Returns items with originalContent and modifiedContent for comparison
        def extract_modified_content(
            content_items: Sequence[object],
        ) -> list[dict[str, object]]:  # mutable-ok: decoded provider content is normalized into JSON modifications
            modified_messages: Final[list[dict[str, object]]] = []  # mutable-ok: local modification accumulator
            for index, item in enumerate(content_items):
                if not _is_json_object(item):
                    continue

                original_content = item.get("originalContent")
                modified_content = item.get("modifiedContent")
                if not isinstance(original_content, str) or not original_content:
                    continue
                if not isinstance(modified_content, str):
                    continue

                # Return with both original and modified content for apply_guardrail to use
                modified_messages.append(
                    {
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
        if output_modified:
            output_data: Final = response_data.get("output")
            if _is_json_object(output_data) and "content" in output_data:
                content_items: Final = output_data.get("content")
                if _is_json_array(content_items):
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

        if input_modified:
            input_data: Final = response_data.get("input")
            if _is_json_object(input_data) and "content" in input_data:
                input_content_items: Final = input_data.get("content")
                if _is_json_array(input_content_items):
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

        error_messages: Final = {
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
        request_data: Mapping[str, object],
        new_messages: Sequence[Mapping[str, object]],
        response_string: str | None = None,
        response_strings: Sequence[str] | None = None,
    ) -> list[dict[str, object]] | None:  # mutable-ok: modifications are returned as LiteLLM JSON objects
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

            response: Final = await self.async_handler.post(  # pyright: ignore[reportUnknownMemberType]  # AsyncHTTPHandler.post is untyped upstream
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
                    decoded_response: Final[object] = response.json()
                except json.JSONDecodeError as e:
                    verbose_proxy_logger.error("Failed to parse PointGuardAI response JSON: %s", e)
                    raise HTTPException(
                        status_code=502,
                        detail="Invalid JSON response from PointGuardAI",
                    )

                response_data: Final = self._validate_response_data(decoded_response, output_present)

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
        request_data: dict[str, object],  # mutable-ok: inherited LiteLLM guardrail hook contract
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
            [],
        )
        structured_messages: Final = inputs.get(
            "structured_messages",
            [],
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
                passthrough_inputs: Final[GenericGuardrailAPIInputs] = {}
                passthrough_inputs.update(inputs)
                return passthrough_inputs
            raise

    async def _apply_guardrail_on_request(
        self,
        inputs: GenericGuardrailAPIInputs,
        texts: list[str],  # mutable-ok: inherited LiteLLM guardrail hook payload
        structured_messages: list["AllMessageValues"],  # mutable-ok: inherited LiteLLM guardrail hook payload
        request_data: Mapping[str, object],
    ) -> GenericGuardrailAPIInputs:
        """Handle request-side (pre-call) guardrail checks for input messages."""
        request_messages: Final = self._get_input_messages_from_request_data(request_data)
        messages: Final = (
            request_messages
            if request_messages is not None
            else (
                self.transform_messages(self._select_input_messages(structured_messages))
                if structured_messages
                else [{"role": "user", "content": text} for text in texts[-1:]]
            )
        )

        scan_only_tool_results: Final = effective_scan_only_tool_results_for_guardrail(self)
        tools: Final = (
            []
            if scan_only_tool_results
            else inputs.get(
                "tools",
                [],
            )
        )
        serialized_tools: Final = tuple(
            (index, serialized)
            for index, tool in enumerate(tools)
            if (serialized := self._serialize_tool_payload(tool)) is not None
        )

        if not messages and not serialized_tools:
            return inputs

        new_messages: Final = self.transform_messages(messages=messages) + [
            {"role": "tool", "content": serialized} for _, serialized in serialized_tools
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

                matching_tool_index = next(
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
                modified_inputs: Final[GenericGuardrailAPIInputs] = {}
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
        messages: Sequence[object],
    ) -> list[dict[str, object]]:  # mutable-ok: selected LiteLLM messages are redacted in place
        skip_system: Final = effective_skip_system_message_for_guardrail(self)
        skip_tool: Final = effective_skip_tool_message_for_guardrail(self)
        scan_only_tool_results: Final = effective_scan_only_tool_results_for_guardrail(self)
        scoped_messages: Final = [
            message
            for message in messages
            if _is_json_object(message)
            and not role_out_of_guardrail_scope(
                str(message.get("role") or "").lower(),
                skip_system_message=skip_system,
                skip_tool_message=skip_tool,
                scan_only_tool_results=scan_only_tool_results,
            )
        ]
        context_roles: Final = ("system", "developer")
        context_messages: Final = (
            []
            if skip_system
            else [message for message in scoped_messages if str(message.get("role") or "").lower() in context_roles]
        )
        latest_assistant_index: Final = next(
            (
                index
                for index in range(len(scoped_messages) - 1, -1, -1)
                if str(scoped_messages[index].get("role") or "").lower() == "assistant"
            ),
            -1,
        )
        current_turn_messages: Final = [
            message
            for message in scoped_messages[latest_assistant_index + 1 :]
            if str(message.get("role") or "").lower() not in context_roles
        ]
        return context_messages + current_turn_messages

    def _get_input_messages_from_request_data(
        self,
        request_data: Mapping[str, object],
    ) -> list[dict[str, object]] | None:  # mutable-ok: PointGuard input schema requires a JSON array of objects
        """Read input messages from the request payload without copying them into metadata."""

        messages: Final = request_data.get("messages")
        if _is_json_array(messages):
            return self.transform_messages(self._select_input_messages(messages))

        request_input: Final = request_data.get("input")
        if isinstance(request_input, str):
            return [{"role": "user", "content": request_input}]
        input_messages: Final[list[dict[str, object]]] = []  # mutable-ok: local JSON message accumulator
        for item in self._responses_input_items(request_input):
            if isinstance(item, str):
                input_messages.append({"role": "user", "content": item})
            elif _is_json_object(item) and "content" in item:
                input_messages.append(item)
        if input_messages:
            return self.transform_messages(self._select_input_messages(input_messages))

        return None

    @staticmethod
    def _responses_input_items(request_input: object) -> Sequence[object]:
        """A Responses API ``input`` as its items; a single object is one item."""
        if _is_json_object(request_input):
            return (request_input,)
        return request_input if _is_json_array(request_input) else ()

    async def _apply_guardrail_on_response(
        self,
        inputs: GenericGuardrailAPIInputs,
        texts: list[str],  # mutable-ok: inherited LiteLLM guardrail hook payload
        request_data: Mapping[str, object],
    ) -> GenericGuardrailAPIInputs:
        """Handle response-side (post-call) guardrail checks for output."""
        output_indices: Final = tuple(index for index, text in enumerate(texts) if text)
        tool_calls: Final = inputs.get(
            "tool_calls",
            [],
        )
        serialized_tool_calls: Final = tuple(
            (index, serialized)
            for index, tool_call in enumerate(tool_calls)
            if (serialized := self._serialize_tool_payload(tool_call)) is not None
        )
        if not output_indices and not serialized_tool_calls:
            return inputs
        output_texts: Final = [texts[index] for index in output_indices] + [
            serialized for _, serialized in serialized_tool_calls
        ]

        # For /output endpoint, we need both input and output
        request_input_messages: Final = self._get_input_messages_from_request_data(request_data)
        input_messages: Final = request_input_messages if request_input_messages is not None else []

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
                    text_index = output_indices[response_index]
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
                modified_inputs: Final[GenericGuardrailAPIInputs] = {}
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
    def get_config_model() -> type["PointGuardAIGuardrailConfigModel"]:
        from litellm.types.proxy.guardrails.guardrail_hooks.pointguardai import (
            PointGuardAIGuardrailConfigModel,
        )

        return PointGuardAIGuardrailConfigModel
