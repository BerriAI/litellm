from collections.abc import AsyncGenerator, Mapping, Sequence
from enum import Enum, auto
from typing import TYPE_CHECKING, Any, Final, Literal

import httpx
from fastapi import HTTPException

if TYPE_CHECKING:
    from litellm.types.proxy.guardrails.guardrail_hooks.base import GuardrailConfigModel

import json

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.caching import DualCache
from litellm.constants import DEFAULT_MAX_RECURSE_DEPTH
from litellm.integrations.custom_guardrail import (
    CustomGuardrail,
    log_guardrail_information,
)
from litellm.litellm_core_utils.core_helpers import (
    get_metadata_variable_name_from_kwargs,
    get_or_create_metadata_bucket,
)
from litellm.llms.custom_httpx.http_handler import (
    get_async_httpx_client,
    httpxSpecialProvider,
)
from litellm.llms.vertex_ai.vertex_llm_base import VertexBase
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.guardrails.anthropic_sse import (
    anthropic_sse_chunks_from_response,
    anthropic_sse_error_frames,
    assemble_anthropic_sse_stream,
    is_anthropic_sse_stream,
    is_raw_sse_stream,
    is_sse_error_stream,
)
from litellm.proxy.guardrails.guardrail_hooks.model_armor.file_scanning import (
    MODEL_ARMOR_MAX_FILE_SIZE_BYTES,
    plan_file_scans,
)
from litellm.types.guardrails import GuardrailEventHooks, LitellmParams
from litellm.types.llms.openai import (
    AllMessageValues,
    ChatCompletionToolCallChunk,
    ResponsesAPIResponse,
    ResponsesAPIStreamEvents,
)
from litellm.types.utils import (
    CallTypes,
    CallTypesLiteral,
    Choices,
    GuardrailStatus,
    ModelResponse,
    ModelResponseStream,
    StandardLoggingGuardrailInformation,
    TextCompletionResponse,
)

GUARDRAIL_NAME: Final = "model_armor"

# Only these carry the finished output; response.created carries an empty body
_RESPONSES_TERMINAL_EVENT_TYPES: Final = frozenset({"response.completed", "response.incomplete", "response.failed"})

# Every event whose ``delta`` is model output already on its way to the client. Read off the event
# enum rather than listed, so an event added there cannot quietly fall out of the scan
_RESPONSES_DELTA_EVENT_TYPES: Final = frozenset(
    event.value for event in ResponsesAPIStreamEvents if event.value.endswith(".delta")
)


class _StreamSurface(Enum):
    """Wire format of a buffered streaming response, which decides how it is read and how it is refused."""

    CHAT_COMPLETIONS = auto()
    ANTHROPIC_MESSAGES = auto()
    RESPONSES = auto()
    OPAQUE_SSE = auto()


class ModelArmorAPIError(Exception):
    """Model Armor API failure (non-2xx), distinct from a content-block decision so
    hooks can honor fail_on_error. The detail is already sanitized per configuration."""

    def __init__(self, detail: str):
        super().__init__(detail)
        self.detail = detail


_SCANNED_CONTENT_KEYS: Final = frozenset({"text", "sanitizedText", "findings", "maliciousUriMatchedItems"})

RedactablePayload = dict | list | str | int | float | bool | None


def _redact_scanned_content(payload: RedactablePayload, depth: int = 0) -> RedactablePayload:
    if depth >= DEFAULT_MAX_RECURSE_DEPTH:
        return "[REDACTED]"
    if isinstance(payload, dict):
        return {
            key: "[REDACTED]" if key in _SCANNED_CONTENT_KEYS else _redact_scanned_content(value, depth + 1)
            for key, value in payload.items()
        }
    if isinstance(payload, list):
        return [_redact_scanned_content(item, depth + 1) for item in payload]
    return payload


class ModelArmorGuardrail(CustomGuardrail, VertexBase):
    """
    Google Cloud Model Armor Guardrail integration for LiteLLM.

    Supports:
    - Pre-call sanitization (sanitizeUserPrompt)
    - Post-call sanitization (sanitizeModelResponse)
    """

    @classmethod
    def get_supported_event_hooks(cls) -> list[GuardrailEventHooks]:
        return [
            GuardrailEventHooks.pre_call,
            GuardrailEventHooks.during_call,
            GuardrailEventHooks.post_call,
            GuardrailEventHooks.pre_mcp_call,
            GuardrailEventHooks.during_mcp_call,
        ]

    def __init__(
        self,
        template_id: str | None = None,
        project_id: str | None = None,
        location: str | None = None,
        credentials: Any | None = None,
        api_endpoint: str | None = None,
        sanitize_error_detail: "bool | None" = True,
        **kwargs,
    ):
        # Set supported event hooks if not already provided
        if "event_hook" not in kwargs:
            kwargs["event_hook"] = [
                GuardrailEventHooks.pre_call,
                GuardrailEventHooks.during_call,
                GuardrailEventHooks.post_call,
            ]
        kwargs.setdefault("supported_event_hooks", list(self.get_supported_event_hooks()))

        # Initialize parent classes first
        super().__init__(**kwargs)
        VertexBase.__init__(self)

        # Then set our attributes (this ensures project_id is not overwritten)
        self.async_handler = get_async_httpx_client(llm_provider=httpxSpecialProvider.GuardrailCallback)
        self.template_id = template_id
        self.project_id = project_id
        self.location = location or "us-central1"
        self.credentials = credentials
        self.api_endpoint = api_endpoint
        self.sanitize_error_detail = sanitize_error_detail is not False

        # Store optional params
        self.optional_params = kwargs

        verbose_proxy_logger.debug(
            "Model Armor Guardrail initialized with template_id: %s, project_id: %s, location: %s",
            self.template_id,
            self.project_id,
            self.location,
        )

    def _get_api_endpoint(self) -> str:
        """Get the API endpoint for Model Armor."""
        if self.api_endpoint:
            return self.api_endpoint
        return f"https://modelarmor.{self.location}.rep.googleapis.com"

    def _create_sanitize_request(self, content: str, source: Literal["user_prompt", "model_response"]) -> dict:
        """Create request body for Model Armor API with correct camelCase field names."""
        if source == "user_prompt":
            return {"userPromptData": {"text": content}}
        else:
            return {"modelResponseData": {"text": content}}

    def _extract_content_from_response(self, response: Any | ModelResponse) -> str:
        """
        Extract text content from model response.

        Returns empty string for non-text responses (TTS, images, etc.) to skip guardrail processing.
        """
        from litellm.litellm_core_utils.prompt_templates.common_utils import (
            get_content_from_model_response,
        )

        # Handle ModelResponse objects
        if isinstance(response, litellm.ModelResponse):
            return get_content_from_model_response(response)

        # For non-ModelResponse types (e.g., TTS, images), return empty string
        # These response types are not text-based and shouldn't be processed by text guardrails
        verbose_proxy_logger.debug("Model Armor: Skipping non-ModelResponse type: %s", type(response).__name__)
        return ""

    def _build_api_error_detail(self, status_code: int, response_text: str) -> str:
        if self.sanitize_error_detail:
            return f"Model Armor API error (upstream {status_code})"
        return f"Model Armor API error (upstream {status_code}): {response_text}"

    def _build_block_error_detail(self, message: str, armor_response: RedactablePayload) -> dict:
        if self.sanitize_error_detail:
            return {"error": message}
        return {"error": message, "model_armor_response": armor_response}

    def _build_logging_response(self, armor_response: RedactablePayload) -> RedactablePayload:
        if self.sanitize_error_detail:
            return _redact_scanned_content(armor_response)
        return armor_response

    def _raise_if_fail_closed(self, e: ModelArmorAPIError) -> None:
        if self.optional_params.get("fail_on_error", True):
            raise e from None

    def update_in_memory_litellm_params(self, litellm_params: LitellmParams) -> None:
        super().update_in_memory_litellm_params(litellm_params)
        self.sanitize_error_detail = self.sanitize_error_detail is not False

    def _log_request_debug(
        self,
        url: str,
        body: dict,
        file_bytes: "bytes | None",
        file_type: "str | None",
    ) -> None:
        # Never log byteData: it is the full base64 of the scanned document. Log only its
        # type and size so debug deployments cannot leak the contents the guardrail inspects.
        if file_bytes is not None and file_type is not None:
            verbose_proxy_logger.debug(
                "Model Armor file request - URL: %s, byteDataType: %s, bytes: %d",
                url,
                file_type,
                len(file_bytes),
            )
        elif self.sanitize_error_detail:
            verbose_proxy_logger.debug("Model Armor request - URL: %s", url)
        else:
            verbose_proxy_logger.debug(
                "Model Armor request - URL: %s, Body: %s",
                url,
                body,
            )

    def _log_response_debug(self, status_code: int, response_text: str) -> None:
        if self.sanitize_error_detail:
            verbose_proxy_logger.debug(
                "Model Armor response - Status: %s",
                status_code,
            )
        else:
            verbose_proxy_logger.debug(
                "Model Armor response - Status: %s, Body: %s",
                status_code,
                response_text,
            )

    async def make_model_armor_request(
        self,
        content: str | None = None,
        source: Literal["user_prompt", "model_response"] = "user_prompt",
        request_data: dict | None = None,
        file_bytes: bytes | None = None,
        file_type: str | None = None,
    ) -> dict:
        """
        Make request to Model Armor API. Supports both text and file prompt sanitization.
        If file_bytes and file_type are provided, file prompt sanitization is performed.
        """
        # Get access token using VertexBase auth
        access_token, resolved_project_id = await self._ensure_access_token_async(
            credentials=self.credentials,
            project_id=self.project_id,
            custom_llm_provider="vertex_ai",
        )

        # Use resolved project ID if not explicitly set
        if not self.project_id and resolved_project_id:
            self.project_id = resolved_project_id

        # Construct URL
        endpoint: Final = self._get_api_endpoint()
        if source == "user_prompt":
            url = f"{endpoint}/v1/projects/{self.project_id}/locations/{self.location}/templates/{self.template_id}:sanitizeUserPrompt"
        else:
            url = f"{endpoint}/v1/projects/{self.project_id}/locations/{self.location}/templates/{self.template_id}:sanitizeModelResponse"

        # Create request body
        if file_bytes is not None and file_type is not None:
            body = self.sanitize_file_prompt(file_bytes, file_type, source)
        elif content is not None:
            body = self._create_sanitize_request(content, source)
        else:
            raise ValueError("Either content or file_bytes and file_type must be provided.")

        # Set headers
        headers: Final = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {access_token}",
        }

        self._log_request_debug(url=url, body=body, file_bytes=file_bytes, file_type=file_type)

        # Make request
        if self.async_handler is None:
            raise ValueError("Async handler not initialized")

        try:
            response: Final = await self.async_handler.post(
                url=url,
                json=body,
                headers=headers,
            )
        except httpx.HTTPStatusError as e:
            detail = self._build_api_error_detail(e.response.status_code, e.response.text)
            verbose_proxy_logger.error(
                "Model Armor API error - Status: %s, Detail: %s",
                e.response.status_code,
                detail,
            )
            raise ModelArmorAPIError(detail) from None

        self._log_response_debug(status_code=response.status_code, response_text=response.text)

        if response.status_code != 200:
            detail = self._build_api_error_detail(response.status_code, response.text)
            verbose_proxy_logger.error(
                "Model Armor API error - Status: %s, Detail: %s",
                response.status_code,
                detail,
            )
            raise ModelArmorAPIError(detail)

        json_response: Final = response.json()
        if hasattr(json_response, "__await__"):
            return await json_response
        return json_response

    def sanitize_file_prompt(self, file_bytes: bytes, file_type: str, source: str = "user_prompt") -> dict:
        """
        Helper to build the request body for file prompt sanitization for Model Armor.
        file_type should be one of: PLAINTEXT_UTF8, PDF, WORD_DOCUMENT, EXCEL_DOCUMENT, POWERPOINT_DOCUMENT, TXT, CSV
        Returns the request body dict.
        """
        import base64

        base64_data: Final = base64.b64encode(file_bytes).decode("utf-8")
        if source == "user_prompt":
            return {"userPromptData": {"byteItem": {"byteDataType": file_type, "byteData": base64_data}}}
        else:
            return {"modelResponseData": {"byteItem": {"byteDataType": file_type, "byteData": base64_data}}}

    def _should_block_content(self, armor_response: Mapping[str, Any], allow_sanitization: bool = False) -> bool:
        """Check if Model Armor response indicates content should be blocked, including both inspectResult and deidentifyResult."""
        for filt in self._filter_result_items(armor_response):
            # Check RAI, PI/Jailbreak, Malicious URI, CSAM, Virus scan as before
            if filt.get("raiFilterResult", {}).get("matchState") == "MATCH_FOUND":
                return True
            if filt.get("piAndJailbreakFilterResult", {}).get("matchState") == "MATCH_FOUND":
                return True
            if filt.get("maliciousUriFilterResult", {}).get("matchState") == "MATCH_FOUND":
                return True
            if filt.get("csamFilterFilterResult", {}).get("matchState") == "MATCH_FOUND":
                return True
            if filt.get("virusScanFilterResult", {}).get("matchState") == "MATCH_FOUND":
                return True
            # Check sdpFilterResult for both inspectResult and deidentifyResult
            sdp = filt.get("sdpFilterResult")
            if sdp:
                if sdp.get("inspectResult", {}).get("matchState") == "MATCH_FOUND":
                    return True
                # Only block on deidentifyResult if sanitization is not allowed
                if sdp.get("deidentifyResult", {}).get("matchState") == "MATCH_FOUND":
                    if not allow_sanitization:
                        return True
        # Fallback dict code removed; all cases handled above
        return False

    def _get_sanitized_content(self, armor_response: Mapping[str, Any]) -> str | None:
        """
        Get the sanitized content from a Model Armor response, if available.
        Looks for sanitized text in deidentifyResult, and falls back to root-level fields if not found.
        """
        filters: Final = self._filter_result_items(armor_response)

        # Prefer sanitized text from deidentifyResult if present
        for filter_entry in filters:
            sdp = filter_entry.get("sdpFilterResult")
            if sdp:
                deid = sdp.get("deidentifyResult", {})
                sanitized = deid.get("data", {}).get("text", "")
                # If Model Armor found something and returned a sanitized version, use it
                if deid.get("matchState") == "MATCH_FOUND" and sanitized:
                    return sanitized

        # If no deidentifyResult, optionally check for inspectResult (rare, but could have findings)
        for filter_entry in filters:
            sdp = filter_entry.get("sdpFilterResult")
            if sdp:
                inspect = sdp.get("inspectResult", {})
                # If Model Armor flagged something but didn't sanitize, return None
                if inspect.get("matchState") == "MATCH_FOUND":
                    return None

        # Fallback: if Model Armor put sanitized text at the root, use it
        return armor_response.get("sanitizedText") or armor_response.get("text")

    @staticmethod
    def _filter_result_items(armor_response: Mapping[str, Any]) -> Sequence[Any]:
        """Every filter result in a scan response.

        filterResults is a dict of named filters on most templates and a list on some, so both
        shapes are flattened to the same list of filter entries.
        """
        filter_results: Final = armor_response.get("sanitizationResult", {}).get("filterResults", {})
        if isinstance(filter_results, dict):
            return list(filter_results.values())
        if isinstance(filter_results, list):
            return filter_results
        return []

    def _has_deidentify_match(self, armor_response: Mapping[str, Any]) -> bool:
        """Whether an SDP de-identify filter matched, i.e. Model Armor owes this response a redaction."""
        for filter_entry in self._filter_result_items(armor_response):
            sdp = filter_entry.get("sdpFilterResult")
            if sdp and sdp.get("deidentifyResult", {}).get("matchState") == "MATCH_FOUND":
                return True
        return False

    def _resolve_streaming_outcome(
        self,
        armor_response: Mapping[str, Any],
        assembled_response: object,
        content: str,
    ) -> tuple[bool, str | None]:
        """Whether to block the buffered stream, and the rewrite to emit when it is not blocked.

        A de-identify match only reaches here unblocked because masking is on, so the redaction it
        stands for has to be both resolvable and emittable. Where it is neither, the buffered
        original still carries what Model Armor matched on, so this fails closed instead of
        releasing it.
        """
        if self._should_block_content(armor_response, allow_sanitization=self.mask_response_content):
            return True, None
        if not self.mask_response_content:
            return False, None

        sanitized_content: Final = self._get_sanitized_content(armor_response)
        if not sanitized_content:
            # No rewrite to apply. Harmless unless a match is outstanding, in which case applying
            # nothing would hand back the very content that matched
            return self._has_deidentify_match(armor_response), None
        if sanitized_content == content:
            return False, None
        if not isinstance(assembled_response, ModelResponse):
            verbose_proxy_logger.warning(
                "Model Armor: sanitized content cannot be re-emitted on this streaming endpoint, "
                "blocking the response instead"
            )
            return True, None
        return False, sanitized_content

    @staticmethod
    def _append_armor_response(existing: object, armor_response: Mapping[str, object]) -> object:
        """Accumulate scan responses so a later text scan does not drop an earlier file scan.

        Returns the single response on its own (backward compatible) and a list once a request
        carries more than one scan. A list (not a tuple) is required because the guardrail logging
        pipeline (redact_nested_match_and_regex_keys and the StandardLoggingGuardrailInformation
        dict | list[dict] contract) only recurses into dicts and lists when redacting and serializing.
        """
        if existing is None:
            return armor_response
        if isinstance(existing, list):
            return [*existing, armor_response]  # mutable-ok: logging pipeline requires list[dict], not tuple
        return [existing, armor_response]  # mutable-ok: logging pipeline requires list[dict], not tuple

    def _process_response(
        self,
        response: dict | None,
        request_data: dict,
        start_time: float | None = None,
        end_time: float | None = None,
        duration: float | None = None,
        event_type: GuardrailEventHooks | None = None,
        original_inputs: dict | None = None,
    ):
        """
        Override to store only the Model Armor API response, not the entire data dict.
        This prevents circular references in logging.
        """
        metadata: Final = (
            request_data.get(get_metadata_variable_name_from_kwargs(request_data)) or {}
            if isinstance(request_data, dict)
            else {}
        )
        guardrail_response: Final = metadata.get("_model_armor_response", {})

        # Determine status – default to "success" but prefer the explicit value if present.
        guardrail_status: Final[GuardrailStatus] = metadata.get("_model_armor_status", "success")

        self.add_standard_logging_guardrail_information_to_request_data(
            guardrail_json_response=guardrail_response,
            request_data=request_data,
            guardrail_status=guardrail_status,
            duration=duration,
            start_time=start_time,
            end_time=end_time,
            event_type=event_type,
        )
        return response

    @staticmethod
    def _unscannable_block_error(reason: str) -> HTTPException:
        return HTTPException(
            status_code=400,
            detail={"error": f"Model Armor could not scan an attachment and blocked the request: {reason}"},
        )

    async def _scan_request_files(self, messages: Sequence[AllMessageValues], data: dict) -> None:
        """Submit inline document/file attachments to Model Armor and block on any findings.

        Each attachment is sent through the byte API and a MATCH_FOUND raises a 400 before the
        request reaches the LLM. File scanning does not support masking (Model Armor returns
        findings, not a sanitized document), so it only blocks. A file_id or remote URL reference
        with no inline bytes and a document over the 4 MB byte limit are guardrail failures that
        block unless the operator has opted into fail-open via fail_on_error=False.

        skip_unscannable_attachments decouples reference-only attachments from fail_on_error: when
        enabled, attachments Model Armor cannot scan (file_id, gs://, or http(s) references with no
        inline bytes, and inline content whose base64 will not decode) pass through instead of
        blocking, while fail_on_error still governs real Model Armor API errors.
        """
        from litellm.proxy.common_utils.callback_utils import (
            add_guardrail_to_applied_guardrails_header,
        )

        plan: Final = plan_file_scans(messages)
        attachments: Final = plan.attachments
        skip_unscannable: Final = bool(self.optional_params.get("skip_unscannable_attachments", False))
        if skip_unscannable and plan.unscannable_count > 0:
            verbose_proxy_logger.warning(
                "Model Armor: allowing %d unscannable attachment(s) through because "
                "skip_unscannable_attachments is enabled",
                plan.unscannable_count,
            )
        unscannable_references: Final = 0 if skip_unscannable else plan.unscannable_count
        if not attachments and unscannable_references == 0:
            return

        add_guardrail_to_applied_guardrails_header(request_data=data, guardrail_name=self.guardrail_name)
        # Use the same metadata bucket the header helper writes to, so the logged Model Armor
        # payload and status land where _process_response reads them on every route.
        _, metadata = get_or_create_metadata_bucket(data)
        fail_on_error: Final = bool(self.optional_params.get("fail_on_error", True))

        if unscannable_references > 0:
            reason = (
                f"{unscannable_references} attachment(s) reference a document with no inline bytes "
                "(file_id or remote URL) that Model Armor cannot scan"
            )
            verbose_proxy_logger.warning("Model Armor: %s", reason)
            if fail_on_error:
                metadata["_model_armor_status"] = "blocked"
                raise self._unscannable_block_error(reason)

        for attachment in attachments:
            if len(attachment.file_bytes) > MODEL_ARMOR_MAX_FILE_SIZE_BYTES:
                reason = (
                    f"attachment of {len(attachment.file_bytes)} bytes exceeds Model Armor's "
                    f"{MODEL_ARMOR_MAX_FILE_SIZE_BYTES} byte scan limit"
                )
                verbose_proxy_logger.warning("Model Armor: %s", reason)
                if not fail_on_error:
                    continue
                metadata["_model_armor_status"] = "blocked"
                raise self._unscannable_block_error(reason)

            try:
                armor_response = await self.make_model_armor_request(
                    source="user_prompt",
                    request_data=data,
                    file_bytes=attachment.file_bytes,
                    file_type=attachment.byte_data_type,
                )
            except ModelArmorAPIError as e:
                self._raise_if_fail_closed(e)
                continue
            except HTTPException:
                raise
            except Exception as e:
                # Isolate transient errors per attachment so one failure does not leave the
                # remaining attachments in the same request unscanned.
                verbose_proxy_logger.error("Model Armor file scan error: %s", str(e), exc_info=True)
                if fail_on_error:
                    raise
                continue

            # Model Armor returns findings for documents, not a sanitized file, so there is no
            # masking fallback. Any finding must block, even when mask_request_content is enabled,
            # otherwise a PII-only (SDP deidentify) document would pass through unscrubbed.
            blocked = self._should_block_content(armor_response, allow_sanitization=False)
            metadata["_model_armor_response"] = self._append_armor_response(
                metadata.get("_model_armor_response"),
                self._build_logging_response(armor_response),
            )
            if blocked or metadata.get("_model_armor_status") == "blocked":
                metadata["_model_armor_status"] = "blocked"
            else:
                metadata["_model_armor_status"] = "success"

            if blocked:
                raise HTTPException(
                    status_code=400,
                    detail=self._build_block_error_detail("Content blocked by Model Armor", armor_response),
                )

    @log_guardrail_information
    async def async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: DualCache,
        data: dict,
        call_type: CallTypesLiteral,
    ) -> Exception | str | dict | None:
        """Pre-call hook to sanitize user prompts."""
        verbose_proxy_logger.debug("Inside Model Armor Pre-Call Hook")

        from litellm.proxy.common_utils.callback_utils import (
            add_guardrail_to_applied_guardrails_header,
        )

        event_type = GuardrailEventHooks.pre_call
        if call_type == CallTypes.call_mcp_tool.value:
            event_type = GuardrailEventHooks.pre_mcp_call
        if self.should_run_guardrail(data=data, event_type=event_type) is not True:
            return data

        messages: Final = data.get("messages")
        if not messages:
            verbose_proxy_logger.warning("Model Armor: not running guardrail. No messages in data")
            return data

        # Extract content from messages using helper from common_utils
        from litellm.litellm_core_utils.prompt_templates.common_utils import (
            get_last_user_message,
        )

        await self._scan_request_files(messages=messages, data=data)

        content: Final = get_last_user_message(messages)
        if not content:
            return data

        # Make Model Armor request
        try:
            armor_response: Final = await self.make_model_armor_request(
                content=content,
                source="user_prompt",
                request_data=data,
            )

            # Store the armor response for logging
            # Attach Model Armor response + evaluation status directly to the per-request metadata to avoid
            #   race-conditions between concurrent requests which share the same guardrail instance.
            #   This ensures each request logs its own Model Armor response instead of a potentially stale value
            #   overwritten by another coroutine.
            blocked: Final = self._should_block_content(armor_response, allow_sanitization=self.mask_request_content)
            if isinstance(data, dict):
                _, metadata = get_or_create_metadata_bucket(data)  # ensures metadata exists and is unique per request
                # Accumulate so a prior file scan on the same request is not overwritten by this text scan.
                metadata["_model_armor_response"] = self._append_armor_response(
                    metadata.get("_model_armor_response"),
                    self._build_logging_response(armor_response),
                )
                # Pre-compute guardrail status for downstream logging. A blocked response will eventually raise
                #   an HTTPException, however in scenarios where the caller decides to ignore the exception (e.g.
                #   fail_on_error=False) we still want the correct status reflected.
                if blocked or metadata.get("_model_armor_status") == "blocked":
                    metadata["_model_armor_status"] = "blocked"
                else:
                    metadata["_model_armor_status"] = "success"

            # Add guardrail to applied_guardrails BEFORE potential blocking
            # This ensures guardrail is recorded even when it blocks the request
            add_guardrail_to_applied_guardrails_header(request_data=data, guardrail_name=self.guardrail_name)

            # Check if content should be blocked
            if blocked:
                raise HTTPException(
                    status_code=400,
                    detail=self._build_block_error_detail("Content blocked by Model Armor", armor_response),
                )

            # If mask_request_content is enabled, update messages with sanitized content
            if self.mask_request_content:
                sanitized_content: Final = self._get_sanitized_content(armor_response)
                if sanitized_content and sanitized_content != content:
                    # Use the helper to set the last user message with sanitized content
                    from litellm.litellm_core_utils.prompt_templates.common_utils import (
                        set_last_user_message,
                    )

                    data["messages"] = set_last_user_message(messages, sanitized_content)

        except ModelArmorAPIError as e:
            self._raise_if_fail_closed(e)
        except HTTPException:
            raise
        except Exception as e:
            verbose_proxy_logger.error("Model Armor pre-call error: %s", str(e), exc_info=True)
            # Depending on configuration, either fail or continue
            if self.optional_params.get("fail_on_error", True):
                raise

        return data

    @log_guardrail_information
    async def async_moderation_hook(
        self,
        data: dict,
        user_api_key_dict: UserAPIKeyAuth,
        call_type: CallTypesLiteral,
    ) -> Exception | str | dict | None:
        """During-call hook to sanitize user prompts in parallel with LLM call."""
        verbose_proxy_logger.debug("Inside Model Armor Moderation Hook")

        from litellm.proxy.common_utils.callback_utils import (
            add_guardrail_to_applied_guardrails_header,
        )

        event_type = GuardrailEventHooks.during_call
        if call_type == CallTypes.call_mcp_tool.value:
            event_type = GuardrailEventHooks.during_mcp_call
        if self.should_run_guardrail(data=data, event_type=event_type) is not True:
            return data

        messages: Final = data.get("messages")
        if not messages:
            verbose_proxy_logger.warning("Model Armor: not running guardrail. No messages in data")
            return data

        # Extract content from messages
        from litellm.litellm_core_utils.prompt_templates.common_utils import (
            get_last_user_message,
        )

        await self._scan_request_files(messages=messages, data=data)

        content: Final = get_last_user_message(messages)
        if not content:
            return data

        # Make Model Armor request
        try:
            armor_response: Final = await self.make_model_armor_request(
                content=content,
                source="user_prompt",
                request_data=data,
            )

            blocked: Final = self._should_block_content(armor_response, allow_sanitization=self.mask_request_content)
            # Store the armor response for logging
            if isinstance(data, dict):
                _, metadata = get_or_create_metadata_bucket(data)
                # Accumulate so a prior file scan on the same request is not overwritten by this text scan.
                metadata["_model_armor_response"] = self._append_armor_response(
                    metadata.get("_model_armor_response"),
                    self._build_logging_response(armor_response),
                )
                if blocked or metadata.get("_model_armor_status") == "blocked":
                    metadata["_model_armor_status"] = "blocked"
                else:
                    metadata["_model_armor_status"] = "success"

            # Add guardrail to applied_guardrails BEFORE potential blocking
            # This ensures guardrail is recorded even when it blocks the request
            add_guardrail_to_applied_guardrails_header(request_data=data, guardrail_name=self.guardrail_name)

            # Check if content should be blocked
            if blocked:
                raise HTTPException(
                    status_code=400,
                    detail=self._build_block_error_detail("Content blocked by Model Armor", armor_response),
                )

            # If mask_request_content is enabled, update messages with sanitized content
            if self.mask_request_content:
                sanitized_content: Final = self._get_sanitized_content(armor_response)
                if sanitized_content and sanitized_content != content:
                    from litellm.litellm_core_utils.prompt_templates.common_utils import (
                        set_last_user_message,
                    )

                    data["messages"] = set_last_user_message(messages, sanitized_content)

        except ModelArmorAPIError as e:
            self._raise_if_fail_closed(e)
        except HTTPException:
            raise
        except Exception as e:
            verbose_proxy_logger.error("Model Armor moderation error: %s", str(e), exc_info=True)
            if self.optional_params.get("fail_on_error", True):
                raise

        return data

    @log_guardrail_information
    async def async_post_call_success_hook(
        self,
        data: dict,
        user_api_key_dict: UserAPIKeyAuth,
        response,
    ):
        """Post-call hook to sanitize model responses."""
        from litellm.proxy.common_utils.callback_utils import (
            add_guardrail_response_to_standard_logging_object,
            add_guardrail_to_applied_guardrails_header,
        )

        if self.should_run_guardrail(data=data, event_type=GuardrailEventHooks.post_call) is not True:
            return

        # Extract content from response
        content: Final = self._extract_content_from_response(response)
        if not content:
            verbose_proxy_logger.debug("Model Armor: No text content to process in response, skipping guardrail")
            return

        # Make Model Armor request
        try:
            armor_response: Final = await self.make_model_armor_request(
                content=content,
                source="model_response",
                request_data=data,
            )

            # Attach Model Armor response & status to this request's metadata to prevent race conditions
            if isinstance(armor_response, dict):
                model_armor_logged_object: Final = {
                    "model_armor_response": self._build_logging_response(armor_response),
                    "model_armor_status": (
                        "blocked"
                        if self._should_block_content(
                            armor_response,
                            allow_sanitization=self.mask_response_content,
                        )
                        else "success"
                    ),
                }
                standard_logging_guardrail_information: Final = StandardLoggingGuardrailInformation(
                    guardrail_name=self.guardrail_name,
                    guardrail_provider="model_armor",
                    guardrail_mode=GuardrailEventHooks.post_call,
                    guardrail_response=model_armor_logged_object,
                    guardrail_status="success",
                    start_time=data.get("start_time"),
                )
                add_guardrail_response_to_standard_logging_object(
                    litellm_logging_obj=data.get("litellm_logging_obj"),
                    guardrail_response=standard_logging_guardrail_information,
                )

            # Add guardrail to applied_guardrails BEFORE potential blocking
            # This ensures guardrail is recorded even when it blocks the request
            add_guardrail_to_applied_guardrails_header(request_data=data, guardrail_name=self.guardrail_name)

            # Check if content should be blocked
            if self._should_block_content(armor_response, allow_sanitization=self.mask_response_content):
                raise HTTPException(
                    status_code=400,
                    detail=self._build_block_error_detail("Response blocked by Model Armor", armor_response),
                )

            # If mask_response_content is enabled, update response with sanitized content
            if self.mask_response_content:
                sanitized_content: Final = self._get_sanitized_content(armor_response)
                if sanitized_content and sanitized_content != content:
                    # Update response content
                    if isinstance(response, litellm.ModelResponse):
                        for choice in response.choices:
                            if isinstance(choice, Choices):
                                if choice.message.content:
                                    choice.message.content = sanitized_content

        except ModelArmorAPIError as e:
            self._raise_if_fail_closed(e)
        except HTTPException:
            raise
        except Exception as e:
            verbose_proxy_logger.error("Model Armor post-call error: %s", str(e), exc_info=True)
            if self.optional_params.get("fail_on_error", True):
                raise

        return response

    @staticmethod
    def _is_terminal_error_stream(all_chunks: Sequence[object]) -> bool:
        """Whether the buffered stream is only the refusal an earlier guardrail in the chain emitted.

        post_call guardrails are composed, so this hook can be handed the terminal error items a
        preceding one produced. They carry no message to scan, and replacing them would hide the
        refusal the client is owed.
        """
        if all(getattr(chunk, "type", None) == "error" for chunk in all_chunks):
            return True
        return is_sse_error_stream(all_chunks)

    @staticmethod
    def _classify_stream(all_chunks: Sequence[object]) -> _StreamSurface:
        """Wire format the buffered chunks belong to."""
        if is_raw_sse_stream(all_chunks):
            return (
                _StreamSurface.ANTHROPIC_MESSAGES if is_anthropic_sse_stream(all_chunks) else _StreamSurface.OPAQUE_SSE
            )
        if any(
            isinstance(event_type := getattr(chunk, "type", None), str) and event_type.startswith("response.")
            for chunk in all_chunks
        ):
            return _StreamSurface.RESPONSES
        return _StreamSurface.CHAT_COMPLETIONS

    @staticmethod
    def _final_responses_api_response(all_chunks: Sequence[object]) -> ResponsesAPIResponse | None:
        """Response body carried by a terminal ``/v1/responses`` event.

        A stream cut short before it completes has to read as unassembled rather than as a clean
        empty response: ``response.created`` also carries a body, but an empty one, and scanning
        that would release every buffered delta unscanned.
        """
        return next(
            (
                body
                for chunk in reversed(all_chunks)
                if getattr(chunk, "type", None) in _RESPONSES_TERMINAL_EVENT_TYPES
                and isinstance(body := getattr(chunk, "response", None), ResponsesAPIResponse)
            ),
            None,
        )

    @staticmethod
    def _responses_api_response_text(response: ResponsesAPIResponse) -> str:
        """Text to scan in a Responses API response, tool-call arguments included.

        Tool calls are folded in because ``get_content_from_model_response`` folds them into what
        the chat surface scans, and a Responses turn can carry its whole payload in them.
        """
        from litellm.llms.openai.responses.guardrail_translation.handler import (
            OpenAIResponsesHandler,
        )

        texts: Final[list[str]] = []  # mutable-ok: the shared extractor below appends into caller-owned lists
        tool_calls: Final[list[ChatCompletionToolCallChunk]] = []  # mutable-ok: the same extractor's tool-call sink
        handler: Final = OpenAIResponsesHandler()
        for output_idx, output_item in enumerate(response.output or ()):
            handler._extract_output_text_and_images(  # pyright: ignore[reportPrivateUsage]  # the shared Responses output extractor; forking it would duplicate per-item parsing
                output_item=output_item,
                output_idx=output_idx,
                texts_to_check=texts,
                images_to_check=[],  # mutable-ok: the extractor's images sink, unused here
                task_mappings=[],  # mutable-ok: the extractor's task-mapping sink, unused here
                tool_calls_to_check=tool_calls,
            )
        return "".join((*texts, *(json.dumps(tool_call) for tool_call in tool_calls)))

    def _extract_streaming_content(self, assembled_response: object) -> str:
        """Text to scan from an assembled stream, for every endpoint shape this hook serves."""
        if isinstance(assembled_response, ResponsesAPIResponse):
            return self._responses_api_response_text(assembled_response)
        return self._extract_content_from_response(assembled_response)

    @staticmethod
    def _responses_delta_text(all_chunks: Sequence[object]) -> str:
        """Text a ``/v1/responses`` stream has already spelled out in its delta events."""
        return "".join(
            delta
            for chunk in all_chunks
            if getattr(chunk, "type", None) in _RESPONSES_DELTA_EVENT_TYPES
            and isinstance(delta := getattr(chunk, "delta", None), str)
        )

    def _streaming_content_to_scan(
        self,
        assembled_response: object,
        all_chunks: Sequence[object],
        surface: _StreamSurface,
    ) -> str:
        """Text to scan for a buffered stream, which is everything the client is about to receive.

        A ``/v1/responses`` stream also spells out reasoning summaries and tool-call arguments in
        delta events that its terminal body never repeats, so body and deltas are scanned together.
        """
        content: Final = self._extract_streaming_content(assembled_response)
        if surface is not _StreamSurface.RESPONSES:
            return content
        delta_text: Final = self._responses_delta_text(all_chunks)
        if delta_text in content:
            return content
        if content in delta_text:
            return delta_text
        return f"{content}\n{delta_text}"

    @staticmethod
    def _apply_sanitized_content(assembled_response: ModelResponse, sanitized_content: str) -> None:
        """Replace every non-empty choice message with the Model Armor sanitized text."""
        for choice in assembled_response.choices:
            if isinstance(choice, Choices) and choice.message.content:
                choice.message.content = sanitized_content

    @staticmethod
    def _assemble_chat_completion_stream(
        all_chunks: list[object],  # mutable-ok: stream_chunk_builder only accepts a mutable list
    ) -> ModelResponse | TextCompletionResponse | None:
        """Assemble chat-completion chunks, returning ``None`` when they cannot be assembled."""
        from litellm.main import stream_chunk_builder

        try:
            return stream_chunk_builder(chunks=all_chunks)
        except Exception as exc:
            verbose_proxy_logger.warning("Model Armor: chat-completion stream assembly failed (%s)", exc)
            return None

    def _assemble_stream(
        self, all_chunks: Sequence[object], surface: _StreamSurface
    ) -> ModelResponse | TextCompletionResponse | ResponsesAPIResponse | None:
        """Assemble the buffered stream into the scannable response its surface produces."""
        if surface is _StreamSurface.ANTHROPIC_MESSAGES:
            return assemble_anthropic_sse_stream(all_chunks, restore_identity=True)
        if surface is _StreamSurface.RESPONSES:
            return self._final_responses_api_response(all_chunks)
        if surface is _StreamSurface.OPAQUE_SSE:
            return None
        return self._assemble_chat_completion_stream(list(all_chunks))

    @staticmethod
    def _error_payload(exc: HTTPException) -> Mapping[str, object]:
        """Error object for a terminal stream item, carrying the status the frame would otherwise lose."""
        detail: Final = exc.detail if isinstance(exc.detail, Mapping) else {"message": str(exc.detail)}
        error_value: Final = detail.get("error", detail)
        return {
            **(dict(error_value) if isinstance(error_value, Mapping) else {"message": str(error_value)}),
            "code": str(exc.status_code),
        }

    @staticmethod
    def _build_responses_error_items(exc: HTTPException) -> Sequence[object] | None:
        """Responses API error events for a failure discovered after the stream started."""
        from litellm.llms.openai.responses.guardrail_translation.handler import (
            OpenAIResponsesHandler,
        )

        return OpenAIResponsesHandler().build_stream_error_items(exc, responses_so_far=None)

    def _stream_error_items(self, exc: HTTPException, *, surface: _StreamSurface) -> Sequence[object]:
        """Frame a guardrail failure as terminal stream items in this endpoint's wire format."""
        payload: Final = self._error_payload(exc)
        if surface is _StreamSurface.ANTHROPIC_MESSAGES:
            return anthropic_sse_error_frames(str(payload.get("message", "")))
        if surface is _StreamSurface.RESPONSES and (responses_items := self._build_responses_error_items(exc)):
            return responses_items
        # Also the fallback when a surface cannot frame its own error: create_response() reads the
        # status back out of this form, so the refusal keeps its code instead of arriving as a 200
        return (f"data: {json.dumps({'error': payload})}\n\n",)

    async def async_post_call_streaming_iterator_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        response: Any,
        request_data: dict,
    ) -> AsyncGenerator[ModelResponseStream, None]:
        """Process streaming response chunks."""

        from litellm.llms.base_llm.base_model_iterator import MockResponseIterator
        from litellm.proxy.common_utils.callback_utils import (
            add_guardrail_to_applied_guardrails_header,
        )

        # Collect all chunks
        all_chunks: Final[list[Any]] = []
        async for chunk in response:
            all_chunks.append(chunk)

        if not all_chunks or self._is_terminal_error_stream(all_chunks):
            for chunk in all_chunks:
                yield chunk
            return

        surface: Final = self._classify_stream(all_chunks)

        # Build complete response
        assembled_response: Final = self._assemble_stream(all_chunks, surface)

        if assembled_response is None:
            if not self.optional_params.get("fail_on_error", True):
                verbose_proxy_logger.warning(
                    "Model Armor: streamed response could not be assembled for scanning, "
                    "forwarding it unscanned because fail_on_error is disabled"
                )
                for chunk in all_chunks:
                    yield chunk
                return

            # Forwarding an unscannable stream would silently disable the guardrail, so fail closed
            add_guardrail_to_applied_guardrails_header(request_data=request_data, guardrail_name=self.guardrail_name)
            for error_item in self._stream_error_items(
                HTTPException(
                    status_code=500,
                    detail=f"{self.guardrail_name}: streamed response could not be assembled for scanning, blocking it",
                ),
                surface=surface,
            ):
                yield error_item
            return

        # Extract content
        content: Final = self._streaming_content_to_scan(
            assembled_response=assembled_response, all_chunks=all_chunks, surface=surface
        )

        if not content:
            verbose_proxy_logger.debug("Model Armor: No text content in streaming response, skipping guardrail")
            for chunk in all_chunks:
                yield chunk
            return

        try:
            # Check with Model Armor
            armor_response: Final = await self.make_model_armor_request(
                content=content,
                source="model_response",
                request_data=request_data,
            )

            # Decide the outcome before recording it. Mirrors the non-streaming sibling: with
            # masking on, a de-identify match is a redaction to apply rather than a refusal, but
            # that only holds while the redaction can actually be delivered
            blocked, sanitized_content = self._resolve_streaming_outcome(
                armor_response=armor_response,
                assembled_response=assembled_response,
                content=content,
            )

            # Attach Model Armor response & status to this request's metadata to avoid race conditions
            if isinstance(request_data, dict):
                _, metadata = get_or_create_metadata_bucket(request_data)
                metadata["_model_armor_response"] = self._build_logging_response(armor_response)
                metadata["_model_armor_status"] = "blocked" if blocked else "success"

            # Add guardrail to applied_guardrails BEFORE potential blocking
            # This ensures guardrail is recorded even when it blocks the request
            add_guardrail_to_applied_guardrails_header(request_data=request_data, guardrail_name=self.guardrail_name)

            if blocked:
                raise HTTPException(
                    status_code=400,
                    detail=self._build_block_error_detail(
                        "Streaming response blocked by Model Armor",
                        armor_response,
                    ),
                )

            if sanitized_content is not None and isinstance(assembled_response, ModelResponse):
                self._apply_sanitized_content(assembled_response, sanitized_content)

                # Return sanitized stream
                if surface is _StreamSurface.ANTHROPIC_MESSAGES:
                    for sse_chunk in anthropic_sse_chunks_from_response(assembled_response):
                        yield sse_chunk
                    return
                mock_response: Final = MockResponseIterator(model_response=assembled_response)
                async for chunk in mock_response:
                    yield chunk
                return

        except ModelArmorAPIError as e:
            if self.optional_params.get("fail_on_error", True):
                for error_item in self._stream_error_items(
                    HTTPException(status_code=500, detail=e.detail), surface=surface
                ):
                    yield error_item
                return
        except HTTPException as e:
            # Yield the error as a terminal stream item so create_response() detects it and returns
            # a proper JSON error response with the correct status code. Raising from a generator
            # instead hits create_response's generic except and becomes a 500.
            for error_item in self._stream_error_items(e, surface=surface):
                yield error_item
            return
        except Exception as e:
            verbose_proxy_logger.error("Model Armor streaming error: %s", str(e), exc_info=True)
            if self.optional_params.get("fail_on_error", True):
                raise

        # Return original chunks if no sanitization needed
        for chunk in all_chunks:
            yield chunk

    @staticmethod
    def get_config_model() -> type["GuardrailConfigModel"] | None:
        """
        Get the config model for the Model Armor guardrail.
        """
        from litellm.types.proxy.guardrails.guardrail_hooks.model_armor import (
            ModelArmorGuardrailConfigModel,
        )

        return ModelArmorGuardrailConfigModel
