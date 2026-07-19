from typing import (
    TYPE_CHECKING,
    Any,
    AsyncGenerator,
    List,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Type,
    Union,
)

import httpx
from fastapi import HTTPException

if TYPE_CHECKING:
    from litellm.types.proxy.guardrails.guardrail_hooks.base import GuardrailConfigModel

import json

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.caching import DualCache
from litellm.integrations.custom_guardrail import (
    CustomGuardrail,
    log_guardrail_information,
)
from litellm.llms.custom_httpx.http_handler import (
    get_async_httpx_client,
    httpxSpecialProvider,
)
from litellm.llms.vertex_ai.vertex_llm_base import VertexBase
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.guardrails.guardrail_hooks.model_armor.file_scanning import (
    MODEL_ARMOR_MAX_FILE_SIZE_BYTES,
    plan_file_scans,
)
from litellm.types.guardrails import GuardrailEventHooks
from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import (
    CallTypes,
    CallTypesLiteral,
    Choices,
    GuardrailStatus,
    ModelResponse,
    ModelResponseStream,
    StandardLoggingGuardrailInformation,
)

GUARDRAIL_NAME = "model_armor"


_SCANNED_CONTENT_KEYS = frozenset({"text", "sanitizedText", "findings"})

_REDACT_MAX_DEPTH = 20

RedactablePayload = Union[dict, list, str, int, float, bool, None]


def _redact_scanned_content(payload: RedactablePayload, depth: int = 0) -> RedactablePayload:
    if depth >= _REDACT_MAX_DEPTH:
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
    def get_supported_event_hooks(cls) -> List[GuardrailEventHooks]:
        return [
            GuardrailEventHooks.pre_call,
            GuardrailEventHooks.during_call,
            GuardrailEventHooks.post_call,
            GuardrailEventHooks.pre_mcp_call,
            GuardrailEventHooks.during_mcp_call,
        ]

    def __init__(
        self,
        template_id: Optional[str] = None,
        project_id: Optional[str] = None,
        location: Optional[str] = None,
        credentials: Optional[Any] = None,
        api_endpoint: Optional[str] = None,
        sanitize_error_detail: bool = True,
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

    def _extract_content_from_response(self, response: Union[Any, ModelResponse]) -> str:
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

    async def make_model_armor_request(
        self,
        content: Optional[str] = None,
        source: Literal["user_prompt", "model_response"] = "user_prompt",
        request_data: Optional[dict] = None,
        file_bytes: Optional[bytes] = None,
        file_type: Optional[str] = None,
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
        endpoint = self._get_api_endpoint()
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
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {access_token}",
        }

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

        # Make request
        if self.async_handler is None:
            raise ValueError("Async handler not initialized")

        try:
            response = await self.async_handler.post(
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
            raise HTTPException(status_code=400, detail=detail) from None

        if self.sanitize_error_detail:
            verbose_proxy_logger.debug(
                "Model Armor response - Status: %s",
                response.status_code,
            )
        else:
            verbose_proxy_logger.debug(
                "Model Armor response - Status: %s, Body: %s",
                response.status_code,
                response.text,
            )

        if response.status_code != 200:
            detail = self._build_api_error_detail(response.status_code, response.text)
            verbose_proxy_logger.error(
                "Model Armor API error - Status: %s, Detail: %s",
                response.status_code,
                detail,
            )
            raise HTTPException(status_code=400, detail=detail)

        json_response = response.json()
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

        base64_data = base64.b64encode(file_bytes).decode("utf-8")
        if source == "user_prompt":
            return {"userPromptData": {"byteItem": {"byteDataType": file_type, "byteData": base64_data}}}
        else:
            return {"modelResponseData": {"byteItem": {"byteDataType": file_type, "byteData": base64_data}}}

    def _should_block_content(self, armor_response: dict, allow_sanitization: bool = False) -> bool:
        """Check if Model Armor response indicates content should be blocked, including both inspectResult and deidentifyResult."""
        sanitization_result = armor_response.get("sanitizationResult", {})
        filter_results = sanitization_result.get("filterResults", {})

        # filterResults can be a dict (named keys) or a list (array of filter result dicts)
        filter_result_items = []
        if isinstance(filter_results, dict):
            filter_result_items = list(filter_results.values())
        elif isinstance(filter_results, list):
            filter_result_items = filter_results

        for filt in filter_result_items:
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

    def _get_sanitized_content(self, armor_response: dict) -> Optional[str]:
        """
        Get the sanitized content from a Model Armor response, if available.
        Looks for sanitized text in deidentifyResult, and falls back to root-level fields if not found.
        """
        result = armor_response.get("sanitizationResult", {})
        filter_results = result.get("filterResults", {})

        # filterResults can be a dict (single filter) or a list (multiple filters)
        filters = (
            list(filter_results.values())
            if isinstance(filter_results, dict)
            else filter_results
            if isinstance(filter_results, list)
            else []
        )

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
        response: Optional[dict],
        request_data: dict,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        duration: Optional[float] = None,
        event_type: Optional[GuardrailEventHooks] = None,
        original_inputs: Optional[dict] = None,
    ):
        """
        Override to store only the Model Armor API response, not the entire data dict.
        This prevents circular references in logging.
        """
        metadata = request_data.get("metadata", {}) if isinstance(request_data, dict) else {}
        guardrail_response = self._build_logging_response(metadata.get("_model_armor_response", {}))

        # Determine status – default to "success" but prefer the explicit value if present.
        guardrail_status: GuardrailStatus = metadata.get("_model_armor_status", "success")  # type: ignore

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
            _get_or_create_proxy_metadata_bucket,
            add_guardrail_to_applied_guardrails_header,
        )

        plan = plan_file_scans(messages)
        attachments = plan.attachments
        skip_unscannable = bool(self.optional_params.get("skip_unscannable_attachments", False))
        if skip_unscannable and plan.unscannable_count > 0:
            verbose_proxy_logger.warning(
                "Model Armor: allowing %d unscannable attachment(s) through because "
                "skip_unscannable_attachments is enabled",
                plan.unscannable_count,
            )
        unscannable_references = 0 if skip_unscannable else plan.unscannable_count
        if not attachments and unscannable_references == 0:
            return

        add_guardrail_to_applied_guardrails_header(request_data=data, guardrail_name=self.guardrail_name)
        # Use the same metadata bucket the header helper writes to, so the logged Model Armor
        # payload and status land where _process_response reads them on every route.
        _, metadata = _get_or_create_proxy_metadata_bucket(data)
        fail_on_error = bool(self.optional_params.get("fail_on_error", True))

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
    ) -> Union[Exception, str, dict, None]:
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

        messages = data.get("messages")
        if not messages:
            verbose_proxy_logger.warning("Model Armor: not running guardrail. No messages in data")
            return data

        # Extract content from messages using helper from common_utils
        from litellm.litellm_core_utils.prompt_templates.common_utils import (
            get_last_user_message,
        )

        await self._scan_request_files(messages=messages, data=data)

        content = get_last_user_message(messages)
        if not content:
            return data

        # Make Model Armor request
        try:
            armor_response = await self.make_model_armor_request(
                content=content,
                source="user_prompt",
                request_data=data,
            )

            # Store the armor response for logging
            # Attach Model Armor response + evaluation status directly to the per-request metadata to avoid
            #   race-conditions between concurrent requests which share the same guardrail instance.
            #   This ensures each request logs its own Model Armor response instead of a potentially stale value
            #   overwritten by another coroutine.
            blocked = self._should_block_content(armor_response, allow_sanitization=self.mask_request_content)
            if isinstance(data, dict):
                metadata = data.setdefault("metadata", {})  # ensures metadata exists and is unique per request
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
                sanitized_content = self._get_sanitized_content(armor_response)
                if sanitized_content and sanitized_content != content:
                    # Use the helper to set the last user message with sanitized content
                    from litellm.litellm_core_utils.prompt_templates.common_utils import (
                        set_last_user_message,
                    )

                    data["messages"] = set_last_user_message(messages, sanitized_content)

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
    ) -> Union[Exception, str, dict, None]:
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

        messages = data.get("messages")
        if not messages:
            verbose_proxy_logger.warning("Model Armor: not running guardrail. No messages in data")
            return data

        # Extract content from messages
        from litellm.litellm_core_utils.prompt_templates.common_utils import (
            get_last_user_message,
        )

        await self._scan_request_files(messages=messages, data=data)

        content = get_last_user_message(messages)
        if not content:
            return data

        # Make Model Armor request
        try:
            armor_response = await self.make_model_armor_request(
                content=content,
                source="user_prompt",
                request_data=data,
            )

            blocked = self._should_block_content(armor_response, allow_sanitization=self.mask_request_content)
            # Store the armor response for logging
            if isinstance(data, dict):
                metadata = data.setdefault("metadata", {})
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
                sanitized_content = self._get_sanitized_content(armor_response)
                if sanitized_content and sanitized_content != content:
                    from litellm.litellm_core_utils.prompt_templates.common_utils import (
                        set_last_user_message,
                    )

                    data["messages"] = set_last_user_message(messages, sanitized_content)

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
        content = self._extract_content_from_response(response)
        if not content:
            verbose_proxy_logger.debug("Model Armor: No text content to process in response, skipping guardrail")
            return

        # Make Model Armor request
        try:
            armor_response = await self.make_model_armor_request(
                content=content,
                source="model_response",
                request_data=data,
            )

            # Attach Model Armor response & status to this request's metadata to prevent race conditions
            if isinstance(armor_response, dict):
                model_armor_logged_object = {
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
                standard_logging_guardrail_information = StandardLoggingGuardrailInformation(
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
                sanitized_content = self._get_sanitized_content(armor_response)
                if sanitized_content and sanitized_content != content:
                    # Update response content
                    if isinstance(response, litellm.ModelResponse):
                        for choice in response.choices:
                            if isinstance(choice, Choices):
                                if choice.message.content:
                                    choice.message.content = sanitized_content

        except HTTPException:
            raise
        except Exception as e:
            verbose_proxy_logger.error("Model Armor post-call error: %s", str(e), exc_info=True)
            if self.optional_params.get("fail_on_error", True):
                raise

        return response

    async def async_post_call_streaming_iterator_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        response: Any,
        request_data: dict,
    ) -> AsyncGenerator[ModelResponseStream, None]:
        """Process streaming response chunks."""

        from litellm.llms.base_llm.base_model_iterator import MockResponseIterator
        from litellm.main import stream_chunk_builder

        # Collect all chunks
        all_chunks: List[ModelResponseStream] = []
        async for chunk in response:
            all_chunks.append(chunk)

        # Build complete response
        assembled_response = stream_chunk_builder(chunks=all_chunks)

        if isinstance(assembled_response, ModelResponse):
            # Extract content
            content = self._extract_content_from_response(assembled_response)

            if content:
                try:
                    # Check with Model Armor
                    armor_response = await self.make_model_armor_request(
                        content=content,
                        source="model_response",
                        request_data=request_data,
                    )

                    # Attach Model Armor response & status to this request's metadata to avoid race conditions
                    if isinstance(request_data, dict):
                        metadata = request_data.setdefault("metadata", {})
                        metadata["_model_armor_response"] = self._build_logging_response(armor_response)
                        metadata["_model_armor_status"] = (
                            "blocked" if self._should_block_content(armor_response) else "success"
                        )

                    # Add guardrail to applied_guardrails BEFORE potential blocking
                    # This ensures guardrail is recorded even when it blocks the request
                    from litellm.proxy.common_utils.callback_utils import (
                        add_guardrail_to_applied_guardrails_header,
                    )

                    add_guardrail_to_applied_guardrails_header(
                        request_data=request_data, guardrail_name=self.guardrail_name
                    )

                    # Check if blocked
                    if self._should_block_content(armor_response):
                        raise HTTPException(
                            status_code=400,
                            detail=self._build_block_error_detail(
                                "Streaming response blocked by Model Armor",
                                armor_response,
                            ),
                        )

                    # Apply sanitization if enabled
                    if self.mask_response_content:
                        sanitized_content = self._get_sanitized_content(armor_response)
                        if sanitized_content and sanitized_content != content:
                            # Update assembled response
                            for choice in assembled_response.choices:
                                if isinstance(choice, Choices):
                                    if choice.message.content:
                                        choice.message.content = sanitized_content

                            # Return sanitized stream
                            mock_response = MockResponseIterator(model_response=assembled_response)
                            async for chunk in mock_response:
                                yield chunk
                            return

                except HTTPException as e:
                    # Yield error as SSE event so create_response() detects it and
                    # returns a proper JSON error response with the correct status code.
                    # (Raising from a generator hits create_response's generic except → 500.)
                    detail = e.detail if isinstance(e.detail, dict) else {"message": str(e.detail)}
                    error_value = detail.get("error", detail)
                    if isinstance(error_value, dict):
                        error_obj = dict(error_value)
                    else:
                        error_obj = {"message": str(error_value)}
                    error_obj["code"] = str(e.status_code)
                    yield f"data: {json.dumps({'error': error_obj})}\n\n"  # type: ignore[misc]
                    return
                except Exception as e:
                    verbose_proxy_logger.error("Model Armor streaming error: %s", str(e), exc_info=True)
                    if self.optional_params.get("fail_on_error", True):
                        raise
            else:
                verbose_proxy_logger.debug("Model Armor: No text content in streaming response, skipping guardrail")

        # Return original chunks if no sanitization needed
        for chunk in all_chunks:
            yield chunk

    @staticmethod
    def get_config_model() -> Optional[Type["GuardrailConfigModel"]]:
        """
        Get the config model for the Model Armor guardrail.
        """
        from litellm.types.proxy.guardrails.guardrail_hooks.model_armor import (
            ModelArmorGuardrailConfigModel,
        )

        return ModelArmorGuardrailConfigModel
