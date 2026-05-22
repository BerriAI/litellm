"""
Microsoft Purview DLP Guardrail for LiteLLM.

Supports three modes:
- pre_call:      Block sensitive data in prompts before they reach the LLM.
- post_call:     Block sensitive data in LLM responses.
- logging_only:  Log interactions to Purview for audit/compliance without blocking.
"""

import asyncio
import threading
import uuid
from datetime import datetime
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncGenerator,
    Dict,
    List,
    Optional,
    Tuple,
    Type,
    Union,
    cast,
)

from fastapi import HTTPException

from litellm._logging import verbose_proxy_logger
from litellm.integrations.custom_guardrail import (
    CustomGuardrail,
    log_guardrail_information,
)
from litellm.types.guardrails import GuardrailEventHooks
from litellm.types.utils import (
    Choices,
    GuardrailStatus,
    ModelResponse,
    ModelResponseStream,
    ResponsesAPIResponse,
    TextChoices,
    TextCompletionResponse,
)

from .base import PurviewGuardrailBase

if TYPE_CHECKING:
    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.types.proxy.guardrails.guardrail_hooks.base import (
        GuardrailConfigModel,
    )
    from litellm.types.utils import (
        CallTypesLiteral,
        EmbeddingResponse,
        ImageResponse,
    )


class MicrosoftPurviewDLPGuardrail(PurviewGuardrailBase, CustomGuardrail):
    """
    Microsoft Purview DLP guardrail.

    Evaluates prompts and responses against Microsoft Purview DLP policies
    via the Microsoft Graph ``processContent`` API.
    """

    def __init__(
        self,
        guardrail_name: str,
        tenant_id: str,
        client_id: str,
        client_secret: str,
        purview_app_name: str = "LiteLLM",
        user_id_field: str = "user_id",
        logging_only: bool = False,
        **kwargs: Any,
    ):
        supported_event_hooks = [
            GuardrailEventHooks.pre_call,
            GuardrailEventHooks.post_call,
            GuardrailEventHooks.logging_only,
        ]

        super().__init__(
            tenant_id=tenant_id,
            client_id=client_id,
            client_secret=client_secret,
            purview_app_name=purview_app_name,
            user_id_field=user_id_field,
            guardrail_name=guardrail_name,
            supported_event_hooks=supported_event_hooks,
            **kwargs,
        )
        self.guardrail_provider = "microsoft_purview"
        verbose_proxy_logger.info(
            "Initialized Microsoft Purview DLP Guardrail: %s (logging_only=%s)",
            guardrail_name,
            logging_only,
        )

    @staticmethod
    def get_config_model() -> Optional[Type["GuardrailConfigModel"]]:
        return None  # Config model can be added later for UI support

    # ------------------------------------------------------------------
    # Core DLP check
    # ------------------------------------------------------------------

    async def _check_content(
        self,
        user_id: str,
        text: str,
        activity: str,
        request_data: Dict[str, Any],
        block_on_violation: bool = True,
    ) -> Dict[str, Any]:
        """Evaluate content against Purview DLP policies.

        Args:
            user_id: Entra object ID.
            text: Content to evaluate.
            activity: ``"uploadText"`` or ``"downloadText"``.
            request_data: Original request dict (used for logging metadata).
            block_on_violation: If False, log only — do not raise.

        Returns:
            The processContent response dict.
        """
        start_time = datetime.now()
        status: GuardrailStatus = "success"
        response: Dict[str, Any] = {}

        try:
            etag, _ = await self._compute_protection_scopes(user_id)
            correlation_id = request_data.get("litellm_call_id") or str(uuid.uuid4())
            response = await self._process_content(
                user_id=user_id,
                text=text,
                activity=activity,
                etag=etag,
                correlation_id=correlation_id,
            )

            if self._should_block(response):
                status = "guardrail_intervened"
        except Exception as exc:
            status = "guardrail_failed_to_respond"
            if block_on_violation:
                raise
            verbose_proxy_logger.warning(
                "Purview DLP: API/network error in logging-only mode (not re-raised): %s",
                exc,
            )
        finally:
            end_time = datetime.now()
            self.add_standard_logging_guardrail_information_to_request_data(
                guardrail_provider=self.guardrail_provider,
                guardrail_json_response=response,
                request_data=request_data,
                guardrail_status=status,
                start_time=start_time.timestamp(),
                end_time=end_time.timestamp(),
                duration=(end_time - start_time).total_seconds(),
            )

        if block_on_violation and status == "guardrail_intervened":
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Microsoft Purview DLP: Content blocked by policy",
                    "activity": activity,
                },
            )

        return response

    def _completion_response_text_parts(self, result: Any) -> List[str]:
        """Collect non-empty text segments from chat, text completions, or responses API.

        Includes assistant message content *and* model-generated tool-call
        arguments so that sensitive data returned inside function calls is not
        missed by the DLP scan.
        """
        parts: List[str] = []
        if isinstance(result, TextCompletionResponse) and result.choices:
            for text_choice in result.choices:
                if not isinstance(text_choice, TextChoices):
                    continue
                raw = text_choice.get("text")
                if isinstance(raw, str) and raw.strip():
                    parts.append(raw)
        elif isinstance(result, ResponsesAPIResponse):
            text = result.output_text
            if text and text.strip():
                parts.append(text)
        elif isinstance(result, ModelResponse) and result.choices:
            for chat_choice in result.choices:
                if not isinstance(chat_choice, Choices):
                    continue
                msg = chat_choice.message
                if msg is None:
                    continue
                raw = (
                    msg.get("content")
                    if isinstance(msg, dict)
                    else getattr(msg, "content", None)
                )
                if isinstance(raw, str) and raw.strip():
                    parts.append(raw)
                # Include tool-call arguments returned by the model
                parts.extend(self._extract_tool_call_args_from_message(msg))
        return parts

    def _responses_api_input_to_str(self, data: Dict[str, Any]) -> Optional[str]:
        """Extract DLP-scannable text from a Responses API request ``input`` field.

        ``input`` may be a plain string or a list of input items (messages).  In
        the latter case the items are converted to chat messages via the standard
        LiteLLM transformation and then concatenated by ``get_prompt_text_for_dlp``.
        """
        from litellm.responses.litellm_completion_transformation.transformation import (
            LiteLLMCompletionResponsesConfig,
        )

        input_data = data.get("input")
        if input_data is None and not data.get("instructions"):
            return None
        try:
            # Always transform via messages so ``instructions`` become a system message
            # (string ``input`` alone would skip instructions and bypass DLP).
            messages = LiteLLMCompletionResponsesConfig.transform_responses_api_input_to_messages(
                input=input_data if input_data is not None else "",
                responses_api_request=data,
            )
            return self.get_prompt_text_for_dlp(cast(List[Any], messages))
        except Exception:
            verbose_proxy_logger.debug(
                "Purview DLP: failed to transform responses API input; skipping scan",
                exc_info=True,
            )
            return None

    # ------------------------------------------------------------------
    # Identity resolution for blocking modes
    # ------------------------------------------------------------------

    def _resolve_user_id_for_blocking(
        self,
        data: Dict[str, Any],
        user_api_key_dict: Any,
    ) -> Optional[str]:
        """Resolve user ID for blocking (pre_call / post_call) DLP hooks.

        Uses only trusted proxy-authenticated sources (``_resolve_trusted_user_id``).
        Caller-supplied ``metadata[user_id_field]`` is rejected (fail closed) because
        it can impersonate another Entra user's Purview policy.

        Returns ``None`` when no trusted identity exists — hooks skip the DLP check.
        """
        trusted_id = self._resolve_trusted_user_id(data, user_api_key_dict)
        if trusted_id:
            return trusted_id

        if self._resolve_user_id(data, user_api_key_dict):
            raise HTTPException(
                status_code=400,
                detail={
                    "error": (
                        "Microsoft Purview DLP: No proxy-authenticated user identity; "
                        "bind user_id to the API key (caller-supplied metadata cannot "
                        "be used for blocking DLP)"
                    ),
                },
            )

        verbose_proxy_logger.warning(
            "Purview DLP: no trusted user_id resolved; skipping DLP check"
        )
        return None

    # ------------------------------------------------------------------
    # Pre-call hook — DLP on prompts
    # ------------------------------------------------------------------

    @log_guardrail_information
    async def async_pre_call_hook(
        self,
        user_api_key_dict: "UserAPIKeyAuth",
        cache: Any,
        data: Dict[str, Any],
        call_type: "CallTypesLiteral",
    ) -> Optional[Dict[str, Any]]:
        """Check user prompt against Purview DLP policies before LLM call."""
        user_id = self._resolve_user_id_for_blocking(data, user_api_key_dict)
        if not user_id:
            return data

        prompt_text: Optional[str] = None
        is_text_completion = call_type in ("text_completion", "atext_completion")
        messages: Optional[List] = data.get("messages")
        if messages:
            prompt_text = self.get_prompt_text_for_dlp(cast(List[Any], messages))
        elif is_text_completion:
            raw_prompt = data.get("prompt")
            prompt_text = self.completion_prompt_to_str(raw_prompt)
            if raw_prompt is not None and prompt_text is None:
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": (
                            "Microsoft Purview DLP: Token-id completion prompts "
                            "cannot be scanned for DLP in blocking mode"
                        ),
                    },
                )
        elif call_type in ("responses", "aresponses"):
            prompt_text = self._responses_api_input_to_str(data)

        if not prompt_text:
            return data

        await self._check_content(
            user_id=user_id,
            text=prompt_text,
            activity="uploadText",
            request_data=data,
            block_on_violation=True,
        )
        return data

    # ------------------------------------------------------------------
    # Post-call hook — DLP on responses
    # ------------------------------------------------------------------

    async def async_post_call_success_hook(
        self,
        data: dict,
        user_api_key_dict: "UserAPIKeyAuth",
        response: Union[Any, ModelResponse, "EmbeddingResponse", "ImageResponse"],
    ) -> Any:
        """Check LLM response against Purview DLP policies (non-streaming only).

        Streaming responses are handled by ``async_post_call_streaming_iterator_hook``
        which buffers all chunks before scanning.  The proxy automatically skips
        this hook for requests that have a streaming iterator hook defined.
        """
        user_id = self._resolve_user_id_for_blocking(data, user_api_key_dict)
        if not user_id:
            return response

        parts = self._completion_response_text_parts(response)

        if parts:
            combined = "\n\n---\n\n".join(parts)
            await self._check_content(
                user_id=user_id,
                text=combined,
                activity="downloadText",
                request_data=data,
                block_on_violation=True,
            )
        return response

    async def async_post_call_streaming_iterator_hook(
        self,
        user_api_key_dict: "UserAPIKeyAuth",
        response: Any,
        request_data: dict,
    ) -> AsyncGenerator[ModelResponseStream, None]:
        """Check streaming LLM responses against Purview DLP policies.

        All chunks are buffered before the DLP scan so that no content is
        delivered to the client if a policy violation is detected.  After a
        clean scan the assembled response is re-yielded chunk-by-chunk via a
        ``MockResponseIterator`` so the caller receives normal streaming output.

        The proxy automatically skips ``async_post_call_success_hook`` for
        guardrails that define this method, preventing duplicate scans.
        """
        from litellm.llms.base_llm.base_model_iterator import MockResponseIterator
        from litellm.main import stream_chunk_builder

        # Buffer the entire stream before any DLP scan.
        all_chunks: List[ModelResponseStream] = []
        async for chunk in response:
            all_chunks.append(chunk)

        assembled_response = stream_chunk_builder(chunks=all_chunks)

        if not isinstance(assembled_response, ModelResponse):
            # Non-chat response (e.g. embeddings) — pass through unchanged.
            for chunk in all_chunks:
                yield chunk
            return

        user_id = self._resolve_user_id_for_blocking(request_data, user_api_key_dict)
        if user_id:
            parts = self._completion_response_text_parts(assembled_response)
            if parts:
                combined = "\n\n---\n\n".join(parts)
                # Raises HTTPException(400) on violation — no chunks are yielded.
                await self._check_content(
                    user_id=user_id,
                    text=combined,
                    activity="downloadText",
                    request_data=request_data,
                    block_on_violation=True,
                )

        # DLP passed (or skipped) — re-yield chunks from the assembled response.
        mock_response = MockResponseIterator(model_response=assembled_response)
        async for chunk in mock_response:
            yield chunk

    # ------------------------------------------------------------------
    # Logging-only hook — audit without blocking
    # ------------------------------------------------------------------

    def logging_hook(
        self, kwargs: dict, result: Any, call_type: str
    ) -> Tuple[dict, Any]:
        """Fire-and-forget async audit logging; returns original (kwargs, result) immediately.

        Unlike the Presidio pattern (which does local text manipulation),
        ``async_logging_hook`` makes two sequential network calls to the
        Microsoft Graph API.  Blocking the calling thread — or worse, the
        event loop thread — until those HTTP round-trips complete would
        significantly degrade throughput.  Since the hook is audit-only and
        always returns ``(kwargs, result)`` unchanged, we can schedule the
        work without waiting and return immediately.
        """

        async def _log_safe() -> None:
            try:
                await self.async_logging_hook(
                    kwargs=kwargs, result=result, call_type=call_type
                )
            except Exception as exc:
                verbose_proxy_logger.error(
                    "Purview audit background logging error: %s", exc
                )

        try:
            loop = asyncio.get_running_loop()
            loop.create_task(_log_safe())
        except RuntimeError:
            # No running event loop — run in a background daemon thread so
            # the caller still isn't blocked.
            def _run_in_new_loop() -> None:
                new_loop = asyncio.new_event_loop()
                try:
                    asyncio.set_event_loop(new_loop)
                    new_loop.run_until_complete(_log_safe())
                finally:
                    new_loop.close()
                    asyncio.set_event_loop(None)

            thread = threading.Thread(target=_run_in_new_loop, daemon=True)
            thread.start()

        return kwargs, result

    async def async_logging_hook(
        self, kwargs: dict, result: Any, call_type: str
    ) -> Tuple[dict, Any]:
        """Send both prompt and response to Purview for audit logging.

        Errors are logged but never raised — this mode is non-blocking.
        Each audit call (prompt and response) is wrapped in its own try/except
        so a failure on the first does not prevent the second from running.
        """
        user_id = self._resolve_user_id_from_logging_kwargs(kwargs)
        if not user_id:
            verbose_proxy_logger.debug("Purview audit: no user_id, skipping")
            return kwargs, result

        # Log prompt (uploadText)
        try:
            prompt_text: Optional[str] = None
            messages = kwargs.get("messages")
            if messages:
                prompt_text = self.get_prompt_text_for_dlp(cast(List[Any], messages))
            elif call_type in ("text_completion", "atext_completion"):
                prompt_text = self.completion_prompt_to_str(kwargs.get("prompt"))
            elif call_type in ("responses", "aresponses"):
                prompt_text = self._responses_api_input_to_str(kwargs)

            if prompt_text:
                await self._check_content(
                    user_id=user_id,
                    text=prompt_text,
                    activity="uploadText",
                    request_data=kwargs,
                    block_on_violation=False,
                )
        except Exception as e:
            verbose_proxy_logger.error("Purview audit logging error (prompt): %s", e)

        # Log response (downloadText) — runs regardless of prompt audit outcome
        try:
            parts = self._completion_response_text_parts(result)
            if parts:
                combined = "\n\n---\n\n".join(parts)
                await self._check_content(
                    user_id=user_id,
                    text=combined,
                    activity="downloadText",
                    request_data=kwargs,
                    block_on_violation=False,
                )
        except Exception as e:
            verbose_proxy_logger.error("Purview audit logging error (response): %s", e)

        return kwargs, result
