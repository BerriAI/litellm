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

import httpx
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
            "Initialized Microsoft Purview DLP Guardrail: %s",
            guardrail_name,
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
        except HTTPException:
            status = "guardrail_failed_to_respond"
            raise
        except httpx.HTTPStatusError as exc:
            # Preserve the upstream Graph API status code (e.g. 429, 503) so
            # callers can distinguish a transient infrastructure error from a
            # DLP policy block (signaled separately as HTTP 400 below) and can
            # implement retry-after handling on rate limits.  401/403 upstream
            # responses indicate a proxy-side credential / consent problem the
            # caller can do nothing about, so they are mapped to 502.
            status = "guardrail_failed_to_respond"
            if block_on_violation:
                upstream_status = exc.response.status_code
                client_status = (
                    502 if upstream_status in (401, 403) else upstream_status
                )
                headers: Optional[Dict[str, str]] = None
                retry_after = exc.response.headers.get("retry-after")
                if retry_after:
                    headers = {"Retry-After": retry_after}
                raise HTTPException(
                    status_code=client_status,
                    detail={
                        "error": "Microsoft Purview DLP: upstream policy evaluation failed",
                        "activity": activity,
                        "upstream_status": upstream_status,
                        "exception": str(exc),
                    },
                    headers=headers,
                ) from exc
            verbose_proxy_logger.warning(
                "Purview DLP: API/network error in logging-only mode (not re-raised): %s",
                exc,
            )
        except Exception as exc:
            status = "guardrail_failed_to_respond"
            if block_on_violation:
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": "Microsoft Purview DLP: upstream policy evaluation failed",
                        "activity": activity,
                        "exception": str(exc),
                    },
                ) from exc
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

    @staticmethod
    def _extract_responses_api_function_call_args(result: Any) -> List[str]:
        """Return tool-call argument strings from a ``ResponsesAPIResponse.output``.

        ``ResponsesAPIResponse.output_text`` only aggregates ``output_text``
        content blocks and ignores ``function_call`` items.  Model-generated
        tool-call arguments can themselves contain sensitive data, so we
        extract them explicitly to keep DLP coverage consistent with the
        chat (``ModelResponse``) path.
        """
        args: List[str] = []
        output = getattr(result, "output", None)
        if not output:
            return args
        for item in output:
            if isinstance(item, dict):
                item_type = item.get("type")
                arguments = item.get("arguments")
            else:
                item_type = getattr(item, "type", None)
                arguments = getattr(item, "arguments", None)
            if item_type == "function_call" and isinstance(arguments, str):
                if arguments.strip():
                    args.append(arguments)
        return args

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
            # Include tool-call arguments from ``function_call`` output items
            # (``output_text`` ignores them).
            parts.extend(self._extract_responses_api_function_call_args(result))
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

    def _assemble_responses_api_from_chunks(
        self, chunks: List[Any]
    ) -> Tuple[bool, Optional[ResponsesAPIResponse]]:
        """Extract the final ``ResponsesAPIResponse`` from a buffered Responses API stream.

        Returns a ``(is_responses_api_stream, assembled)`` tuple so the caller
        can distinguish "not a Responses API stream" (fall through to
        ``stream_chunk_builder``) from "Responses API stream but no final
        response event was received" (fail closed with an accurate error).
        When the stream is a Responses API stream the latest event carrying a
        ``ResponsesAPIResponse`` body is returned (``response.completed``, or
        ``response.failed`` / ``response.incomplete`` as fallbacks).
        """
        looks_like_responses_api = False
        final: Optional[ResponsesAPIResponse] = None
        for chunk in chunks:
            event_type = getattr(chunk, "type", None)
            if isinstance(event_type, str) and event_type.startswith("response."):
                looks_like_responses_api = True
            candidate = getattr(chunk, "response", None)
            if isinstance(candidate, ResponsesAPIResponse):
                final = candidate
        return looks_like_responses_api, final

    def _responses_api_input_to_str(
        self, data: Dict[str, Any], raise_on_failure: bool = False
    ) -> Optional[str]:
        """Extract DLP-scannable text from a Responses API request ``input`` field.

        ``input`` may be a plain string or a list of input items (messages).  In
        the latter case the items are converted to chat messages via the standard
        LiteLLM transformation and then concatenated by ``get_prompt_text_for_dlp``.

        When ``raise_on_failure`` is True (blocking mode), a transformation error
        raises ``HTTPException`` so the request is fail-closed.  In logging-only
        mode the error is swallowed and ``None`` is returned so audit attempts on
        the response side can still run.
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
            verbose_proxy_logger.warning(
                "Purview DLP: failed to transform responses API input",
                exc_info=True,
            )
            if raise_on_failure:
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": (
                            "Microsoft Purview DLP: Responses API input could "
                            "not be transformed for DLP scanning in blocking mode"
                        ),
                    },
                )
            return None

    # ------------------------------------------------------------------
    # Identity resolution for blocking modes
    # ------------------------------------------------------------------

    def _resolve_user_id_for_blocking(
        self,
        data: Dict[str, Any],
        user_api_key_dict: Any,
    ) -> str:
        """Resolve user ID for blocking (pre_call / post_call) DLP hooks.

        Uses only trusted proxy-authenticated sources (``_resolve_trusted_user_id``).
        Caller-supplied ``UserAPIKeyAuth.end_user_id`` (from request ``user``,
        ``metadata.user_id``, ``safety_identifier``, etc.) and
        ``metadata[user_id_field]`` are rejected (fail closed) because they can
        impersonate another Entra user's Purview policy.

        Raises ``HTTPException`` when no API-key-bound ``user_id`` exists or when
        only caller-influenceable identity fields are available (fail closed).
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

        raise HTTPException(
            status_code=400,
            detail={
                "error": (
                    "Microsoft Purview DLP: No proxy-authenticated user identity; "
                    "bind user_id to the API key for blocking DLP"
                ),
            },
        )

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

        prompt_text: Optional[str] = None
        if call_type in ("responses", "aresponses"):
            # Route Responses API calls to the responses-specific extractor
            # before the generic ``messages`` branch.  This mirrors
            # ``async_logging_hook`` and ensures ``instructions`` (system
            # prompt) content is included in the DLP scan, and prevents a
            # crafted ``messages`` key in the request from being scanned in
            # place of the actual ``input``.
            prompt_text = self._responses_api_input_to_str(data, raise_on_failure=True)
        elif call_type in ("text_completion", "atext_completion"):
            raw_prompt = data.get("prompt")
            # Reject every token-id prompt shape Purview cannot evaluate —
            # flat ``list[int]`` (single prompt), ``list[list[int]]`` (multi-prompt
            # batches), and mixed lists that include any token-id sub-array.
            # Empty/whitespace-only strings also yield ``prompt_text is None`` but
            # contain no sensitive data and pass through harmlessly below.
            if self.is_token_id_prompt(raw_prompt):
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": (
                            "Microsoft Purview DLP: Token-id completion prompts "
                            "cannot be scanned for DLP in blocking mode"
                        ),
                    },
                )
            prompt_text = self.completion_prompt_to_str(raw_prompt)
        else:
            messages: Optional[List] = data.get("messages")
            if messages:
                prompt_text = self.get_prompt_text_for_dlp(cast(List[Any], messages))

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

    @log_guardrail_information
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

        # Resolve user ID up-front so identity failures don't waste work
        # buffering and assembling the stream.
        user_id = self._resolve_user_id_for_blocking(request_data, user_api_key_dict)

        # Buffer the entire stream before any DLP scan.
        all_chunks: List[ModelResponseStream] = []
        async for chunk in response:
            all_chunks.append(chunk)

        # Responses API streams emit typed events (e.g. ``response.completed``)
        # whose final event carries the full ``ResponsesAPIResponse`` — these
        # are not understood by ``stream_chunk_builder`` (which is built for
        # chat/text-completion deltas). Detect and scan them via the same
        # ``_completion_response_text_parts`` path used by non-streaming.
        (
            is_responses_api_stream,
            responses_api_assembled,
        ) = self._assemble_responses_api_from_chunks(all_chunks)
        if is_responses_api_stream:
            if responses_api_assembled is None:
                # Fail closed: Responses API events were seen but no final
                # ``response.completed`` / ``response.failed`` /
                # ``response.incomplete`` event carrying a ``ResponsesAPIResponse``
                # body was received, so we cannot scan the content.
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": (
                            "Microsoft Purview DLP: Incomplete Responses API "
                            "stream — no final response event received for "
                            "DLP scanning; blocking response."
                        ),
                    },
                )
            parts = self._completion_response_text_parts(responses_api_assembled)
            if parts:
                combined = "\n\n---\n\n".join(parts)
                await self._check_content(
                    user_id=user_id,
                    text=combined,
                    activity="downloadText",
                    request_data=request_data,
                    block_on_violation=True,
                )
            for chunk in all_chunks:
                yield chunk
            return

        assembled_response = stream_chunk_builder(chunks=all_chunks)

        if assembled_response is None and all_chunks:
            # Fail closed: stream_chunk_builder dropped all chunks, so we cannot
            # scan the content. Refuse to release the buffered chunks.
            raise HTTPException(
                status_code=400,
                detail={
                    "error": (
                        "Microsoft Purview DLP: Unable to assemble streamed "
                        "response for scanning; blocking response."
                    ),
                },
            )

        if isinstance(
            assembled_response, (TextCompletionResponse, ResponsesAPIResponse)
        ):
            parts = self._completion_response_text_parts(assembled_response)
            if parts:
                combined = "\n\n---\n\n".join(parts)
                await self._check_content(
                    user_id=user_id,
                    text=combined,
                    activity="downloadText",
                    request_data=request_data,
                    block_on_violation=True,
                )
            for chunk in all_chunks:
                yield chunk
            return

        if not isinstance(assembled_response, ModelResponse):
            # Non-content response (e.g. embeddings) — pass through unchanged.
            for chunk in all_chunks:
                yield chunk
            return

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

        # DLP passed — re-yield chunks from the assembled chat response.
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

        In the proxy's async success path, litellm independently calls both
        ``logging_hook`` (sync) and ``async_logging_hook`` (async) for every
        ``CustomGuardrail`` callback.  To avoid making two complete sets of
        Purview API calls per request, this sync hook is a no-op whenever an
        event loop is running — the framework's async path will invoke
        ``async_logging_hook`` directly.

        For genuine sync-only call paths (no running event loop, so the async
        success handler will not fire either), schedule ``async_logging_hook``
        on a short-lived background daemon thread so audit logging still runs
        without blocking the caller on two Graph API round-trips.
        """

        try:
            asyncio.get_running_loop()
            # Async context — let the framework's async success handler invoke
            # async_logging_hook to avoid duplicate Purview API calls.  Log so
            # the deferral is observable if the framework ever stops dispatching
            # async_logging_hook on a given code path (otherwise audit silently
            # drops).
            verbose_proxy_logger.debug(
                "Purview audit: deferring to async_logging_hook (running event loop detected)"
            )
            return kwargs, result
        except RuntimeError:
            pass

        async def _log_safe() -> None:
            try:
                await self.async_logging_hook(
                    kwargs=kwargs, result=result, call_type=call_type
                )
            except Exception as exc:
                verbose_proxy_logger.error(
                    "Purview audit background logging error: %s", exc
                )

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
            if call_type in ("responses", "aresponses"):
                # Responses API: route to the responses-specific extractor
                # before the generic ``messages`` branch.  litellm's logging
                # pipeline stores the raw responses ``input`` (a string or a
                # list of input items) under ``model_call_details["messages"]``
                # via ``function_setup``, which is NOT the chat message format
                # ``get_prompt_text_for_dlp`` expects.  Use the original
                # ``input`` / ``instructions`` keys that ``pre_call`` and
                # ``update_environment_variables`` persist on the call details.
                prompt_text = self._responses_api_input_to_str(kwargs)
            elif call_type in ("text_completion", "atext_completion"):
                prompt_text = self.completion_prompt_to_str(kwargs.get("prompt"))
            else:
                messages = kwargs.get("messages")
                if messages:
                    prompt_text = self.get_prompt_text_for_dlp(
                        cast(List[Any], messages)
                    )

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
