"""Rubrik LiteLLM Plugin for prompt/response moderation and batch logging."""

import asyncio
import os
import random
import time
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Optional

import httpx
from litellm._logging import verbose_logger
from litellm.integrations.custom_batch_logger import CustomBatchLogger
from litellm.integrations.custom_guardrail import (
    CustomGuardrail,
    ModifyResponseException,
)
from litellm.litellm_core_utils.core_helpers import safe_deep_copy
from litellm.litellm_core_utils.prompt_templates.common_utils import (
    convert_content_list_to_str,
)
from litellm.llms.custom_httpx.http_handler import (
    get_async_httpx_client,
    httpxSpecialProvider,
)
from litellm.types.guardrails import GuardrailEventHooks
from litellm.types.utils import (
    ChatCompletionMessageToolCall,
    Function,
    GenericGuardrailAPIInputs,
    StandardLoggingPayload,
)

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import (
        Logging as LiteLLMLoggingObj,
    )

_WEBHOOK_PATH_RESPONSE_MODERATION = "/v1/after_completion/openai/v1"
_WEBHOOK_PATH_PROMPT_MODERATION = "/v1/before_prompt/openai/v1"
_WEBHOOK_PATH_LOGGING_BATCH = "/v1/litellm/batch"
_MAX_QUEUE_SIZE = 10_000
_DROP_WARNING_INTERVAL_SECONDS = 60.0


class _MalformedToolBlockingResponseError(Exception):
    """Raised when the response moderation service returns a structurally invalid
    response (e.g. empty ``choices``).

    Distinct from transient network/HTTP errors so callers can surface a
    louder, misconfiguration-style log instead of treating it as a routine
    fail-open.
    """


@dataclass
class BlockedResponseResult:
    """Returned by _extract_response_block when the response was blocked
    (response text replaced, or at least one tool call removed)."""

    explanation: str


class RubrikLogger(CustomGuardrail, CustomBatchLogger):
    def __init__(
        self,
        api_key: str | None = None,
        api_base: str | None = None,
        **kwargs,
    ):
        self.flush_lock = asyncio.Lock()
        kwargs.setdefault("guardrail_name", "rubrik")
        # `initialize_guardrail` always passes these kwargs explicitly, with
        # value `None` when the user omits `mode` / `default_on` from the
        # guardrail config. Coerce None (omitted) to the desired default
        # while preserving any explicit value the caller did set --
        # in particular `default_on=False` if the user wants the guardrail
        # off by default.
        kwargs["event_hook"] = kwargs.get("event_hook") or GuardrailEventHooks.post_call
        if kwargs.get("default_on") is None:
            kwargs["default_on"] = True
        super().__init__(
            flush_lock=self.flush_lock,
            supported_event_hooks=list(self.get_supported_event_hooks()),
            **kwargs,
        )

        verbose_logger.debug("initializing rubrik logger")

        # Defining ``apply_guardrail`` routes streaming responses through
        # litellm's ``unified_guardrail.async_post_call_streaming_iterator_hook``.
        # By default that hook samples intermediate chunks
        # (``streaming_sampling_rate``, default 5) and also moderates at
        # end-of-stream, so a streamed response costs ~ceil(N/5)+1 Rubrik
        # webhook round-trips. litellm reads this attribute via
        # ``getattr(guardrail, "streaming_end_of_stream_only", False)``; when
        # True it yields chunks unprocessed and only moderates the fully
        # assembled response once at end of stream.
        self.streaming_end_of_stream_only = True

        # ``streaming_end_of_stream_only`` is detect-only: it releases every
        # chunk to the client *before* moderating, so a block can only append a
        # trailing message -- the original content has already been delivered.
        # ``streaming_buffer_until_moderated`` (litellm >= BerriAI/litellm#31389)
        # withholds all chunks until end-of-stream moderation passes, then
        # releases the original response (clean) or only the block message
        # (blocked). On older litellm this attribute is ignored and we fall
        # back to the detect-only behavior above.
        self.streaming_buffer_until_moderated = True

        self._parse_sampling_rate()

        self.key = api_key or os.getenv("RUBRIK_API_KEY")
        if not self.key:
            verbose_logger.warning("Rubrik: No API key configured. Requests will be unauthenticated.")

        self._parse_batch_size()

        # Cap the in-memory retry queue so a Rubrik webhook outage cannot let
        # authenticated traffic accumulate prompt/response payloads until the
        # proxy runs out of memory. Once the cap is reached, oldest events are
        # dropped to make room for fresh ones (drop-oldest backpressure).
        self.max_queue_size = _MAX_QUEUE_SIZE
        self._dropped_since_warning = 0
        self._last_drop_warning_time = 0.0

        _webhook_url = api_base or os.getenv("RUBRIK_WEBHOOK_URL")
        if not _webhook_url:
            raise ValueError("Rubrik webhook URL not configured. Set RUBRIK_WEBHOOK_URL or pass api_base.")

        _webhook_url = _webhook_url.rstrip("/").removesuffix("/v1")
        self._setup_clients(_webhook_url)

        self._headers: dict[str, str] = {"Content-Type": "application/json"}
        if self.key:
            self._headers["Authorization"] = f"Bearer {self.key}"

        self._periodic_flush_task: Optional[asyncio.Task[Any]] = self._start_periodic_flush_task()

    @classmethod
    def get_supported_event_hooks(cls) -> list[GuardrailEventHooks]:
        """Return the guardrail event hooks this integration supports.

        Prompt moderation (``pre_call``) evaluates the user's message before
        the LLM is called. Response moderation (``post_call``) evaluates the
        assistant's reply and tool calls after the LLM returns.
        """
        return [GuardrailEventHooks.pre_call, GuardrailEventHooks.post_call]

    def _parse_sampling_rate(self) -> None:
        self.sampling_rate = 1.0
        rbrk_sampling_rate = os.getenv("RUBRIK_SAMPLING_RATE")
        if rbrk_sampling_rate is not None:
            try:
                parsed_rate = float(rbrk_sampling_rate.strip())
                self.sampling_rate = max(0.0, min(1.0, parsed_rate))
                if parsed_rate != self.sampling_rate:
                    verbose_logger.warning(f"RUBRIK_SAMPLING_RATE={parsed_rate} clamped to {self.sampling_rate}")
            except ValueError:
                verbose_logger.warning(f"Invalid RUBRIK_SAMPLING_RATE: {rbrk_sampling_rate!r}, using 1.0")

    def _parse_batch_size(self) -> None:
        _batch_size = os.getenv("RUBRIK_BATCH_SIZE")
        if _batch_size:
            try:
                parsed_size = int(_batch_size)
                if parsed_size <= 0:
                    verbose_logger.warning(f"RUBRIK_BATCH_SIZE={_batch_size!r} must be > 0, using default")
                else:
                    self.batch_size = parsed_size
            except ValueError:
                verbose_logger.warning(f"Invalid RUBRIK_BATCH_SIZE: {_batch_size!r}, using default")

    def _setup_clients(self, webhook_url: str) -> None:
        self.response_moderation_endpoint = f"{webhook_url}{_WEBHOOK_PATH_RESPONSE_MODERATION}"
        self.prompt_moderation_endpoint = f"{webhook_url}{_WEBHOOK_PATH_PROMPT_MODERATION}"
        self.logging_endpoint = f"{webhook_url}{_WEBHOOK_PATH_LOGGING_BATCH}"

        self.async_httpx_client = get_async_httpx_client(llm_provider=httpxSpecialProvider.LoggingCallback)

        self.moderation_client = get_async_httpx_client(
            llm_provider=httpxSpecialProvider.LoggingCallback,
            params={"timeout": httpx.Timeout(5.0, connect=2.0)},
        )

    def _start_periodic_flush_task(self) -> Optional[asyncio.Task[Any]]:
        """Start the periodic flush task only when an event loop is already running."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return None
        return loop.create_task(self.periodic_flush())

    def _ensure_periodic_flush_task(self) -> None:
        if self._periodic_flush_task is None or self._periodic_flush_task.done():
            self._periodic_flush_task = self._start_periodic_flush_task()

    async def aclose(self):
        """Cancel the periodic flush task.

        ``moderation_client`` and ``async_httpx_client`` are shared objects
        from LiteLLM's global HTTP-client cache (``get_async_httpx_client``
        uses the same cache key for all instances with equal parameters).
        Closing them here would close the shared connection pool for every
        other logger instance; let LiteLLM manage their lifecycle instead.
        """
        task = getattr(self, "_periodic_flush_task", None)
        if task is not None:
            task.cancel()

    # -- Guardrail hook --------------------------------------------------------

    async def apply_guardrail(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict,
        input_type: Literal["request", "response"],
        logging_obj: Optional["LiteLLMLoggingObj"] = None,
    ) -> GenericGuardrailAPIInputs:
        """Moderate prompts (request) and responses (response); fail-open.

        - ``request``: evaluate the prompt via the before_prompt webhook and
          block disallowed prompts before the model is called.
        - ``response``: evaluate the assistant's response text and tool calls
          via the after_completion webhook and block on a policy violation.

        litellm's guardrail-translation layer normalizes Anthropic and OpenAI
        requests/responses into ``inputs`` before this runs, so a single code
        path covers both wire formats. The configured guardrail ``mode``
        selects which surface(s) run.
        """
        if input_type == "request":
            return await self._guarded(
                self._moderate_prompt(inputs, request_data, logging_obj),
                inputs,
                "Prompt moderation",
            )
        if input_type == "response":
            return await self._guarded(
                self._moderate_response(inputs, request_data, logging_obj),
                inputs,
                "Response moderation",
            )
        return inputs

    @staticmethod
    async def _guarded(
        coro: Any,
        inputs: GenericGuardrailAPIInputs,
        label: str,
    ) -> GenericGuardrailAPIInputs:
        """Await a moderation coroutine fail-open: re-raise an intentional
        block, log at critical for malformed service responses, and swallow
        any other error returning ``inputs`` unchanged."""
        try:
            return await coro
        except ModifyResponseException:
            raise
        except _MalformedToolBlockingResponseError as e:
            # The service responded but the payload was structurally invalid,
            # which usually indicates a misconfigured webhook or a breaking
            # change in its response format. Log loudly so operators notice
            # their moderation policy is not actually being enforced.
            verbose_logger.critical(
                "Response moderation service returned a malformed response: %s. "
                "Requests are NOT being checked -- verify the webhook "
                "configuration. Returning original inputs unchanged.",
                e,
                exc_info=True,
            )
            return inputs
        except Exception as e:
            verbose_logger.error(
                f"{label} hook failed: {e}. Returning original inputs unchanged.",
                exc_info=True,
            )
            return inputs

    async def _moderate_response(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict,
        logging_obj: Optional["LiteLLMLoggingObj"],
    ) -> GenericGuardrailAPIInputs:
        """Send response text + tool calls to the after_completion webhook and
        raise if either the response text or any tool call is blocked."""
        tool_calls = inputs.get("tool_calls")
        texts = inputs.get("texts")
        if not tool_calls and not texts:
            return inputs

        message_tool_calls = self._normalize_tool_calls(tool_calls or [])
        sent_content = self._join_texts(texts)

        call_details = getattr(logging_obj, "model_call_details", {}) if logging_obj else {}
        if logging_obj and not call_details:
            verbose_logger.warning(
                "Rubrik: logging_obj present but model_call_details is empty -- request context will be missing"
            )

        # The moderation payload's ``id`` becomes the tool-blocking log's
        # correlation key (the S3 filename), so it must match the failure
        # (response) log written for the same blocked request. Both use
        # ``litellm_call_id`` -- see ``_correlation_id``.
        request_id = self._correlation_id(call_details, request_data)

        response_data = self._build_response_moderation_payload(message_tool_calls, sent_content, request_id)
        req_data = self._extract_request_data(call_details, request_data)

        service_response = await self._post_to_response_moderation_endpoint(response_data, req_data)
        blocked = self._extract_response_block(service_response, message_tool_calls, sent_content)

        if blocked:
            model = self._resolve_model(request_data, call_details)
            self._stash_block_context(logging_obj, request_data)
            raise ModifyResponseException(
                message=blocked.explanation,
                model=model,
                request_data=request_data,
                guardrail_name=self.guardrail_name,
            )

        return inputs

    async def _moderate_prompt(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict,
        logging_obj: Optional["LiteLLMLoggingObj"],
    ) -> GenericGuardrailAPIInputs:
        """Send the (normalized) prompt to the before_prompt webhook and raise
        if the prompt is blocked."""
        messages = inputs.get("structured_messages")
        if not messages:
            return inputs

        payload = self._build_prompt_moderation_payload(inputs, request_data)
        service_response = await self._post_to_prompt_moderation_endpoint(payload)
        refusal = self._extract_prompt_refusal(service_response)
        if refusal is None:
            return inputs

        model = inputs.get("model") or request_data.get("model") or "unknown"
        self._stash_block_context(logging_obj, request_data)
        raise ModifyResponseException(
            message=refusal,
            model=model,
            request_data=request_data,
            guardrail_name=self.guardrail_name,
        )

    @staticmethod
    def _stash_block_context(
        logging_obj: Optional["LiteLLMLoggingObj"],
        request_data: dict,
    ) -> None:
        """Stash signals so the deferred success-event skips this request and
        ``async_post_call_failure_hook`` can build the failure payload.

        - Sets a flag on ``logging_obj.model_call_details`` so the deferred
          success-event handler short-circuits.
        - Stashes a reference to ``logging_obj`` on ``request_data`` under a
          custom key. ``ProxyLogging.post_call_failure_hook`` pops only
          ``litellm_logging_obj`` before iterating callbacks, so this key
          survives.

        When ``logging_obj`` is ``None`` the success-event has no way to
        observe the block (the flag has nowhere to live), so we log an error
        instead of silently dropping the signal.
        """
        if logging_obj is None:
            verbose_logger.error(
                "Rubrik: moderation block fired with logging_obj=None for "
                f"litellm_call_id={request_data.get('litellm_call_id')}; "
                "cannot suppress success event or attach failure payload."
            )
            request_data["_rubrik_logging_obj"] = None
            return
        logging_obj.model_call_details["_rubrik_blocked"] = True
        request_data["_rubrik_logging_obj"] = logging_obj

    @staticmethod
    def _normalize_tool_calls(tool_calls: Any) -> list[ChatCompletionMessageToolCall]:
        """Convert tool_calls from inputs to ChatCompletionMessageToolCall objects."""
        result = []
        for tc in tool_calls:
            if isinstance(tc, ChatCompletionMessageToolCall):
                result.append(tc)
            elif isinstance(tc, dict):
                func = tc.get("function", {})
                result.append(
                    ChatCompletionMessageToolCall(
                        id=tc.get("id", ""),
                        type=tc.get("type", "function"),
                        function=Function(
                            name=func.get("name", ""),
                            arguments=func.get("arguments", ""),
                        ),
                    )
                )
            elif hasattr(tc, "id") and hasattr(tc, "function"):
                result.append(
                    ChatCompletionMessageToolCall(
                        id=tc.id or "",
                        type=getattr(tc, "type", None) or "function",
                        function=tc.function,
                    )
                )
            else:
                raise TypeError(f"Cannot normalize tool_call of type {type(tc).__name__}: {tc!r}")
        return result

    @staticmethod
    def _join_texts(texts: Any) -> str:
        """Join response text segments into the single content string the
        webhook evaluates. Empty when there is no assistant text."""
        if not texts:
            return ""
        return "\n".join(t for t in texts if t)

    @staticmethod
    def _build_response_moderation_payload(
        tool_calls: list[ChatCompletionMessageToolCall],
        content: str,
        request_id: str | None,
    ) -> dict[str, Any]:
        """Build an OpenAI ChatCompletion-format dict (assistant text + tool
        calls) for the after_completion webhook.

        ``content`` is sent so the webhook can moderate the response text;
        ``None`` when the assistant produced no text (tool-call-only response).
        """
        message: dict[str, Any] = {
            "role": "assistant",
            "content": content or None,
        }
        if tool_calls:
            message["tool_calls"] = [tc.model_dump(exclude_none=True) for tc in tool_calls]
        return {
            "id": request_id or f"chatcmpl-{uuid.uuid4()}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "",
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": "tool_calls" if tool_calls else "stop",
                }
            ],
        }

    @staticmethod
    def _flatten_messages_for_moderation(messages: Any) -> list[dict[str, Any]]:
        """Collapse each message's content to a plain string for the webhook.

        litellm normalizes Anthropic ``/v1/messages`` requests to OpenAI shape,
        but a turn sent as content-parts (``[{"type": "text", ...}]``) stays a
        list. The before_prompt webhook reads ``content`` as a string and drops
        non-string content, so we flatten text parts here (images skipped, per
        ``convert_content_list_to_str``) -- otherwise block-content prompts
        would pass through unmoderated. Builds a new list; never mutates the
        shared ``structured_messages``.
        """
        flattened: list[dict[str, Any]] = []
        for message in messages or []:
            if not isinstance(message, dict):
                continue
            flattened.append(
                {
                    "role": message.get("role"),
                    "content": convert_content_list_to_str(message),  # pyright: ignore[reportArgumentType]
                }
            )
        return flattened

    @staticmethod
    def _build_prompt_moderation_payload(
        inputs: GenericGuardrailAPIInputs,
        request_data: dict[str, Any],
    ) -> dict[str, Any]:
        """Build the bare OpenAI request the before_prompt webhook consumes.

        Unlike the after_completion envelope, this endpoint takes a raw OpenAI
        chat-completions request. ``structured_messages`` is litellm's
        OpenAI-normalized view of the prompt, so this works for Anthropic
        ``/v1/messages`` requests too. Optional fields are sent only when
        present so the payload stays clean.
        """
        payload: dict[str, Any] = {
            "model": inputs.get("model") or request_data.get("model") or "",
            "messages": RubrikLogger._flatten_messages_for_moderation(inputs.get("structured_messages")),
        }
        tools = inputs.get("tools")
        if tools is not None:
            payload["tools"] = tools
        user = request_data.get("user")
        if user:
            payload["user"] = user
        # Fall back to litellm_call_id, the stable cross-provider join key the
        # response/tool path uses (see _correlation_id). LiteLLM does not
        # populate request_data["correlation_key"]; it carries litellm_call_id.
        # The before_prompt webhook skips the *_prompt_moderation.json S3 write
        # when correlation_key is empty, so without this the block fires but no
        # log is ever written. An explicit correlation_key still wins.
        correlation_key = request_data.get("correlation_key") or request_data.get("litellm_call_id")
        if correlation_key:
            payload["correlation_key"] = correlation_key
        return payload

    @staticmethod
    def _extract_request_data(
        call_details: dict[str, Any],
        request_data: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Extract original request data from model_call_details for the
        response moderation service envelope.

        Includes the agent's declared ``tools`` (OpenAI-format) when available
        so the webhook's hallucination evaluator can compare returned tool calls
        against the declared tool list.
        """
        if not call_details and not request_data:
            return {}
        call_details = call_details or {}
        request_data = request_data or {}
        optional_params = call_details.get("optional_params", {}) or {}

        # Use ``in`` rather than truthy ``or`` so an explicit empty list
        # (caller declared the agent has NO tools) is forwarded as-is.
        # The response moderation service uses that signal to flag tool-call
        # hallucinations -- ``or`` would mask it by falling through to
        # optional_params.
        if "tools" in request_data:
            tools = request_data["tools"]
        else:
            tools = optional_params.get("tools")

        # The response moderation service consumes only messages/model/tools.
        # Don't forward proxy_server_request -- in litellm >=1.83 its ``body``
        # snapshot carries a UserAPIKeyAuth instance that breaks json.dumps,
        # silently fail-opening the guardrail.
        return {
            "messages": call_details.get("messages"),
            "model": call_details.get("model"),
            "tools": tools,
        }

    @staticmethod
    def _sanitize_proxy_server_request(proxy_server_request: Any) -> Any:
        """Allowlist only routing fields (``url``, ``method``) when forwarding
        ``proxy_server_request`` to an external webhook, dropping inbound
        ``headers`` (Authorization, Cookie, x-api-key, ...) and the raw
        request ``body`` so proxy credentials are not exfiltrated."""
        if not isinstance(proxy_server_request, dict):
            return proxy_server_request
        return {key: proxy_server_request[key] for key in ("url", "method") if key in proxy_server_request}

    @staticmethod
    def _resolve_model(request_data: dict[str, Any], call_details: dict[str, Any]) -> str:
        """Get the model name for the ModifyResponseException."""
        response = request_data.get("response")
        if response and hasattr(response, "model"):
            return response.model or "unknown"
        return call_details.get("model", "unknown")

    # -- Logging hooks ---------------------------------------------------------

    @staticmethod
    def _correlation_id(call_details: dict, request_data: dict | None = None) -> str | None:
        """The id that joins a blocked request's two S3 logs by filename: the
        moderation (``_blocking``) log and the failure (response) log.

        Always ``litellm_call_id``. It is assigned at request start and is
        present identically in both the guardrail path (``model_call_details``
        / ``request_data``) and the failure-hook path. Unlike ``response.id``
        or ``standard_logging_object["id"]`` it is immune to the race where a
        block fires before the response/logging object is populated, so the
        two logs correlate for every provider (OpenAI and Anthropic alike).
        """
        return call_details.get("litellm_call_id") or (request_data or {}).get("litellm_call_id")

    @classmethod
    def _apply_correlation_id(cls, payload: dict[str, Any], source: dict[str, Any]) -> None:
        """Pin ``payload["id"]`` to ``litellm_call_id`` in place so this log
        shares its S3 filename id with the moderation (``_blocking``) and
        failure logs for the same request -- for every provider.

        ``standard_logging_object["id"]`` is the provider response id
        (``response_obj.get("id", litellm_call_id)``), a ``chatcmpl-*`` value
        for OpenAI, which would not correlate. ``litellm_call_id`` is assigned
        at request start and is identical across all log paths. Falls back to
        the existing id when ``litellm_call_id`` is somehow absent rather than
        writing a null filename key.

        ``source`` may be ``model_call_details`` directly or a ``kwargs`` dict
        that aliases it -- same shape either way.
        """
        correlated = cls._correlation_id(source)
        if correlated:
            payload["id"] = correlated

    @staticmethod
    def _prepend_system_prompt(payload: dict[str, Any], source: dict[str, Any]) -> None:
        """Prepend ``source["system"]`` onto ``payload["messages"]``.

        Builds a NEW messages list rather than mutating ``payload["messages"]``
        in place. The fallback branch of ``_prepare_block_failure_payload``
        aliases ``call_details["messages"]`` directly, so an in-place
        ``list.insert(0, ...)`` would mutate the shared source dict.

        No-op if no system prompt is present. Tolerates list/dict/str
        message shapes; on unexpected shape, leaves payload alone.
        """
        system_prompt = source.get("system")
        if not system_prompt:
            return
        try:
            system_scaffold = {"role": "system", "content": system_prompt}
            messages = payload.get("messages")
            if isinstance(messages, list):
                payload["messages"] = [system_scaffold, *messages]
            elif isinstance(messages, (dict, str)):
                payload["messages"] = [system_scaffold, messages]
        except Exception as e:
            verbose_logger.warning(
                f"Rubrik: failed to prepend system prompt: {e}",
                exc_info=True,
            )

    async def _prepare_log_payload(self, kwargs: dict, event_type: str) -> StandardLoggingPayload | None:
        """Shared logic for success logging (sampled)."""
        if random.random() > self.sampling_rate:
            verbose_logger.debug(f"Skipping Rubrik {event_type} logging (sampling_rate={self.sampling_rate})")
            return None

        # Deep-copy so mutations don't affect other callbacks sharing this object
        standard_logging_payload: StandardLoggingPayload = safe_deep_copy(kwargs["standard_logging_object"])

        self._apply_correlation_id(standard_logging_payload, kwargs)  # pyright: ignore[reportArgumentType]
        self._prepend_system_prompt(standard_logging_payload, kwargs)  # pyright: ignore[reportArgumentType]

        return standard_logging_payload

    async def _append_and_maybe_flush(self, payload) -> None:
        self._ensure_periodic_flush_task()
        self.log_queue.append(payload)
        self._enforce_max_queue_size()
        if len(self.log_queue) >= self.batch_size:
            await self.flush_queue()

    def _enforce_max_queue_size(self) -> None:
        overflow = len(self.log_queue) - self.max_queue_size
        if overflow <= 0:
            return
        del self.log_queue[:overflow]
        self._dropped_since_warning += overflow
        now = time.time()
        if now - self._last_drop_warning_time >= _DROP_WARNING_INTERVAL_SECONDS:
            verbose_logger.warning(
                "Rubrik: log queue exceeded max_queue_size=%s; dropped %s "
                "oldest events since the last warning. The Rubrik webhook may "
                "be unhealthy or undersized for current traffic.",
                self.max_queue_size,
                self._dropped_since_warning,
            )
            self._dropped_since_warning = 0
            self._last_drop_warning_time = now

    async def _enqueue_log_event(self, kwargs: dict, event_type: str):
        try:
            payload = await self._prepare_log_payload(kwargs, event_type)
            if payload is None:
                return
            await self._append_and_maybe_flush(payload)
        except Exception as e:
            verbose_logger.error(
                f"Rubrik {event_type} logging hook failed: {e}. Skipping logging for this event.",
                exc_info=True,
            )

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        # Blocked requests are logged via async_post_call_failure_hook;
        # skip here to avoid double-logging the pre-block response.
        if kwargs.get("_rubrik_blocked"):
            verbose_logger.debug(
                f"Rubrik: skipping success event for blocked request litellm_call_id={kwargs.get('litellm_call_id')}"
            )
            return
        await self._enqueue_log_event(kwargs, "success")

    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
        # Log regular LLM failures (timeouts, upstream errors, etc.) to Rubrik.
        # NOTE: ``ModifyResponseException`` blocks are NOT routed here; they
        # bypass ``Logging.async_failure_handler`` entirely and reach
        # ``async_post_call_failure_hook`` instead. So there is no risk of
        # double-logging a block through this path.
        await self._enqueue_log_event(kwargs, "failure")

    async def async_post_call_failure_hook(
        self,
        request_data: dict,
        original_exception: Exception,
        user_api_key_dict: Any,
        traceback_str: Optional[str] = None,
    ) -> None:
        """Log blocked requests signalled via ``ModifyResponseException``
        (prompt blocks, response/tool blocks, streaming blocks).

        Carries the stashed ``_rubrik_logging_obj``. For every other
        exception we no-op; LiteLLM's standard failure plumbing handles those.
        """
        if not isinstance(original_exception, ModifyResponseException):
            return

        logging_obj = request_data.pop("_rubrik_logging_obj", None)
        if logging_obj is None:
            # Legitimate when a non-Rubrik guardrail raised the block;
            # problematic if Rubrik did and the stash was lost (e.g.
            # ``_stash_block_context`` ran with ``logging_obj=None``). Either
            # way we cannot build the payload.
            verbose_logger.warning(
                "Rubrik: block exception without stashed logging_obj. "
                f"litellm_call_id={request_data.get('litellm_call_id')}, "
                f"model={request_data.get('model')}, "
                f"user_id={getattr(user_api_key_dict, 'user_id', None)}, "
                f"raising_guardrail="
                f"{getattr(original_exception, 'guardrail_name', None)}"
            )
            return

        call_id: Optional[str] = None
        await self._build_and_enqueue_block_event(logging_obj, original_exception, call_id)

    async def _build_and_enqueue_block_event(
        self,
        logging_obj: "LiteLLMLoggingObj",
        exception: "ModifyResponseException",
        call_id: Optional[str],
    ) -> None:
        try:
            call_details = logging_obj.model_call_details
            # Do NOT pop "_rubrik_blocked" here. The deferred success-handler
            # task may still be iterating callbacks, and popping mid-iteration
            # (between two awaited callback invocations) would cause this
            # plugin's success-event callback to read the flag as absent and
            # log the pre-block response -- the exact bug this hook exists to
            # prevent. The flag dies with model_call_details when the request
            # completes; there's nothing to clean up.
            call_id = call_details.get("litellm_call_id")
            payload = self._prepare_block_failure_payload(logging_obj, exception)
        except (AttributeError, KeyError, TypeError) as e:
            verbose_logger.error(
                f"Rubrik: failed to build blocked-tool payload for "
                f"litellm_call_id={call_id}: {e}. Event will NOT be logged.",
                exc_info=True,
            )
            return

        try:
            await self._append_and_maybe_flush(payload)
        except Exception as e:
            verbose_logger.error(
                f"Rubrik: failed to enqueue blocked-tool event for litellm_call_id={call_id}: {e}.",
                exc_info=True,
            )

    def _prepare_block_failure_payload(
        self,
        logging_obj: "LiteLLMLoggingObj",
        exception: "ModifyResponseException",
    ) -> StandardLoggingPayload:
        """Build a failure-style payload using the exception text as response.

        Blocked-tool events are security-relevant and **bypass sampling**:
        every block is logged.

        The deferred success-handler runs as a separately-scheduled task and
        races with this hook, so ``standard_logging_object`` on
        ``model_call_details`` may not yet be populated. If present we reuse
        it; otherwise we fall back to a best-effort payload built from the
        fields available at block time.

        For prompt blocks the LLM is never called, so ``standard_logging_object``
        is never populated. The fallback therefore must carry enough fields to
        pass the log processor's ``LogEntry`` schema (``BaseLogEntry`` requires
        ``metadata``, ``model_id``, ``model_group``, ``model_parameters``,
        ``startTime``, ``endTime``, and ``completionStartTime``). Without a
        parseable payload the log processor discards the entry with a parse
        error and no session is created, so prompt-moderation violations are
        silently dropped even though the ``_prompt_moderation.json`` forensic
        log is written correctly.

        Field sourcing for the fallback path:
        - ``model`` / ``model_group``: ``call_details["model"]`` -- this is the
          model-group name (e.g. "gpt-4o") set by the proxy before the guardrail
          fires. The router writes ``metadata["model_group"]`` only inside
          ``acompletion()``, which hasn't run yet for a prompt block.
        - ``model_id``: not available before the LLM returns hidden_params;
          defaults to empty string.
        - ``user_api_key_hash``: ``call_details["metadata"]["user_api_key"]`` --
          the hashed token written by ``add_user_information_to_request_data``
          before ``pre_call_hook`` fires.
        - time fields: ``call_details["start_time"]`` reused for all three;
          end/completion times are meaningless for a prompt block.
        """
        call_details = logging_obj.model_call_details
        exception_text = f"{type(exception).__name__}: {exception.message}"

        base = call_details.get("standard_logging_object")
        if base is not None:
            payload: dict = safe_deep_copy(base)
        else:
            verbose_logger.debug(
                "Rubrik: standard_logging_object not yet on model_call_details "
                f"for litellm_call_id={call_details.get('litellm_call_id')}; "
                "using best-effort fallback payload."
            )
            payload = self._build_fallback_payload(call_details)

        payload["response"] = exception_text

        # Pin the correlation key to litellm_call_id so this failure log shares
        # its S3 filename id with the moderation (``_blocking``) log for the
        # same request. The copied ``standard_logging_object["id"]`` is
        # ``response_obj.get("id", litellm_call_id)`` -- a provider ``chatcmpl-*``
        # value for OpenAI -- which would not correlate; overwrite it.
        payload["id"] = self._correlation_id(call_details) or f"chatcmpl-{uuid.uuid4()}"
        self._prepend_system_prompt(payload, call_details)

        return payload  # type: ignore[return-value]

    @staticmethod
    def _build_fallback_payload(call_details: dict) -> dict:
        _metadata: dict = call_details.get("metadata") or {}
        # Convert datetime to a Unix float so json.dumps can serialize it.
        # httpx's json= parameter uses stdlib json.dumps with no custom encoder.
        _raw_start = call_details.get("start_time")
        _start = _raw_start.timestamp() if _raw_start is not None else None
        return {
            "id": call_details.get("litellm_call_id"),
            "model": call_details.get("model") or "",
            # model_group is set by the router inside acompletion(), which
            # hasn't run for a prompt block; use the model name instead.
            "model_group": call_details.get("model") or "",
            # model_id comes from response.hidden_params -- unavailable here.
            "model_id": "",
            "model_parameters": call_details.get("optional_params") or {},
            "startTime": _start,
            "endTime": _start,
            "completionStartTime": _start,
            "messages": call_details.get("messages") or [],
            "metadata": {
                # "user_api_key" is the hashed token written by
                # add_user_information_to_request_data before guardrails fire.
                "user_api_key_hash": _metadata.get("user_api_key_hash") or _metadata.get("user_api_key") or "",
            },
            "status": "failure",
        }

    # -- Batch logging ---------------------------------------------------------

    async def _log_batch_to_rubrik(self, data):
        # NOTE: this method intentionally re-raises on failure so flush_queue
        # can preserve the unsent events for the next flush attempt instead of
        # silently dropping them.
        try:
            response = await self.async_httpx_client.post(
                url=self.logging_endpoint,
                json=data,
                headers=self._headers,
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            verbose_logger.exception(f"Rubrik HTTP Error: {e.response.status_code} - {e.response.text}")
            raise
        except Exception:
            verbose_logger.exception("Rubrik Layer Error")
            raise

    async def async_send_batch(self):
        """Handles sending batches of responses to Rubrik.

        Note: the canonical flush path is :meth:`flush_queue`, which takes a
        single snapshot used for both sending and queue draining. This method
        is kept for direct callers / tests; it intentionally does NOT remove
        events from the queue.
        """
        if not self.log_queue:
            return

        await self._log_batch_to_rubrik(
            data=self.log_queue,
        )

    async def flush_queue(self):
        """Snapshot, send, and drain in one consistent step.

        Overrides the base implementation so the same snapshot drives both
        the HTTP send and the queue truncation. This avoids the subtle
        coupling where the base class captures ``len(self.log_queue)``
        separately from the snapshot taken inside ``async_send_batch``,
        which could otherwise drift in a future refactor and cause
        duplicate deliveries to Rubrik.
        """
        if self.flush_lock is None:
            return

        async with self.flush_lock:
            if not self.log_queue:
                return
            snapshot = list(self.log_queue)
            verbose_logger.debug("Rubrik: Flushing batch of %s events", len(snapshot))
            try:
                await self._log_batch_to_rubrik(data=snapshot)
            except Exception:
                # Already logged with traceback inside _log_batch_to_rubrik.
                # Preserve the in-flight events for retry on the next flush.
                return
            del self.log_queue[: len(snapshot)]
            self.last_flush_time = time.time()

    # -- Webhook services ------------------------------------------------------

    async def _post_json(self, endpoint: str, payload: dict[str, Any], service_name: str) -> dict[str, Any]:
        """POST ``payload`` to a Rubrik webhook and return its dict response.

        Raises:
            Exception: If the service is unavailable or returns an error.
            TypeError: If the response JSON is not a dict.
        """
        verbose_logger.debug(f"Sending request to {service_name}: {endpoint}")
        http_response = await self.moderation_client.post(
            endpoint,
            json=payload,
            headers=self._headers,
        )
        http_response.raise_for_status()
        result = http_response.json()
        if not isinstance(result, dict):
            raise TypeError(
                f"{service_name} returned non-dict JSON "
                f"({type(result).__name__}); expected OpenAI chat completion "
                "shape or empty object."
            )
        return result

    async def _post_to_response_moderation_endpoint(
        self,
        response_data: dict[str, Any],
        request_data: dict[str, Any],
    ) -> dict[str, Any]:
        """Post the ``{request, response}`` envelope to the after_completion
        webhook and return its (possibly rewritten) response.

        Args:
            response_data: The OpenAI-formatted response payload to send.
            request_data: Original LLM request data to include alongside
                the response for additional context. Empty dict if unavailable.
        """
        envelope = {"request": request_data, "response": response_data}
        return await self._post_json(
            self.response_moderation_endpoint,
            envelope,
            "Response moderation service",
        )

    async def _post_to_prompt_moderation_endpoint(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Post a bare OpenAI request to the before_prompt webhook.

        Returns ``{}`` (passthrough) or a synthetic chat.completion (block).
        """
        return await self._post_json(self.prompt_moderation_endpoint, payload, "Prompt moderation service")

    @staticmethod
    def _extract_prompt_refusal(service_response: dict[str, Any]) -> str | None:
        """Return the refusal text when the prompt was blocked, else None.

        The before_prompt webhook returns ``{}`` (passthrough) or a synthetic
        chat.completion whose ``choices[0].message.content`` is the refusal
        explanation.
        """
        choices = service_response.get("choices")
        if not choices:
            return None
        message = choices[0].get("message", {})
        content = message.get("content")
        return content or "Request blocked by policy."

    @staticmethod
    def _extract_response_block(
        service_response: dict[str, Any],
        all_tool_calls: list[ChatCompletionMessageToolCall],
        sent_content: str,
    ) -> BlockedResponseResult | None:
        """Detect whether the webhook moderated the response text or tool calls.

        The after_completion webhook rewrites the response in place with no
        explicit "blocked" flag, so we infer a block by diffing what we sent
        against what came back:

        - Tool block: a tool call we sent is absent from the returned (allowed)
          set.
        - Text block: the returned content was REPLACED wholesale (a text
          violation), as opposed to having a tool-block explanation APPENDED to
          the original content. We tell them apart with ``startswith``, which
          mirrors the webhook's own append-vs-replace behavior.

        Returns None when nothing was moderated. A text block supersedes a tool
        block (mirroring the webhook, which drops tool calls on a text block).

        Expects service_response in OpenAI chat completion format:
            {"choices": [{"message": {"tool_calls": [...], "content": "..."}}]}
        """
        choices = service_response.get("choices", [])
        if not choices:
            raise _MalformedToolBlockingResponseError("Response moderation service returned empty response")

        message = choices[0].get("message", {})
        returned_tool_calls = message.get("tool_calls") or []
        returned_content = message.get("content") or ""

        allowed_ids = {tc["id"] for tc in returned_tool_calls if tc.get("id")}
        allowed_tools = [tc for tc in all_tool_calls if tc.id in allowed_ids]
        tools_blocked = len(allowed_tools) < len(all_tool_calls)

        # The webhook either replaces content wholesale (text block) or appends
        # a tool-block explanation to the original text. ``appended`` tells the
        # two apart, and is reused below to recover just the explanation. A text
        # block requires there to have been assistant text to block.
        appended = bool(sent_content) and returned_content.startswith(sent_content)
        text_blocked = bool(sent_content) and not appended

        if text_blocked:
            return BlockedResponseResult(explanation=returned_content or "Response blocked by policy.")

        if tools_blocked:
            if appended:
                # Recover just the appended explanation: drop the original text
                # and the leading separator the webhook inserted before it.
                explanation = returned_content[len(sent_content) :].lstrip("\n")
            else:
                explanation = returned_content
            return BlockedResponseResult(explanation=explanation or "Tool call blocked by policy.")

        return None
