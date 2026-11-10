import os
import uuid
from collections.abc import Mapping
from typing import (
    Final,
    Literal,
)

import httpx
from pydantic import TypeAdapter
from typing_extensions import TypedDict, Unpack

from litellm._logging import verbose_proxy_logger
from litellm.exceptions import GuardrailRaisedException
from litellm.integrations.custom_guardrail import (
    CustomGuardrail,
    log_guardrail_information,  # pyright: ignore[reportUnknownVariableType]  # the decorator is untyped upstream
)
from litellm.litellm_core_utils.api_route_to_call_types import get_call_types_for_route
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.custom_httpx.http_handler import (
    AsyncHTTPHandler,
    get_async_httpx_client,  # pyright: ignore[reportUnknownVariableType]  # the factory's params are untyped upstream
    httpxSpecialProvider,
)
from litellm.types.guardrails import GuardrailEventHooks
from litellm.types.proxy.guardrails.guardrail_hooks.airia import AiriaGuardrailConfigModel
from litellm.types.utils import GenericGuardrailAPIInputs

GUARDRAIL_PATH: Final = "/v1/guardrails/litellm"

ACTION_NONE: Final = "NONE"
ACTION_BLOCKED: Final = "BLOCKED"
ACTION_INTERVENED: Final = "GUARDRAIL_INTERVENED"

DEFAULT_BLOCKED_MESSAGE: Final = "Blocked by your organization's content policy."

SUPPORTED_EVENT_HOOKS: Final = (GuardrailEventHooks.pre_call, GuardrailEventHooks.post_call)

_DECODED_JSON: Final = TypeAdapter(object)


class _CustomGuardrailOptions(TypedDict, total=False, extra_items=object):
    pass


def _stream_cannot_carry_a_rewrite(request_data: Mapping[str, object] | None) -> bool:
    """Whether this is a streamed response on a route the framework can only block, not rewrite.

    ``incremental_diff``, the mode that writes a rewrite into the stream, exists only for OpenAI chat
    completions: ``UnifiedLLMGuardrails._resolve_transform_call_type`` sends every other streamed route
    to ``block_only``, which drops the rewrite and replays the buffered original. On those routes a
    REDACT verdict can only be honoured as a block. The route check mirrors that resolver so the two
    cannot disagree, and a route it cannot resolve fails closed the same way the resolver does.
    """
    if request_data is None or not _is_streamed(request_data):
        return False
    route: Final = _request_route(request_data)
    call_types: Final = get_call_types_for_route(route) if route else None
    if not call_types:
        return True
    # Deferred like the resolver's own import: the handler package imports the proxy.
    from litellm.llms import load_guardrail_translation_mappings  # noqa: PLC0415  # circular import
    from litellm.llms.openai.chat.guardrail_translation.handler import (  # noqa: PLC0415  # circular import
        OpenAIChatCompletionsHandler,
    )

    handler: Final[object] = load_guardrail_translation_mappings().get(call_types[0])
    return not (isinstance(handler, type) and issubclass(handler, OpenAIChatCompletionsHandler))


def _is_streamed(request_data: Mapping[str, object]) -> bool:
    if request_data.get("stream") is True:
        return True
    proxy_request: Final = _as_object(request_data.get("proxy_server_request"))
    body: Final = _as_object(proxy_request.get("body")) if proxy_request is not None else None
    return body is not None and body.get("stream") is True


def _request_route(request_data: Mapping[str, object]) -> str | None:
    """The route the proxy stamps as ``user_api_key_request_route``, in whichever metadata bucket the
    route uses: ``metadata`` for chat completions, ``litellm_metadata`` for Responses and Anthropic."""
    for bucket_name in ("metadata", "litellm_metadata"):
        bucket = _as_object(request_data.get(bucket_name))
        route = bucket.get("user_api_key_request_route") if bucket is not None else None
        if isinstance(route, str) and route:
            return route
    return None


def _as_object(value: object) -> Mapping[str, object] | None:
    """`value` as a JSON object, or None. States the key type once so the checker does not see
    `dict[Unknown, Unknown]` at every read."""
    return value if isinstance(value, dict) else None  # pyright: ignore[reportUnknownVariableType]  # isinstance narrows to dict[Unknown, Unknown]; JSON keys are str


class AiriaGuardrail(CustomGuardrail):
    """Evaluates prompts and responses against your Airia guardrail policy."""

    def __init__(
        self,
        api_base: str | None = None,
        api_key: str | None = None,
        timeout: float | None = None,
        supported_event_hooks: list[GuardrailEventHooks] | None = None,  # mutable-ok: matches CustomGuardrail.__init__
        async_handler: AsyncHTTPHandler | None = None,  # dependency-injected in tests; defaults to a real client
        **kwargs: Unpack[_CustomGuardrailOptions],  # kwargs-ok: passed straight through to CustomGuardrail.__init__
    ) -> None:
        resolved_timeout: Final = timeout or float(os.getenv("AIRIA_TIMEOUT", "10"))

        self.async_handler = async_handler or get_async_httpx_client(
            llm_provider=httpxSpecialProvider.GuardrailCallback,
            params={"timeout": httpx.Timeout(timeout=resolved_timeout, connect=5.0)},
        )

        self.api_base = (api_base or os.getenv("AIRIA_GATEWAY_URL", "")).rstrip("/")
        self.api_key = api_key or os.getenv("AIRIA_API_KEY")
        self.streaming_transform_mode: Final[Literal["block_only", "incremental_diff"]] = "incremental_diff"
        self.streaming_end_of_stream_only: Final = True
        # incremental_diff exists only for streamed chat completions. Every other streamed route falls
        # back to block_only, where the framework releases the original text live unless told to hold
        # it; holding it keeps the text behind the end-of-stream verdict, and apply_guardrail turns a
        # REDACT that block_only cannot deliver into a block. Never read on the incremental_diff path.
        self.streaming_buffer_until_moderated: Final = True

        if not self.api_base:
            raise ValueError("AiriaGuardrail requires api_base, or the AIRIA_GATEWAY_URL environment variable.")
        if not self.api_key:
            raise ValueError("AiriaGuardrail requires api_key, or the AIRIA_API_KEY environment variable.")

        self.optional_params = kwargs
        super().__init__(  # pyright: ignore[reportUnknownMemberType]  # parent takes untyped **kwargs
            supported_event_hooks=supported_event_hooks or [*SUPPORTED_EVENT_HOOKS],
            **kwargs,
        )
        verbose_proxy_logger.info("AiriaGuardrail initialized with gateway: %s", self.api_base)

    def supports_structured_messages(self) -> bool:
        return False

    @log_guardrail_information
    async def apply_guardrail(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict[str, object],  # mutable-ok: the base class declares this parameter as a dict
        input_type: Literal["request", "response"],
        logging_obj: "LiteLLMLoggingObj | None" = None,
    ) -> GenericGuardrailAPIInputs:
        call_id: Final = (
            logging_obj.litellm_call_id
            if logging_obj
            else (request_data.get("litellm_call_id") if request_data else None)
        ) or str(uuid.uuid4())

        try:
            response: Final = await self.async_handler.post(  # pyright: ignore[reportUnknownMemberType]  # the handler's `data` param is an untyped dict upstream
                f"{self.api_base}{GUARDRAIL_PATH}",
                json={
                    "input_type": input_type,
                    "texts": inputs.get("texts") or (),
                    "images": inputs.get("images") or (),
                    "structured_messages": inputs.get("structured_messages") or (),
                    "tools": inputs.get("tools") or (),
                    "tool_calls": inputs.get("tool_calls") or (),
                    "model": inputs.get("model"),
                    "litellm_call_id": call_id,
                },
                headers={"Authorization": f"Bearer {self.api_key}"},
            )
            response.raise_for_status()
            raw_body: Final = _DECODED_JSON.validate_python(response.json())
            body: Final = _as_object(raw_body)
            if body is None:
                raise TypeError(f"expected a JSON object in the response body, got {type(raw_body).__name__}")
        except Exception as error:
            verbose_proxy_logger.error(
                "Airia guardrail could not evaluate the request (litellm_call_id=%s, input_type=%s): %s",
                call_id,
                input_type,
                error,
            )
            raise GuardrailRaisedException(
                guardrail_name=self.guardrail_name,
                message=f"Airia guardrail could not evaluate the request: {error}",
                blocked_content=False,
            ) from error

        action: Final = body.get("action")

        if action == ACTION_BLOCKED:
            reason: Final = body.get("blocked_reason")
            raise GuardrailRaisedException(
                guardrail_name=self.guardrail_name,
                message=reason if isinstance(reason, str) and reason else DEFAULT_BLOCKED_MESSAGE,
                blocked_content=True,
            )

        if action == ACTION_INTERVENED:
            if input_type == "response" and _stream_cannot_carry_a_rewrite(request_data):
                raise self._blocked()
            return self._rewritten(body, inputs)

        if action != ACTION_NONE:
            raise self._blocked()

        return inputs

    def _rewritten(
        self,
        body: Mapping[str, object],
        inputs: GenericGuardrailAPIInputs,
    ) -> GenericGuardrailAPIInputs:
        texts: Final = body.get("texts")
        structured_messages: Final = body.get("structured_messages")
        if not self._is_applicable_rewrite(texts, inputs.get("texts")):
            raise self._blocked()
        if not self._is_applicable_rewrite(structured_messages, inputs.get("structured_messages")):
            raise self._blocked()
        # An intervention that rewrites nothing would hand the original content through under a
        # verdict that said it must not pass.
        if texts is None and structured_messages is None:
            raise self._blocked()

        rewritten: Final[GenericGuardrailAPIInputs] = {**inputs}
        if isinstance(texts, list):
            rewritten["texts"] = texts
        if isinstance(structured_messages, list):
            rewritten["structured_messages"] = structured_messages
        return rewritten

    @staticmethod
    def _is_applicable_rewrite(rewrite: object, original: object) -> bool:
        """A rewrite is safe to apply positionally only when it is absent (this field was not
        touched) or a list the same length as what was sent. A different count would misalign
        the positional write-back downstream, dropping content or leaving other fields unchanged.

        A rewrite for a field that was never sent is a mismatch too, not a no-op: post_call sends
        only `texts`, so `structured_messages: []` would otherwise count as an applied rewrite
        while the `texts` that carried the content passed through untouched.
        """
        if rewrite is None:
            return True
        if not isinstance(original, list) or not isinstance(rewrite, list):
            return False
        return len(rewrite) == len(original)  # pyright: ignore[reportUnknownArgumentType]  # only the counts matter; the items are the gateway's

    def _blocked(self) -> GuardrailRaisedException:
        return GuardrailRaisedException(
            guardrail_name=self.guardrail_name,
            message=DEFAULT_BLOCKED_MESSAGE,
            blocked_content=True,
        )

    @staticmethod
    def get_config_model() -> type[AiriaGuardrailConfigModel] | None:
        return AiriaGuardrailConfigModel
