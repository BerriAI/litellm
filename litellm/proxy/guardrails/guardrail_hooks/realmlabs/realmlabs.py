"""RealmLabs MLS guardrail.

On ``pre_call`` MLS checks the request. A matching ``hazard_prompt`` verdict above ``hazard_threshold`` blocks
the request. On ``post_call`` MLS checks the assistant reply with the request's conversation as context.
Both hooks mask detected PII as ``[type]``, or block when ``pii_mask`` is False.

Conversation context comes from Chat Completions-style ``request_data.messages``; nothing is kept between
hooks. Streaming uses LiteLLM's existing delivery settings, which do not enable text rewrites by default.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Literal

from httpx import HTTPError
from pydantic import TypeAdapter, ValidationError

from litellm._logging import verbose_proxy_logger
from litellm.exceptions import GuardrailRaisedException
from litellm.integrations.custom_guardrail import (
    CustomGuardrail,
    log_guardrail_information,  # pyright: ignore[reportUnknownVariableType]  # decorator is untyped in custom_guardrail
)
from litellm.llms.custom_httpx.http_handler import (
    get_async_httpx_client,  # pyright: ignore[reportUnknownVariableType]  # helper is untyped in http_handler
    httpxSpecialProvider,
)
from litellm.proxy.guardrails.guardrail_hooks.content_text import content_to_text
from litellm.proxy.guardrails.guardrail_hooks.realmlabs.pii_masking import mask_pii_in_text
from litellm.secret_managers.main import get_secret_str
from litellm.types.guardrails import GuardrailEventHooks
from litellm.types.proxy.guardrails.guardrail_hooks.realmlabs import (
    RealmLabsChatMessage,
    RealmLabsGuardrailConfigModel,
    RealmLabsGuardrailRequest,
    RealmLabsGuardrailResponse,
    RealmLabsPIISpan,
)

if TYPE_CHECKING:
    from httpx import Response as HttpxResponse

    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
    from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
    from litellm.types.guardrails import Mode
    from litellm.types.utils import GenericGuardrailAPIInputs

_DEFAULT_API_BASE: Final = "https://mls.realmlabs.ai"
_GUARDRAIL_ENDPOINT: Final = "/guardrail"
_HAZARD_PROBE: Final = "hazard_prompt"
_DEFAULT_HAZARD_THRESHOLD: Final = 0.703
_DEFAULT_TIMEOUT: Final = 15.0

_RESPONSE_ADAPTER: Final = TypeAdapter(RealmLabsGuardrailResponse)
_MESSAGE_ADAPTER: Final = TypeAdapter(Mapping[str, object])
_MESSAGES_ADAPTER: Final = TypeAdapter(tuple[object, ...])


class RealmLabsMissingCredentials(Exception):
    """Raised at startup when no MLS API key is configured."""


@dataclass(frozen=True, slots=True)
class _InvalidResponse:
    reason: str


class RealmLabsGuardrail(CustomGuardrail):
    """Blocks hazardous prompts and masks PII using the RealmLabs MLS endpoint."""

    def __init__(
        self,
        api_key: str | None = None,
        api_base: str | None = None,
        probes: Sequence[str] | str | None = None,
        hazard_threshold: float | None = None,
        pii: bool | None = None,
        pii_mask: bool | None = None,
        block_on_error: bool | None = None,
        enable_thinking: bool | None = None,
        timeout: float | None = None,
        guardrail_name: str | None = None,
        event_hook: (  # mutable-ok: same type as CustomGuardrail
            GuardrailEventHooks | list[GuardrailEventHooks] | Mode | None
        ) = None,
        default_on: bool = False,
    ) -> None:
        """Resolve each setting from its argument, then the ``REALMLABS_*`` env vars, then the module defaults."""
        self.api_key = api_key or get_secret_str("REALMLABS_API_KEY")
        if not self.api_key:
            raise RealmLabsMissingCredentials(
                "RealmLabs API key is required. Set REALMLABS_API_KEY in the environment "
                "or pass api_key in the guardrail config."
            )

        self.api_base = (api_base or get_secret_str("REALMLABS_API_BASE") or _DEFAULT_API_BASE).rstrip("/")
        self.probes: Sequence[str] | str = (_HAZARD_PROBE,) if probes is None else probes
        self.hazard_threshold = _DEFAULT_HAZARD_THRESHOLD if hazard_threshold is None else hazard_threshold
        self.pii = True if pii is None else pii
        self.pii_mask = True if pii_mask is None else pii_mask
        self.block_on_error = False if block_on_error is None else block_on_error
        self.enable_thinking = False if enable_thinking is None else enable_thinking
        self.timeout = _DEFAULT_TIMEOUT if timeout is None else timeout
        self.async_handler: AsyncHTTPHandler = get_async_httpx_client(
            llm_provider=httpxSpecialProvider.GuardrailCallback,
        )
        super().__init__(  # pyright: ignore[reportUnknownMemberType]  # CustomGuardrail.__init__ is untyped
            guardrail_name=guardrail_name,
            supported_event_hooks=self.get_supported_event_hooks(),
            event_hook=event_hook,
            default_on=default_on,
        )

    @staticmethod
    def get_config_model() -> type[RealmLabsGuardrailConfigModel] | None:
        """Config model the admin UI uses to render and validate this guardrail's settings."""
        return RealmLabsGuardrailConfigModel

    @classmethod
    def get_supported_event_hooks(cls) -> list[GuardrailEventHooks]:  # mutable-ok: base class returns a list
        """Inspect requests on ``pre_call`` and replies on ``post_call``."""
        return [GuardrailEventHooks.pre_call, GuardrailEventHooks.post_call]

    @staticmethod
    def _hazard_score(response: RealmLabsGuardrailResponse) -> float | None:
        """Score of the first ``hazard_prompt`` verdict, unless MLS reports a role mismatch."""
        for result in response["results"]:
            if result["probe"] == _HAZARD_PROBE:
                return None if result.get("role_mismatch") else result["prob"]
        return None

    @staticmethod
    def _span_types(spans: Sequence[RealmLabsPIISpan]) -> str:
        """Distinct span types in first-seen order, for block messages and logs; ``"unknown"`` if none."""
        entity_types: Final = (span.get("type") for span in spans)
        unique_types: Final = tuple(dict.fromkeys(entity_type for entity_type in entity_types if entity_type))
        return ", ".join(unique_types) or "unknown"

    def _build_request(self, messages: Sequence[Mapping[str, object]]) -> RealmLabsGuardrailRequest:
        return RealmLabsGuardrailRequest(
            messages=messages,
            probes=self.probes,
            pii=self.pii,
            enable_thinking=self.enable_thinking,
        )

    def _parse_response(self, content: bytes) -> RealmLabsGuardrailResponse | _InvalidResponse:
        try:
            result: Final = _RESPONSE_ADAPTER.validate_json(content)
        except ValidationError as exc:
            return _InvalidResponse(exc.json(include_input=False, include_context=False, include_url=False))

        if any(not span.get("type") for span in result["pii_spans"]):
            return _InvalidResponse("pii_spans must include a nonempty type")
        if self.pii_mask and any(not span.get("text") for span in result["pii_spans"]):
            return _InvalidResponse("pii_spans must include nonempty text when masking is enabled")
        return result

    async def _call_mls(
        self, messages: Sequence[Mapping[str, object]]
    ) -> RealmLabsGuardrailResponse | _InvalidResponse:
        endpoint: Final = f"{self.api_base}{_GUARDRAIL_ENDPOINT}"
        verbose_proxy_logger.debug(
            "RealmLabs MLS: %s msgs=%d probes=%s pii=%s",
            endpoint,
            len(messages),
            self.probes,
            self.pii,
        )
        response: Final[HttpxResponse] = await self.async_handler.post(  # pyright: ignore[reportUnknownMemberType]  # AsyncHTTPHandler.post is untyped
            url=endpoint,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            content=json.dumps(self._build_request(messages)),
            timeout=self.timeout,
        )
        response.raise_for_status()
        return self._parse_response(response.content)

    def _handle_mls_error(self, inputs: GenericGuardrailAPIInputs, reason: str) -> GenericGuardrailAPIInputs:
        verbose_proxy_logger.error("RealmLabs MLS error: %s", reason)
        if self.block_on_error:
            raise GuardrailRaisedException(
                guardrail_name=self.guardrail_name,
                message=f"RealmLabs MLS error (block_on_error=True): {reason}",
            )
        return inputs

    @log_guardrail_information
    async def apply_guardrail(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: Mapping[str, object],
        input_type: Literal["request", "response"],
        logging_obj: LiteLLMLoggingObj | None = None,
    ) -> GenericGuardrailAPIInputs:
        """Inspect request or response text through LiteLLM's unified guardrail layer.

        Enforce hazard scores only on requests with a matching role, then mask or block PII on either side.
        Only the current hook's texts are rewritten. MLS failures pass through unless ``block_on_error`` is set.
        """
        texts: Final = tuple(inputs.get("texts") or ())
        messages: Final = _messages_for_scan(inputs, request_data, input_type)
        if not messages:
            return inputs

        try:
            result: Final = await self._call_mls(messages)
        except (HTTPError, TypeError, ValueError) as exc:
            return self._handle_mls_error(inputs, str(exc))

        if isinstance(result, _InvalidResponse):
            return self._handle_mls_error(inputs, f"Invalid RealmLabs guardrail response: {result.reason}")

        # Hazard is checked before PII, so a hazardous prompt is rejected rather than masked and forwarded.
        hazard_score: Final = self._hazard_score(result) if input_type == "request" else None
        if hazard_score is not None and hazard_score > self.hazard_threshold:
            verbose_proxy_logger.warning(
                "RealmLabs MLS blocked request: %s=%s > %s",
                _HAZARD_PROBE,
                hazard_score,
                self.hazard_threshold,
            )
            raise GuardrailRaisedException(
                guardrail_name=self.guardrail_name,
                message=(
                    f"Blocked by RealmLabs {_HAZARD_PROBE} probe: "
                    f"score={hazard_score} exceeds threshold={self.hazard_threshold}"
                ),
                blocked_content=True,
            )

        spans: Final = result["pii_spans"]
        if not spans:
            return inputs

        if not self.pii_mask:
            raise GuardrailRaisedException(
                guardrail_name=self.guardrail_name,
                message=f"Blocked by RealmLabs: PII detected in the {input_type} ({self._span_types(spans)})",
                blocked_content=True,
            )

        masked_texts: Final = tuple(mask_pii_in_text(text, spans) for text in texts)
        if masked_texts == texts:
            return inputs
        verbose_proxy_logger.debug("RealmLabs MLS masked PII types in the %s: %s", input_type, self._span_types(spans))
        return {**inputs, "texts": list(masked_texts)}


def _conversation_message(message: object) -> RealmLabsChatMessage | None:
    """Extract a plain-text turn without carrying images or unrelated message fields to MLS."""

    if not isinstance(message, dict):
        return None
    row: Final = _MESSAGE_ADAPTER.validate_python(message)
    role: Final = row.get("role")
    text: Final = content_to_text(row.get("content"))
    if isinstance(role, str) and role and text:
        return RealmLabsChatMessage(role=role, content=text)
    return None


def _conversation_messages(request_data: Mapping[str, object]) -> tuple[RealmLabsChatMessage, ...]:
    """Preserve roles and text from request history, skipping empty or non-message entries."""

    raw_messages: Final = request_data.get("messages")
    if not isinstance(raw_messages, list):
        return ()

    messages: Final = _MESSAGES_ADAPTER.validate_python(raw_messages)
    return tuple(message for item in messages if (message := _conversation_message(item)) is not None)


def _messages_for_scan(
    inputs: GenericGuardrailAPIInputs,
    request_data: Mapping[str, object],
    input_type: Literal["request", "response"],
) -> Sequence[Mapping[str, object]]:
    """Prefer scoped request messages; append response texts as assistant turns to the request history."""

    if input_type == "request" and (structured_messages := inputs.get("structured_messages")):
        return tuple(structured_messages)

    conversation: Final = _conversation_messages(request_data)
    texts: Final = inputs.get("texts") or ()
    if input_type == "request":
        return conversation or tuple(RealmLabsChatMessage(role="user", content=text) for text in texts)
    return (*conversation, *(RealmLabsChatMessage(role="assistant", content=text) for text in texts))
