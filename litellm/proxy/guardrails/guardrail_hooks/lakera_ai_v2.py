import copy
import os
from collections.abc import Mapping, Sequence
from datetime import datetime
from string import Formatter
from types import MappingProxyType
from typing import Final, Literal

from fastapi import HTTPException

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.integrations.custom_guardrail import (
    DEFAULT_ADVISORY_MESSAGE,
    CustomGuardrail,
)
from litellm.llms.base_llm.guardrail_translation.utils import (
    filter_messages_by_skip_flags,
)
from litellm.llms.custom_httpx.http_handler import (
    get_async_httpx_client,
    httpxSpecialProvider,
)
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.guardrails._content_utils import (
    apply_redacted_messages_back,
    build_inspection_messages,
    has_non_string_content,
)
from litellm.secret_managers.main import get_secret_str
from litellm.types.guardrails import GuardrailEventHooks, LitellmParams, Mode
from litellm.types.llms.openai import AllMessageValues
from litellm.types.proxy.guardrails.guardrail_hooks.lakera_ai_v2 import (
    LakeraAIBreakdownItem,
    LakeraAIRequest,
    LakeraAIResponse,
)
from litellm.types.utils import CallTypesLiteral, GuardrailStatus, ModelResponse

_DETECTOR_CATEGORY_PHRASES: Final[Mapping[str, str]] = MappingProxyType(
    {
        "prompt_injection": "a potential prompt injection attempt",
        "prompt_attack": "a potential prompt injection attempt",
        "pii": "personally identifiable information",
        "moderated_content": "policy-violating content",
    }
)


def humanize_lakera_block_reasons(breakdown: Sequence[LakeraAIBreakdownItem] | None) -> str:
    """
    Turn a Lakera v2 ``breakdown`` list into a plain-language reason string
    suitable for an advisory message shown to the LLM (e.g. "a potential
    prompt injection attempt, personally identifiable information").

    Falls back to a generic phrase when breakdown is empty or every detected
    detector_type is unrecognized.
    """
    if not breakdown:
        return "a content safety concern"

    categories: Final = (
        (item.get("detector_type") or "").split("/")[0] for item in breakdown if item.get("detected", False)
    )
    phrases: Final = tuple(
        dict.fromkeys(
            _DETECTOR_CATEGORY_PHRASES.get(category) or category.replace("_", " ")
            for category in categories
            if category
        )
    )
    return ", ".join(phrases) if phrases else "a content safety concern"


def _template_uses_reason_placeholder(template: str) -> bool:
    """True if ``template`` has a real ``{reason}`` format field, not just the
    literal substring -- an escaped ``{{reason}}`` contains the substring but
    formats to a literal "{reason}", never substituting the actual value."""
    return any(field_name == "reason" for _, field_name, _, _ in Formatter().parse(template))


def _event_hook_includes_during_call(
    event_hook: GuardrailEventHooks | Sequence[GuardrailEventHooks] | Mode | str | Sequence[str] | None,
) -> bool:
    """True if ``event_hook`` could ever resolve to during_call, covering a plain
    value, a list of values, or a tag-based Mode (checked across every tag value
    and the default)."""
    candidates: Final = (
        tuple(event_hook.tags.values()) + (event_hook.default,)
        if isinstance(event_hook, Mode)
        else tuple(event_hook)
        if isinstance(event_hook, list)
        else (event_hook,)
    )

    flattened: Final = tuple(
        value
        for candidate in candidates
        for value in (tuple(candidate) if isinstance(candidate, list) else (candidate,))
    )
    return any(value == GuardrailEventHooks.during_call for value in flattened if value is not None)


_MASKABLE_MESSAGE_KEYS: Final[frozenset[str]] = frozenset({"role", "content"})


def _has_non_maskable_message_fields(data: Mapping[str, object]) -> bool:
    """True if any message in ``data["messages"]`` carries a field besides
    role/content (e.g. tool_call_id, name, function_call). Mask-in-place
    rewrites data["messages"] from a synthetic {role, content}-only list built
    by build_inspection_messages, which drops every other field -- masking a
    tool message would silently strip its tool_call_id, producing a malformed
    outgoing request."""
    messages: Final = data.get("messages")
    if not isinstance(messages, list):
        return False
    return any(
        isinstance(message, dict) and any(key not in _MASKABLE_MESSAGE_KEYS for key in message) for message in messages
    )


def _has_combined_messages_and_input(data: Mapping[str, object]) -> bool:
    """True if ``data`` carries both ``messages`` and ``input``.
    build_inspection_messages flattens both into one synthetic list, so
    mask-in-place would write input-derived content into data["messages"]
    (and vice versa) even when a message dropped for having no text
    coincidentally keeps the raw message count unchanged."""
    return isinstance(data.get("messages"), list) and data.get("input") is not None


def _has_responses_instructions(data: Mapping[str, object]) -> bool:
    """True if ``data`` carries a Responses-API ``instructions`` field.
    _build_lakera_inspection_messages includes ``instructions`` as a
    synthetic system message so Lakera can inspect it, but
    apply_redacted_messages_back has no path to rewrite
    ``data["instructions"]`` -- masking here would either leave unredacted
    content in the real instructions field the model reads, or write a
    redacted duplicate into data["messages"] instead, which the Responses
    API never consumes."""
    instructions = data.get("instructions")
    return isinstance(instructions, str) and bool(instructions)


def _build_lakera_inspection_messages(data: Mapping[str, object]) -> Sequence[Mapping[str, str]]:
    """Like build_inspection_messages, but also covers the Responses-API
    ``instructions`` field, placed first since litellm later converts it
    into the model's leading system message and a prompt-injection detector
    should see the same conversation order the model actually receives.

    Kept local to Lakera rather than folded into the shared
    _content_utils.build_inspection_messages helper: doing that once made
    ``instructions`` visible to every guardrail sharing that helper (AIM,
    presidio, bedrock, ...), but only Lakera has a masking-safety-guard
    (_has_responses_instructions) accounting for apply_redacted_messages_back
    having no write-back path for data["instructions"] -- other guardrails
    would have silently mishandled a PII/redaction hit found there."""
    instructions: Final = data.get("instructions")
    leading: Final[Sequence[Mapping[str, str]]] = (
        [{"role": "system", "content": instructions}]  # mutable-ok: fresh list/dict, not stored
        if isinstance(instructions, str) and instructions
        else []  # mutable-ok: fresh empty list, not stored
    )
    return [  # mutable-ok: fresh list, not stored
        *leading,
        *build_inspection_messages(dict(data)),  # mutable-ok: fresh shallow copy for the dict[str, Any] param
    ]


class LakeraAIGuardrail(CustomGuardrail):
    @classmethod
    def get_supported_event_hooks(cls) -> list[GuardrailEventHooks]:
        return [
            GuardrailEventHooks.pre_call,
            GuardrailEventHooks.during_call,
            GuardrailEventHooks.post_call,
        ]

    def __init__(
        self,
        api_key: str | None = None,
        api_base: str | None = None,
        project_id: str | None = None,
        payload: bool | None = True,
        breakdown: bool | None = True,
        metadata: dict | None = None,
        dev_info: bool | None = True,
        on_flagged: Literal["block", "monitor", "inject_system_message"] | None = "block",
        skip_system_message_in_guardrail: bool | None = None,
        skip_tool_message_in_guardrail: bool | None = None,
        advisory_system_message: str | None = None,
        **kwargs,
    ):
        """
        Initialize the LakeraAIGuardrail class.

        This guardrail only supports the chat completions endpoint (/v1/chat/completions).
        It is not supported for the Responses API, /v1/messages, MCP, A2A, or other endpoints.

        This calls: https://api.lakera.ai/v2/guard

        Args:
            api_key: Optional[str] = None,
            api_base: Optional[str] = None,
            project_id: Optional[str] = None,
            payload: Optional[bool] = True,
            breakdown: Optional[bool] = True,
            metadata: Optional[Dict] = None,
            dev_info: Optional[bool] = True,
            on_flagged: Optional[str] = "block", Action to take when content is flagged:
                "block", "monitor", or "inject_system_message"
            skip_system_message_in_guardrail: Optional[bool] = None,
            skip_tool_message_in_guardrail: Optional[bool] = None,
            advisory_system_message: Optional[str] = None, custom advisory message template
                (must contain a {reason} placeholder) used when on_flagged="inject_system_message".
                Defaults to a generic message when unset.
        """
        self.async_handler = get_async_httpx_client(llm_provider=httpxSpecialProvider.GuardrailCallback)
        self.lakera_api_key = api_key or os.environ.get("LAKERA_API_KEY") or ""
        self.project_id = project_id
        self.api_base = api_base or get_secret_str("LAKERA_API_BASE") or "https://api.lakera.ai"
        self.payload: bool | None = payload
        self.breakdown: bool | None = breakdown
        self.metadata: dict | None = metadata
        self.dev_info: bool | None = dev_info
        self.skip_system_message_in_guardrail = skip_system_message_in_guardrail
        self.skip_tool_message_in_guardrail = skip_tool_message_in_guardrail
        self.on_flagged = on_flagged or "block"
        self.advisory_system_message = advisory_system_message
        kwargs.setdefault("supported_event_hooks", list(self.get_supported_event_hooks()))
        super().__init__(**kwargs)
        self._validate_advisory_config(
            on_flagged=self.on_flagged,
            advisory_system_message=self.advisory_system_message,
            event_hook=self.event_hook,
        )

    def update_in_memory_litellm_params(self, litellm_params: LitellmParams) -> None:
        """
        The base implementation blindly ``setattr``s every field on ``litellm_params``
        (including ``on_flagged``/``advisory_system_message``) onto this live instance
        with no revalidation, so an in-place config update (via the DB/UI, without a
        restart) could otherwise reintroduce the exact invalid on_flagged/event_hook
        combinations __init__ rejects. Validate the prospective post-update state
        *before* mutating, so a rejected update leaves the live instance untouched
        instead of raising after it's already been corrupted.

        The base setattr also writes ``litellm_params.mode`` onto a new
        ``self.mode`` attribute rather than the ``self.event_hook`` dispatch
        actually reads (LitellmParams has no field literally named
        ``event_hook``), so without the explicit sync below a hot reload that
        moves this guardrail off during_call would pass validation but still
        dispatch as during_call afterward -- inject_system_message would then
        run against a live instance validation had confirmed was safe, but
        whose real dispatch hook never changed.
        """
        new_event_hook: Final = getattr(litellm_params, "mode", None) or self.event_hook
        self._validate_advisory_config(
            on_flagged=getattr(litellm_params, "on_flagged", None) or self.on_flagged,
            advisory_system_message=getattr(litellm_params, "advisory_system_message", None),
            event_hook=new_event_hook,
        )
        super().update_in_memory_litellm_params(litellm_params=litellm_params)
        self.event_hook = new_event_hook

    def _validate_advisory_config(
        self,
        on_flagged: str,
        advisory_system_message: str | None,
        event_hook: GuardrailEventHooks | Sequence[GuardrailEventHooks] | Mode | str | Sequence[str] | None,
    ) -> None:
        if advisory_system_message is not None:
            if not _template_uses_reason_placeholder(advisory_system_message):
                raise ValueError(
                    "Invalid advisory_system_message template: must include a real {reason} "
                    "placeholder (not an escaped {{reason}}) so the LLM sees why the request was flagged."
                )
            try:
                advisory_system_message.format(reason="placeholder")
            except (KeyError, IndexError, ValueError) as e:
                raise ValueError(
                    f"Invalid advisory_system_message template: {e}. The template must be a valid "
                    "str.format() string using only the {reason} placeholder."
                ) from e
        if on_flagged == "inject_system_message" and _event_hook_includes_during_call(event_hook):
            raise ValueError(
                "on_flagged='inject_system_message' is not supported for mode='during_call': during_call "
                "runs concurrently with the LLM dispatch with no pre-call barrier, so the advisory message "
                "cannot reliably reach the request. Use mode='pre_call' instead."
            )

    def _build_advisory_message(self, lakera_response: LakeraAIResponse | None) -> str:
        """Format the advisory message shown to the LLM when on_flagged='inject_system_message'."""
        reason: Final = humanize_lakera_block_reasons(lakera_response.get("breakdown") if lakera_response else None)
        template: Final = self.advisory_system_message or DEFAULT_ADVISORY_MESSAGE
        return template.format(reason=reason)

    def _filter_skipped_messages(
        self, messages: Sequence[AllMessageValues]
    ) -> tuple[tuple[AllMessageValues, ...], bool]:
        return filter_messages_by_skip_flags(self, messages)

    async def call_v2_guard(
        self,
        messages: list[AllMessageValues],
        request_data: dict,
        event_type: GuardrailEventHooks,
    ) -> tuple[LakeraAIResponse, dict]:
        """
        Call the Lakera AI v2 guard API.
        """
        status: GuardrailStatus = "success"
        exception_str: str = ""
        start_time: Final[datetime] = datetime.now()
        lakera_response: LakeraAIResponse | None = None
        request: dict = {}
        masked_entity_count: Final[dict] = {}
        try:
            request = dict(
                LakeraAIRequest(
                    messages=messages,
                    project_id=self.project_id,
                    payload=self.payload,
                    breakdown=self.breakdown,
                    metadata=self.metadata,
                    dev_info=self.dev_info,
                )
            )
            verbose_proxy_logger.debug("Lakera AI v2 guard request: %s", request)
            response: Final = await self.async_handler.post(
                url=f"{self.api_base}/v2/guard",
                headers={"Authorization": f"Bearer {self.lakera_api_key}"},
                json=request,
            )
            verbose_proxy_logger.debug("Lakera AI v2 guard response: %s", response.json())
            lakera_response = LakeraAIResponse(**response.json())
            return lakera_response, masked_entity_count
        except Exception as e:
            status = "guardrail_failed_to_respond"
            exception_str = str(e)
            raise e
        finally:
            ####################################################
            # Create Guardrail Trace for logging on Langfuse, Datadog, etc.
            ####################################################
            guardrail_json_response: Exception | str | dict | list[dict] = {}
            if status == "success":
                copy_lakera_response_dict: Final = dict(copy.deepcopy(lakera_response)) if lakera_response else {}
                # payload contains PII, we don't want to log it
                copy_lakera_response_dict.pop("payload")
                guardrail_json_response = copy_lakera_response_dict
            else:
                guardrail_json_response = exception_str
            self.add_standard_logging_guardrail_information_to_request_data(
                guardrail_json_response=guardrail_json_response,
                guardrail_status=status,
                request_data=request_data,
                start_time=start_time.timestamp(),
                end_time=datetime.now().timestamp(),
                duration=(datetime.now() - start_time).total_seconds(),
                masked_entity_count=masked_entity_count,
                event_type=event_type,
            )

    def _mask_pii_in_messages(
        self,
        messages: list[AllMessageValues],
        lakera_response: LakeraAIResponse | None,
        masked_entity_count: dict,
    ) -> list[AllMessageValues]:
        """
        Return a copy of messages with any detected PII replaced by
        “[MASKED <TYPE>]” tokens.
        """
        payload: Final = lakera_response.get("payload") if lakera_response else None
        if not payload:
            return messages

        messages = copy.deepcopy(messages)
        # For each message, find its detections on the fly
        for idx, msg in enumerate(messages):
            content = msg.get("content", "")
            if not content:
                continue

            # For v1, we only support masking content strings
            if not isinstance(content, str):
                continue

            # Filter only detections for this message
            detected_modifications = [d for d in payload if d.get("message_id") == idx]
            if not detected_modifications:
                continue

            # Apply masks from end to start so earlier indices remain valid after each replacement
            detected_modifications = sorted(
                detected_modifications,
                key=lambda d: (d.get("start", 0), d.get("end", 0)),
                reverse=True,
            )

            for modification in detected_modifications:
                start, end = modification.get("start", 0), modification.get("end", 0)

                # Extract the type (e.g. 'credit_card' → 'CREDIT_CARD')
                detector_type = modification.get("detector_type", "")
                if not detector_type:
                    continue

                typ = detector_type.split("/")[-1].upper() or "PII"
                mask = f"[MASKED {typ}]"
                if start is not None and end is not None:
                    content = self.mask_content_in_string(
                        content_string=content,
                        mask_string=mask,
                        start_index=start,
                        end_index=end,
                    )
                    masked_entity_count[typ] = masked_entity_count.get(typ, 0) + 1

            msg["content"] = content
        return messages

    async def async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: litellm.DualCache,
        data: dict,
        call_type: CallTypesLiteral,
    ) -> Exception | str | dict | None:
        from litellm.proxy.common_utils.callback_utils import (
            add_guardrail_to_applied_guardrails_header,
        )

        verbose_proxy_logger.debug("Lakera AI: pre_call_hook")

        event_type: Final[GuardrailEventHooks] = GuardrailEventHooks.pre_call
        if self.should_run_guardrail(data=data, event_type=event_type) is not True:
            verbose_proxy_logger.debug("Lakera AI: not running guardrail. Guardrail is disabled.")
            return data

        # Raw count before build_inspection_messages drops any message with no
        # inspectable text — needed below to detect that drop too, not just
        # skip-flag-driven drops.
        raw_message_count: Final = len(data.get("messages") or ())

        # Covers multimodal list content + Responses-API input/instructions.
        inspection_messages: Final = _build_lakera_inspection_messages(data)
        if not inspection_messages:
            verbose_proxy_logger.warning("Lakera AI: not running guardrail. No inspectable text in data")
            return data

        new_messages, messages_were_skipped = self._filter_skipped_messages(
            inspection_messages  # pyright: ignore[reportArgumentType]  # build_inspection_messages returns plain dicts, not typed message unions
        )
        if not new_messages:
            verbose_proxy_logger.warning(
                "Lakera AI: not running guardrail. All inspectable text was excluded by "
                "skip_system_message_in_guardrail/skip_tool_message_in_guardrail"
            )
            return data

        # Mask-in-place uses offsets returned by Lakera and can only
        # preserve non-text parts (images, audio, …) when the original
        # content is a plain string. For multimodal/Responses-API input
        # we degrade to block-on-detect so we never silently strip image
        # parts while attempting to redact text. The same applies when any
        # message was excluded from ``new_messages`` before masking — whether
        # by the skip flags or by build_inspection_messages dropping a
        # no-text message — since masking would rewrite data["messages"]
        # from the shorter inspected list, silently dropping the excluded
        # message from the actual outgoing request. Also degrade when any
        # message carries fields beyond role/content (e.g. a tool message's
        # tool_call_id), since masking would rewrite it from a role/content-only
        # synthetic dict, silently stripping those fields. Also degrade when
        # both messages and input are present, since build_inspection_messages
        # flattens both into one list and the raw-count check above can miss a
        # dropped no-text message when input backfills the count.
        is_multimodal_input: Final = (
            has_non_string_content(data)
            or messages_were_skipped
            or len(new_messages) < raw_message_count
            or _has_non_maskable_message_fields(data)
            or _has_combined_messages_and_input(data)
            or _has_responses_instructions(data)
        )

        #########################################################
        ########## 1. Make the Lakera AI v2 guard API request ##########
        #########################################################
        lakera_guardrail_response, masked_entity_count = await self.call_v2_guard(
            messages=new_messages,
            request_data=data,
            event_type=GuardrailEventHooks.pre_call,
        )

        #########################################################
        ########## 2. Handle flagged content ##########
        #########################################################
        if lakera_guardrail_response.get("flagged") is True:
            if self.on_flagged == "inject_system_message":
                advisory_delivered: Final = self.inject_advisory_message(
                    data, self._build_advisory_message(lakera_guardrail_response)
                )
                if advisory_delivered:
                    verbose_proxy_logger.warning(
                        "Lakera Guardrail: Advisory mode - violation detected, appended advisory system message"
                    )
                else:
                    # Structured Responses-API input (a list, not a plain string)
                    # has no field this can safely append into -- degrade to
                    # blocking rather than silently letting the flagged request
                    # through with no advisory ever reaching the model.
                    raise self._get_http_exception_for_blocked_guardrail(lakera_guardrail_response)
            # If only PII violations exist, mask the PII (string input only).
            elif self._is_only_pii_violation(lakera_guardrail_response) and not is_multimodal_input:
                redacted_messages: Final = self._mask_pii_in_messages(
                    messages=new_messages,
                    lakera_response=lakera_guardrail_response,
                    masked_entity_count=masked_entity_count,
                )
                # Write back to ``messages`` AND ``input``. The Responses-API
                # backend reads ``input``; writing only to ``messages``
                # would let unredacted PII reach the LLM for /v1/responses.
                apply_redacted_messages_back(data, list(redacted_messages))
                verbose_proxy_logger.debug("Lakera AI: Masked PII in messages instead of blocking request")
            else:
                # Check on_flagged setting
                if self.on_flagged == "monitor":
                    verbose_proxy_logger.warning(
                        "Lakera Guardrail: Monitoring mode - violation detected but allowing request"
                    )
                    # Log violation but continue
                elif self.on_flagged == "block":
                    # Either non-PII violations, or PII on multimodal input
                    # (which cannot be masked in place without dropping
                    # image/audio parts) — raise the standard block error.
                    raise self._get_http_exception_for_blocked_guardrail(lakera_guardrail_response)

        #########################################################
        ########## 3. Add the guardrail to the applied guardrails header ##########
        #########################################################
        add_guardrail_to_applied_guardrails_header(request_data=data, guardrail_name=self.guardrail_name)

        return data

    async def async_moderation_hook(
        self,
        data: dict,
        user_api_key_dict: UserAPIKeyAuth,
        call_type: CallTypesLiteral,
    ):
        from litellm.proxy.common_utils.callback_utils import (
            add_guardrail_to_applied_guardrails_header,
        )

        event_type: Final[GuardrailEventHooks] = GuardrailEventHooks.during_call
        if self.should_run_guardrail(data=data, event_type=event_type) is not True:
            return

        raw_message_count: Final = len(data.get("messages") or ())

        # Covers multimodal list content + Responses-API input/instructions.
        inspection_messages: Final = _build_lakera_inspection_messages(data)
        if not inspection_messages:
            verbose_proxy_logger.warning("Lakera AI: not running guardrail. No inspectable text in data")
            return

        new_messages, messages_were_skipped = self._filter_skipped_messages(
            inspection_messages  # pyright: ignore[reportArgumentType]  # build_inspection_messages returns plain dicts, not typed message unions
        )
        if not new_messages:
            verbose_proxy_logger.warning(
                "Lakera AI: not running guardrail. All inspectable text was excluded by "
                "skip_system_message_in_guardrail/skip_tool_message_in_guardrail"
            )
            return

        # See ``async_pre_call_hook`` — multimodal input degrades to
        # block-on-detect because mask-in-place would drop image parts; the
        # same applies to any message excluded from ``new_messages`` before
        # masking, whether by the skip flags or by build_inspection_messages
        # dropping a no-text message, since writing the masked (shorter) list
        # back would drop it from the outgoing request; and to any message
        # carrying fields beyond role/content, since masking would rewrite it
        # from a role/content-only synthetic dict; and to both messages and
        # input being present together, per the same reasoning.
        is_multimodal_input: Final = (
            has_non_string_content(data)
            or messages_were_skipped
            or len(new_messages) < raw_message_count
            or _has_non_maskable_message_fields(data)
            or _has_combined_messages_and_input(data)
            or _has_responses_instructions(data)
        )

        #########################################################
        ########## 1. Make the Lakera AI v2 guard API request ##########
        #########################################################
        lakera_guardrail_response, masked_entity_count = await self.call_v2_guard(
            messages=new_messages,
            request_data=data,
            event_type=GuardrailEventHooks.during_call,
        )

        #########################################################
        ########## 2. Handle flagged content ##########
        #########################################################
        if lakera_guardrail_response.get("flagged") is True:
            if self.on_flagged == "inject_system_message":
                # during_call runs concurrently with the LLM dispatch (see
                # ProxyLogging.during_call_hook / common_request_processing.py),
                # with no pre-call barrier -- mutating data["messages"] here races
                # against the outgoing request already being built from the same
                # dict, so the advisory message can silently fail to reach the
                # LLM. Degrade to monitor-equivalent (log only) instead, matching
                # how post_call also can't reliably influence a request that's
                # already been dispatched.
                verbose_proxy_logger.warning(
                    "Lakera Guardrail: Advisory mode has no effect during during_call; "
                    "violation detected but allowing request"
                )
            elif self._is_only_pii_violation(lakera_guardrail_response) and not is_multimodal_input:
                redacted_messages: Final = self._mask_pii_in_messages(
                    messages=new_messages,
                    lakera_response=lakera_guardrail_response,
                    masked_entity_count=masked_entity_count,
                )
                # Write back to ``messages`` AND ``input``. The Responses-API
                # backend reads ``input``; writing only to ``messages``
                # would let unredacted PII reach the LLM for /v1/responses.
                apply_redacted_messages_back(data, list(redacted_messages))
                verbose_proxy_logger.debug("Lakera AI: Masked PII in messages instead of blocking request")
            else:
                if self.on_flagged == "monitor":
                    verbose_proxy_logger.warning(
                        "Lakera Guardrail: Monitoring mode - violation detected but allowing request"
                    )
                elif self.on_flagged == "block":
                    raise self._get_http_exception_for_blocked_guardrail(lakera_guardrail_response)

        #########################################################
        ########## 3. Add the guardrail to the applied guardrails header ##########
        #########################################################
        add_guardrail_to_applied_guardrails_header(request_data=data, guardrail_name=self.guardrail_name)

        return data

    async def async_post_call_success_hook(
        self,
        data: dict,
        user_api_key_dict: UserAPIKeyAuth,
        response,
    ):
        """
        Post-call hook for Lakera guardrail.
        """
        from litellm.proxy.common_utils.callback_utils import (
            add_guardrail_to_applied_guardrails_header,
        )

        event_type: Final[GuardrailEventHooks] = GuardrailEventHooks.post_call
        if self.should_run_guardrail(data=data, event_type=event_type) is not True:
            return response

        original_messages: list[AllMessageValues] | None = data.get("messages", [])
        if original_messages is None:
            original_messages = []
        original_messages, _ = self._filter_skipped_messages(original_messages)

        # Extract assistant messages from the response, keeping only role/content.
        # Track choice indices so we write masked content back to the correct choice
        # when some choices have null content (e.g. tool-call-only).
        response_messages: Final[list[AllMessageValues]] = []
        choice_indices: Final[list[int]] = []
        response_dict: Final = response.model_dump() if hasattr(response, "model_dump") else {}
        for i, choice in enumerate(response_dict.get("choices", [])):
            msg = choice.get("message")
            if not msg:
                continue
            role = msg.get("role")
            content = msg.get("content")
            if role and content:
                response_messages.append({"role": role, "content": content})
                choice_indices.append(i)

        # Use a copy of original_messages so _mask_pii_in_messages does not mutate data["messages"]
        post_call_messages: Final = list(copy.deepcopy(original_messages)) + response_messages  # mutable-ok: needs list

        # Call Lakera guardrail
        lakera_guardrail_response, _ = await self.call_v2_guard(
            messages=post_call_messages,
            request_data=data,
            event_type=GuardrailEventHooks.post_call,
        )

        # Handle flagged content
        if lakera_guardrail_response.get("flagged") is True:
            # If only PII violations exist, mask the PII in the response and allow
            if self._is_only_pii_violation(lakera_guardrail_response):
                masked_entity_count: Final[dict[str, int]] = {}
                masked_messages: Final = self._mask_pii_in_messages(
                    messages=post_call_messages,
                    lakera_response=lakera_guardrail_response,
                    masked_entity_count=masked_entity_count,
                )
                assistant_messages: Final = masked_messages[len(original_messages) :]
                for idx, msg in enumerate(assistant_messages):
                    if idx < len(choice_indices):
                        choice_idx = choice_indices[idx]
                        response_dict["choices"][choice_idx]["message"]["content"] = msg.get("content", "")
                add_guardrail_to_applied_guardrails_header(request_data=data, guardrail_name=self.guardrail_name)
                return ModelResponse(**response_dict)

            # inject_system_message has nothing left to inject into once a response
            # already exists, so it is treated the same as monitor: log and allow.
            if self.on_flagged in ("monitor", "inject_system_message"):
                verbose_proxy_logger.warning(
                    "Lakera Guardrail: Post-call violation detected (on_flagged=%s) - allowing response",
                    self.on_flagged,
                )
            elif self.on_flagged == "block":
                raise self._get_http_exception_for_blocked_guardrail(lakera_guardrail_response)

        # Record applied guardrail
        add_guardrail_to_applied_guardrails_header(request_data=data, guardrail_name=self.guardrail_name)

        return response

    def _is_only_pii_violation(self, lakera_response: LakeraAIResponse | None) -> bool:
        """
        Returns True if there are only PII violations in the response.
        """
        if not lakera_response:
            return False

        # Check breakdown field for detected violations
        breakdown: Final = lakera_response.get("breakdown", []) or []
        if not breakdown:
            return False

        has_violations = False
        for item in breakdown:
            if item.get("detected", False):
                has_violations = True
                detector_type = item.get("detector_type", "") or ""
                if not detector_type.startswith("pii/"):
                    return False

        # Return True only if there are violations and they are all PII
        return has_violations

    def _get_http_exception_for_blocked_guardrail(self, lakera_response: LakeraAIResponse | None) -> HTTPException:
        """
        Get the HTTP exception for a blocked guardrail, similar to Bedrock's implementation.
        """
        return HTTPException(
            status_code=400,
            detail={
                "error": "Violated guardrail policy",
                "lakera_guardrail_response": lakera_response,
            },
        )
