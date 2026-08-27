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
    effective_skip_system_message_for_guardrail,
    effective_skip_tool_message_for_guardrail,
    filter_messages_by_skip_flags,
    merge_guardrailed_scoped_messages,
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
from litellm.types.guardrails import GuardrailEventHooks, LitellmParams
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


def _pre_masking_scope_indices(
    guardrail: "LakeraAIGuardrail",
    messages: Sequence[object],
) -> tuple[int, ...]:
    """Indices into ``messages`` that mask-in-place can safely target: has
    non-empty string content, and survives the same skip_system_message_in_guardrail
    / skip_tool_message_in_guardrail scoping ``filter_messages_by_skip_flags``
    applies. Content is guaranteed to already be a plain string here -- masking
    is only attempted when ``has_non_string_content(data)`` is False.

    Preserved in original order, so it lines up positionally with the
    ``messages_for_lakera`` list _build_lakera_inspection_messages/skip-filtering
    produces from the same input: both apply the identical "has text" and
    "not skipped by role" predicates over the same original sequence. Role
    comparison is lowercased to match filter_messages_by_skip_flags's own
    normalization (via its _message_role helper) -- an uppercase-cased
    "System"/"TOOL" role must be excluded by both or the two lists disagree
    on length and the caller's strict positional zip raises."""
    skip_system: Final = effective_skip_system_message_for_guardrail(guardrail)
    skip_tool: Final = effective_skip_tool_message_for_guardrail(guardrail)
    return tuple(
        idx
        for idx, message in enumerate(messages)
        if isinstance(message, dict)
        and isinstance(message.get("content"), str)
        and message["content"]
        and not (skip_system and str(message.get("role") or "").lower() == "system")
        and not (skip_tool and str(message.get("role") or "").lower() == "tool")
    )


def _apply_redacted_messages_back_preserving_fields(
    guardrail: "LakeraAIGuardrail",
    data: dict[str, object],  # mutable-ok: writes the redacted result back into the caller's request dict in place
    redacted_messages: Sequence[AllMessageValues],
) -> None:
    """Write masked content back to ``data["messages"]`` without losing fields
    the synthetic role/content-only ``redacted_messages`` never carried (e.g. a
    tool message's tool_call_id, an assistant message's tool_calls, name,
    cache_control). Falls back to the shared, wholesale-replacing
    apply_redacted_messages_back when ``data["messages"]`` isn't a list (a pure
    Responses-API ``input`` string, with no chat messages to merge into)."""
    original_messages: Final = data.get("messages")
    if not isinstance(original_messages, list):
        redacted_list: Final = list(redacted_messages)  # mutable-ok: apply_redacted_messages_back requires a list
        apply_redacted_messages_back(data, redacted_list)
        return
    scope_indices: Final = _pre_masking_scope_indices(guardrail, original_messages)
    guardrailed_scoped: Final = tuple(
        {  # mutable-ok: fresh dict per iteration, not stored beyond this comprehension
            **original_messages[original_idx],
            "content": redacted["content"],
        }
        for original_idx, redacted in zip(scope_indices, redacted_messages, strict=True)
    )
    data["messages"] = merge_guardrailed_scoped_messages(
        full_messages=original_messages,
        scoped_indices=scope_indices,
        guardrailed_scoped=guardrailed_scoped,  # pyright: ignore[reportArgumentType]  # plain dicts satisfy AllMessageValues's TypedDict shape at runtime
    )


def _has_combined_messages_and_input(data: Mapping[str, object]) -> bool:
    """True if ``data`` carries both ``messages`` and ``input``.
    build_inspection_messages flattens both into one synthetic list, so
    mask-in-place would write input-derived content into data["messages"]
    (and vice versa) even when a message dropped for having no text
    coincidentally keeps the raw message count unchanged."""
    return isinstance(data.get("messages"), list) and data.get("input") is not None


def _has_responses_instructions(guardrail: "LakeraAIGuardrail", data: Mapping[str, object]) -> bool:
    """True if ``data`` carries a Responses-API ``instructions`` field that
    Lakera actually inspected. _build_lakera_inspection_messages includes
    ``instructions`` as a synthetic system message so Lakera can inspect it,
    but apply_redacted_messages_back has no path to rewrite
    ``data["instructions"]`` -- masking here would either leave unredacted
    content in the real instructions field the model reads, or write a
    redacted duplicate into data["messages"] instead, which the Responses
    API never consumes.

    When skip_system_message_in_guardrail excludes that synthetic system
    message before it ever reaches Lakera, none of this applies: Lakera never
    saw ``instructions``, so it can't have flagged anything there, and
    forcing a hard block anyway would defeat the whole point of the skip
    flag for a response that only carries PII in the (maskable) non-system
    content."""
    instructions: Final = data.get("instructions")
    return (
        isinstance(instructions, str)
        and bool(instructions)
        and not effective_skip_system_message_for_guardrail(guardrail)
    )


def _breakdown_has_pii_violation(lakera_response: LakeraAIResponse | None) -> bool:
    """True if any PII-category detector fired, regardless of whether other,
    non-PII detectors (prompt injection, moderated content) also fired.
    Unlike ``_is_only_pii_violation``, this doesn't require PII to be the
    *only* thing detected -- it's used to decide whether masking/blocking is
    even relevant at all before advisory mode's own logic runs."""
    if not lakera_response:
        return False
    breakdown: Final = lakera_response.get("breakdown") or ()
    return any(
        item.get("detected", False) and (item.get("detector_type") or "").startswith("pii/") for item in breakdown
    )


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
            payload=self.payload,
            breakdown=self.breakdown,
        )

    def update_in_memory_litellm_params(self, litellm_params: LitellmParams) -> None:
        """
        The base implementation blindly ``setattr``s every field on ``litellm_params``
        (including ``on_flagged``/``advisory_system_message``/``payload``/``breakdown``)
        onto this live instance with no revalidation, so an in-place config update (via
        the DB/UI, without a restart) could otherwise reintroduce the exact invalid
        on_flagged combinations __init__ rejects. Validate the prospective post-update
        state *before* mutating, so a rejected update leaves the live instance untouched
        instead of raising after it's already been corrupted.

        The base setattr also writes ``litellm_params.mode`` onto a new ``self.mode``
        attribute rather than the ``self.event_hook`` dispatch actually reads
        (LitellmParams has no field literally named ``event_hook``), so without the
        explicit sync below a hot reload that changes mode would pass validation but
        keep dispatching on the stale event_hook.
        """
        new_event_hook: Final = getattr(litellm_params, "mode", None) or self.event_hook
        prospective_payload: Final = getattr(litellm_params, "payload", None)
        prospective_breakdown: Final = getattr(litellm_params, "breakdown", None)
        self._validate_advisory_config(
            on_flagged=getattr(litellm_params, "on_flagged", None) or self.on_flagged,
            advisory_system_message=getattr(litellm_params, "advisory_system_message", None),
            payload=self.payload if prospective_payload is None else prospective_payload,
            breakdown=self.breakdown if prospective_breakdown is None else prospective_breakdown,
        )
        super().update_in_memory_litellm_params(litellm_params=litellm_params)
        self.event_hook = new_event_hook

    def _validate_advisory_config(
        self,
        on_flagged: str,
        advisory_system_message: str | None,
        payload: bool | None,
        breakdown: bool | None,
    ) -> None:
        if on_flagged == "inject_system_message" and advisory_system_message is not None:
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
        if on_flagged == "inject_system_message" and not (payload and breakdown):
            raise ValueError(
                "on_flagged='inject_system_message' requires payload=True and breakdown=True: advisory "
                "mode masks any detected PII before appending the advisory note, and that masking can "
                "only happen when Lakera's response carries both the violation breakdown and the "
                "payload location data. Without them, PII would be forwarded to the model unredacted."
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
        messages: Sequence[AllMessageValues],
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
        messages: Sequence[AllMessageValues],
        lakera_response: LakeraAIResponse | None,
        masked_entity_count: dict,
    ) -> Sequence[AllMessageValues]:
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

        # Covers multimodal list content + Responses-API input/instructions.
        inspection_messages: Final = _build_lakera_inspection_messages(data)
        if not inspection_messages:
            verbose_proxy_logger.warning("Lakera AI: not running guardrail. No inspectable text in data")
            return data

        new_messages, _ = self._filter_skipped_messages(
            inspection_messages  # pyright: ignore[reportArgumentType]  # build_inspection_messages returns plain dicts, not typed message unions
        )
        if not new_messages:
            verbose_proxy_logger.warning(
                "Lakera AI: not running guardrail. All inspectable text was excluded by "
                "skip_system_message_in_guardrail/skip_tool_message_in_guardrail"
            )
            return data

        # Mask-in-place can only preserve non-text parts (images, audio) when
        # the original content is a plain string, and can only merge a
        # redacted result back into data["messages"] by position when
        # messages and input aren't both present at once (build_inspection_messages
        # flattens both into one list, so a position could mean either).
        # Degrade to block-on-detect in either case. Skip-flag-excluded and
        # no-text messages, and messages carrying fields beyond role/content
        # (tool_call_id, name, tool_calls, cache_control), are otherwise
        # handled safely by _apply_redacted_messages_back_preserving_fields's
        # scope-index merge, which never touches a message outside the scope
        # it actually redacted instead of reconstructing the list from scratch.
        is_multimodal_input: Final = (
            has_non_string_content(data)
            or _has_combined_messages_and_input(data)
            or _has_responses_instructions(self, data)
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
            # PII-only violations get masked in place regardless of on_flagged: there's
            # no reason to expose raw PII to satisfy an advisory note, and masking is
            # strictly safer than either blocking or appending an advisory message next
            # to unredacted PII.
            if self._is_only_pii_violation(lakera_guardrail_response) and not is_multimodal_input:
                redacted_messages: Final = self._mask_pii_in_messages(
                    messages=new_messages,
                    lakera_response=lakera_guardrail_response,
                    masked_entity_count=masked_entity_count,
                )
                _apply_redacted_messages_back_preserving_fields(self, data, redacted_messages)
                verbose_proxy_logger.debug("Lakera AI: Masked PII in messages instead of blocking request")
            elif self.on_flagged == "inject_system_message":
                if _breakdown_has_pii_violation(lakera_guardrail_response) and is_multimodal_input:
                    # There's PII in the mix and nothing here can be safely masked,
                    # so an advisory note next to this raw, unredacted PII would be
                    # no safer than a note next to nothing. Degrade to blocking
                    # instead, same as this on_flagged setting already does when
                    # the advisory itself has no field it can be delivered into.
                    raise self._get_http_exception_for_blocked_guardrail(lakera_guardrail_response)
                masked_pii_before_advisory: Final = _breakdown_has_pii_violation(lakera_guardrail_response)
                if masked_pii_before_advisory:
                    # A mixed violation (PII plus something else, e.g. prompt
                    # injection): mask whatever Lakera returned location data for
                    # before advising about what remains, so the advisory is never
                    # shown next to raw PII that could have been redacted.
                    mixed_redacted_messages: Final = self._mask_pii_in_messages(
                        messages=new_messages,
                        lakera_response=lakera_guardrail_response,
                        masked_entity_count=masked_entity_count,
                    )
                    _apply_redacted_messages_back_preserving_fields(self, data, mixed_redacted_messages)
                advisory_delivered: Final = self.inject_advisory_message(
                    data, self._build_advisory_message(lakera_guardrail_response)
                )
                if advisory_delivered:
                    verbose_proxy_logger.warning(
                        "Lakera Guardrail: Advisory mode - violation detected, %sappended advisory system message",
                        "masked PII and " if masked_pii_before_advisory else "",
                    )
                else:
                    # Structured Responses-API input (a list, not a plain string)
                    # has no field this can safely append into -- degrade to
                    # blocking rather than silently letting the flagged request
                    # through with no advisory ever reaching the model.
                    raise self._get_http_exception_for_blocked_guardrail(lakera_guardrail_response)
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

        # Covers multimodal list content + Responses-API input/instructions.
        inspection_messages: Final = _build_lakera_inspection_messages(data)
        if not inspection_messages:
            verbose_proxy_logger.warning("Lakera AI: not running guardrail. No inspectable text in data")
            return

        new_messages, _ = self._filter_skipped_messages(
            inspection_messages  # pyright: ignore[reportArgumentType]  # build_inspection_messages returns plain dicts, not typed message unions
        )
        if not new_messages:
            verbose_proxy_logger.warning(
                "Lakera AI: not running guardrail. All inspectable text was excluded by "
                "skip_system_message_in_guardrail/skip_tool_message_in_guardrail"
            )
            return

        # See async_pre_call_hook for the full rationale: mask-in-place
        # degrades to block-on-detect only for multimodal content or when
        # messages and input are both present; everything else (skipped/no-text
        # messages, extra chat fields) is handled safely by
        # _apply_redacted_messages_back_preserving_fields's scope-index merge.
        is_multimodal_input: Final = (
            has_non_string_content(data)
            or _has_combined_messages_and_input(data)
            or _has_responses_instructions(self, data)
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
            # See async_pre_call_hook: PII-only violations get masked regardless of
            # on_flagged, including inject_system_message, before any advisory logic.
            if self._is_only_pii_violation(lakera_guardrail_response) and not is_multimodal_input:
                redacted_messages: Final = self._mask_pii_in_messages(
                    messages=new_messages,
                    lakera_response=lakera_guardrail_response,
                    masked_entity_count=masked_entity_count,
                )
                _apply_redacted_messages_back_preserving_fields(self, data, redacted_messages)
                verbose_proxy_logger.debug("Lakera AI: Masked PII in messages instead of blocking request")
            elif self.on_flagged == "inject_system_message":
                if not is_multimodal_input:
                    # A mixed violation (PII plus something else): mask whatever's
                    # maskable even though the advisory note below has no effect
                    # here, so raw PII doesn't pass through untouched just because
                    # this violation wasn't PII-only.
                    mixed_redacted_messages: Final = self._mask_pii_in_messages(
                        messages=new_messages,
                        lakera_response=lakera_guardrail_response,
                        masked_entity_count=masked_entity_count,
                    )
                    _apply_redacted_messages_back_preserving_fields(self, data, mixed_redacted_messages)
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

        messages_or_none: Final[list[AllMessageValues] | None] = data.get("messages")
        original_messages, _ = self._filter_skipped_messages(messages_or_none or [])

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
