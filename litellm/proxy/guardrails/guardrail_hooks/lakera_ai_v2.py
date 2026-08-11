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
        self.on_flagged = on_flagged or "block"
        self.advisory_system_message = advisory_system_message
        kwargs.setdefault("supported_event_hooks", list(self.get_supported_event_hooks()))
        super().__init__(**kwargs)
        self._validate_advisory_config()

    def update_in_memory_litellm_params(self, litellm_params: LitellmParams) -> None:
        """
        The base implementation blindly ``setattr``s every field on ``litellm_params``
        (including ``on_flagged``/``advisory_system_message``) onto this live instance
        with no revalidation, so an in-place config update (via the DB/UI, without a
        restart) could otherwise reintroduce the exact invalid on_flagged/event_hook
        combinations __init__ rejects. Re-run the same validation after every update.
        """
        super().update_in_memory_litellm_params(litellm_params=litellm_params)
        self._validate_advisory_config()

    def _validate_advisory_config(self) -> None:
        if self.advisory_system_message is not None:
            if not _template_uses_reason_placeholder(self.advisory_system_message):
                raise ValueError(
                    "Invalid advisory_system_message template: must include a real {reason} "
                    "placeholder (not an escaped {{reason}}) so the LLM sees why the request was flagged."
                )
            try:
                self.advisory_system_message.format(reason="placeholder")
            except (KeyError, IndexError, ValueError) as e:
                raise ValueError(
                    f"Invalid advisory_system_message template: {e}. The template must be a valid "
                    "str.format() string using only the {reason} placeholder."
                ) from e
        if self.on_flagged == "inject_system_message" and _event_hook_includes_during_call(self.event_hook):
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

        # Covers multimodal list content + Responses-API input.
        new_messages: Final = build_inspection_messages(data)
        if not new_messages:
            verbose_proxy_logger.warning("Lakera AI: not running guardrail. No inspectable text in data")
            return data

        # Mask-in-place uses offsets returned by Lakera and can only
        # preserve non-text parts (images, audio, …) when the original
        # content is a plain string. For multimodal/Responses-API input
        # we degrade to block-on-detect so we never silently strip image
        # parts while attempting to redact text.
        is_multimodal_input: Final = has_non_string_content(data)

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
                self.inject_advisory_message(data, self._build_advisory_message(lakera_guardrail_response))
                verbose_proxy_logger.warning(
                    "Lakera Guardrail: Advisory mode - violation detected, appended advisory system message"
                )
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

        new_messages: Final = build_inspection_messages(data)
        if not new_messages:
            verbose_proxy_logger.warning("Lakera AI: not running guardrail. No inspectable text in data")
            return

        # See ``async_pre_call_hook`` — multimodal input degrades to
        # block-on-detect because mask-in-place would drop image parts.
        is_multimodal_input: Final = has_non_string_content(data)

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
        post_call_messages: Final = copy.deepcopy(original_messages) + response_messages

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
