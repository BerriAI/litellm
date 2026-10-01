#!/usr/bin/env python3
"""
Azure Prompt Shield Native Guardrail Integrationfor LiteLLM
"""

import math
from collections.abc import Mapping, MutableMapping, Sequence
from contextvars import ContextVar
from dataclasses import dataclass
from functools import reduce
from itertools import chain, zip_longest
from typing import TYPE_CHECKING, Any, ClassVar, Final, Literal, NoReturn

from fastapi import HTTPException
from pydantic import TypeAdapter, ValidationError

from litellm._logging import verbose_proxy_logger
from litellm.integrations.custom_guardrail import (
    CustomGuardrail,
    log_guardrail_information,
)
from litellm.litellm_core_utils.llm_cost_calc.guardrail_cost import (
    AZURE_PROMPT_SHIELD_TEXT_RECORD_UNIT,
    azure_prompt_shield_guardrail_cost,
)
from litellm.llms.base_llm.guardrail_translation.attachments import content_attachments, request_attachments
from litellm.llms.base_llm.guardrail_translation.utils import message_slot_texts
from litellm.secret_managers.main import get_secret_str
from litellm.types.guardrails import GuardrailEventHooks
from litellm.types.proxy.guardrails.guardrail_hooks.azure.azure_prompt_shield import (
    AzurePromptShieldGuardrailRequestBody,
    AzurePromptShieldGuardrailResponse,
)
from litellm.types.utils import (
    CallTypesLiteral,
    GenericGuardrailAPIInputs,
    GuardrailTracingDetail,
)

from .base import (
    AZURE_CONTENT_SAFETY_MAX_TEXT_LENGTH,
    AZURE_CONTENT_SAFETY_TEXT_RECORD_LENGTH,
    AzureGuardrailBase,
)

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.types.guardrails import LitellmParams
    from litellm.types.llms.openai import AllMessageValues
    from litellm.types.proxy.guardrails.guardrail_hooks.base import GuardrailConfigModel


# Per-invocation billing counters. A ContextVar rather than request metadata: the
# decorator can swap out ``request_data``, metadata is client-forgeable, and
# concurrent guardrails run in separate tasks with their own context copy.
_billing_usage_stash: Final[ContextVar[dict[str, int] | None]] = ContextVar(  # mutable-ok: task-local stash
    "azure_prompt_shield_billing_usage", default=None
)

AZURE_PROMPT_SHIELD_MAX_DOCUMENTS: Final = 5
TOOL_OUTPUT_ROLES: Final = frozenset({"tool", "function"})
TOOL_RESULT_PART_TYPE: Final = "tool_result"

_CONTENT_PARTS: Final = TypeAdapter(tuple[Mapping[str, object], ...])


@dataclass(frozen=True, slots=True)
class _ShieldRequest:
    user_prompt: str | None
    documents: tuple[str, ...]

    @property
    def texts(self) -> tuple[str, ...]:
        return (*(() if self.user_prompt is None else (self.user_prompt,)), *self.documents)

    def body(self) -> AzurePromptShieldGuardrailRequestBody:
        if self.user_prompt is None:
            return AzurePromptShieldGuardrailRequestBody(documents=self.documents)
        return AzurePromptShieldGuardrailRequestBody(userPrompt=self.user_prompt, documents=self.documents)


def _parts(content: object) -> tuple[Mapping[str, object], ...]:
    if content is None or isinstance(content, str):
        return ()
    try:
        return _CONTENT_PARTS.validate_python(content)
    except ValidationError:
        return ()


def _is_tool_result(part: Mapping[str, object]) -> bool:
    return part.get("type") == TOOL_RESULT_PART_TYPE


def _is_user_turn(message: Mapping[str, object]) -> bool:
    if message.get("role") != "user":
        return False
    content: Final = message.get("content")
    return isinstance(content, str) or any(not _is_tool_result(part) for part in _parts(content))


def _current_turn_start(messages: Sequence[Mapping[str, object]]) -> int | None:
    return next((index for index in reversed(range(len(messages))) if _is_user_turn(messages[index])), None)


def _tool_output_nodes(message: Mapping[str, object]) -> tuple[Mapping[str, object], ...]:
    role: Final = message.get("role")
    if role in TOOL_OUTPUT_ROLES:
        return (message,)
    if role != "user":
        return ()
    return tuple(part for part in _parts(message.get("content")) if _is_tool_result(part))


def _tool_output_documents(node: Mapping[str, object]) -> tuple[str, ...]:
    text: Final = "\n".join(message_slot_texts(node))
    attachments: Final = content_attachments(node.get("content")).texts
    return (*(() if not text else (text,)), *(attachment.text for attachment in attachments))


def _prompt_and_documents(messages: Sequence[Mapping[str, object]]) -> tuple[str, tuple[str, ...]]:
    start: Final = _current_turn_start(messages)
    user_message: Final = None if start is None else messages[start]
    user_prompt: Final = "" if user_message is None else "\n".join(message_slot_texts(user_message))
    own_parts: Final = () if user_message is None else _parts(user_message.get("content"))
    own_attachments: Final = content_attachments(tuple(part for part in own_parts if not _is_tool_result(part))).texts
    tool_nodes: Final = chain.from_iterable(map(_tool_output_nodes, messages[start or 0 :]))
    tool_documents: Final = chain.from_iterable(map(_tool_output_documents, tool_nodes))
    return user_prompt, (*(attachment.text for attachment in own_attachments), *tool_documents)


def _fits(batch: tuple[str, ...], piece: str) -> bool:
    within_count: Final = len(batch) < AZURE_PROMPT_SHIELD_MAX_DOCUMENTS
    within_length: Final = sum(map(len, batch)) + len(piece) <= AZURE_CONTENT_SAFETY_MAX_TEXT_LENGTH
    return within_count and within_length


def _with_piece(batches: tuple[tuple[str, ...], ...], piece: str) -> tuple[tuple[str, ...], ...]:
    if batches and _fits(batches[-1], piece):
        return (*batches[:-1], (*batches[-1], piece))
    return (*batches, (piece,))


def _document_batches(documents: Sequence[str]) -> tuple[tuple[str, ...], ...]:
    non_empty: Final = (document for document in documents if document)
    pieces: Final = chain.from_iterable(
        AzureGuardrailBase.split_text_by_words(document, AZURE_CONTENT_SAFETY_MAX_TEXT_LENGTH) for document in non_empty
    )
    empty: Final[tuple[tuple[str, ...], ...]] = ()
    return reduce(_with_piece, pieces, empty)


def _paired_requests(chunk: str | None, batch: tuple[str, ...] | None) -> tuple[_ShieldRequest, ...]:
    documents: Final = batch or ()
    if len(chunk or "") + sum(map(len, documents)) <= AZURE_CONTENT_SAFETY_MAX_TEXT_LENGTH:
        return (_ShieldRequest(user_prompt=chunk, documents=documents),)
    return (_ShieldRequest(user_prompt=chunk, documents=()), _ShieldRequest(user_prompt=None, documents=documents))


def _shield_requests(user_prompt: str, documents: Sequence[str]) -> tuple[_ShieldRequest, ...]:
    prompt_chunks: Final = (
        tuple(AzureGuardrailBase.split_text_by_words(user_prompt, AZURE_CONTENT_SAFETY_MAX_TEXT_LENGTH))
        if user_prompt
        else ()
    )
    pairs: Final = zip_longest(prompt_chunks, _document_batches(documents), fillvalue=None)
    return tuple(chain.from_iterable(_paired_requests(chunk, batch) for chunk, batch in pairs))


def _add_usage(usage_accumulator: MutableMapping[str, int], texts: Sequence[str]) -> None:  # mutable-ok: accumulator
    text_records: Final = sum(math.ceil(len(text) / AZURE_CONTENT_SAFETY_TEXT_RECORD_LENGTH) for text in texts)
    usage_accumulator.update(
        (
            ("requests", usage_accumulator.get("requests", 0) + 1),
            ("input_characters", usage_accumulator.get("input_characters", 0) + sum(map(len, texts))),
            (
                AZURE_PROMPT_SHIELD_TEXT_RECORD_UNIT,
                usage_accumulator.get(AZURE_PROMPT_SHIELD_TEXT_RECORD_UNIT, 0) + text_records,
            ),
        )
    )


def _require_complete_analysis(response: AzurePromptShieldGuardrailResponse, request: _ShieldRequest) -> None:
    if request.user_prompt is not None and response.userPromptAnalysis is None:
        raise ValueError("Azure Prompt Shield: response carries no userPromptAnalysis for the submitted user prompt")
    if len(response.documentsAnalysis) < len(request.documents):
        raise ValueError(
            f"Azure Prompt Shield: response analyzed {len(response.documentsAnalysis)} of "
            f"{len(request.documents)} submitted documents"
        )


def _detection_message(response: AzurePromptShieldGuardrailResponse) -> str | None:
    if response.userPromptAnalysis is not None and response.userPromptAnalysis.attackDetected:
        return f"Attack detected in user prompt: {response.userPromptAnalysis.model_dump()}"
    document_attack: Final = next(
        (analysis for analysis in response.documentsAnalysis if analysis.attackDetected),
        None,
    )
    if document_attack is None:
        return None
    return f"Attack detected in a document (attachment or tool output): {document_attack.model_dump()}"


def _resolved_secret_value(value: object) -> object:
    """Resolve ``os.environ/<VAR>`` references the way guardrail api_key/api_base
    are resolved; any other value passes through unchanged. A reference that
    resolves to nothing raises instead of silently disabling pricing, so an
    intended-paid deployment fails fast rather than starting in usage-only mode."""
    if isinstance(value, str) and value.startswith("os.environ/"):
        resolved: Final = get_secret_str(value)
        if resolved is None or not resolved.strip():
            raise ValueError(f"Azure Prompt Shield: {value!r} resolves to an unset or blank environment variable")
        return resolved
    return value


def _updated_param(litellm_params: "LitellmParams | dict", key: str) -> object:  # mutable-ok: DB dict
    """Read one param from a Mapping or a pydantic object, including pydantic
    extras (cost_tier / price_per_1000_text_records live there), which the base
    class ``vars()`` loop never sees."""
    if isinstance(litellm_params, Mapping):
        return litellm_params.get(key)
    return getattr(litellm_params, key, None)


def _resolved_cost_tier(raw: object) -> str | None:
    """Normalize the configured cost_tier to 'free' / 'paid' / None."""
    value: Final = _resolved_secret_value(raw)
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    tier: Final = str(value).strip().lower()
    if tier not in ("free", "paid"):
        raise ValueError(f"Azure Prompt Shield: cost_tier must be 'free' or 'paid', got {value!r}")
    return tier


def _resolved_price(raw: object, cost_tier: str | None) -> float | None:
    """Normalize price_per_1000_text_records and validate it against the tier.

    A 'paid' tier requires a positive price so a misconfigured deployment fails at
    startup instead of silently reporting a wrong cost; an omitted price with no
    tier means usage-only tracking (no cost estimate)."""
    value: Final = _resolved_secret_value(raw)
    price: Final = _price_from_value(value)
    if cost_tier == "paid" and (price is None or price <= 0):
        raise ValueError("Azure Prompt Shield: cost_tier 'paid' requires a positive price_per_1000_text_records")
    return price


def _price_from_value(value: object) -> float | None:
    """Parse a resolved price value into a float; None for an unset/blank value."""
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise TypeError(f"Azure Prompt Shield: price_per_1000_text_records must be a number, got {value!r}")
    try:
        price: Final = float(value)
    except ValueError as e:
        raise ValueError(f"Azure Prompt Shield: price_per_1000_text_records must be a number, got {value!r}") from e
    if not math.isfinite(price) or price < 0:
        raise ValueError(
            f"Azure Prompt Shield: price_per_1000_text_records must be a finite, non-negative number, got {value!r}"
        )
    return price


class AzureContentSafetyPromptShieldGuardrail(AzureGuardrailBase, CustomGuardrail):
    """
    LiteLLM Built-in Guardrail for Azure Content Safety Guardrail (Prompt Shield).

    This guardrail scans prompts and responses using the Azure Prompt Shield API to detect
    malicious content, injection attempts, and policy violations.

    Configuration:
        guardrail_name: Name of the guardrail instance
        api_key: Azure Prompt Shield API key
        api_base: Azure Prompt Shield API endpoint
        default_on: Whether to enable by default
    """

    use_native_lifecycle_hooks: ClassVar[bool] = True

    def __init__(
        self,
        guardrail_name: str,
        api_key: str,
        api_base: str,
        **kwargs,
    ):
        """Initialize Azure Prompt Shield guardrail handler."""
        # AzureGuardrailBase.__init__ stores api_key, api_base, api_version,
        # async_handler and forwards the rest to CustomGuardrail.
        super().__init__(
            api_key=api_key,
            api_base=api_base,
            guardrail_name=guardrail_name,
            supported_event_hooks=list(self.get_supported_event_hooks()),
            **kwargs,
        )

        # Plain (non-Final) attributes: ``update_in_memory_litellm_params``
        # re-resolves them when the guardrail is updated in place.
        self.cost_tier: str | None = _resolved_cost_tier(kwargs.get("cost_tier"))
        self.price_per_1000_text_records: float | None = _resolved_price(
            kwargs.get("price_per_1000_text_records"), self.cost_tier
        )

        verbose_proxy_logger.debug("Initialized Azure Prompt Shield Guardrail: %s", guardrail_name)

    async def async_make_request(
        self,
        user_prompt: str,
        usage_accumulator: MutableMapping[str, int],  # mutable-ok: callee-filled accumulator
        documents: Sequence[str] = (),
    ) -> AzurePromptShieldGuardrailResponse:
        """
        Scan a user prompt and its documents (attachments and tool outputs) with
        the Azure Prompt Shield API.

        The prompt is split at word boundaries into chunks within the Azure
        Content Safety text limit; each document is split the same way and the
        pieces are packed into batches within Azure's per-request document count
        and total length limits. Request ``i`` carries prompt chunk ``i`` and
        document batch ``i`` when they exist. An attack in any chunk or document
        raises an HTTPException immediately.

        ``usage_accumulator`` collects billable usage per SUBMITTED request:
        ``requests`` (Azure API calls), ``input_characters`` (prompt and document
        characters), and ``text_records`` (ceil(chars / 1000) per submitted text,
        Azure's billing unit). A request that triggers an intervention was still
        submitted and billed, so it is counted before the block is raised; requests
        after it are never submitted and never counted.
        """
        requests: Final = _shield_requests(user_prompt, documents)
        responses: Final = tuple([await self._scan(request, usage_accumulator) for request in requests])
        return responses[-1] if responses else AzurePromptShieldGuardrailResponse()

    async def _scan(
        self,
        request: _ShieldRequest,
        usage_accumulator: MutableMapping[str, int],  # mutable-ok: callee-filled accumulator
    ) -> AzurePromptShieldGuardrailResponse:
        response_json: Final = await self._post_to_content_safety(
            "text:shieldPrompt",
            dict(request.body()),  # mutable-ok: _post_to_content_safety takes the JSON body as a dict
        )
        _add_usage(usage_accumulator, request.texts)
        response: Final = AzurePromptShieldGuardrailResponse.model_validate(response_json)
        _require_complete_analysis(response, request)
        detection: Final = _detection_message(response)
        if detection is None:
            return response
        verbose_proxy_logger.warning("Azure Prompt Shield: %s", detection)
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Violated Azure Prompt Shield guardrail policy",
                "detection_message": detection,
            },
        )

    @log_guardrail_information
    async def apply_guardrail(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict,
        input_type: Literal["request", "response"],
        logging_obj: "LiteLLMLoggingObj | None" = None,
    ) -> GenericGuardrailAPIInputs:
        _billing_usage_stash.set(None)
        texts: Final = tuple(text for text in inputs.get("texts") or () if text)
        attachments: Final = request_attachments(request_data).texts if input_type == "request" else ()
        documents: Final = tuple(attachment.text for attachment in attachments)
        scans: Final = tuple(zip_longest(texts, (documents,) if documents else (), fillvalue=None))
        usage: Final[dict[str, int]] = {}  # mutable-ok: per-invocation billing accumulator
        try:
            for text, scan_documents in scans:
                await self.async_make_request(
                    user_prompt=text or "",
                    usage_accumulator=usage,
                    documents=scan_documents or (),
                )
        finally:
            self._record_billing_usage(usage)
        return inputs

    @log_guardrail_information
    async def async_pre_call_hook(
        self,
        user_api_key_dict: "UserAPIKeyAuth",
        cache: Any,
        data: dict[str, Any],
        call_type: CallTypesLiteral,
    ) -> dict[str, Any] | None:
        """
        Pre-call hook to scan the current turn (the user's prompt, its attachments,
        and the tool outputs that follow it) before sending to the LLM.

        Raises HTTPException if content should be blocked.
        """
        _billing_usage_stash.set(None)
        verbose_proxy_logger.debug(
            "Azure Prompt Shield: Running pre-call prompt scan, on call_type: %s",
            call_type,
        )
        new_messages: Final[list[AllMessageValues] | None] = data.get("messages")
        if new_messages is None:
            verbose_proxy_logger.warning("Azure Prompt Shield: not running guardrail. No messages in data")
            return data
        user_prompt, documents = _prompt_and_documents(new_messages)
        if not user_prompt and not documents:
            verbose_proxy_logger.warning("Azure Prompt Shield: No user prompt found")
            return None
        verbose_proxy_logger.debug(
            "Azure Prompt Shield: User prompt: %s, with %d documents", user_prompt, len(documents)
        )
        usage: Final[dict[str, int]] = {}  # mutable-ok: per-invocation billing accumulator
        try:
            await self.async_make_request(
                user_prompt=user_prompt,
                usage_accumulator=usage,
                documents=documents,
            )
        finally:
            self._record_billing_usage(usage)
        return None

    def update_in_memory_litellm_params(self, litellm_params: "LitellmParams | dict") -> None:  # mutable-ok: DB dict
        """Apply updated params in place, re-resolving billing and credentials.

        Pricing is read via ``_updated_param`` (the values are pydantic extras, and
        the immediate PUT sync hands this method the raw DB dict). Pricing and any
        ``os.environ/`` credential references are validated and resolved BEFORE any
        state is mutated, so an invalid update leaves the running guardrail
        untouched and a raw reference never overwrites a resolved credential.
        """
        cost_tier: Final = _resolved_cost_tier(_updated_param(litellm_params, "cost_tier"))
        price: Final = _resolved_price(_updated_param(litellm_params, "price_per_1000_text_records"), cost_tier)
        resolved_credentials: dict[str, object] = {}  # mutable-ok: staged before mutation
        for cred_key in ("api_key", "api_base"):
            cred_value = _updated_param(litellm_params, cred_key)
            if isinstance(cred_value, str) and cred_value.startswith("os.environ/"):
                resolved_credentials[cred_key] = _resolved_secret_value(cred_value)
        if isinstance(litellm_params, Mapping):
            for key, value in litellm_params.items():
                setattr(self, key, resolved_credentials.get(key, value))
        else:
            super().update_in_memory_litellm_params(litellm_params)
            for cred_key, cred_value in resolved_credentials.items():
                setattr(self, cred_key, cred_value)
        self.cost_tier = cost_tier
        self.price_per_1000_text_records = price

    def _record_billing_usage(self, usage: Mapping[str, int]) -> None:
        """Stash this invocation's usage counters for the ``_process_*`` call the
        decorator runs next in the same asyncio task; overwrites any leftover."""
        _billing_usage_stash.set(dict(usage) if usage else None)  # mutable-ok: fresh snapshot, popped by _process_*

    def _pop_billing_tracing_detail(self) -> GuardrailTracingDetail | None:
        """Build the billing tracing detail from the stashed usage counters, priced
        with the configured tier/price. ``guardrail_cost_in_spend=False`` keeps the
        estimated cost out of ``response_cost`` and budget enforcement: Azure
        guardrail cost is reported on logs, OTEL spans, and the UI, never billed
        against team/user/key budgets (LIT-5917)."""
        usage: Final = _billing_usage_stash.get()
        _billing_usage_stash.set(None)
        if not usage:
            return None
        cost: Final = azure_prompt_shield_guardrail_cost(
            usage_units=usage,
            cost_tier=self.cost_tier,
            price_per_1000_text_records=self.price_per_1000_text_records,
        )
        if cost is None:
            return GuardrailTracingDetail(guardrail_usage=usage)
        return GuardrailTracingDetail(
            guardrail_usage=usage,
            guardrail_cost=cost,
            guardrail_cost_in_spend=False,
        )

    def _process_response(
        self,
        response: dict | None,  # mutable-ok: matches CustomGuardrail._process_response signature
        request_data: dict,  # mutable-ok: matches CustomGuardrail._process_response signature
        start_time: float | None = None,
        end_time: float | None = None,
        duration: float | None = None,
        event_type: GuardrailEventHooks | None = None,
        original_inputs: dict | None = None,  # mutable-ok: matches CustomGuardrail._process_response signature
    ) -> dict | None:  # mutable-ok: matches CustomGuardrail._process_response return
        """Override to attach the Azure billing tracing detail (usage counters and
        estimated cost) and the ``azure`` provider label to the recorded guardrail
        information. Follows the OpenAI moderation override pattern
        (openai/moderations.py)."""
        guardrail_response: Final = self._summarize_guardrail_response(
            response=response,
            original_inputs=original_inputs,
            event_type=event_type,
        )
        self.add_standard_logging_guardrail_information_to_request_data(
            guardrail_json_response=guardrail_response,
            request_data=request_data,
            guardrail_status="success",
            duration=duration,
            start_time=start_time,
            end_time=end_time,
            event_type=event_type,
            guardrail_provider="azure",
            tracing_detail=self._pop_billing_tracing_detail(),
        )
        return response

    def _process_error(
        self,
        e: Exception,
        request_data: dict,  # mutable-ok: matches CustomGuardrail._process_error signature
        start_time: float | None = None,
        end_time: float | None = None,
        duration: float | None = None,
        event_type: GuardrailEventHooks | None = None,
    ) -> NoReturn:
        """Override to attach the Azure billing tracing detail to the blocked/error
        guardrail record; a chunk that triggered an intervention was still submitted
        to (and billed by) Azure, so its usage is recorded on this path too."""
        guardrail_status: Final = (
            "guardrail_intervened" if self._is_guardrail_intervention(e) else "guardrail_failed_to_respond"
        )
        self.add_standard_logging_guardrail_information_to_request_data(
            guardrail_json_response=e,
            request_data=request_data,
            guardrail_status=guardrail_status,
            duration=duration,
            start_time=start_time,
            end_time=end_time,
            event_type=event_type,
            guardrail_provider="azure",
            tracing_detail=self._pop_billing_tracing_detail(),
        )
        raise e

    @staticmethod
    def get_config_model() -> type["GuardrailConfigModel"] | None:
        """
        Get the config model for the Azure Prompt Shield guardrail.
        """
        from litellm.types.proxy.guardrails.guardrail_hooks.azure.azure_prompt_shield import (
            AzurePromptShieldGuardrailConfigModel,
        )

        return AzurePromptShieldGuardrailConfigModel

    @classmethod
    def get_supported_event_hooks(cls) -> list[GuardrailEventHooks]:
        return [
            GuardrailEventHooks.pre_call,
            GuardrailEventHooks.during_call,
        ]
