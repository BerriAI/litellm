#!/usr/bin/env python3
"""
Azure Prompt Shield Native Guardrail Integrationfor LiteLLM
"""

import math
from collections.abc import Mapping, MutableMapping
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any, ClassVar, Final, Literal, NoReturn, cast

from fastapi import HTTPException

from litellm._logging import verbose_proxy_logger
from litellm.integrations.custom_guardrail import (
    CustomGuardrail,
    log_guardrail_information,
)
from litellm.litellm_core_utils.llm_cost_calc.guardrail_cost import (
    AZURE_PROMPT_SHIELD_TEXT_RECORD_UNIT,
    azure_prompt_shield_guardrail_cost,
)
from litellm.secret_managers.main import get_secret_str
from litellm.types.guardrails import GuardrailEventHooks
from litellm.types.utils import (
    CallTypesLiteral,
    GenericGuardrailAPIInputs,
    GuardrailTracingDetail,
)

from .base import AZURE_CONTENT_SAFETY_TEXT_RECORD_LENGTH, AzureGuardrailBase

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.types.guardrails import LitellmParams
    from litellm.types.llms.openai import AllMessageValues
    from litellm.types.proxy.guardrails.guardrail_hooks.azure.azure_prompt_shield import (
        AzurePromptShieldGuardrailResponse,
    )
    from litellm.types.proxy.guardrails.guardrail_hooks.base import GuardrailConfigModel


# Per-invocation billing counters. A ContextVar rather than request metadata: the
# decorator can swap out ``request_data``, metadata is client-forgeable, and
# concurrent guardrails run in separate tasks with their own context copy.
_billing_usage_stash: Final[ContextVar[dict[str, int] | None]] = ContextVar(  # mutable-ok: task-local stash
    "azure_prompt_shield_billing_usage", default=None
)


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
    ) -> "AzurePromptShieldGuardrailResponse":
        """
        Make a request to the Azure Prompt Shield API.

        Long prompts are automatically split at word boundaries into chunks
        that respect the Azure Content Safety 10 000-character limit.  Each
        chunk is analysed independently; an attack in *any* chunk raises
        an HTTPException immediately.

        ``usage_accumulator`` collects billable usage per SUBMITTED chunk:
        ``requests`` (Azure API calls), ``input_characters``, and
        ``text_records`` (ceil(chunk_chars / 1000), Azure's billing unit).
        A chunk that triggers an intervention was still submitted and billed,
        so it is counted before the block is raised; chunks after it are
        never submitted and never counted.
        """
        from litellm.types.proxy.guardrails.guardrail_hooks.azure.azure_prompt_shield import (
            AzurePromptShieldGuardrailRequestBody,
            AzurePromptShieldGuardrailResponse,
        )

        from .base import AZURE_CONTENT_SAFETY_MAX_TEXT_LENGTH

        chunks: Final = self.split_text_by_words(user_prompt, AZURE_CONTENT_SAFETY_MAX_TEXT_LENGTH)

        last_response: AzurePromptShieldGuardrailResponse | None = None

        for chunk in chunks:
            request_body = AzurePromptShieldGuardrailRequestBody(documents=[], userPrompt=chunk)
            response_json = await self._post_to_content_safety("text:shieldPrompt", cast(dict, request_body))

            last_response = cast(AzurePromptShieldGuardrailResponse, response_json)

            usage_accumulator["requests"] = usage_accumulator.get("requests", 0) + 1
            usage_accumulator["input_characters"] = usage_accumulator.get("input_characters", 0) + len(chunk)
            usage_accumulator[AZURE_PROMPT_SHIELD_TEXT_RECORD_UNIT] = usage_accumulator.get(
                AZURE_PROMPT_SHIELD_TEXT_RECORD_UNIT, 0
            ) + math.ceil(len(chunk) / AZURE_CONTENT_SAFETY_TEXT_RECORD_LENGTH)

            if last_response["userPromptAnalysis"].get("attackDetected"):
                verbose_proxy_logger.warning(
                    "Azure Prompt Shield: Attack detected in chunk of length %d",
                    len(chunk),
                )
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": "Violated Azure Prompt Shield guardrail policy",
                        "detection_message": f"Attack detected: {last_response['userPromptAnalysis']}",
                    },
                )

        # chunks is always non-empty (split_text_by_words guarantees ≥1 element)
        assert last_response is not None
        return last_response

    @log_guardrail_information
    async def apply_guardrail(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict,
        input_type: Literal["request", "response"],
        logging_obj: "LiteLLMLoggingObj | None" = None,
    ) -> GenericGuardrailAPIInputs:
        _billing_usage_stash.set(None)
        usage: Final[dict[str, int]] = {}  # mutable-ok: per-invocation billing accumulator
        try:
            for text in inputs.get("texts") or ():
                if text:
                    await self.async_make_request(user_prompt=text, usage_accumulator=usage)
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
        Pre-call hook to scan user prompts before sending to LLM.

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
        user_prompt: Final = self.get_user_prompt(new_messages)

        if user_prompt:
            verbose_proxy_logger.debug("Azure Prompt Shield: User prompt: %s", user_prompt)
            usage: Final[dict[str, int]] = {}  # mutable-ok: per-invocation billing accumulator
            try:
                await self.async_make_request(
                    user_prompt=user_prompt,
                    usage_accumulator=usage,
                )
            finally:
                self._record_billing_usage(usage)
        else:
            verbose_proxy_logger.warning("Azure Prompt Shield: No user prompt found")
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
        guardrail_response: Final[dict | str] = (  # mutable-ok: mirrors CustomGuardrail._process_response
            ("mask" if self._inputs_were_modified(original_inputs, response) else "allow")
            if original_inputs is not None and isinstance(response, dict)
            else ({} if response is None else response)  # mutable-ok: empty placeholder, never mutated
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
