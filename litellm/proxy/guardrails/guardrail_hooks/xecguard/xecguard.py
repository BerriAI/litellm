"""
XecGuard guardrail integration for LiteLLM.

Calls the CyCraft XecGuard API (https://api-xecguard.cycraft.ai)
to scan the full conversation history against configured policies
(prompt-injection, PII, harmful-content, custom rules) and, when
grounding documents are supplied via request metadata, also validates
the assistant response against those reference documents via the
/grounding endpoint.

Design notes (intentional divergences from the framework defaults):
  * The full conversation history (system + user + assistant) is always
    forwarded to XecGuard regardless of ``scan_type``. This bypasses the
    framework's optional ``skip_system_message_in_guardrail`` behaviour
    on purpose - policy enforcement depends on system-prompt visibility.
  * ``apply_guardrail`` is defined directly on this class so the
    ``during_call`` dispatch (proxy/utils.py checks for the method on
    ``type(callback).__dict__``) reaches our implementation.
  * ``async_logging_hook`` is overridden because the framework calls it
    directly for ``logging_only`` mode - it does NOT bridge to
    ``apply_guardrail``. Our override runs the scan non-blockingly and
    swallows every exception.
  * When ``send_meta`` is enabled the scan payload carries a ``meta``
    object identifying the calling virtual key. It is correlation data
    for XecGuard's SIEM export only and never affects the verdict; the
    backend's flat-scalar contract for it is enforced client-side so a
    malformed key metadata entry cannot fail an otherwise valid scan.
"""

import asyncio
import json
import os
import re
from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import TYPE_CHECKING, Any, Final, Literal, Optional

from fastapi.exceptions import HTTPException

from litellm._logging import verbose_proxy_logger
from litellm.integrations.custom_guardrail import (
    CustomGuardrail,
    log_guardrail_information,
)
from litellm.litellm_core_utils.core_helpers import redact_nested_match_and_regex_keys
from litellm.litellm_core_utils.sensitive_data_masker import mask_credentials_in_payload
from litellm.llms.custom_httpx.http_handler import (
    get_async_httpx_client,
    httpxSpecialProvider,
)
from litellm.proxy._types import (
    LiteLLM_ManagementEndpoint_MetadataFields,
    LiteLLM_ManagementEndpoint_MetadataFields_Premium,
)
from litellm.types.guardrails import GuardrailEventHooks
from litellm.types.utils import (
    GenericGuardrailAPIInputs,
    GuardrailStatus,
    StandardLoggingGuardrailInformation,
)

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import (
        Logging as LiteLLMLoggingObj,
    )
    from litellm.types.proxy.guardrails.guardrail_hooks.base import (
        GuardrailConfigModel,
    )


def _sanitize_scan_result_for_logging(scan_result: dict) -> dict:
    without_secrets: Final = {key: value for key, value in scan_result.items() if key != "secret_fields"}
    redacted: Final = redact_nested_match_and_regex_keys(without_secrets)
    masked: Final = mask_credentials_in_payload(redacted if isinstance(redacted, dict) else without_secrets)
    return masked if isinstance(masked, dict) else without_secrets


_DEFAULT_API_BASE: Final = "https://api-xecguard.cycraft.ai"
_SCAN_ENDPOINT: Final = "/xecguard/v1/scan"
_GROUNDING_ENDPOINT: Final = "/xecguard/v1/grounding"
_DEFAULT_MODEL: Final = "xecguard_v2"
_DEFAULT_GROUNDING_STRICTNESS: Final = "BALANCED"
_METADATA_GROUNDING_KEY: Final = "xecguard_grounding_documents"
_RATIONALE_TRUNCATE_CHARS: Final = 200
_DEFAULT_POLICIES: Final = [
    "Default_Policy_SystemPromptEnforcement",
    "Default_Policy_HarmfulContentProtection",
    "Default_Policy_GeneralPromptAttackProtection",
]

# ``meta`` contract of POST /xecguard/v1/scan: an optional object carrying caller
# context that takes no part in detection. XecGuard flattens it into the SIEM
# event (``virtualkey`` -> ``ctx_virtualkey``, ``data.X`` -> ``ctx_X``), and SIEM
# index fields only accept flat scalars - anything else is rejected with 400. So
# every value is coerced or dropped here rather than risking a scan failure.
_METADATA_KEY_METADATA_FIELD: Final = "user_api_key_metadata"
_META_NAME_PATTERN: Final = re.compile(r"^[A-Za-z_][A-Za-z0-9_.\-]{0,63}$")
_META_CONTROL_CHARS: Final = re.compile(r"[\x00-\x1f\x7f]")
_META_MAX_DATA_FIELDS: Final = 32
_META_MAX_VALUE_CHARS: Final = 512
_META_MAX_SERIALIZED_BYTES: Final = 4096
# Never forwarded, and ``meta_data_fields`` cannot opt them back in: these slots
# hold credentials, so there is no configuration under which shipping them to an
# external SIEM is right.
_META_EXCLUDED_DATA_FIELDS: Final = frozenset({"logging", "callback_settings", "secret_manager_settings"})

# The proxy stores its own per-key control settings inside key metadata - rate
# limits, budget knobs, enforced params, ``disable_global_guardrails``. They sit
# in the same dict as the admin's own fields but they are proxy configuration,
# not caller identity: noise in a SIEM, they eat the 32-field / 4096-byte budget,
# and a couple of them describe the key's security posture. Skipped by default,
# but an admin who explicitly names one in ``meta_data_fields`` gets it - unlike
# the credential slots above, forwarding these is a judgement call, not a bug.
#
# Taken from the proxy's own lists rather than copied, so a field litellm adds
# later is covered without an edit here.
_META_CONTROL_DATA_FIELDS: Final = (
    frozenset(LiteLLM_ManagementEndpoint_MetadataFields) | frozenset(LiteLLM_ManagementEndpoint_MetadataFields_Premium)
) - _META_EXCLUDED_DATA_FIELDS

# Two shapes for ``meta.virtualkey``. "string" is the identity as a bare string,
# which is all the currently deployed backend accepts. "object" carries the alias
# and the key id side by side, so a SIEM event is attributable even when the alias
# is absent, renamed, or reused - it needs a backend that validates the object
# form, hence the switch rather than a straight cutover.
_META_IDENTITY_FORMATS: Final = ("string", "object")
_DEFAULT_META_IDENTITY_FORMAT: Final = "string"

# Virtual-key attributes the proxy injects alongside every request, forwarded as
# ``meta.data`` so a SIEM event can be attributed without a lookup back into the
# proxy database. Ordered: identity first, then tenancy, then commercials, so the
# fields that survive the 32-field / 4096-byte caps are the ones worth keeping.
#
# This set deliberately includes PII (``user_email``) and commercial figures
# (``spend``, ``max_budget``). Both leave the proxy only when ``send_meta`` is
# explicitly enabled, and ``meta_data_fields`` narrows the set for deployments
# that must not egress them.
_META_AUTO_DATA_FIELDS: Final[tuple[tuple[str, str], ...]] = (
    ("key_id", "user_api_key_hash"),
    ("key_alias", "user_api_key_alias"),
    ("team_id", "user_api_key_team_id"),
    ("team_alias", "user_api_key_team_alias"),
    ("user_id", "user_api_key_user_id"),
    ("user_email", "user_api_key_user_email"),
    ("org_id", "user_api_key_org_id"),
    ("org_alias", "user_api_key_org_alias"),
    ("project_id", "user_api_key_project_id"),
    ("project_alias", "user_api_key_project_alias"),
    ("end_user_id", "user_api_key_end_user_id"),
    ("spend", "user_api_key_spend"),
    ("max_budget", "user_api_key_max_budget"),
    ("request_route", "user_api_key_request_route"),
)


def _sanitized_meta_text(text: str) -> str | None:
    """Strip control characters and cap the length, or None when nothing is left."""
    cleaned: Final = _META_CONTROL_CHARS.sub("", text)[:_META_MAX_VALUE_CHARS]
    return cleaned or None


class XecGuardMissingCredentials(Exception):
    pass


class XecGuardGuardrail(CustomGuardrail):
    def __init__(
        self,
        api_key: str | None = None,
        api_base: str | None = None,
        xecguard_model: str | None = None,
        policy_names: list[str] | None = None,
        apply_to_aliases: Sequence[str] | None = None,
        except_aliases: Sequence[str] | None = None,
        send_meta: bool | None = None,
        meta_data_fields: Sequence[str] | None = None,
        meta_identity_format: str | None = None,
        block_on_error: bool | None = None,
        grounding_strictness: str | None = None,
        **kwargs: Any,
    ) -> None:
        self.api_key = api_key or os.environ.get("XECGUARD_API_KEY")
        if not self.api_key:
            raise XecGuardMissingCredentials(
                "XecGuard API key is required. "
                "Set XECGUARD_API_KEY in the "
                "environment or pass api_key in "
                "the guardrail config."
            )

        self.api_base = (api_base or os.environ.get("XECGUARD_API_BASE") or _DEFAULT_API_BASE).rstrip("/")

        self.xecguard_model = xecguard_model or _DEFAULT_MODEL
        self.policy_names = policy_names
        # Guardrail-side key targeting (free, OSS). Normalized to lists.
        self.apply_to_aliases = apply_to_aliases or ()
        self.except_aliases = except_aliases or ()

        # Caller context forwarded as the scan payload's ``meta``. Opt-in: turning
        # it on sends the calling key's alias and its admin-set metadata to
        # XecGuard, which is a data-egress change no upgrade should make silently.
        if send_meta is None:
            self.send_meta = os.environ.get("XECGUARD_SEND_META", "false").lower() in (
                "true",
                "1",
                "yes",
            )
        else:
            self.send_meta = send_meta
        self.meta_data_fields = tuple(meta_data_fields) if meta_data_fields else ()

        # Wire shape of ``meta.virtualkey``. Defaults to the string form: a backend
        # that only accepts strings answers the object form with 400, and with
        # ``block_on_error`` on (the default) that turns every request into a block.
        # An unknown value falls back rather than raising - a typo in the UI should
        # not take the gateway down.
        requested_format: Final = (
            (meta_identity_format or os.environ.get("XECGUARD_META_IDENTITY_FORMAT") or "").strip().lower()
        )
        if requested_format and requested_format not in _META_IDENTITY_FORMATS:
            verbose_proxy_logger.warning(
                "XecGuard: unknown meta_identity_format %r - falling back to %r (valid: %s)",
                requested_format,
                _DEFAULT_META_IDENTITY_FORMAT,
                ", ".join(_META_IDENTITY_FORMATS),
            )
        self.meta_identity_format = (
            requested_format if requested_format in _META_IDENTITY_FORMATS else _DEFAULT_META_IDENTITY_FORMAT
        )

        if block_on_error is None:
            env: Final = os.environ.get("XECGUARD_BLOCK_ON_ERROR", "true")
            self.block_on_error = env.lower() in (
                "true",
                "1",
                "yes",
            )
        else:
            self.block_on_error = block_on_error

        self.grounding_strictness = grounding_strictness or _DEFAULT_GROUNDING_STRICTNESS

        self.async_handler = get_async_httpx_client(
            llm_provider=httpxSpecialProvider.GuardrailCallback,
        )

        kwargs.setdefault("supported_event_hooks", list(self.get_supported_event_hooks()))

        super().__init__(**kwargs)

    @staticmethod
    def get_config_model() -> type["GuardrailConfigModel"] | None:
        from litellm.types.proxy.guardrails.guardrail_hooks.xecguard import (
            XecGuardConfigModel,
        )

        return XecGuardConfigModel

    @classmethod
    def get_supported_event_hooks(cls) -> list[GuardrailEventHooks]:
        return [
            GuardrailEventHooks.pre_call,
            GuardrailEventHooks.during_call,
            GuardrailEventHooks.post_call,
            GuardrailEventHooks.logging_only,
        ]

    @staticmethod
    def _calling_key_identity(
        request_data: Mapping[str, Any] | None,
    ) -> tuple[str | None, str | None]:
        """Return (key_alias, key_hash) of the calling virtual key from the
        proxy-injected request metadata. Both may be None (e.g. master key)."""
        if not isinstance(request_data, dict):
            return None, None
        sources: Final = tuple(
            md
            for md in (request_data.get(meta_key) for meta_key in ("metadata", "litellm_metadata"))
            if isinstance(md, dict)
        )
        alias: Final = next((md["user_api_key_alias"] for md in sources if md.get("user_api_key_alias")), None)
        key_hash: Final = next((md["user_api_key_hash"] for md in sources if md.get("user_api_key_hash")), None)
        return alias, key_hash

    def _key_is_targeted(self, request_data: Mapping[str, Any] | None) -> bool:
        """Guardrail-side key targeting. With no allow/block list configured,
        every key is scanned. Otherwise the calling key is matched by alias
        (preferred) or hashed token:
          * blocklist (except_aliases): listed keys are NOT scanned;
          * allowlist (apply_to_aliases): only listed keys are scanned.
        When both are set, a key is scanned iff it is in the allowlist AND not
        in the blocklist.
        """
        allowlist: Final = self.apply_to_aliases or ()
        blocklist: Final = self.except_aliases or ()
        if not allowlist and not blocklist:
            return True

        alias, key_hash = self._calling_key_identity(request_data)
        identifiers: Final = tuple(ident for ident in (alias, key_hash) if ident)

        # Deny wins, and it is checked first so that precedence stays visible
        # rather than folded into the allowlist expression below.
        if blocklist and any(ident in blocklist for ident in identifiers):
            return False
        if not allowlist:
            return True
        return any(ident in allowlist for ident in identifiers)

    # Metadata fields the proxy injects to identify the calling virtual key.
    _KEY_IDENTITY_FIELDS = ("user_api_key_alias", "user_api_key_hash")

    @classmethod
    def _key_context(cls, data: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
        """Return a mapping ``_calling_key_identity`` can read the key fields from.

        That reader looks for top-level ``metadata`` / ``litellm_metadata``. On the
        pre/during/post_call paths the proxy already puts the injected key fields
        there, so ``data`` is handed back untouched -- reshaping to a single key would
        drop the other location it also reads. Only the logging path needs help: there
        ``data`` is ``model_call_details``, which carries the same fields one level
        down under ``litellm_params``.
        """
        if not isinstance(data, dict):
            return data
        for meta_key in ("metadata", "litellm_metadata"):
            md = data.get(meta_key)
            if isinstance(md, dict) and any(field in md for field in cls._KEY_IDENTITY_FIELDS):
                return data
        nested: Final = data.get("litellm_params")
        if isinstance(nested, dict):
            for meta_key in ("metadata", "litellm_metadata"):
                md = nested.get(meta_key)
                if isinstance(md, dict) and any(field in md for field in cls._KEY_IDENTITY_FIELDS):
                    return {meta_key: md}  # mutable-ok: lifts nested metadata to the readers' shape
        return data

    def should_run_guardrail(self, data: Mapping[str, object], event_type: GuardrailEventHooks) -> bool:
        """Gate on the calling virtual key in addition to the native checks.

        Deciding here rather than inside ``apply_guardrail`` is what makes LiteLLM
        record the guardrail as not having run for a key this guardrail does not
        cover, instead of logging a "success"/"allow" entry for a request it never
        evaluated. ``super()`` is consulted first so the native decisions -- global
        opt-outs, event-hook matching, tag-based modes -- keep precedence.

        The gates are still enforced inside ``apply_guardrail`` and
        ``async_logging_hook`` as well: ``POST /guardrails/apply_guardrail`` invokes
        ``apply_guardrail`` directly and never reaches this method.
        """
        if not super().should_run_guardrail(data, event_type):
            return False
        return self._key_is_targeted(self._key_context(data))

    @log_guardrail_information
    async def apply_guardrail(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict,
        input_type: Literal["request", "response"],
        logging_obj: Optional["LiteLLMLoggingObj"] = None,
    ) -> GenericGuardrailAPIInputs:
        # Guardrail-side key targeting (allowlist / blocklist by key alias):
        # skip scanning entirely for keys this guardrail does not cover.
        # should_run_guardrail already gates the proxy's own dispatch paths; this
        # also covers POST /guardrails/apply_guardrail, which calls straight in.
        if not self._key_is_targeted(self._key_context(request_data)):
            return inputs

        messages: Final = self._build_full_history(
            request_data=request_data,
            inputs=inputs,
            input_type=input_type,
        )
        if not messages:
            return inputs

        scan_type: Final = "input" if input_type == "request" else "response"
        scan_result: Final = await self._call_scan(
            messages=messages,
            scan_type=scan_type,
            request_data=request_data,
        )
        if scan_result is None:
            return inputs

        if scan_result.get("decision") == "UNSAFE":
            raise HTTPException(
                status_code=400,
                detail={
                    "error": self._format_scan_block_message(scan_result),
                    "guardrail_name": self.guardrail_name or "xecguard",
                    "xecguard_response": scan_result,
                },
            )

        if input_type == "response":
            documents: Final = self._extract_grounding_documents(request_data)
            if documents:
                grounding_result: Final = await self._call_grounding(
                    messages=messages,
                    documents=documents,
                )
                if grounding_result is not None and grounding_result.get("decision") == "UNSAFE":
                    raise HTTPException(
                        status_code=400,
                        detail={
                            "error": self._format_grounding_block_message(grounding_result),
                            "guardrail_name": self.guardrail_name or "xecguard",
                            "xecguard_response": grounding_result,
                        },
                    )

        return inputs

    async def async_logging_hook(
        self,
        kwargs: dict,
        result: Any,
        call_type: str,
    ) -> tuple[dict, Any]:
        """Observe-only scan for logging_only mode.

        Never blocks, never raises - all errors are swallowed. Records a
        StandardLoggingGuardrailInformation entry so the scan decision
        reaches downstream loggers (Langfuse, DataDog, etc.).
        """
        if (
            isinstance(kwargs, dict)
            and "litellm_params" in kwargs
            and "metadata" in kwargs["litellm_params"]
            and "standard_logging_guardrail_information" in kwargs["litellm_params"]["metadata"]
            and kwargs["litellm_params"]["metadata"]["standard_logging_guardrail_information"]
        ):
            return kwargs, result

        # Same key targeting as apply_guardrail. logging_only reaches the guardrail
        # through this hook rather than apply_guardrail, so the gate is repeated here;
        # without it an excluded key's content would still be sent to XecGuard.
        if not self._key_is_targeted(self._key_context(kwargs)):
            return kwargs, result

        start_time: Final = datetime.now()
        try:
            assistant_text: Final = self._extract_assistant_text_from_response(result)
            request_data: Final = {**kwargs}
            if assistant_text is not None:
                request_data["response"] = result
                messages = self._build_full_history(
                    request_data=request_data,
                    inputs={},
                    input_type="response",
                )
                scan_type = "response"
            else:
                messages = self._build_full_history(
                    request_data=request_data,
                    inputs={},
                    input_type="request",
                )
                scan_type = "input"

            if not messages:
                return kwargs, result

            scan_result: Final = await self._call_scan(
                messages=messages,
                scan_type=scan_type,
                request_data=request_data,
                suppress_errors=True,
            )
            if scan_result is None:
                return kwargs, result

            guardrail_status: Final[GuardrailStatus] = (
                "guardrail_intervened" if scan_result.get("decision") == "UNSAFE" else "success"
            )
            end_time: Final = datetime.now()
            slg: Final = StandardLoggingGuardrailInformation(
                guardrail_name=self.guardrail_name or "xecguard",
                guardrail_mode=GuardrailEventHooks.logging_only,
                guardrail_response=_sanitize_scan_result_for_logging(scan_result),
                guardrail_status=guardrail_status,
                start_time=start_time.timestamp(),
                end_time=end_time.timestamp(),
                duration=(end_time - start_time).total_seconds(),
                masked_entity_count=None,
            )
            existing: Final = kwargs["standard_logging_object"].get("guardrail_information")
            if isinstance(existing, list):
                existing.append(slg)
            else:
                kwargs["standard_logging_object"]["guardrail_information"] = [slg]

        except Exception as exc:
            verbose_proxy_logger.debug(
                "XecGuard logging_only swallowed exception: %s",
                str(exc),
            )
        return kwargs, result

    def logging_hook(
        self,
        kwargs: dict,
        result: Any,
        call_type: str,
    ) -> tuple[dict, Any]:
        """Sync counterpart to ``async_logging_hook``.

        Runs the async version on an available loop, swallowing every
        exception. Mirrors the pattern used by the Presidio guardrail
        for sync logging callbacks.
        """
        try:
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            if loop.is_running():
                return kwargs, result
            loop.run_until_complete(self.async_logging_hook(kwargs=kwargs, result=result, call_type=call_type))
        except Exception as exc:
            verbose_proxy_logger.debug(
                "XecGuard sync logging_hook swallowed exception: %s",
                str(exc),
            )
        return kwargs, result

    # ------------------------------------------------------------------
    # Caller context (scan payload ``meta``) - SIEM correlation only
    # ------------------------------------------------------------------

    def _build_scan_meta(self, context: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
        """Assemble the scan payload's ``meta`` object, or None to omit it.

        ``virtualkey`` is the identity this guardrail filtered on and ``data`` is
        the calling key's proxy-injected attributes plus its own metadata as set on
        the Virtual Keys page. Neither participates in detection - XecGuard forwards
        them to the SIEM so a scan can be traced back to the virtual key that caused
        it.

        ``meta`` is optional in the contract, so anything that cannot be made to
        satisfy it is left out instead of turning a scan into a 400.
        """
        if not self.send_meta:
            return None

        virtualkey: Final[str | Mapping[str, str] | None] = (
            self._scan_meta_virtualkey_object(context)
            if self.meta_identity_format == "object"
            else self._scan_meta_virtualkey(context)
        )
        if not virtualkey:
            verbose_proxy_logger.debug(
                "XecGuard: omitting scan meta - the calling key has no alias or hash matching the "
                "backend's virtualkey pattern (give the key a key_alias to enable SIEM correlation)"
            )
            return None

        meta: Final[dict[str, Any]] = {"virtualkey": virtualkey}  # mutable-ok: the JSON object being assembled
        data: Final = self._build_scan_meta_data(context, virtualkey=virtualkey)
        if data:
            meta["data"] = data
        return meta

    def _scan_meta_virtualkey(self, context: Mapping[str, Any] | None) -> str | None:
        """The key identity to report, or None when there is no usable one.

        Alias first: that is what an operator types into ``apply_to_aliases`` /
        ``except_aliases``, so the value in the SIEM matches the value in the
        guardrail config. The hashed token is the fallback for keys created
        without an alias; a master-key call has neither. Note that the token
        hash only satisfies the backend's pattern when it happens to start with
        a hex letter - aliasless keys are not reliably correlatable.
        """
        for candidate in self._calling_key_identity(context):
            if isinstance(candidate, str) and _META_NAME_PATTERN.match(candidate):
                return candidate
        return None

    def _scan_meta_virtualkey_object(self, context: Mapping[str, Any] | None) -> Mapping[str, str] | None:
        """The object form of ``virtualkey``: ``{"alias": ..., "key_id": ...}``.

        Either member may be absent - a key created without an alias has only an
        id, and a master-key call has neither (in which case ``meta`` is omitted).
        Unlike the string form this does not require the alias to satisfy the
        backend's identifier pattern: the pattern exists because a bare string
        becomes a SIEM field *value* directly, whereas here each member is
        sanitized the same way ``meta.data`` values are. That makes keys whose
        alias contains spaces or CJK correlatable, which the string form cannot do.
        """
        alias, key_hash = self._calling_key_identity(context)
        obj: Final[dict[str, str]] = {}  # mutable-ok: the JSON object being assembled
        for name, raw in (("alias", alias), ("key_id", key_hash)):
            value = self._coerce_meta_value(raw)
            if value is not None:
                obj[name] = value
        return obj or None

    @staticmethod
    def _calling_key_metadata(context: Mapping[str, Any] | None) -> Mapping[object, Any]:
        """The calling virtual key's own metadata, as injected by the proxy.

        This is the JSON an admin typed into the key's Metadata box on the
        Virtual Keys page (minus the callback-credential slots, which the proxy
        strips before injecting). Team metadata is deliberately not merged in:
        ``meta.data`` is meant to describe the key that made the call.

        The key type is ``object``, not ``str``: nothing between the database and
        here validates it, and the caller drops a non-str key rather than letting
        it reach ``re.match`` and raise. Narrowing this to ``str`` would make that
        guard look redundant to a type checker and invite its removal.
        """
        if not isinstance(context, dict):
            return {}  # mutable-ok: "this key has no metadata"; the caller only reads it
        for meta_key in ("metadata", "litellm_metadata"):
            md = context.get(meta_key)
            if isinstance(md, dict):
                key_metadata = md.get(_METADATA_KEY_METADATA_FIELD)
                if isinstance(key_metadata, dict):
                    return key_metadata
        return {}  # mutable-ok: same empty result, no metadata field was injected

    @classmethod
    def _auto_meta_data_items(cls, context: Mapping[str, Any] | None) -> tuple[tuple[str, Any], ...]:
        """The proxy-injected virtual-key attributes, in ``_META_AUTO_DATA_FIELDS``
        order regardless of how the proxy ordered its metadata dict.

        Absent and null fields are skipped, so a key with no team contributes no
        ``team_id`` rather than an empty one - a SIEM query for "scans with no
        team" then means it, instead of matching every key.
        """
        injected: Final[dict[str, Any]] = {}  # mutable-ok: accumulator keyed by meta.data name
        if isinstance(context, dict):
            for meta_key in ("metadata", "litellm_metadata"):
                md = context.get(meta_key)
                if not isinstance(md, dict):
                    continue
                for name, source_field in _META_AUTO_DATA_FIELDS:
                    if name not in injected and md.get(source_field) is not None:
                        injected[name] = md[source_field]
        return tuple((name, injected[name]) for name, _ in _META_AUTO_DATA_FIELDS if name in injected)

    def _build_scan_meta_data(
        self, context: Mapping[str, Any] | None, virtualkey: str | Mapping[str, str]
    ) -> Mapping[str, str]:
        """Coerce the calling key's attributes and metadata into ``meta.data``.

        Two sources, in this order: the attributes the proxy injects about the
        calling key (identity, tenancy, budget), then the free-form metadata an
        admin typed into the key's Metadata box. Proxy-injected attributes go
        first and win a name collision, so an admin cannot shadow ``key_id`` with
        a field of their own and mislead an investigation.

        Fields are kept while they satisfy the contract: a name matching the
        backend's pattern, a flat scalar value, at most 32 fields, and a
        serialized ``meta`` within the 4096-byte cap. Oversize fields are skipped
        rather than ending the scan, so a later small field still gets through.
        Dropped names are logged without their values - both sources can hold
        sensitive strings.
        """
        source: Final = self._calling_key_metadata(context)
        data: Final[dict[str, str]] = {}  # mutable-ok: accumulator, re-measured as it grows
        # Re-measured against the real payload shape each time, so the cap holds
        # regardless of how long the virtualkey and the field names are.
        probe: Final[dict[str, Any]] = {"virtualkey": virtualkey, "data": data}  # mutable-ok: views `data`
        dropped: Final[list[str]] = []  # mutable-ok: skipped field names, for one debug line

        for name, raw_value in (*self._auto_meta_data_items(context), *source.items()):
            if self.meta_data_fields:
                if name not in self.meta_data_fields:
                    continue
            elif name in _META_CONTROL_DATA_FIELDS:
                # proxy config rather than caller identity - opt in by name
                continue
            if name in _META_EXCLUDED_DATA_FIELDS:
                continue
            if name in data:  # a proxy-injected attribute already claimed this name
                dropped.append(str(name))
                continue
            if not isinstance(name, str) or not _META_NAME_PATTERN.match(name):
                dropped.append(str(name))
                continue
            if len(data) >= _META_MAX_DATA_FIELDS:
                dropped.append(name)
                continue
            value = self._coerce_meta_value(raw_value)
            if value is None:
                dropped.append(name)
                continue
            data[name] = value
            if len(json.dumps(probe, ensure_ascii=False).encode("utf-8")) > _META_MAX_SERIALIZED_BYTES:
                del data[name]
                dropped.append(name)

        if dropped:
            verbose_proxy_logger.debug(
                "XecGuard: scan meta.data dropped %d field(s) (names only): %s",
                len(dropped),
                dropped,
            )
        return data

    @staticmethod
    def _coerce_meta_value(value: object) -> str | None:
        """Coerce one key-metadata value to the contract, or None to drop it.

        Scalars are stringified so an admin writing ``{"tier": 3}`` still gets a
        usable ``ctx_tier``. Nested objects and lists have no flat representation
        a SIEM index field can hold, so they are dropped.
        """
        if isinstance(value, bool):
            return _sanitized_meta_text("true" if value else "false")
        if isinstance(value, str):
            return _sanitized_meta_text(value)
        if isinstance(value, (int, float)):
            return _sanitized_meta_text(str(value))
        return None

    # ------------------------------------------------------------------
    # HTTP helpers
    # ------------------------------------------------------------------

    async def _call_scan(
        self,
        messages: list[dict],
        scan_type: str,
        request_data: Mapping[str, Any] | None = None,
        suppress_errors: bool = False,
    ) -> dict | None:
        payload: Final[dict[str, Any]] = {
            "model": self.xecguard_model,
            "scan_type": scan_type,
            "messages": messages,
            "policy_names": (self.policy_names if self.policy_names else _DEFAULT_POLICIES),
        }
        meta: Final = self._build_scan_meta(self._key_context(request_data))
        if meta is not None:
            payload["meta"] = meta
        return await self._post(
            path=_SCAN_ENDPOINT,
            payload=payload,
            suppress_errors=suppress_errors,
        )

    async def _call_grounding(
        self,
        messages: list[dict],
        documents: list[dict],
    ) -> dict | None:
        prompt: Final = self._extract_last_text_by_role(messages, "user")
        response_text: Final = self._extract_last_text_by_role(messages, "assistant")
        if prompt is None or response_text is None:
            return None
        payload: Final = {
            "model": self.xecguard_model,
            "prompt": prompt,
            "response": response_text,
            "documents": documents,
            "strictness": self.grounding_strictness,
        }
        return await self._post(path=_GROUNDING_ENDPOINT, payload=payload)

    async def _post(
        self,
        path: str,
        payload: dict,
        suppress_errors: bool = False,
    ) -> dict | None:
        endpoint: Final = f"{self.api_base}{path}"
        verbose_proxy_logger.debug(
            "XecGuard: POST %s payload_keys=%s",
            endpoint,
            list(payload.keys()),
        )
        try:
            response: Final = await self.async_handler.post(
                url=endpoint,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=10.0,
            )
            response.raise_for_status()
            return response.json()
        except Exception as exc:
            verbose_proxy_logger.error("XecGuard API error: %s", str(exc))
            if suppress_errors:
                return None
            if self.block_on_error:
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": (f"XecGuard API unreachable (block_on_error=True): {exc}"),
                        "guardrail_name": self.guardrail_name or "xecguard",
                    },
                ) from exc
            return None

    # ------------------------------------------------------------------
    # Message-assembly helpers (respect the full-history requirement)
    # ------------------------------------------------------------------

    def _build_full_history(
        self,
        request_data: dict,
        inputs: Any,
        input_type: str,
    ) -> list[dict]:
        """Assemble the full message list that will be sent to XecGuard.

        Always reads from ``request_data['messages']`` so the framework's
        optional ``skip_system_message_in_guardrail`` filter cannot strip
        system prompts. Synthesises a trailing user/assistant message when
        the request data is incomplete.
        """
        raw_messages: Final = request_data.get("messages") or []
        messages: Final[list[dict]] = [self._normalize_message(m) for m in raw_messages if isinstance(m, dict)]

        if input_type == "request":
            if not messages:
                return []
            if messages[-1].get("role") != "user":
                synthesized: Final = self._synthesize_user_from_inputs(inputs)
                if synthesized is None:
                    return []
                messages.append(synthesized)
            return messages

        # input_type == "response"
        assistant_text: Final = self._extract_assistant_text_from_response(request_data.get("response"))
        if assistant_text is None:
            return []
        messages.append({"role": "assistant", "content": assistant_text})
        return messages

    @staticmethod
    def _normalize_message(message: dict) -> dict:
        """Flatten multimodal content to a plain string for XecGuard."""
        role: Final = message.get("role") or "user"
        content: Final = message.get("content")
        if isinstance(content, str):
            return {"role": role, "content": content}
        if isinstance(content, list):
            parts: Final[list[str]] = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text = item.get("text")
                    if isinstance(text, str):
                        parts.append(text)
            return {"role": role, "content": "\n".join(parts)}
        return {"role": role, "content": ""}

    @staticmethod
    def _synthesize_user_from_inputs(inputs: Any) -> dict | None:
        if not isinstance(inputs, dict):
            return None
        texts: Final = inputs.get("texts")
        if not texts:
            return None
        joined: Final = "\n".join(t for t in texts if isinstance(t, str) and t)
        if not joined:
            return None
        return {"role": "user", "content": joined}

    @staticmethod
    def _extract_last_text_by_role(messages: list[dict], role: str) -> str | None:
        for message in reversed(messages):
            if message.get("role") == role:
                content = message.get("content")
                if isinstance(content, str) and content:
                    return content
                return None
        return None

    @staticmethod
    def _extract_assistant_text_from_response(response: Any) -> str | None:
        if response is None:
            return None
        choices = None
        if hasattr(response, "choices"):
            choices = response.choices
        elif isinstance(response, dict):
            choices = response.get("choices")
        if not choices:
            return None
        text_parts: Final[list[str]] = []
        for choice in choices:
            content = XecGuardGuardrail._extract_choice_content(choice)
            text = XecGuardGuardrail._content_to_text(content)
            if text:
                text_parts.append(text)
        return "\n".join(text_parts) or None

    @staticmethod
    def _extract_choice_content(choice: Any) -> Any:
        if hasattr(choice, "message"):
            message = choice.message
        elif isinstance(choice, dict):
            message = choice.get("message")
        else:
            return None
        if message is None:
            return None
        if hasattr(message, "content"):
            return message.content
        if isinstance(message, dict):
            return message.get("content")
        return None

    @staticmethod
    def _content_to_text(content: Any) -> str | None:
        if isinstance(content, str) and content:
            return content
        if isinstance(content, list):
            parts: Final = [
                item.get("text")
                for item in content
                if isinstance(item, dict) and item.get("type") == "text" and isinstance(item.get("text"), str)
            ]
            joined: Final = "\n".join(p for p in parts if p)
            return joined or None
        return None

    # ------------------------------------------------------------------
    # Grounding document extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_grounding_documents(request_data: dict) -> list[dict]:
        metadata: Final = request_data.get("metadata") or request_data.get("litellm_metadata")
        if not isinstance(metadata, dict):
            return []
        raw_docs: Final = metadata.get(_METADATA_GROUNDING_KEY)
        if not isinstance(raw_docs, list) or not raw_docs:
            return []
        valid_docs: Final[list[dict]] = []
        for doc in raw_docs:
            if (
                isinstance(doc, dict)
                and isinstance(doc.get("document_id"), str)
                and isinstance(doc.get("context"), str)
            ):
                valid_docs.append(
                    {
                        "document_id": doc["document_id"],
                        "context": doc["context"],
                    }
                )
            else:
                verbose_proxy_logger.debug(
                    "XecGuard: dropping malformed grounding document: %r",
                    doc,
                )
        return valid_docs

    # ------------------------------------------------------------------
    # Error-message formatting
    # ------------------------------------------------------------------

    @staticmethod
    def _format_scan_block_message(result: dict) -> str:
        trace_id: Final = result.get("trace_id", "")
        violations = result.get("xecguard_result")
        if not isinstance(violations, list):
            violations = []
        seen: Final[list[str]] = []
        for v in violations:
            if not isinstance(v, dict):
                continue
            name = v.get("violated_policy_name")
            if isinstance(name, str) and name and name not in seen:
                seen.append(name)
        policies: Final = ",".join(seen) if seen else "unknown"
        rationale = ""
        for v in violations:
            if isinstance(v, dict):
                candidate = v.get("rationale")
                if isinstance(candidate, str) and candidate:
                    rationale = candidate[:_RATIONALE_TRUNCATE_CHARS]
                    break
        return f"Blocked by XecGuard: policies=[{policies}] trace_id={trace_id} rationale={rationale}"

    @staticmethod
    def _format_grounding_block_message(result: dict) -> str:
        trace_id: Final = result.get("trace_id", "")
        detail: Final = result.get("xecguard_result")
        rules: list[str] = []
        rationale = ""
        if isinstance(detail, dict):
            raw_rules: Final = detail.get("violated_rules_list")
            if isinstance(raw_rules, list):
                rules = [r for r in raw_rules if isinstance(r, str)]
            candidate: Final = detail.get("rationale")
            if isinstance(candidate, str):
                rationale = candidate[:_RATIONALE_TRUNCATE_CHARS]
        rules_str: Final = ",".join(rules) if rules else "unknown"
        return f"Blocked by XecGuard grounding: rules=[{rules_str}] trace_id={trace_id} rationale={rationale}"
