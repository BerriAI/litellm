# +-------------------------------------------------------------+
#
#                    Use Alice for your LLM calls
#                         https://alice.io/
#
# +-------------------------------------------------------------+

import json
import os
from collections.abc import Mapping
from typing import (
    TYPE_CHECKING,
    Any,  # noqa: TID251  # **kwargs forwards verbatim to CustomGuardrail.__init__; see ruff-strict.toml
    Final,
    Literal,
    Optional,
)

import httpx
from typing_extensions import NotRequired, ReadOnly, TypedDict

from litellm._logging import verbose_proxy_logger
from litellm.exceptions import GuardrailRaisedException, Timeout
from litellm.integrations.custom_guardrail import (
    CustomGuardrail,
    log_guardrail_information,
)
from litellm.llms.custom_httpx.http_handler import (
    get_async_httpx_client,
    httpxSpecialProvider,
)
from litellm.types.guardrails import GuardrailEventHooks
from litellm.types.utils import GenericGuardrailAPIInputs

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj

GUARDRAIL_NAME: Final = "alice"

_DEFAULT_API_BASE: Final = "https://api.alice.io"
_EVALUATE_PATH: Final = "/v2/evaluate/litellm"

_VERDICT_ALLOW: Final = "ALLOW"
_VERDICT_BLOCK: Final = "BLOCK"
_VERDICT_MASK: Final = "MASK"
_VERDICT_DETECT: Final = "DETECT"
_KNOWN_VERDICTS: Final = frozenset({_VERDICT_ALLOW, _VERDICT_BLOCK, _VERDICT_MASK, _VERDICT_DETECT})

_DEFAULT_BLOCK_MESSAGE: Final = "Blocked by your organization's content policy."

# apply_guardrail selects nothing: it forwards whichever of these came populated and lets Alice
# decide what is worth evaluating. Only skip the call when every one of them is empty — there is
# then genuinely nothing to send.
_SELECTABLE_INPUT_FIELDS: Final = ("texts", "images", "tools", "tool_calls", "structured_messages")

# Caps on the outbound copy of request_data. A payload deeper or wider than this is malformed
# rather than large, and serializing it would cost more than the evaluation it feeds.
_MAX_DEPTH: Final = 12
_MAX_ITEMS: Final = 5000

# request_data carries the caller's raw credentials under these keys, at any nesting depth —
# a real captured payload puts inbound headers at request_data["proxy_server_request"]["headers"],
# again under ["metadata"]["headers"] / ["litellm_metadata"]["headers"], and again under
# ["metadata"]["requester_metadata"]["headers"], any of which can carry an Authorization or
# x-api-key value. LiteLLM's own spend-log sanitizer excludes `secret_fields` for the same reason
# (spend_tracking_utils._SENSITIVE_REQUEST_BODY_KEYS): `secret_fields.raw_headers` holds the
# caller's Authorization / x-api-key in the clear, and `api_key` can carry a forwarded provider
# credential. Stripping by key name rather than by path means a new nesting path can never
# reintroduce the leak. Posting any of these to a third-party guardrail endpoint would be worse
# than what the proxy already refuses to persist in its own audit trail — so none of them leave
# the process.
_CREDENTIAL_KEYS_TO_STRIP: Final = frozenset(
    {"secret_fields", "api_key", "raw_headers", "headers", "provider_specific_header"}
)


class AliceReplacement(TypedDict):
    """A masked substitution, positional against the texts that were submitted."""

    index: ReadOnly[NotRequired[int]]
    text: ReadOnly[NotRequired[str]]


class AliceVerdict(TypedDict):
    """Body returned by Alice's LiteLLM evaluate endpoint."""

    verdict: ReadOnly[NotRequired[str]]
    categories: ReadOnly[NotRequired["tuple[str, ...]"]]
    correlation_id: ReadOnly[NotRequired[str]]
    message: ReadOnly[NotRequired[str]]
    replacements: ReadOnly[NotRequired["tuple[AliceReplacement, ...]"]]


class AliceGuardrailMissingSecrets(Exception):
    """Raised when the Alice API key is not configured."""


class AliceGuardrail(CustomGuardrail):
    """
    Alice — policy-based guardrails for prompts and model responses.

    This forwards the hook's arguments as it received them and enforces the verdict that comes
    back, with one deliberate exception: any key named `secret_fields`, `api_key`, `raw_headers`,
    `headers`, or `provider_specific_header` is dropped from `request_data` at any nesting depth
    before it is serialized, and never reaches Alice. Short of that, it selects nothing and
    renames nothing: which parts of a conversation are worth evaluating, and how a verdict is
    reached, are decided by Alice — so changing either is a change on their side rather than a
    LiteLLM upgrade. A batch with nothing selectable at all (no `texts`, `images`, `tools`,
    `tool_calls`, or `structured_messages`) still skips the call, since there would be nothing to
    send.

    Known limitation: the unified guardrail's `streaming_transform_mode` defaults to
    `block_only`, whose streaming path discards any returned text rewrite. A MASK verdict is
    therefore a no-op on a streamed response — the original, unmasked text still reaches the
    caller — while BLOCK continues to function on both streamed and non-streamed responses.
    This is `during_call`'s documented behavior generally, not specific to Alice; configure a
    masking-aware `streaming_transform_mode` if that gap matters for your traffic.

    Alice evaluates against policies configured per *application*, and one proxy typically fronts
    several, so the application is named on the virtual key rather than in this config:

        curl $PROXY/key/generate -H "Authorization: Bearer $LITELLM_MASTER_KEY" \\
          -d '{"key_alias": "payments-bot",
               "metadata": {"alice_app_id": "payments-bot"}}'

    Alice reads that off the authenticated key. Because the proxy strips caller-supplied
    `user_api_key_*` from the request before a guardrail sees it, a caller cannot point its own
    traffic at an application with laxer policies than the one its key was issued for.

    Configuration example (litellm config YAML):
        guardrails:
          - guardrail_name: alice
            litellm_params:
              guardrail: alice
              mode: [pre_call, post_call]
              api_key: os.environ/ALICE_API_KEY
              api_base: https://api.alice.io          # optional
              unreachable_fallback: fail_closed       # optional
    """

    def __init__(
        self,
        api_key: str | None = None,
        api_base: str | None = None,
        unreachable_fallback: Literal["fail_closed", "fail_open"] = "fail_closed",
        **kwargs: Any,  # kwargs-ok: forwarded verbatim to CustomGuardrail.__init__, whose param list is wide and evolving
    ) -> None:
        self.async_handler = get_async_httpx_client(llm_provider=httpxSpecialProvider.GuardrailCallback)

        alice_api_key: Final = api_key or os.environ.get("ALICE_API_KEY")
        if not alice_api_key:
            raise AliceGuardrailMissingSecrets(
                "Alice API key is required. Set the `ALICE_API_KEY` environment variable or "
                "pass `api_key` in the guardrail config."
            )
        self.alice_api_key: str = alice_api_key

        base: Final = (api_base or os.environ.get("ALICE_API_BASE") or _DEFAULT_API_BASE).rstrip("/")
        self.api_base: str = f"{base}{_EVALUATE_PATH}"
        self.unreachable_fallback: Literal["fail_closed", "fail_open"] = unreachable_fallback

        if "supported_event_hooks" not in kwargs:
            kwargs["supported_event_hooks"] = [  # mutable-ok: CustomGuardrail.__init__ requires a list here
                GuardrailEventHooks.pre_call,
                GuardrailEventHooks.during_call,
                GuardrailEventHooks.post_call,
            ]

        super().__init__(**kwargs)

    @log_guardrail_information
    async def apply_guardrail(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict[str, object],  # mutable-ok: overrides CustomGuardrail.apply_guardrail's plain-dict contract
        input_type: Literal["request", "response"],
        logging_obj: Optional["LiteLLMLoggingObj"] = None,
    ) -> GenericGuardrailAPIInputs:
        if not any(inputs.get(field) for field in _SELECTABLE_INPUT_FIELDS):
            return inputs

        try:
            verdict: AliceVerdict = await self._evaluate(
                inputs=inputs, request_data=request_data, input_type=input_type
            )
        except Timeout as e:
            return self._on_unreachable(e, inputs)
        except httpx.HTTPStatusError as e:
            status_code: Final = getattr(getattr(e, "response", None), "status_code", None)
            # Any 5xx is an outage on Alice's side, not our misconfiguration — route the whole
            # class through the configured policy. A 4xx (rejected credential, bad request) is
            # ours to fix and must never fail open, so it is deliberately left to propagate.
            if isinstance(status_code, int) and 500 <= status_code < 600:
                return self._on_unreachable(e, inputs)
            raise
        except httpx.RequestError as e:
            return self._on_unreachable(e, inputs)
        except (json.JSONDecodeError, UnicodeDecodeError, TypeError) as e:
            # A body that cannot be decoded, cannot be parsed as JSON, or parses to something
            # other than an object, is as unreachable as a dropped connection: this deployment's
            # policy decides, not a raw exception. UnicodeDecodeError is named explicitly because
            # it is a sibling of JSONDecodeError under ValueError, not a subclass of it.
            return self._on_unreachable(e, inputs)

        return self._enforce(verdict, inputs)

    async def _evaluate(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: Mapping[str, object],
        input_type: str,
    ) -> AliceVerdict:
        response: Final = await self.async_handler.post(
            url=self.api_base,
            json={  # mutable-ok: one-shot HTTP request body, never mutated after construction
                "input_type": input_type,
                "inputs": _json_safe(inputs),
                "request_data": _json_safe(request_data, strip_keys=_CREDENTIAL_KEYS_TO_STRIP),
            },
            headers={  # mutable-ok: one-shot HTTP headers, never mutated after construction
                "Content-Type": "application/json",
                "af-api-key": self.alice_api_key,
            },
        )
        response.raise_for_status()
        body = response.json()
        if not isinstance(body, dict):
            raise TypeError("Alice returned a non-object body")
        return body

    def _enforce(self, verdict: AliceVerdict, inputs: GenericGuardrailAPIInputs) -> GenericGuardrailAPIInputs:
        """Act on the verdict. An answer we cannot read is treated as unavailable, never as a pass."""
        name: Final = verdict.get("verdict")
        if name not in _KNOWN_VERDICTS:
            return self._on_unreachable(ValueError(f"unrecognized verdict: {name!r}"), inputs)

        if name == _VERDICT_BLOCK:
            raise GuardrailRaisedException(
                guardrail_name=GUARDRAIL_NAME,
                message=verdict.get("message") or _DEFAULT_BLOCK_MESSAGE,
                should_wrap_with_default_message=False,
                blocked_content=True,
            )

        if name == _VERDICT_DETECT:
            # Recorded by Alice and allowed through. The correlation id is what ties this request
            # to that record; the evaluated text itself is never logged.
            verbose_proxy_logger.warning(
                "Alice guardrail: detection recorded, request allowed (correlation_id=%s, categories=%s)",
                verdict.get("correlation_id"),
                verdict.get("categories"),
            )
            return inputs

        if name == _VERDICT_MASK:
            self._apply_replacements(verdict, inputs)

        return inputs

    def _apply_replacements(self, verdict: AliceVerdict, inputs: GenericGuardrailAPIInputs) -> None:
        """
        Write each replacement onto the text it names.

        Only `texts` is touched. The chat translation layer maps a returned `texts` list back onto
        the request positionally, but takes a different branch entirely when `structured_messages`
        comes back as a new object — which would drop these edits.

        All-or-nothing: a single out-of-range or malformed replacement blocks the whole verdict
        rather than being silently skipped, so content Alice meant to replace can never reach the
        model unmasked alongside content that was replaced.
        """
        texts: Final = inputs.get("texts") or []  # mutable-ok: empty-list fallback, replaced wholesale below
        replacements: Final = verdict.get("replacements") or []  # mutable-ok: empty-list fallback for iteration only

        if not replacements:
            raise self._mask_rejected(verdict)

        for replacement in replacements:
            index = replacement.get("index")
            text = replacement.get("text")
            if not (isinstance(index, int) and isinstance(text, str) and 0 <= index < len(texts)):
                raise self._mask_rejected(verdict)
            texts[index] = text  # mutable-ok: item assignment into the local working copy above

        inputs["texts"] = texts

    def _mask_rejected(self, verdict: AliceVerdict) -> GuardrailRaisedException:
        """A MASK verdict that cannot be applied in full is refused outright, never partially —
        see `_apply_replacements`."""
        return GuardrailRaisedException(
            guardrail_name=GUARDRAIL_NAME,
            message=verdict.get("message") or _DEFAULT_BLOCK_MESSAGE,
            should_wrap_with_default_message=False,
            blocked_content=True,
        )

    def _on_unreachable(self, error: Exception, inputs: GenericGuardrailAPIInputs) -> GenericGuardrailAPIInputs:
        """Apply the configured policy when Alice cannot be reached or cannot be understood."""
        if self.unreachable_fallback == "fail_open":
            verbose_proxy_logger.critical(
                "Alice guardrail unreachable, allowing request per unreachable_fallback: %s",
                error,
            )
            return inputs
        raise GuardrailRaisedException(
            guardrail_name=GUARDRAIL_NAME,
            message="Alice guardrail is unavailable and this request cannot be checked",
            should_wrap_with_default_message=False,
        ) from error

    @staticmethod
    def get_config_model() -> type | None:
        from litellm.types.proxy.guardrails.guardrail_hooks.alice import (
            AliceGuardrailConfigModel,
        )

        return AliceGuardrailConfigModel


def _json_safe(
    value: object,
    depth: int = 0,
    seen: frozenset[int] = frozenset(),
    strip_keys: frozenset[str] = frozenset(),
) -> object:
    """
    Copy `value` into something `json.dumps` accepts, dropping only what cannot cross.

    `request_data` carries live Python objects — an OpenTelemetry span among them — so it cannot
    be serialized as it stands. What is dropped is decided by a mechanical rule rather than a
    field list: a list drifts from what the far side needs, a rule cannot. Serializing naively
    raises, and that error would be read as "guardrail unavailable" on every single request.

    `strip_keys` drops a dict key by name at every depth it appears, not just the root — a caller
    passes `_CREDENTIAL_KEYS_TO_STRIP` here so a credential nested under any path is caught the
    same way a top-level one is, without maintaining a list of paths. The source object is never
    mutated: every branch below builds a new container.
    """
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if depth >= _MAX_DEPTH or id(value) in seen:
        return None

    nested: Final = seen | {id(value)}  # mutable-ok: one-shot set literal, unioned into a frozenset immediately

    if isinstance(value, dict):
        out: dict[str, object] = {}  # mutable-ok: bounded accumulator local to this call, never escapes as-is
        for key, item in list(value.items())[:_MAX_ITEMS]:  # mutable-ok: list() only to slice an unordered view
            if isinstance(key, str) and key not in strip_keys:
                out[key] = _json_safe(item, depth + 1, nested, strip_keys)
        return out

    if isinstance(value, (list, tuple, set, frozenset)):
        return [  # mutable-ok: return value is a one-shot list, discarded by the caller after use
            _json_safe(item, depth + 1, nested, strip_keys)
            for item in list(value)[:_MAX_ITEMS]  # mutable-ok: list() only to slice an unordered view
        ]

    dump: Final = getattr(value, "model_dump", None)
    if callable(dump):
        try:
            return _json_safe(dump(mode="json"), depth + 1, nested, strip_keys)
        except Exception:  # noqa: BLE001  # a model that will not dump is one we drop
            return None

    # Everything json.dumps handles natively — str, int, float, bool, None, dict, list — is
    # caught above, and a dict/list subclass is caught by isinstance. So whatever reaches here
    # (bytes, datetime, an OpenTelemetry span) cannot cross the wire.
    return None
