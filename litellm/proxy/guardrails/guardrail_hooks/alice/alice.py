# +-------------------------------------------------------------+
#
#              Use Alice by ActiveFence for your LLM calls
#                     https://www.activefence.com/
#
# +-------------------------------------------------------------+

import json
import os
from typing import TYPE_CHECKING, Any, Final, Literal, Optional

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

# Caps on the outbound copy of request_data. A payload deeper or wider than this is malformed
# rather than large, and serializing it would cost more than the evaluation it feeds.
_MAX_DEPTH: Final = 12
_MAX_ITEMS: Final = 5000


class AliceReplacement(TypedDict):
    """A masked substitution, positional against the texts that were submitted."""

    index: ReadOnly[NotRequired[int]]
    text: ReadOnly[NotRequired[str]]


class AliceVerdict(TypedDict):
    """Body returned by Alice's LiteLLM evaluate endpoint."""

    verdict: ReadOnly[NotRequired[str]]
    categories: ReadOnly[NotRequired["list[str]"]]
    correlation_id: ReadOnly[NotRequired[str]]
    message: ReadOnly[NotRequired[str]]
    replacements: ReadOnly[NotRequired["list[AliceReplacement]"]]


class AliceGuardrailMissingSecrets(Exception):
    """Raised when the Alice API key is not configured."""


class AliceGuardrail(CustomGuardrail):
    """
    Alice by ActiveFence — policy-based guardrails for prompts and model responses.

    This forwards the hook's arguments as it received them and enforces the verdict that comes
    back. It selects nothing and renames nothing: which parts of a conversation are worth
    evaluating, and how a verdict is reached, are decided by Alice — so changing either is a
    change on their side rather than a LiteLLM upgrade.

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
        **kwargs: Any,
    ):
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
            kwargs["supported_event_hooks"] = [
                GuardrailEventHooks.pre_call,
                GuardrailEventHooks.during_call,
                GuardrailEventHooks.post_call,
            ]

        super().__init__(**kwargs)

    @log_guardrail_information
    async def apply_guardrail(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict,
        input_type: Literal["request", "response"],
        logging_obj: Optional["LiteLLMLoggingObj"] = None,
    ) -> GenericGuardrailAPIInputs:
        if not inputs.get("texts"):
            return inputs

        try:
            verdict: AliceVerdict = await self._evaluate(
                inputs=inputs, request_data=request_data, input_type=input_type
            )
        except GuardrailRaisedException:
            raise
        except Timeout as e:
            return self._on_unreachable(e, inputs)
        except httpx.HTTPStatusError as e:
            status_code: Final = getattr(getattr(e, "response", None), "status_code", None)
            if status_code in (502, 503, 504):
                return self._on_unreachable(e, inputs)
            raise
        except httpx.RequestError as e:
            return self._on_unreachable(e, inputs)

        return self._enforce(verdict, inputs)

    async def _evaluate(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict,
        input_type: str,
    ) -> AliceVerdict:
        response: Final = await self.async_handler.post(
            url=self.api_base,
            json={
                "input_type": input_type,
                "inputs": _json_safe(inputs),
                "request_data": _json_safe(request_data),
            },
            headers={
                "Content-Type": "application/json",
                "af-api-key": self.alice_api_key,
            },
        )
        response.raise_for_status()
        body = response.json()
        if not isinstance(body, dict):
            raise ValueError("Alice returned a non-object body")
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
        """
        texts: Final = inputs.get("texts") or []
        applied = 0
        for replacement in verdict.get("replacements") or []:
            index = replacement.get("index")
            text = replacement.get("text")
            if isinstance(index, int) and isinstance(text, str) and 0 <= index < len(texts):
                texts[index] = text
                applied += 1

        if applied == 0:
            # A mask that wrote nothing is an allow wearing a mask's name. Refuse it rather than
            # let the text through unmasked under a verdict that said it should not go.
            raise GuardrailRaisedException(
                guardrail_name=GUARDRAIL_NAME,
                message=verdict.get("message") or _DEFAULT_BLOCK_MESSAGE,
                should_wrap_with_default_message=False,
                blocked_content=True,
            )
        inputs["texts"] = texts

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


def _json_safe(value: Any, depth: int = 0, seen: frozenset[int] = frozenset()) -> Any:
    """
    Copy `value` into something `json.dumps` accepts, dropping only what cannot cross.

    `request_data` carries live Python objects — an OpenTelemetry span among them — so it cannot
    be serialized as it stands. What is dropped is decided by a mechanical rule rather than a
    field list: a list drifts from what the far side needs, a rule cannot. Serializing naively
    raises, and that error would be read as "guardrail unavailable" on every single request.
    """
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if depth >= _MAX_DEPTH or id(value) in seen:
        return None

    nested: Final = seen | {id(value)}

    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, item in list(value.items())[:_MAX_ITEMS]:
            if isinstance(key, str):
                out[key] = _json_safe(item, depth + 1, nested)
        return out

    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_safe(item, depth + 1, nested) for item in list(value)[:_MAX_ITEMS]]

    dump: Final = getattr(value, "model_dump", None)
    if callable(dump):
        try:
            return _json_safe(dump(mode="json"), depth + 1, nested)
        except Exception:  # noqa: BLE001  # a model that will not dump is one we drop
            return None

    try:
        json.dumps(value)
    except (TypeError, ValueError):
        return None
    return value
