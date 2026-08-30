# +-------------------------------------------------------------+
#
#              Use Alice by ActiveFence for your LLM calls
#                     https://www.activefence.com/
#
# +-------------------------------------------------------------+

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
_EVALUATE_PATH: Final = "/v2/evaluate/message"

# The key a customer sets on a virtual key's metadata to name the application whose policies
# apply. Read only from the authenticated key's metadata, never from the request body.
_APP_ID_METADATA_KEY: Final = "alice_app_id"

# Alice enforces these on the evaluate contract; exceeding either is a 400.
_MAX_TEXT_CHARS: Final = 10_240
_MAX_ID_CHARS: Final = 100

# What a request is attributed to when the gateway names no end user. The contract requires a
# non-empty id holding a word character, and a constant is honest about what is not known.
_FALLBACK_USER_ID: Final = "litellm-unknown"


class AliceDetection(TypedDict):
    """One policy that matched, as Alice reports it."""

    type: ReadOnly[NotRequired[str]]
    score: ReadOnly[NotRequired[float]]


class AliceEvaluateResponse(TypedDict):
    """Body returned by Alice's evaluate endpoint."""

    correlation_id: ReadOnly[NotRequired[str]]
    action: ReadOnly[NotRequired[str]]
    action_text: ReadOnly[NotRequired[str]]
    detections: ReadOnly[NotRequired["list[AliceDetection]"]]
    errors: ReadOnly[NotRequired["list[dict[str, object]]"]]


class AliceGuardrailMissingSecrets(Exception):
    """Raised when the Alice API key is not configured."""


class AliceGuardrail(CustomGuardrail):
    """
    Alice by ActiveFence — policy-based guardrails for prompts and model responses.

    Alice evaluates against policies configured per *application*, so one proxy can enforce a
    different policy set per team or product while sharing a single project credential. The
    application is named on the LiteLLM virtual key rather than in this config, because a proxy
    typically fronts several of them:

        curl $PROXY/key/generate -H "Authorization: Bearer $LITELLM_MASTER_KEY" \\
          -d '{"key_alias": "payments-bot",
               "metadata": {"alice_app_id": "payments-bot"}}'

    `alice_app_id` is read first and the key's `key_alias` is the fallback, so naming the key
    after the application is enough. Either value must match the Application ID configured on
    that application in Alice. Only the authenticated key is consulted — a value a caller puts in
    its own request body is ignored, so a caller cannot select a laxer application than the one
    its key was issued for.

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
        texts: Final = inputs.get("texts") or []
        if not texts:
            return inputs

        app_id: Final = self._resolve_app_id(request_data)
        if not app_id:
            # Refuse rather than guess: evaluating against an arbitrary application would apply
            # the wrong policy set, which is worse than not evaluating at all.
            raise GuardrailRaisedException(
                guardrail_name=GUARDRAIL_NAME,
                message=(
                    "No Alice application is named by this key. Set "
                    f"`metadata.{_APP_ID_METADATA_KEY}` or `key_alias` on the virtual key."
                ),
                should_wrap_with_default_message=False,
            )

        shared: Final = {
            "app_id": app_id,
            "message_type": "prompt" if input_type == "request" else "response",
            "session_id": self._resolve_session_id(request_data, logging_obj),
            "user_id": self._resolve_user_id(request_data),
        }

        masked: Final[list[str]] = []
        for text in texts:
            masked.append(await self._evaluate_one(text=text, shared=shared))

        if masked != list(texts):
            inputs["texts"] = masked
        return inputs

    async def _evaluate_one(self, text: str, shared: dict[str, str]) -> str:
        """Evaluate one text, returning it unchanged, masked, or raising to block."""
        if not text.strip():
            return text

        try:
            response: Final = await self.async_handler.post(
                url=self.api_base,
                json={"text": text[:_MAX_TEXT_CHARS], **shared},
                headers={
                    "Content-Type": "application/json",
                    "af-api-key": self.alice_api_key,
                },
            )
            response.raise_for_status()
            body: AliceEvaluateResponse = response.json()
        except GuardrailRaisedException:
            raise
        except Timeout as e:
            return self._on_unreachable(e, text)
        except httpx.HTTPStatusError as e:
            status_code: Final = getattr(getattr(e, "response", None), "status_code", None)
            if status_code in (502, 503, 504):
                return self._on_unreachable(e, text)
            raise
        except httpx.RequestError as e:
            return self._on_unreachable(e, text)

        return self._enforce(body, text)

    def _enforce(self, body: AliceEvaluateResponse, text: str) -> str:
        """Turn Alice's verdict into an action on this text."""
        # A verdict carrying errors[] is a failure, not a pass — reporting it as one would let a
        # half-evaluated message through.
        if body.get("errors"):
            raise GuardrailRaisedException(
                guardrail_name=GUARDRAIL_NAME,
                message="Alice reported errors while evaluating this content",
                should_wrap_with_default_message=False,
            )

        action: Final = body.get("action") or ""
        correlation_id: Final = body.get("correlation_id")

        if action == "BLOCK":
            raise GuardrailRaisedException(
                guardrail_name=GUARDRAIL_NAME,
                message=body.get("action_text") or "Blocked by Alice policy",
                should_wrap_with_default_message=False,
                blocked_content=True,
            )

        if action == "MASK":
            # `action_text` is the redacted text. Absent it there is nothing to substitute, and
            # returning the original would defeat the verdict.
            masked: Final = body.get("action_text")
            if masked is None:
                raise GuardrailRaisedException(
                    guardrail_name=GUARDRAIL_NAME,
                    message="Blocked by Alice policy",
                    should_wrap_with_default_message=False,
                    blocked_content=True,
                )
            return masked

        if action == "DETECT":
            # Recorded by Alice and allowed through. The correlation id is what ties this request
            # to that record; the text itself is never logged.
            verbose_proxy_logger.warning(
                "Alice guardrail: detection recorded, request allowed (correlation_id=%s, types=%s)",
                correlation_id,
                [d.get("type") for d in body.get("detections") or []],
            )

        return text

    def _on_unreachable(self, error: Exception, text: str) -> str:
        """Apply the configured policy when Alice cannot be reached."""
        if self.unreachable_fallback == "fail_open":
            verbose_proxy_logger.critical(
                "Alice guardrail unreachable, allowing request per unreachable_fallback: %s",
                error,
            )
            return text
        raise GuardrailRaisedException(
            guardrail_name=GUARDRAIL_NAME,
            message="Alice guardrail is unavailable and this request cannot be checked",
            should_wrap_with_default_message=False,
        ) from error

    def _resolve_app_id(self, request_data: dict) -> str | None:
        """
        The application whose policies apply, taken from the authenticated virtual key.

        `_get_admin_metadata` reads whichever metadata holder the proxy wrote the key's own
        values into, which differs by route, and the proxy strips caller-supplied `user_api_key_*`
        keys from both — so this cannot be set by whoever makes the call.
        """
        admin_metadata: Final = self._get_admin_metadata(request_data)
        configured: Final = admin_metadata.get(_APP_ID_METADATA_KEY)
        if isinstance(configured, str) and configured.strip():
            return configured.strip()[:_MAX_ID_CHARS]

        for holder_name in ("litellm_metadata", "metadata"):
            holder = request_data.get(holder_name)
            if not isinstance(holder, dict):
                continue
            alias = holder.get("user_api_key_alias")
            if isinstance(alias, str) and alias.strip():
                return alias.strip()[:_MAX_ID_CHARS]
        return None

    @staticmethod
    def _resolve_session_id(request_data: dict, logging_obj: Optional["LiteLLMLoggingObj"]) -> str:
        """
        Alice groups a conversation by session id, so the trace id is preferred: it spans every
        call of one conversation where the per-call id does not.
        """
        candidates: Final = (
            getattr(logging_obj, "litellm_trace_id", None) if logging_obj else None,
            request_data.get("litellm_trace_id"),
            getattr(logging_obj, "litellm_call_id", None) if logging_obj else None,
            request_data.get("litellm_call_id"),
        )
        for candidate in candidates:
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip()[:_MAX_ID_CHARS]
        return _FALLBACK_USER_ID

    def _resolve_user_id(self, request_data: dict) -> str:
        """
        The end user comes first: on a multi-tenant proxy the key belongs to a team or an
        application, so keying on its owner would attribute every one of that tenant's users to a
        single identity.
        """
        for holder_name in ("litellm_metadata", "metadata"):
            holder = request_data.get(holder_name)
            if not isinstance(holder, dict):
                continue
            for field in (
                "user_api_key_end_user_id",
                "user_api_key_user_email",
                "user_api_key_user_id",
                "user_api_key_hash",
            ):
                value = holder.get(field)
                if isinstance(value, str) and value.strip():
                    return value.strip()[:_MAX_ID_CHARS]
        return _FALLBACK_USER_ID

    @staticmethod
    def get_config_model() -> type | None:
        from litellm.types.proxy.guardrails.guardrail_hooks.alice import (
            AliceGuardrailConfigModel,
        )

        return AliceGuardrailConfigModel
