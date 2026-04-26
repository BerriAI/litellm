# +-------------------------------------------------------------+
#
#           Use Peyeeye PII redaction & rehydration for LLM calls
#                        https://peyeeye.ai
#
# +-------------------------------------------------------------+

import os
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Literal,
    NoReturn,
    Optional,
    Type,
    Union,
)

try:
    import httpx

    HTTPX_AVAILABLE = True
except ImportError:
    httpx = None  # type: ignore
    HTTPX_AVAILABLE = False

import litellm
from litellm import DualCache
from litellm._logging import verbose_proxy_logger
from litellm.integrations.custom_guardrail import (
    CustomGuardrail,
    dc as global_cache,
    log_guardrail_information,
)
from litellm.llms.custom_httpx.http_handler import (
    get_async_httpx_client,
    httpxSpecialProvider,
)
from litellm.proxy._types import UserAPIKeyAuth
from litellm.types.guardrails import GuardrailEventHooks


if TYPE_CHECKING:
    from litellm.types.proxy.guardrails.guardrail_hooks.base import GuardrailConfigModel


DEFAULT_API_BASE = "https://api.peyeeye.ai"
SESSION_CACHE_TTL_SECONDS = 3600


class PEyeEyeGuardrailMissingSecrets(Exception):
    """Raised when the peyeeye API key is missing."""


class PEyeEyeGuardrailAPIError(Exception):
    """Raised when the peyeeye API returns an error."""


class PEyeEyeGuardrail(CustomGuardrail):
    """Peyeeye PII redaction + rehydration guardrail.

    Pre-call hook redacts PII from each message's ``content`` and stores the
    redaction session id under the request's ``litellm_call_id`` so the
    post-call hook can rehydrate the model's response with the original
    values.

    Two session modes:
      * ``stateful`` (default): peyeeye stores the token→value mapping under
        a ``ses_…`` id; rehydrate references the id.
      * ``stateless``: peyeeye returns a sealed ``skey_…`` blob; nothing is
        retained server-side.
    """

    def __init__(
        self,
        peyeeye_api_key: Optional[str] = None,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        peyeeye_locale: Optional[str] = None,
        peyeeye_entities: Optional[List[str]] = None,
        peyeeye_session_mode: Optional[Literal["stateful", "stateless"]] = None,
        **kwargs,
    ):
        self.async_handler = get_async_httpx_client(
            llm_provider=httpxSpecialProvider.GuardrailCallback
        )
        self.peyeeye_api_key = (
            peyeeye_api_key or api_key or os.environ.get("PEYEEYE_API_KEY")
        )
        if self.peyeeye_api_key is None:
            raise PEyeEyeGuardrailMissingSecrets(
                "Couldn't get peyeeye api key, either set the `PEYEEYE_API_KEY` "
                "environment variable or pass it as `api_key` in the guardrail config."
            )

        self.api_base = (
            api_base or os.environ.get("PEYEEYE_API_BASE") or DEFAULT_API_BASE
        ).rstrip("/")
        self.peyeeye_locale = peyeeye_locale or "auto"
        self.peyeeye_entities = peyeeye_entities
        self.peyeeye_session_mode: Literal["stateful", "stateless"] = (
            peyeeye_session_mode or "stateful"
        )

        verbose_proxy_logger.debug(
            "Peyeeye guardrail initialized: name=%s, mode=%s, session_mode=%s",
            kwargs.get("guardrail_name", "unknown"),
            kwargs.get("event_hook", "unknown"),
            self.peyeeye_session_mode,
        )

        super().__init__(**kwargs)

    # ------------------------------------------------------------------ hooks

    @log_guardrail_information
    async def async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: DualCache,
        data: dict,
        call_type: Literal[
            "completion",
            "text_completion",
            "embeddings",
            "image_generation",
            "moderation",
            "audio_transcription",
            "pass_through_endpoint",
            "rerank",
            "mcp_call",
            "anthropic_messages",
        ],
    ) -> Union[Exception, str, dict, None]:
        """Redact every text message before it reaches the model."""
        event_type: GuardrailEventHooks = GuardrailEventHooks.pre_call
        if self.should_run_guardrail(data=data, event_type=event_type) is not True:
            return data

        messages: List[Dict[str, Any]] = data.get("messages") or []
        text_parts = list(_iter_message_text(messages))
        if not text_parts:
            return data

        redacted_texts, session_id = await self._redact_batch(
            [t for _, _, t in text_parts]
        )
        if len(redacted_texts) != len(text_parts):
            raise PEyeEyeGuardrailAPIError(
                f"peyeeye /v1/redact returned {len(redacted_texts)} texts for "
                f"{len(text_parts)} inputs; refusing to forward partially-"
                "redacted data"
            )
        for (msg_idx, part_path, _), redacted in zip(text_parts, redacted_texts):
            _set_message_text(messages[msg_idx], part_path, redacted)

        if session_id:
            cache_key = self._cache_key(data)
            try:
                global_cache.set_cache(
                    cache_key, session_id, ttl=SESSION_CACHE_TTL_SECONDS
                )
            except Exception as e:
                verbose_proxy_logger.warning(
                    "peyeeye: failed to cache session id: %s", e
                )

        return data

    @log_guardrail_information
    async def async_post_call_success_hook(
        self,
        data: dict,
        user_api_key_dict: UserAPIKeyAuth,
        response,
    ):
        """Rehydrate model output by swapping placeholders back to original PII."""
        event_type: GuardrailEventHooks = GuardrailEventHooks.post_call
        if self.should_run_guardrail(data=data, event_type=event_type) is not True:
            return response

        cache_key = self._cache_key(data)
        try:
            session_id = global_cache.get_cache(cache_key)
        except Exception:
            session_id = None
        if not session_id:
            # No redaction happened (or session expired); nothing to do.
            return response

        if isinstance(response, litellm.ModelResponse):
            for choice in response.choices:
                message = getattr(choice, "message", None)
                if message is None:
                    continue
                content = getattr(message, "content", None)
                if isinstance(content, str) and content:
                    new = await self._rehydrate(content, session_id)
                    message.content = new
                elif isinstance(content, list):
                    new_parts: List[Any] = []
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "text":
                            text = part.get("text", "")
                            new_parts.append(
                                {**part, "text": await self._rehydrate(text, session_id)}
                            )
                        else:
                            new_parts.append(part)
                    message.content = new_parts

        # Clean up: drop the stateful session server-side. Stateless
        # ``skey_…`` blobs hold no server-side state, so skip the DELETE.
        if self.peyeeye_session_mode == "stateful":
            try:
                await self._delete_session(session_id)
            except Exception as e:
                verbose_proxy_logger.debug(
                    "peyeeye: best-effort session cleanup failed: %s", e
                )
        try:
            global_cache.delete_cache(cache_key)
        except Exception:
            pass

        return response

    # --------------------------------------------------------------- internals

    @staticmethod
    def _cache_key(data: dict) -> str:
        return f"peyeeye_session:{data.get('litellm_call_id') or id(data)}"

    async def _redact_batch(self, texts: List[str]) -> tuple[List[str], Optional[str]]:
        """Redact a batch of texts in a single peyeeye session.

        Returns the redacted strings (in order) and the session id (or sealed
        ``skey_…`` blob) so the post-call hook can rehydrate.
        """
        body: Dict[str, Any] = {
            "text": texts,
            "locale": self.peyeeye_locale,
        }
        if self.peyeeye_entities:
            body["entities"] = list(self.peyeeye_entities)
        if self.peyeeye_session_mode == "stateless":
            body["session"] = "stateless"

        payload = await self._post("/v1/redact", body)
        out_text = payload.get("text")
        if isinstance(out_text, str):
            redacted = [out_text]
        elif isinstance(out_text, list):
            redacted = [str(x) for x in out_text]
        else:
            raise PEyeEyeGuardrailAPIError(
                "peyeeye /v1/redact returned unexpected response shape; "
                "refusing to forward unredacted text"
            )

        if self.peyeeye_session_mode == "stateless":
            session_id = payload.get("rehydration_key")
        else:
            session_id = payload.get("session_id") or payload.get("session")

        return redacted, session_id

    async def _rehydrate(self, text: str, session_id: str) -> str:
        if not text:
            return text
        body = {"text": text, "session": session_id}
        try:
            payload = await self._post("/v1/rehydrate", body)
        except Exception as e:
            verbose_proxy_logger.warning("peyeeye: rehydrate failed: %s", e)
            return text
        return payload.get("text", text)

    async def _delete_session(self, session_id: str) -> None:
        url = f"{self.api_base}/v1/sessions/{session_id}"
        await self.async_handler.delete(url=url, headers=self._headers(), timeout=10.0)

    async def _post(self, path: str, body: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self.api_base}{path}"
        try:
            response = await self.async_handler.post(
                url=url, headers=self._headers(), json=body, timeout=15.0
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            self._reraise_api_error(e, path)

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.peyeeye_api_key}",
            "Content-Type": "application/json",
        }

    @staticmethod
    def _reraise_api_error(error: Exception, path: str) -> NoReturn:
        if HTTPX_AVAILABLE and httpx is not None:
            if isinstance(error, httpx.TimeoutException):
                raise PEyeEyeGuardrailAPIError(f"peyeeye {path} timed out") from error
            if isinstance(error, httpx.HTTPStatusError):
                status = error.response.status_code
                if status == 401:
                    raise PEyeEyeGuardrailMissingSecrets(
                        "Invalid peyeeye API key"
                    ) from error
                if status == 429:
                    raise PEyeEyeGuardrailAPIError(
                        "peyeeye rate limit exceeded"
                    ) from error
                raise PEyeEyeGuardrailAPIError(
                    f"peyeeye {path} returned {status}"
                ) from error
        raise PEyeEyeGuardrailAPIError(
            f"peyeeye {path} failed: {error}"
        ) from error

    @staticmethod
    def get_config_model() -> Optional[Type["GuardrailConfigModel"]]:
        from litellm.types.proxy.guardrails.guardrail_hooks.peyeeye import (
            PEyeEyeGuardrailConfigModel,
        )

        return PEyeEyeGuardrailConfigModel


# ---------------------------------------------------------------- text helpers


def _iter_message_text(messages: List[Dict[str, Any]]):
    """Yield (message_index, part_path, text) for every text-bearing chunk.

    ``part_path`` is either ``"content"`` for a plain string message or an
    int index into the multimodal content list.
    """
    for i, msg in enumerate(messages):
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        if isinstance(content, str):
            if content:
                yield i, "content", content
        elif isinstance(content, list):
            for j, part in enumerate(content):
                if isinstance(part, dict) and part.get("type") == "text":
                    text = part.get("text", "")
                    if text:
                        yield i, j, text


def _set_message_text(message: Dict[str, Any], part_path, value: str) -> None:
    if part_path == "content":
        message["content"] = value
        return
    parts = message.get("content")
    if isinstance(parts, list) and isinstance(part_path, int) and part_path < len(parts):
        part = parts[part_path]
        if isinstance(part, dict):
            part["text"] = value
