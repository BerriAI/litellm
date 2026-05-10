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

        # Walk every text-bearing input we know about — chat ``messages``,
        # tool_call arguments, ``text_completion`` ``prompt``, and embeddings
        # ``input``. Anything we don't extract here would bypass redaction.
        text_parts = list(_iter_data_text(data))
        if not text_parts:
            return data

        redacted_texts, session_id = await self._redact_batch(
            [t for _, t in text_parts]
        )
        if len(redacted_texts) != len(text_parts):
            raise PEyeEyeGuardrailAPIError(
                f"peyeeye /v1/redact returned {len(redacted_texts)} texts for "
                f"{len(text_parts)} inputs; refusing to forward partially-"
                "redacted data"
            )
        for (locator, _), redacted in zip(text_parts, redacted_texts):
            _set_data_text(data, locator, redacted)

        if session_id:
            cache_key = self._cache_key(data, user_api_key_dict)
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

        cache_key = self._cache_key(data, user_api_key_dict)
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
                # Mirror the pre-call coverage: if the model echoes placeholders
                # into tool_call arguments, rehydrate those too.
                tool_calls = getattr(message, "tool_calls", None)
                if isinstance(tool_calls, list):
                    for tc in tool_calls:
                        fn = getattr(tc, "function", None) or (
                            tc.get("function") if isinstance(tc, dict) else None
                        )
                        if fn is None:
                            continue
                        args = getattr(fn, "arguments", None) if not isinstance(fn, dict) else fn.get("arguments")
                        if isinstance(args, str) and args:
                            new_args = await self._rehydrate(args, session_id)
                            if isinstance(fn, dict):
                                fn["arguments"] = new_args
                            else:
                                try:
                                    fn.arguments = new_args
                                except Exception:
                                    pass
        elif isinstance(response, litellm.TextCompletionResponse):
            # /v1/completions returns choices[].text rather than choices[].message.
            for choice in response.choices:
                text = getattr(choice, "text", None)
                if isinstance(text, str) and text:
                    try:
                        choice.text = await self._rehydrate(text, session_id)
                    except Exception:
                        pass

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
    def _cache_key(data: dict, user_api_key_dict: UserAPIKeyAuth) -> str:
        # ``litellm_call_id`` is sourced from the inbound ``x-litellm-call-id``
        # header, so it is caller-controlled. Namespace by the authenticated
        # key (server-controlled) so two callers can't collide on the same key
        # and rehydrate each other's PII.
        call_id = data.get("litellm_call_id") or id(data)
        auth_ns = (
            getattr(user_api_key_dict, "api_key", None)
            or getattr(user_api_key_dict, "token", None)
            or "anon"
        )
        return f"peyeeye_session:{auth_ns}:{call_id}"

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


def _iter_data_text(data: Dict[str, Any]):
    """Yield ``(locator, text)`` for every text-bearing input in ``data``.

    Covers chat ``messages`` (incl. tool_call arguments), ``text_completion``
    ``prompt``, and ``embeddings`` ``input``. ``locator`` is opaque to callers
    and is consumed by ``_set_data_text`` to write the redacted value back.
    """
    messages = data.get("messages")
    if isinstance(messages, list):
        for msg_idx, sub_path, text in _iter_message_text(messages):
            yield ("messages", msg_idx, sub_path), text

    prompt = data.get("prompt")
    if isinstance(prompt, str):
        if prompt:
            yield ("prompt",), prompt
    elif isinstance(prompt, list):
        for j, p in enumerate(prompt):
            if isinstance(p, str) and p:
                yield ("prompt", j), p

    inp = data.get("input")
    if isinstance(inp, str):
        if inp:
            yield ("input",), inp
    elif isinstance(inp, list):
        for j, v in enumerate(inp):
            if isinstance(v, str) and v:
                yield ("input", j), v


def _set_data_text(data: Dict[str, Any], locator, value: str) -> None:
    head = locator[0]
    if head == "messages":
        _, msg_idx, sub_path = locator
        messages = data.get("messages") or []
        if isinstance(msg_idx, int) and msg_idx < len(messages):
            _set_message_text(messages[msg_idx], sub_path, value)
        return
    if head == "prompt":
        if len(locator) == 1:
            data["prompt"] = value
            return
        _, j = locator
        prompt = data.get("prompt")
        if isinstance(prompt, list) and isinstance(j, int) and j < len(prompt):
            prompt[j] = value
        return
    if head == "input":
        if len(locator) == 1:
            data["input"] = value
            return
        _, j = locator
        inp = data.get("input")
        if isinstance(inp, list) and isinstance(j, int) and j < len(inp):
            inp[j] = value


def _iter_message_text(messages: List[Dict[str, Any]]):
    """Yield (message_index, part_path, text) for every text-bearing chunk.

    ``part_path`` identifies where the text lives so ``_set_message_text``
    can write the redacted value back:

      * ``"content"`` — plain string ``content``
      * ``("content", j)`` — the ``j``-th item of a multimodal content list
      * ``("tool_call", k)`` — ``tool_calls[k].function.arguments``
      * ``"function_call"`` — legacy ``function_call.arguments``
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
                        yield i, ("content", j), text
        # Tool calls carry model-visible text in ``function.arguments``; if we
        # leave them alone a caller can put PII there and bypass redaction.
        tool_calls = msg.get("tool_calls")
        if isinstance(tool_calls, list):
            for k, tc in enumerate(tool_calls):
                if not isinstance(tc, dict):
                    continue
                fn = tc.get("function")
                if isinstance(fn, dict):
                    args = fn.get("arguments")
                    if isinstance(args, str) and args:
                        yield i, ("tool_call", k), args
        fc = msg.get("function_call")
        if isinstance(fc, dict):
            args = fc.get("arguments")
            if isinstance(args, str) and args:
                yield i, "function_call", args


def _set_message_text(message: Dict[str, Any], part_path, value: str) -> None:
    if part_path == "content":
        message["content"] = value
        return
    if part_path == "function_call":
        fc = message.get("function_call")
        if isinstance(fc, dict):
            fc["arguments"] = value
        return
    if isinstance(part_path, tuple) and len(part_path) == 2:
        kind, idx = part_path
        if kind == "content":
            parts = message.get("content")
            if isinstance(parts, list) and isinstance(idx, int) and idx < len(parts):
                part = parts[idx]
                if isinstance(part, dict):
                    part["text"] = value
            return
        if kind == "tool_call":
            tool_calls = message.get("tool_calls")
            if isinstance(tool_calls, list) and isinstance(idx, int) and idx < len(tool_calls):
                tc = tool_calls[idx]
                if isinstance(tc, dict):
                    fn = tc.get("function")
                    if isinstance(fn, dict):
                        fn["arguments"] = value
