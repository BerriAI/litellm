# SPDX-License-Identifier: Apache-2.0
#
# Veto guardrail for LiteLLM.
#
# Upstream destination (when submitting the PR to BerriAI/litellm):
#   litellm/proxy/guardrails/guardrail_hooks/veto/veto.py
#
# This file contains NO detection logic. It is a thin HTTP client that calls
# the hosted Veto gateway (POST {api_base}/v1/check). All detection lives in
# veto-core; this adapter only maps Veto's verdict onto LiteLLM's hook
# contract. See veto-core/integrations/litellm/README.md for the full PR
# checklist (enum + initializer + registry edits).

import uuid
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncGenerator,
    List,
    Literal,
    Optional,
    Type,
    Union,
)

from fastapi import HTTPException

from litellm import DualCache
from litellm.integrations.custom_guardrail import (
    CustomGuardrail,
    log_guardrail_information,
)
from litellm.llms.custom_httpx.http_handler import (
    get_async_httpx_client,
    httpxSpecialProvider,
)
from litellm.proxy._types import UserAPIKeyAuth
from litellm.types.utils import (
    CallTypesLiteral,
    Delta,
    GenericGuardrailAPIInputs,
    ModelResponse,
    ModelResponseStream,
    StreamingChoices,
)

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
    from litellm.types.proxy.guardrails.guardrail_hooks.base import (
        GuardrailConfigModel,
    )

# Veto returns action ∈ {allow, redact, block}. block → reject the call;
# redact → swap message content for the masked text and continue; allow → pass.
VETO_DEFAULT_CATEGORIES = ["pii", "secrets", "injection"]

# Streaming (SPEC §3.5): the adapter buffers ~N tokens of the upstream
# stream locally, holds them until the gateway clears that buffer, then
# releases. Default 128 tokens (tier-tunable 64/128/256). The gateway has
# no tokenizer either, so token budget is approximated as chars/4 on both
# sides — buffer size is passed back via ?buffer=N so the gateway's
# sliding window matches what the adapter buffered.
VETO_STREAM_BUFFER_TOKENS = 128
VETO_APPROX_CHARS_PER_TOKEN = 4


class VetoGuardrail(CustomGuardrail):
    def __init__(
        self,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        **kwargs,
    ):
        self.api_base = (api_base or "https://api.vetocheck.com").rstrip("/")
        self.api_key = api_key
        self.categories = VETO_DEFAULT_CATEGORIES
        self.buffer_tokens = VETO_STREAM_BUFFER_TOKENS
        self.timeout = 10.0
        super().__init__(**kwargs)

    @staticmethod
    def get_config_model() -> Optional[Type["GuardrailConfigModel"]]:
        from litellm.types.proxy.guardrails.guardrail_hooks.veto import (
            VetoGuardrailConfigModel,
        )

        return VetoGuardrailConfigModel

    async def _check(self, text: str) -> dict:
        """POST one text to the Veto gateway. Returns the verdict JSON.

        Uses litellm's shared async HTTP client (connection pooling, retries,
        observability; lifecycle owned by the global client cache).
        """
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        client = get_async_httpx_client(
            llm_provider=httpxSpecialProvider.GuardrailCallback
        )
        resp = await client.post(
            f"{self.api_base}/v1/check",
            headers=headers,
            json={"text": text, "categories": self.categories},
            timeout=self.timeout,
        )
        # Fail closed on any non-2xx (auth / rate-limit / 5xx). The shared
        # handler raises on its primary path; calling it here also covers the
        # connection-retry path, so a gateway error can never silently pass
        # unscanned text through to the model.
        resp.raise_for_status()
        return resp.json()

    async def _check_stream(
        self, text: str, stream_id: str, chunk_index: int, final: bool
    ) -> dict:
        """POST one buffered chunk to the stateful streaming endpoint
        (POST /v1/check?stream_id=…&buffer=N — SPEC §3.5.2). The gateway
        keeps a Redis-backed sliding window keyed by stream_id; the fast
        lane runs per chunk, the slow lane on final=true. Fail closed on
        any non-2xx, same as the batch path."""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        client = get_async_httpx_client(
            llm_provider=httpxSpecialProvider.GuardrailCallback
        )
        resp = await client.post(
            f"{self.api_base}/v1/check",
            headers=headers,
            params={"stream_id": stream_id, "buffer": self.buffer_tokens},
            json={
                "text": text,
                "chunk_index": chunk_index,
                "final": final,
                "categories": self.categories,
            },
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()

    @staticmethod
    def _stream_chunk_text(chunk: ModelResponseStream) -> str:
        """Concatenate the assistant text delta carried by one stream chunk.
        Tool-call / function deltas and role-only chunks contribute no text."""
        parts: List[str] = []
        for choice in getattr(chunk, "choices", None) or []:
            delta = getattr(choice, "delta", None)
            content = getattr(delta, "content", None) if delta is not None else None
            if isinstance(content, str):
                parts.append(content)
        return "".join(parts)

    @staticmethod
    def _blocked_stream_chunk(model: str) -> ModelResponseStream:
        """Terminal chunk emitted when Veto blocks mid-stream: an empty
        delta with finish_reason="content_filter" (SPEC §3.5.8 chunked-JSON
        fallback). Downstream SDKs read it as a content-policy stop; the
        offending buffered tokens are never yielded."""
        return ModelResponseStream(
            model=model,
            choices=[
                StreamingChoices(
                    index=0,
                    delta=Delta(content=None),
                    finish_reason="content_filter",
                )
            ],
        )

    async def async_post_call_streaming_iterator_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        response: Any,
        request_data: dict,
    ) -> AsyncGenerator[Any, None]:
        """Output guardrail for streaming responses (SPEC §3.5).

        Wraps the upstream LLM iterator. Buffers ~buffer_tokens of assistant
        text locally and HOLDS those chunks until the gateway clears the
        buffer, then releases them. On a fast-lane block the held buffer is
        dropped (never yielded) and a content_filter terminal chunk closes
        the stream — already-yielded tokens cannot be unsaid, so redact is
        not applied mid-stream (SPEC §3.5.1); a final redact verdict is
        treated as allow. Fails closed on any gateway error: blocks the
        remainder of the stream (SPEC §3.6 / §3.5.8)."""
        stream_id = uuid.uuid4().hex
        model = request_data.get("model") or ""
        buffer_chars = self.buffer_tokens * VETO_APPROX_CHARS_PER_TOKEN
        chunk_index = 0
        held: List[ModelResponseStream] = []
        buffered_text = ""

        async for chunk in response:
            # Non-chat stream shapes (raw bytes SSE, /v1/responses events)
            # are passed through unscanned in v1 — the prompt was already
            # scanned by the pre-call + moderation hooks. Documented limit.
            if not isinstance(chunk, ModelResponseStream):
                yield chunk
                continue

            held.append(chunk)
            buffered_text += self._stream_chunk_text(chunk)
            if len(buffered_text) < buffer_chars:
                continue

            try:
                verdict = await self._check_stream(
                    buffered_text, stream_id, chunk_index, final=False
                )
            except Exception:
                yield self._blocked_stream_chunk(model)  # fail closed
                return
            chunk_index += 1
            if verdict.get("action") == "block":
                yield self._blocked_stream_chunk(model)
                return
            for held_chunk in held:
                yield held_chunk
            held = []
            buffered_text = ""

        # EOF: flush the trailing buffer with final=true so the slow lane
        # (AI classifiers) runs on the assembled response.
        try:
            verdict = await self._check_stream(
                buffered_text, stream_id, chunk_index, final=True
            )
        except Exception:
            yield self._blocked_stream_chunk(model)
            return
        if verdict.get("action") == "block":
            yield self._blocked_stream_chunk(model)
            return
        for held_chunk in held:
            yield held_chunk

    def _raise_blocked(self, verdict: dict) -> None:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Request blocked by Veto guardrail",
                "veto": {
                    "action": verdict.get("action"),
                    "findings": verdict.get("findings", []),
                    "degraded": verdict.get("degraded", []),
                },
            },
        )

    async def _scan_text(self, text: str, allow_redact: bool = True) -> str:
        """Scan one string against the Veto gateway. Block raises; redact
        returns the masked text when allow_redact is set (the parallel
        moderation hook passes False — it cannot rewrite); allow returns the
        input. Empty / non-string input is returned untouched."""
        if not isinstance(text, str) or not text.strip():
            return text
        verdict = await self._check(text)
        action = verdict.get("action")
        if action == "block":
            self._raise_blocked(verdict)
        if action == "redact" and allow_redact:
            return verdict.get("redacted", text)
        return text

    async def _scan_content(self, content: Any, allow_redact: bool = True) -> Any:
        """Scan a message ``content`` value — a plain string or a multimodal
        list of parts. Every part carrying a string ``text`` field is scanned
        (and rewritten in place on redact); non-text parts (image, audio) are
        left untouched. Closes the bypass where blocked text rides inside a
        multimodal part."""
        if isinstance(content, str):
            return await self._scan_text(content, allow_redact)
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    scanned = await self._scan_text(part["text"], allow_redact)
                    if allow_redact:
                        part["text"] = scanned
        return content

    async def _scan_messages(
        self, messages: List[dict], allow_redact: bool = True
    ) -> List[dict]:
        """Scan every message's content (string or multimodal). Block raises;
        redact rewrites content in place when allow_redact is set."""
        for msg in messages:
            if isinstance(msg, dict) and "content" in msg:
                msg["content"] = await self._scan_content(
                    msg.get("content"), allow_redact
                )
        return messages

    async def _scan_input(self, value: Any, allow_redact: bool = True) -> Any:
        """Scan the Responses-API ``input`` field — a plain string, or a list
        of items (strings or message dicts carrying ``content``). Mirrors the
        chat ``messages`` path so blocked text cannot bypass via /v1/responses.
        """
        if isinstance(value, str):
            return await self._scan_text(value, allow_redact)
        if isinstance(value, list):
            for i, item in enumerate(value):
                if isinstance(item, str):
                    scanned = await self._scan_text(item, allow_redact)
                    if allow_redact:
                        value[i] = scanned
                elif isinstance(item, dict) and "content" in item:
                    item["content"] = await self._scan_content(
                        item.get("content"), allow_redact
                    )
        return value

    async def async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: DualCache,
        data: dict,
        call_type: str,
    ) -> Optional[Union[Exception, str, dict]]:
        """Input guardrail: scan the prompt — chat ``messages`` and/or the
        Responses ``input`` field — before it reaches the LLM."""
        messages = data.get("messages")
        if isinstance(messages, list):
            data["messages"] = await self._scan_messages(messages)
        if "input" in data:
            data["input"] = await self._scan_input(data.get("input"))
        return data

    async def async_moderation_hook(
        self,
        data: dict,
        user_api_key_dict: UserAPIKeyAuth,
        call_type: CallTypesLiteral,
    ) -> Any:
        """Parallel guardrail (block-only): runs alongside the LLM call. Cannot
        rewrite content, so redact is not applied — only block raises."""
        messages = data.get("messages")
        if isinstance(messages, list):
            await self._scan_messages(messages, allow_redact=False)
        if "input" in data:
            await self._scan_input(data.get("input"), allow_redact=False)
        return data

    async def async_post_call_success_hook(
        self,
        data: dict,
        user_api_key_dict: UserAPIKeyAuth,
        response: Any,
    ) -> Any:
        """Output guardrail: scan the model response. Block raises; redact
        rewrites the assistant message content."""
        if not isinstance(response, ModelResponse):
            return response
        for choice in getattr(response, "choices", []) or []:
            message = getattr(choice, "message", None)
            if message is None:
                continue
            content = getattr(message, "content", None)
            if isinstance(content, str) and content.strip():
                message.content = await self._scan_text(content)
        return response

    @log_guardrail_information
    async def apply_guardrail(
        self,
        inputs: "GenericGuardrailAPIInputs",
        request_data: dict,
        input_type: Literal["request", "response"],
        logging_obj: Optional["LiteLLMLoggingObj"] = None,
    ) -> "GenericGuardrailAPIInputs":
        """Text-in / text-out surface used by the unified guardrail API.
        Scans each entry in ``inputs['texts']``: block raises; redact rewrites
        the text in place; allow returns it untouched."""
        texts = inputs.get("texts") or []
        for i, text in enumerate(texts):
            scanned = await self._scan_text(text)
            if isinstance(scanned, str):
                texts[i] = scanned
        return inputs
