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

from typing import TYPE_CHECKING, Any, List, Literal, Optional, Type, Union

from fastapi import HTTPException

from litellm import DualCache
from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.llms.custom_httpx.http_handler import (
    get_async_httpx_client,
    httpxSpecialProvider,
)
from litellm.proxy._types import UserAPIKeyAuth
from litellm.types.utils import ModelResponse

if TYPE_CHECKING:
    from litellm.types.proxy.guardrails.guardrail_hooks.base import (
        GuardrailConfigModel,
    )

# Veto returns action ∈ {allow, redact, block}. block → reject the call;
# redact → swap message content for the masked text and continue; allow → pass.
VETO_DEFAULT_CATEGORIES = ["pii", "secrets", "injection"]


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
        observability; lifecycle owned by the global client cache). The handler
        raises for non-2xx and retries transient connection errors internally.
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
        return resp.json()

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
        call_type: Literal[
            "completion",
            "embeddings",
            "image_generation",
            "moderation",
            "audio_transcription",
            "responses",
        ],
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
            content = getattr(message, "content", None)
            if isinstance(content, str) and content.strip():
                message.content = await self._scan_text(content)
        return response

    async def apply_guardrail(
        self,
        text: str,
        language: Optional[str] = None,
        entities: Optional[List[Any]] = None,
        request_data: Optional[dict] = None,
    ) -> str:
        """Text-in / text-out surface used by the unified guardrail API.
        Block raises; redact returns the masked text; allow returns input."""
        return await self._scan_text(text)
