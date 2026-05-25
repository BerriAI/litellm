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

import httpx

from litellm import DualCache
from litellm.integrations.custom_guardrail import CustomGuardrail
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
        categories: Optional[List[str]] = None,
        **kwargs,
    ):
        self.api_base = (api_base or "https://api.vetocheck.com").rstrip("/")
        self.api_key = api_key
        self.categories = categories or VETO_DEFAULT_CATEGORIES
        self.client = httpx.AsyncClient(timeout=10.0)
        super().__init__(**kwargs)

    @staticmethod
    def get_config_model() -> Optional[Type["GuardrailConfigModel"]]:
        from litellm.types.proxy.guardrails.guardrail_hooks.veto import (
            VetoGuardrailConfigModel,
        )

        return VetoGuardrailConfigModel

    async def _check(self, text: str) -> dict:
        """POST one text to the Veto gateway. Returns the verdict JSON."""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        resp = await self.client.post(
            f"{self.api_base}/v1/check",
            headers=headers,
            json={"text": text, "categories": self.categories},
        )
        resp.raise_for_status()
        return resp.json()

    def _raise_blocked(self, verdict: dict) -> None:
        from fastapi import HTTPException

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

    async def _scan_messages(self, messages: List[dict]) -> List[dict]:
        """Scan each string-content message. Block raises; redact rewrites the
        content in place. Returns the (possibly redacted) messages."""
        for msg in messages:
            content = msg.get("content")
            if not isinstance(content, str) or not content.strip():
                continue
            verdict = await self._check(content)
            action = verdict.get("action")
            if action == "block":
                self._raise_blocked(verdict)
            elif action == "redact":
                msg["content"] = verdict.get("redacted", content)
        return messages

    async def async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: DualCache,
        data: dict,
        call_type: str,
    ) -> Optional[Union[Exception, str, dict]]:
        """Input guardrail: scan the prompt before it reaches the LLM."""
        messages = data.get("messages")
        if isinstance(messages, list):
            data["messages"] = await self._scan_messages(messages)
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
        rewrite content here, so a redact verdict is logged, not applied."""
        messages = data.get("messages")
        if not isinstance(messages, list):
            return data
        for msg in messages:
            content = msg.get("content")
            if not isinstance(content, str) or not content.strip():
                continue
            verdict = await self._check(content)
            if verdict.get("action") == "block":
                self._raise_blocked(verdict)
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
            if not isinstance(content, str) or not content.strip():
                continue
            verdict = await self._check(content)
            action = verdict.get("action")
            if action == "block":
                self._raise_blocked(verdict)
            elif action == "redact":
                message.content = verdict.get("redacted", content)
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
        verdict = await self._check(text)
        if verdict.get("action") == "block":
            self._raise_blocked(verdict)
        if verdict.get("action") == "redact":
            return verdict.get("redacted", text)
        return text
